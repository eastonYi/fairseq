# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq.models.fairseq_encoder import EncoderOut
from fairseq import utils, checkpoint_utils, tasks
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
    TransposeLast,
    Fp32LayerNorm,
    FairseqDropout,
    Fp32GroupNorm
)

from .wav2vec2_ctc import add_common_args, Wav2VecEncoder, Linear, base_architecture

PAD_IDX = 1
EOS_IDX = 2
eps = 1e-7


def build_embedding(dictionary, embed_dim):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    emb = Embedding(num_embeddings, embed_dim, padding_idx)

    return emb


def add_decoder_args(parser):
    parser.add_argument(
        "--decoder-embed-dim",
        type=int,
        metavar="N",
        help="decoder embedding dimension",
    )
    parser.add_argument(
        "--decoder-ffn-embed-dim",
        type=int,
        metavar="N",
        help="decoder embedding dimension for FFN",
    )
    parser.add_argument(
        "--decoder-layers", type=int, metavar="N", help="num decoder layers"
    )
    parser.add_argument(
        "--decoder-layerdrop",
        type=float,
        metavar="D",
        help="decoder layerdrop chance",
    )
    parser.add_argument(
        "--decoder-attention-heads",
        type=int,
        metavar="N",
        help="num decoder attention heads",
    )
    parser.add_argument(
        "--decoder-learned-pos",
        action="store_true",
        help="use learned positional embeddings in the decoder",
    )
    parser.add_argument(
        "--decoder-normalize-before",
        action="store_true",
        help="apply layernorm before each decoder block",
    )
    parser.add_argument(
        "--no-token-positional-embeddings",
        default=False,
        action="store_true",
        help="if set, disables positional embeddings (outside self attention)",
    )

    parser.add_argument(
        "--decoder-dropout",
        type=float,
        metavar="D",
        help="dropout probability in the decoder",
    )
    parser.add_argument(
        "--decoder-attention-dropout",
        type=float,
        metavar="D",
        help="dropout probability for attention weights inside the decoder",
    )
    parser.add_argument(
        "--decoder-activation-dropout",
        type=float,
        metavar="D",
        help="dropout probability after activation in FFN inside the decoder",
    )
    parser.add_argument(
        "--teacher-forcing-updates", type=int, help="is teacher-forcing"
    )


@register_model("wav2vec_seq2seq")
class TransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.num_updates = 0
        self.teacher_forcing_updates = args.teacher_forcing_updates

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_decoder_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        seq2seq_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary

        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)

        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, tgt_dict=None):
        return Wav2VecEncoder(args, tgt_dict=tgt_dict)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(args, tgt_dict, embed_tokens)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, **kwargs):
        encoder_out = self.encoder(tbc=False, **kwargs)
        # decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)
        if self.num_updates <= self.teacher_forcing_updates:
            decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)
        else:
            with torch.no_grad():
                decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)
                decoded = decoder_out[0].argmax(-1)
            encoder_out.prev_output_tokens = decoded
            decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)

        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output["logits"]
        if log_probs:
            res = utils.log_softmax(logits.float(), dim=-1)
        else:
            res = utils.softmax(logits.float(), dim=-1)
        res.batch_first = True

        return res

    def get_encoder_output(self, net_input):
        encoder_out = self.encoder(tbc=True, **net_input)
        return EncoderOut(
            encoder_out=encoder_out['encoder_out'],  # T x B x C
            encoder_embedding=None,
            encoder_padding_mask=encoder_out['encoder_padding_mask'],  # B x T
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )


@register_model("wav2vec_ctc_seq2seq")
class TransformerCTCModel(TransformerModel):

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        seq2seq_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary

        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)

        encoder = cls.build_encoder(args, tgt_dict)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    def forward(self, **kwargs):
        encoder_out = self.encoder(tbc=False, **kwargs)
        ctc_logits = encoder_out["encoder_out"]
        encoder_out["encoder_out"] = encoder_out["encoded"]
        logits, _ = self.decoder(encoder_out=encoder_out, **kwargs)
        encoder_out["ctc_logits"] = ctc_logits

        return encoder_out, logits

    def get_normalized_probs(self, ctc_logits, logits, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        if log_probs:
            ctc_res = utils.log_softmax(ctc_logits.float(), dim=-1)
            res = utils.log_softmax(logits.float(), dim=-1)
        else:
            ctc_res = utils.softmax(ctc_logits.float(), dim=-1)
            res = utils.softmax(logits.float(), dim=-1)
        ctc_res.batch_first = True
        res.batch_first = True

        return ctc_res, res


@register_model("wav2vec_ctc_shrink_seq2seq")
class TransformerCTCShrinkModel(TransformerModel):

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_decoder_args(parser)
        parser.add_argument(
            "--w2v-ctc-path", type=str, help="is teacher-forcing"
        )

    def __init__(self, args, encoder, decoder, dictionary):
        self.dictionary = dictionary
        self.blank_id = dictionary.index("<ctc_blank>")
        if getattr(args, "w2v_ctc_path", None):
            print('load Wav2VecEncoder from {}'.format(args.w2v_ctc_path))
            state = checkpoint_utils.load_checkpoint_to_cpu(args.w2v_ctc_path)
            w2v_ctc_args = state["args"]
            assert getattr(w2v_ctc_args, "w2v_ctc_path", None) is None # w2v_path is the pretrain model which should not have w2v_path
            task = tasks.setup_task(w2v_ctc_args)
            encoder = task.build_model(w2v_ctc_args)
            print('restore w2v_ctc from {}'.format(args.w2v_ctc_path))
            encoder.load_state_dict(state["model"], strict=True)

        super().__init__(args, encoder, decoder)

    @classmethod
    def build_decoder(cls, args, tgt_dict, input_dim):
        return FCDecoder(args, tgt_dict, input_dim)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        seq2seq_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary
        tgt_dict.add_symbol("<ctc_blank>")

        # decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)
        encoder = cls.build_encoder(args, tgt_dict)
        decoder = cls.build_decoder(args, tgt_dict, args.decoder_embed_dim)
        # decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

        return cls(args, encoder, decoder, tgt_dict)

    def forward(self, **kwargs):
        encoder_output = self.encoder(tbc=False, **kwargs)
        encoder_shrunk_output = self.ctc_shrink(encoder_output)
        decoder_output = self.decoder(encoder_out=encoder_shrunk_output, **kwargs)

        return decoder_output

    def ctc_shrink(self, encoder_output):
        from fairseq.utils import ctc_shrink, sequence_mask

        encoder_padding_mask = encoder_output["encoder_padding_mask"]
        encoded_shrunk, len_decode = ctc_shrink(
            hidden=encoder_output["encoded"],
            logits=encoder_output["encoder_out"],
            len_logits=(~encoder_padding_mask).int().sum(-1),
            blk=self.blank_id)

        encoded_shrunk_padding_mask = ~sequence_mask(len_decode, dtype=torch.bool)

        return EncoderOut(
            encoder_out=encoded_shrunk.transpose(0,1),  # T x B x C
            encoder_embedding=None,
            encoder_padding_mask=encoded_shrunk_padding_mask,  # B x T
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )

    def get_normalized_probs(self, logits_ctc, logits_ce, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        if log_probs:
            ctc_res = utils.log_softmax(logits_ctc.float(), dim=-1)
            res = utils.log_softmax(logits_ce.float(), dim=-1)
        else:
            ctc_res = utils.softmax(logits_ctc.float(), dim=-1)
            res = utils.softmax(logits_ce.float(), dim=-1)
        ctc_res.batch_first = True
        res.batch_first = True

        return ctc_res, res

    def batch_greedy_decode(self, encoder_output, SOS_ID, vocab_size, max_decode_len=100):
        """
        encoder_output:
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        """
        len_encoded = (~encoder_output.encoder_padding_mask).sum(-1)
        max_decode_len = len_encoded.max()
        batch_size = len_encoded.size(0)
        device = encoder_output.encoder_out.device
        d_output = vocab_size

        preds = torch.ones([batch_size, 1]).long().to(device) * SOS_ID
        logits = torch.zeros([batch_size, 0, d_output]).float().to(device)
        incremental_state = {}

        for t in range(max_decode_len):
            # cur_logits: [B, 1, V]
            cur_logits = self.decoder(prev_output_tokens=preds,
                                      encoder_out=encoder_output,
                                      incremental_state=incremental_state)

            cur_logits = cur_logits["logits"]
            assert cur_logits.size(1) == 1
            logits = torch.cat([logits, cur_logits], 1)  # [batch, t, size_output]
            z = F.log_softmax(cur_logits[:, 0, :], dim=-1) # [batch, size_output]

            # rank the combined scores
            next_scores, next_preds = torch.topk(z, k=1, sorted=True, dim=-1)
            next_preds = next_preds.squeeze(-1)

            preds = torch.cat([preds, next_preds[:, None]], axis=1)  # [batch_size, i]

        preds = preds[:, 1:]

        return logits, preds


@register_model("wav2vec_cif")
class CIFModel(TransformerModel):
    def __init__(self, args, encoder, assigner, decoder):
        super().__init__(args, encoder, decoder)
        self.assigner = assigner

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_decoder_args(parser)
        parser.add_argument(
            "--assigner-conv-layers",
            type=str,
            metavar="EXPR",
            help="convolutional feature extraction layers [(dim, kernel_size, stride), ...]",
        )
        parser.add_argument("--lambda-qua", type=float, metavar="D", help="lambda-qua")
        parser.add_argument("--lambda-cp", type=float, metavar="D", help="lambda-cp")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        cif_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary

        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)

        encoder = cls.build_encoder(args)
        assigner = cls.build_assigner(args, encoder.d)
        # decoder = cls.build_decoder(args, tgt_dict, encoder.d)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

        return cls(args, encoder, assigner, decoder)

    @classmethod
    def build_assigner(cls, args, dim_input):
        return Assigner(args, dim_input)

    # @classmethod
    # def build_decoder(cls, args, tgt_dict, input_dim):
        # return FCDecoder(args, tgt_dict, input_dim)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(args, tgt_dict, embed_tokens)

    def forward(self, **kwargs):
        encoder_output = self.encoder(tbc=False, **kwargs)
        alphas = self.assigner(encoder_output)
        _alphas, num_output = self.resize(alphas, kwargs['target_lengths'])
        cif_outputs = self.cif(encoder_output, _alphas)

        if self.training and cif_outputs.abs().sum(-1).ne(0).sum() != kwargs['target_lengths'].sum():
            import pdb; pdb.set_trace()

        if self.num_updates <= self.teacher_forcing_updates:
            logits = self.decode(encoded_shrunk=cif_outputs,
                                 prev_output_tokens=kwargs["prev_output_tokens"])
        else:
            logits = self.decode(encoded_shrunk=cif_outputs)

        return {'logits': logits, 'len_logits': kwargs['target_lengths'],
                'alphas': alphas, 'num_output': num_output}

    def cif(self, encoder_output, alphas, threshold=0.9, log=False):
        hidden = encoder_output['encoder_out']
        device = hidden.device
        B, T, H = hidden.size()

        # loop varss
        integrate = torch.zeros([B]).to(device)
        frame = torch.zeros([B, H]).to(device)
        # intermediate vars along time
        list_fires = []
        list_frames = []

        for t in range(T):
            alpha = alphas[:, t]
            distribution_completion = torch.ones([B]).to(device) - integrate

            integrate += alpha
            list_fires.append(integrate)

            fire_place = integrate > threshold
            integrate = torch.where(fire_place,
                                    integrate - torch.ones([B]).to(device),
                                    integrate)
            cur = torch.where(fire_place,
                              distribution_completion,
                              alpha)
            remainds = alpha - cur

            frame += cur[:, None] * hidden[:, t, :]
            list_frames.append(frame)
            frame = torch.where(fire_place[:, None].repeat(1, H),
                                remainds[:, None] * hidden[:, t, :],
                                frame)
            if log:
                print('t: {}\t{:.3f} -> {:.3f}|{:.3f} fire: {}'.format(
                    t, integrate[log], cur[log], remainds[log], fire_place[log]))

        fires = torch.stack(list_fires, 1)
        frames = torch.stack(list_frames, 1)
        list_ls = []
        len_labels = torch.round(alphas.sum(-1)).int()
        max_label_len = len_labels.max()
        for b in range(B):
            fire = fires[b, :]
            l = torch.index_select(frames[b, :, :], 0, torch.where(fire > threshold)[0])
            pad_l = torch.zeros([max_label_len - l.size(0), H]).to(device)
            list_ls.append(torch.cat([l, pad_l], 0))

        if log:
            print('fire:\n', fires)
            print('fire place:\n', torch.where(fires > threshold))

        return torch.stack(list_ls, 0)

    @staticmethod
    def resize(alphas, target_lengths, noise=0.095, threshold=0.9):
        """
        alpha in thresh=0.9 | (0.91, noise-0.09)
        """
        device = alphas.device
        # sum
        _num = alphas.sum(-1)

        # scaling
        num = target_lengths.float()
        num_noise = num + 2 * noise * torch.rand(alphas.size(0)).to(device) - noise
        if (torch.round(num) < 1).float().sum() > 0 or \
           (torch.round(num_noise) < 1).float().sum() > 0:
            import pdb; pdb.set_trace()
        _alphas = alphas * (num_noise / _num)[:, None].repeat(1, alphas.size(1))

        # rm attention value that exceeds threashold
        while len(torch.where(_alphas >= threshold)[0]):
            xs, ys = torch.where(_alphas >= threshold)
            for x, y in zip(xs, ys):
                if _alphas[x][y] >= threshold:
                    mask = _alphas[x].ne(0).float()
                    mean = 0.5 * _alphas[x].sum() / mask.sum()
                    _alphas[x] = _alphas[x] * 0.5 + mean * mask

        return _alphas, _num

    def decode(self, encoded_shrunk, prev_output_tokens=None, incremental_states=None, t=None):
        encoder_padding_mask = encoded_shrunk.abs().sum(-1).eq(0)
        encoder_output = EncoderOut(
            encoder_out=encoded_shrunk.transpose(0, 1),  # T x B x C
            encoder_embedding=None,
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )

        if incremental_states is not None: # step forward
            assert prev_output_tokens is not None, t is not None
            cur_logits = self.decoder(prev_output_tokens=prev_output_tokens,
                                      encoder_out=encoder_output,
                                      incremental_state=incremental_states)
            logits = cur_logits["logits"]
        elif prev_output_tokens is not None: # teacher-forcing
            logits = self.decoder(prev_output_tokens=prev_output_tokens,
                                  encoder_out=encoder_output)
            logits = logits["logits"]

        else: # self-decode
            B, T, V = encoded_shrunk.size()
            device = encoded_shrunk.device
            prev_decoded = torch.ones([B, 1]).to(device).long() * EOS_IDX
            list_logits = []
            incremental_state = {}
            for _ in torch.range(T):
                cur_logits = self.decoder(prev_output_tokens=prev_decoded,
                                          encoder_out=encoder_output,
                                          incremental_state=incremental_state)
                list_logits.append(cur_logits)
                cur_token = cur_logits.argmax(-1, keepdim=True)
                prev_decoded = torch.cat([prev_decoded, cur_token], 1)

            logits = torch.stack(list_logits, 1)

        return logits


@register_model("wav2vec_cif_rnn")
class CIF_RNN_Model(CIFModel):

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        cif_rnn_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary

        encoder = cls.build_encoder(args)
        assigner = cls.build_assigner(args, encoder.d)
        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)
        decoder = cls.build_decoder(args,
                                    embed_tokens=decoder_embed_tokens,
                                    dim_input=encoder.d,
                                    dim_output=len(tgt_dict))

        return cls(args, encoder, assigner, decoder)

    @classmethod
    def build_decoder(cls, args, dim_input, dim_output, embed_tokens):
        decoder = LSTMDecoder(
            embed_tokens=embed_tokens,
            dim_input=dim_input,
            hidden_size=dim_input,
            dim_output=dim_output,
            num_layers=2,
        )
        return decoder

    def decode(self, encoded_shrunk, prev_output_tokens=None, incremental_states=None, t=None):

        encoder_output = encoded_shrunk
        if incremental_states is not None: # step forward
            assert prev_output_tokens is not None, t is not None
            cur_logits = self.decoder(prev_output_tokens=prev_output_tokens,
                                      encoder_out=encoder_output,
                                      incremental_state=incremental_states)
            logits = cur_logits
        elif prev_output_tokens is not None: # teacher-forcing
            logits = self.decoder(prev_output_tokens=prev_output_tokens,
                                  encoder_out=encoder_output)

        else: # self-decode
            B, T, V = encoded_shrunk.size()
            device = encoded_shrunk.device
            prev_decoded = torch.ones([B, 1]).to(device).long() * EOS_IDX
            list_logits = []
            incremental_state = {}
            for _ in torch.range(T):
                cur_logits = self.decoder(prev_output_tokens=prev_decoded,
                                          encoder_out=encoder_output,
                                          incremental_state=incremental_state)
                list_logits.append(cur_logits)
                cur_token = cur_logits.argmax(-1, keepdim=True)
                prev_decoded = torch.cat([prev_decoded, cur_token], 1)

            logits = torch.stack(list_logits, 1)

        return logits


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~dataload.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)

        self.dropout = args.decoder_dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_embed_dim
        args.encoder_embed_dim = embed_dim

        self.layerdrop = args.decoder_layerdrop

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings else None
        )

        args = copy.deepcopy(args)
        args.dropout = args.decoder_dropout
        args.attention_dropout = args.decoder_attention_dropout
        args.activation_dropout = args.decoder_activation_dropout

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        prev_output_tokens = prev_output_tokens.long()
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        # x = self.output_layer(encoder_out.encoder_out.transpose(0,1) + x)
        x = self.output_layer(x)
        extra["logits"] = x

        return extra

    def extract_features(
        self, prev_output_tokens, encoder_output=None, incremental_state=None, **unused
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, attn, _ = layer(
                    x,
                    encoder_output.encoder_out if encoder_output is not None else None,
                    encoder_output.encoder_padding_mask if encoder_output is not None else None,
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x)
                    if incremental_state is None else None,
                )
                inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": attn, "inner_states": inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


from fairseq.models.lstm import LSTMCell
class LSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""
    def __init__(
        self, embed_tokens, dim_input, hidden_size=512, dim_output=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, residuals=False,
    ):
        super().__init__(None)
        self.embed_tokens = embed_tokens
        self.dropout_in_module = FairseqDropout(dropout_in, module_name=self.__class__.__name__)
        self.dropout_out_module = FairseqDropout(dropout_out, module_name=self.__class__.__name__)
        self.hidden_size = hidden_size
        self.residuals = residuals
        self.num_layers = num_layers
        self.max_target_positions = 200

        self.encoder_output_units = dim_input
        if dim_input != hidden_size and dim_input != 0:
            self.encoder_hidden_proj = Linear(dim_input, hidden_size)
        else:
            self.encoder_hidden_proj = None

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if dim_input == 0 else hidden_size
        embed_dim = embed_tokens.embedding_dim
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=input_feed_size + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])

        if hidden_size != dim_output:
            self.fc_out = Linear(hidden_size, dim_output)
        else:
            self.fc_out = None

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        src_lengths: Optional[Tensor] = None,
    ):
        x = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        return self.output_layer(x)

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Similar to *forward* but only return features.
        """
        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, seqlen = prev_output_tokens.size()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens.long())
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        input_feed = encoder_out.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells = self.get_cached_state(incremental_state)
        else:
            # setup recurrent cells
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(y) for y in prev_hiddens]

        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j], input_feed[j]), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = self.dropout_out_module(hidden)
                if self.residuals:
                    input = input + prev_hiddens[i]

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            out = hidden
            out = self.dropout_out_module(out)

            # save final output
            outs.append(out)

        # Stack all the necessary tensors together and store
        prev_hiddens_tensor = torch.stack(prev_hiddens)
        prev_cells_tensor = torch.stack(prev_cells)
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": prev_hiddens_tensor,
                "prev_cells": prev_cells_tensor,
                "input_feed": input_feed,
            }
        )
        self.set_incremental_state(incremental_state, 'cached_state', cache_state)
        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen

        return x

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if self.fc_out:
            x = self.fc_out(x)

        return x

    def get_cached_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
    ) -> Tuple[List[Tensor], List[Tensor], Optional[Tensor]]:
        cached_state = self.get_incremental_state(incremental_state, 'cached_state')
        assert cached_state is not None
        prev_hiddens_ = cached_state["prev_hiddens"]
        assert prev_hiddens_ is not None
        prev_cells_ = cached_state["prev_cells"]
        assert prev_cells_ is not None
        prev_hiddens = [prev_hiddens_[i] for i in range(self.num_layers)]
        prev_cells = [prev_cells_[j] for j in range(self.num_layers)]
        return prev_hiddens, prev_cells

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        if incremental_state is None or len(incremental_state) == 0:
            return
        prev_hiddens, prev_cells = self.get_cached_state(incremental_state)
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]
        prev_cells = [p.index_select(0, new_order) for p in prev_cells]
        cached_state_new = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": torch.stack(prev_hiddens),
                "prev_cells": torch.stack(prev_cells)
            }
        )
        self.set_incremental_state(incremental_state, 'cached_state', cached_state_new),
        return

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions


class Assigner(FairseqEncoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~dataload.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dim_input):
        super().__init__()
        assigner_conv_layers = eval(args.assigner_conv_layers)
        self.embed = assigner_conv_layers[-1][0]
        self.feature_extractor = Conv2DFeatureExtractionModel(
            dim_input=dim_input,
            conv_layers=assigner_conv_layers,
            dropout=0.1,
            mode=args.extractor_mode,
            conv_bias=True,
            output='same'
        )
        self.proj = Linear(self.embed, 1)

    def forward(self, encoder_output):
        """
        Args:
            encoder_out (FloatTensor): previous decoder outputs of shape
                `(B, T, H)`, for teacher forcing
            encoded_lengths (Tensor): output from the encoder, used for
                encoder-side attention
        Returns:
            the decoder's output of shape `(batch, src_len)`
        """
        encoded, padding_mask = encoder_output['encoder_out'], encoder_output['padding_mask']

        x = self.feature_extractor(encoded)
        x = self.proj(x)[:, :, 0]
        x = torch.sigmoid(x) + eps
        x = x * (~padding_mask)

        return x


class FCDecoder(FairseqDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~dataload.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, input_dim):
        super().__init__(dictionary)
        self.proj = Linear(input_dim, len(dictionary), bias=True)

    def forward(self, encoded):
        """
        Args:
            encoder_out (Tensor): output from the encoder, used for
                encoder-side attention
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
        """
        x = self.proj(encoded)
        return x


class Conv2DFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        dim_input: int,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        output: str = "valid", # ["valid", "same"]
    ):
        super().__init__()
        assert mode in {"default", "layer_norm"}
        assert output in {"valid", "same"}
        self.output = output

        def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = dim_input
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(in_d, dim, k, stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias)
            )
            in_d = dim

    def forward(self, x):
        if self.output == 'same':
            length = x.size(1)
            x = F.pad(x, [0,0,0,20,0,0])
        x = x.transpose(1,2)

        for conv in self.conv_layers:
            x = conv(x)

        x = x.transpose(1,2)

        if self.output == 'same':
            x = x[:, :length, :]

        return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def build_embedding(dictionary, embed_dim):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    emb = Embedding(num_embeddings, embed_dim, padding_idx)
    return emb


@register_model_architecture("wav2vec_seq2seq", "wav2vec_seq2seq")
def seq2seq_architecture(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_layers = getattr(args, "decoder_layers", 10)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.decoder_dropout = getattr(args, "decoder_dropout", 0)
    args.decoder_attention_dropout = getattr(args, "share-decoder-input-output-embed", 0)
    args.decoder_activation_dropout = getattr(args, "decoder_activation_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    base_architecture(args)


@register_model_architecture("wav2vec_ctc_seq2seq", "wav2vec_ctc_seq2seq")
def ctc_seq2seq_architecture(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.decoder_dropout = getattr(args, "decoder_dropout", 0)
    args.decoder_attention_dropout = getattr(args, "share-decoder-input-output-embed", 0)
    args.decoder_activation_dropout = getattr(args, "decoder_activation_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    base_architecture(args)


@register_model_architecture("wav2vec_ctc_shrink_seq2seq", "wav2vec_ctc_shrink_seq2seq")
def ctc_shrink_seq2seq_architecture(args):
    seq2seq_architecture(args)


@register_model_architecture("wav2vec_cif", "wav2vec_cif")
def cif_architecture(args):
    args.extractor_mode = getattr(args, "extractor_mode", 'default')
    args.conv_bias = getattr(args, "conv-bias", False)
    args.lambda_qua = getattr(args, "lambda_qua", 0.05)
    args.lambda_cp = getattr(args, "lambda_cp", 0.1)
    seq2seq_architecture(args)


@register_model_architecture("wav2vec_cif_rnn", "wav2vec_cif_rnn")
def cif_rnn_architecture(args):
    cif_architecture(args)
