# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils, checkpoint_utils, tasks
from fairseq.models import (
    register_model,
    register_model_architecture,
)

from .wav2vec2_seq2seq import (
    TransformerModel,
    Wav2VecEncoder,
    TransformerDecoder,
    TransformerCTCShrinkModel,
    Embedding,
    FCDecoder,
    seq2seq_architecture,
    add_common_args,
    add_decoder_args
)

PAD_IDX = 1
EOS_IDX = 2


def NonlinearLayer(in_features, out_features, bias=True, activation_fn=nn.ReLU):
    """Weight-normalized non-linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return nn.Sequential(m, activation_fn())


def build_embedding(dictionary, embed_dim):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    emb = Embedding(num_embeddings, embed_dim, padding_idx)

    return emb


def add_mixer_args(parser):
    parser.add_argument(
        "--dim-hidden-mixer", type=int, help="dim-hidden-mixer"
    )
    parser.add_argument(
        "--freeze-lm-finetune-updates", type=int, help="freeze_lm_finetune_updates"
    )
    parser.add_argument(
        "--lm-path", type=str, help="dim-hidden-mixer"
    )


@register_model("wav2vec_seq2seq_lm")
class TransformerLMModel(TransformerModel):

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_decoder_args(parser)
        add_mixer_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        w2v_seq2seq_lm_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary

        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)
        encoder = cls.build_encoder(args)
        lm = cls.build_lm(args, tgt_dict)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens, lm)

        return cls(args, encoder, decoder)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, lm):
        return TransformerDecoder_DF(args, tgt_dict, embed_tokens, lm)

    @classmethod
    def build_lm(cls, args, dictionary):
        from fairseq.models.lstm_lm import LSTMLanguageModel
        return LSTMLanguageModel.build_model(args, task=None, dictionary=dictionary)


@register_model("wav2vec_seq2seq_lm_coldfusion")
class TransformerLMColdFusionModel(TransformerLMModel):

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        w2v_seq2seq_lm_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)

            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)
        encoder = cls.build_encoder(args)
        lm = cls.build_lm(args, tgt_dict)
        dim_phone, dim_hidden, vocab_size = encoder.d, 1024, lm.dim_output
        hidden_layer = NonlinearLayer(
            vocab_size, dim_hidden, bias=False, activation_fn=nn.ReLU
        )
        gating_network = NonlinearLayer(
            dim_hidden + dim_phone,
            dim_hidden,
            bias=True,
            activation_fn=nn.Sigmoid,
        )
        output_projections = NonlinearLayer(
            dim_hidden + dim_phone,
            vocab_size,
            bias=False,
            activation_fn=nn.ReLU,
        )
        decoder = cls.build_decoder(
            args, tgt_dict, decoder_embed_tokens,
            hidden_layer, gating_network, output_projections)
        return cls(args, encoder, decoder, lm)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens,
                      hidden_layer, gating_network, output_projections):
        return TransformerDecoder_ColdFusion(args, tgt_dict, embed_tokens,
                                             hidden_layer, gating_network, output_projections)

    @classmethod
    def build_lm(cls, args, dictionary):
        from fairseq.models.lstm_lm import LSTMLanguageModel
        return LSTMLanguageModel.build_model(args, task=None, dictionary=dictionary)


@register_model("wav2vec_ctc_shrink_seq2seq_lm")
class TransformerCTCShrinkLMModel(TransformerCTCShrinkModel):

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_decoder_args(parser)
        add_mixer_args(parser)
        parser.add_argument(
            "--w2v-ctc-path", type=str, help="is teacher-forcing"
        )

    def __init__(self, args, encoder, decoder, dictionary):
        super().__init__(args, encoder, decoder, dictionary)

    @classmethod
    def build_decoder(cls, args, hidden_layer, gating_network, output_projections, lm):
        return ColdFusion(args, hidden_layer, gating_network, output_projections, lm)

    @classmethod
    def build_lm(cls, args, dictionary):
        from fairseq.models.lstm_lm import LSTMLanguageModel
        return LSTMLanguageModel.build_model(args, task=None, dictionary=dictionary)

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

        encoder = cls.build_encoder(args, tgt_dict)
        lm = cls.build_lm(args, tgt_dict)

        dim_phone, dim_hidden, vocab_size = encoder.d, 1024, lm.dim_output
        hidden_layer = NonlinearLayer(
            vocab_size, dim_hidden, bias=False, activation_fn=nn.ReLU
        )
        gating_network = NonlinearLayer(
            dim_hidden + dim_phone,
            dim_hidden,
            bias=True,
            activation_fn=nn.Sigmoid,
        )
        output_projections = NonlinearLayer(
            dim_hidden + dim_phone,
            vocab_size,
            bias=False,
            activation_fn=nn.ReLU,
        )
        decoder = cls.build_decoder(
            args, hidden_layer, gating_network, output_projections, lm)

        return cls(args, encoder, decoder, tgt_dict)

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
        incremental_state = ({}, {})

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


class TransformerDecoder_DF(TransformerDecoder):
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

    def __init__(self, args, dictionary, embed_tokens, lm, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.lm = lm
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.num_updates = 0
        self.output_embed_dim = args.decoder_embed_dim + self.lm.dim_output
        self.embed_out = nn.Parameter(
            torch.Tensor(len(dictionary), self.output_embed_dim)
        )
        nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

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
        if incremental_state:
            decoder_incremental_state, lm_incremental_state = incremental_state
        else:
            decoder_incremental_state, lm_incremental_state = None, None
        prev_output_tokens = prev_output_tokens.long()
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, decoder_incremental_state
        )

        ft = self.freeze_lm_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            logits_lm, _ = self.lm(prev_output_tokens,
                                   incremental_state=lm_incremental_state)
        x = torch.cat([x, logits_lm], -1)
        x = self.output_layer(x)
        extra["logits"] = x

        return extra

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class TransformerDecoder_ColdFusion(TransformerDecoder_DF):

    def __init__(self, args, dictionary, embed_tokens,
                 hidden_layer, gating_network, output_projections,
                 no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        self.hidden_layer = hidden_layer
        self.gating_network = gating_network
        self.output_projections = output_projections

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
        if incremental_state:
            decoder_incremental_state, lm_incremental_state = incremental_state
        else:
            decoder_incremental_state, lm_incremental_state = None, None
        prev_output_tokens = prev_output_tokens.long()
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, decoder_incremental_state
        )

        ft = self.freeze_lm_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            logits_lm, _ = self.lm(prev_output_tokens,
                                   incremental_state=lm_incremental_state)

        pred_lm = F.softmax(logits_lm, -1)
        h_lm = self.hidden_layer(pred_lm)
        h_am_lm = torch.cat([x, h_lm], -1)
        g = self.gating_network(h_am_lm)
        h_cf = torch.cat([x, g * h_lm], -1)
        logits = self.output_projections(h_cf)

        extra["logits"] = logits

        return extra


class ColdFusion():

    def __init__(self, args, hidden_layer, gating_network, output_projections, lm):
        self.lm = lm
        self.hidden_layer = hidden_layer
        self.gating_network = gating_network
        self.output_projections = output_projections

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
        x = encoder_out.encoder_out.transpose(0,1)

        ft = self.freeze_lm_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            logits_lm, _ = self.lm(prev_output_tokens,
                                   incremental_state=incremental_state)

        pred_lm = F.softmax(logits_lm, -1)
        h_lm = self.hidden_layer(pred_lm)
        h_am_lm = torch.cat([x, h_lm], -1)
        g = self.gating_network(h_am_lm)
        h_cf = torch.cat([x, g * h_lm], -1)
        logits = self.output_projections(h_cf)

        return {'logits': logits}

@register_model_architecture("wav2vec_seq2seq_lm", "wav2vec_seq2seq_lm")
def w2v_seq2seq_lm_architecture(args):
    seq2seq_architecture(args)
    args.freeze_lm_finetune_updates = getattr(args, "freeze_lm_finetune_updates", 1000)


@register_model_architecture("wav2vec_seq2seq_lm_coldfusion", "wav2vec_seq2seq_lm_coldfusion")
def w2v_seq2seq_lm_CF_architecture(args):
    w2v_seq2seq_lm_architecture(args)

@register_model_architecture("wav2vec_ctc_shrink_seq2seq_lm", "wav2vec_ctc_shrink_seq2seq_lm")
def w2v_ctc_shrink_seq2seq_lm_architecture(args):
    w2v_seq2seq_lm_architecture(args)
