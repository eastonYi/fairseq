# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Dict, Optional

from fairseq.models.fairseq_encoder import EncoderOut
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    BaseFairseqModel,
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
    Fp32GroupNorm,
    FairseqDropout
)

from .wav2vec2_ctc import add_common_args, base_architecture
from .wav2vec2_cif import (
    CIFFcModel,
    Assigner,
    cif_architecture,
)
from .wav2vec2_seq2seq import (
    LSTMDecoder,
    build_embedding,
    Wav2VecEncoder,
    Linear,
    add_decoder_args
)

PAD_IDX = 1
EOS_IDX = 2


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


@register_model("wav2vec_ctc_lm")
class W2V_CTC_LM(BaseFairseqModel):
    def __init__(self, args, encoder, lm):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.lm = lm
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.proj = Linear(encoder.d, lm.encoder.encoder.sentence_encoder.embedding_dim)

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_decoder_args(parser)
        parser.add_argument(
            "--freeze-lm-finetune-updates", type=int, help="freeze_lm_finetune_updates"
        )
        parser.add_argument(
            "--lm-path", type=str, help="dim-hidden-mixer"
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        global PAD_IDX, BLK_IDX, EOS_IDX
        # make sure all arguments are present in older models
        w2v_lm_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary
        PAD_IDX = tgt_dict.pad()
        EOS_IDX = tgt_dict.eos()
        BLK_IDX = tgt_dict.index("<ctc_blank>")

        encoder = cls.build_encoder(args, tgt_dict)
        lm = cls.build_lm(args, tgt_dict)

        return cls(args, encoder, lm)

    @classmethod
    def build_encoder(cls, args, tgt_dict=None):
        return Wav2VecEncoder(args, tgt_dict)

    @classmethod
    def build_lm(cls, args, dictionary):
        from fairseq.models.lstm_lm import LSTMLanguageModel
        return LSTMLanguageModel.build_model(args, task=None, dictionary=dictionary)

    def forward(self, **kwargs):
        encoder_output = self.encoder(tbc=False, **kwargs)
        encoded = encoder_output['encoded']
        logits = encoder_output['encoder_out']
        encoded_shrunk, len_encoded_shrunk = utils.ctc_shrink(
            hidden=encoded, logits=logits, pad=PAD_IDX, blk=BLK_IDX, at_least_one=True)
        encoder_shrunk_out = {'encoded_shrunk': encoded_shrunk,
                              'len_encoded_shrunk': len_encoded_shrunk}
        logits = self.decode(encoder_shrunk_out)

        return {'logits': logits, 'len_logits': len_encoded_shrunk, 'encoder_output': encoder_output}

    def decode(self, encoder_shrunk_out):
        encoded_shrunk = encoder_shrunk_out["encoded_shrunk"]
        B, T, V = encoded_shrunk.size()
        device = encoded_shrunk.device
        prev_decoded = torch.ones([B, 1]).to(device).long() * EOS_IDX
        list_logits = []
        for encoded_t in torch.unbind(encoded_shrunk, 1):
            ft = self.freeze_lm_finetune_updates <= self.num_updates
            with torch.no_grad() if not ft else contextlib.ExitStack():
                cur_pred_raw, _ = self.lm(prev_decoded)
            cur_pred = self.mixer(torch.cat([encoded_t, cur_pred_raw[:, 0, :]], -1))
            list_logits.append(cur_pred)
            cur_token = cur_pred.argmax(-1, keepdim=True)
            prev_decoded = torch.cat([prev_decoded, cur_token], 1)

        logits = torch.stack(list_logits, 1)
        logits.batch_first = True

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        ctc_logits = net_output["encoder_output"]["encoder_out"]
        logits = net_output["logits"]

        if log_probs:
            ctc_res = utils.log_softmax(ctc_logits.float(), dim=-1)
            res = utils.log_softmax(logits.float(), dim=-1)
        else:
            ctc_res = utils.softmax(ctc_logits.float(), dim=-1)
            res = utils.softmax(logits.float(), dim=-1)

        return ctc_res, res


@register_model("wav2vec_ctc_bert")
class W2V_CTC_BERT(W2V_CTC_LM):

    def __init__(self, args, encoder, lm):
        super().__init__(args, encoder, lm)

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        parser.add_argument(
            "--freeze-lm-finetune-updates", type=int, help="freeze_lm_finetune_updates"
        )
        parser.add_argument(
            "--lm-path", type=str, help="dim-hidden-mixer"
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        global PAD_IDX, BLK_IDX, EOS_IDX
        # make sure all arguments are present in older models
        w2v_lm_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary
        PAD_IDX = tgt_dict.pad()
        EOS_IDX = tgt_dict.eos()
        BLK_IDX = tgt_dict.index("<ctc_blank>")

        encoder = cls.build_encoder(args, tgt_dict)
        lm = cls.build_lm(args, tgt_dict)

        return cls(args, encoder, lm)

    @classmethod
    def build_lm(cls, args, dictionary):
        from fairseq.models.masked_lm import MaskedLMModel as LM
        args.tokens_per_sample = 50
        return LM.build_model(args, task=None, dictionary=dictionary)

    def forward(self, **kwargs):
        encoder_output = self.encoder(tbc=False, **kwargs)
        encoded = encoder_output['encoded']
        logits = encoder_output['encoder_out']
        encoded_shrunk, len_encoded_shrunk = utils.ctc_shrink(
            hidden=logits, logits=logits, pad=PAD_IDX, blk=BLK_IDX, at_least_one=True)
        encoder_shrunk_out = {'encoded_shrunk': encoded_shrunk,
                              'len_encoded_shrunk': len_encoded_shrunk}

        logits = self.decode(encoder_shrunk_out)

        return {'logits': logits, 'len_logits': len_encoded_shrunk, 'encoder_output': encoder_output}

    def decode(self, encoder_shrunk_out):
        encoded_logits = encoder_shrunk_out["encoded_shrunk"]
        prob = torch.softmax(encoded_logits[:, :, :-1], -1)

        ft = self.freeze_lm_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            embedded = prob * self.lm.embedding
            logits = self.lm.forward_embeded(embedded)

        logits.batch_first = True

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        ctc_logits = net_output["encoder_output"]["encoder_out"]
        logits = net_output["logits"]

        if log_probs:
            ctc_res = utils.log_softmax(ctc_logits.float(), dim=-1)
            res = utils.log_softmax(logits.float(), dim=-1)
        else:
            ctc_res = utils.softmax(ctc_logits.float(), dim=-1)
            res = utils.softmax(logits.float(), dim=-1)

        return ctc_res, res


@register_model("wav2vec_lm")
class W2V_MIX_LM(W2V_CTC_MIX_LM):
    def __init__(self, args, encoder, assigner, mixer, lm):
        super().__init__(args, encoder, assigner, lm)
        self.mixer = mixer
        self.lm = lm
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates

    @staticmethod
    def add_args(parser):
        CIFModel.add_args(parser)
        add_mixer_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        global PAD_IDX, EOS_IDX
        # make sure all arguments are present in older models
        w2v_lm_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary
        PAD_IDX = tgt_dict.pad()
        EOS_IDX = tgt_dict.eos()

        encoder = cls.build_encoder(args)
        assigner = cls.build_assigner(args, encoder.d)
        lm = cls.build_lm(args, task)
        mixer = cls.build_mixer(args, encoder.d + len(tgt_dict), len(tgt_dict))

        return cls(args, encoder, assigner, mixer, lm)

    @classmethod
    def build_mixer(cls, args, dim_input, dim_output):

        return nn.Linear(dim_input, dim_output, bias=True)

    @classmethod
    def build_lm(cls, args, task):
        from fairseq.models.lstm_lm import LSTMLanguageModel
        return LSTMLanguageModel.build_model(args, task)

    @classmethod
    def build_bert(cls):
        from transformers import BertModel

        bert = BertModel.from_pretrained("hfl/rbt3")
        return bert

    def forward(self, **kwargs):
        encoder_output = self.encoder(tbc=False, **kwargs)
        alphas = self.assigner(encoder_output)
        _alphas, num_output = self.resize(alphas, kwargs['target_lengths'])
        cif_outputs = self.cif(encoder_output, _alphas)

        if torch.isnan(_alphas.sum() + alphas.sum() + cif_outputs.sum()):
            import pdb; pdb.set_trace()
        if torch.round(_alphas.sum(-1)).eq(0).sum() > 1:
            import pdb; pdb.set_trace()
        if cif_outputs.size(1) != kwargs["prev_output_tokens"].size(1):
            import pdb; pdb.set_trace()
        if cif_outputs.size(1) == 0:
            print('encoder_output', encoder_output['encoder_out'].sum(-1))
            print('alphas', alphas)
            import pdb; pdb.set_trace()

        if self.num_updates <= self.teacher_forcing_updates:
            logits = self.decode(encoded_shrunk=cif_outputs,
                                 prev_output_tokens=kwargs["prev_output_tokens"])
        else:
            logits = self.decode(encoded_shrunk=cif_outputs)

        return {'logits': logits, 'len_logits': kwargs['target_lengths'],
                'alphas': alphas, 'num_output': num_output}

    # teacher-forcing
    def decode(self, encoded_shrunk, prev_output_tokens=None, incremental_states=None, t=None):

        if incremental_states is not None: # step forward
            assert prev_output_tokens is not None, t is not None
            # t, incremental_state = incremental_states # t, ({...}, {...})
            B, T, V = encoded_shrunk.size()
            device = encoded_shrunk.device
            with torch.no_grad():
                encoded_t = encoded_shrunk[:, t, :]
                cur_logits_lm, _ = self.lm(prev_output_tokens, incremental_state=incremental_states[1])
                cur_logits = self.mixer(torch.cat([encoded_t[:, None, :], cur_logits_lm], -1))
                logits = cur_logits

        elif prev_output_tokens is not None: # teacher-forcing
            ft = self.freeze_lm_finetune_updates <= self.num_updates
            with torch.no_grad() if not ft else contextlib.ExitStack():
                logits_lm, _ = self.lm(prev_output_tokens.long())

            logits = self.mixer(torch.cat([encoded_shrunk, logits_lm], -1))

        else: # self-decode
            B, T, V = encoded_shrunk.size()
            device = encoded_shrunk.device
            prev_decoded = torch.ones([B, 1]).to(device).long() * EOS_IDX
            list_logits = []
            incremental_state = {}
            ft = self.freeze_lm_finetune_updates <= self.num_updates
            for encoded_t in torch.unbind(encoded_shrunk, 1):
                with torch.no_grad() if not ft else contextlib.ExitStack():
                    cur_logits_lm, _ = self.lm(prev_decoded, incremental_state=incremental_state)
                cur_logits = self.mixer(torch.cat([encoded_t, cur_logits_lm[:, 0, :]], -1))
                list_logits.append(cur_logits)
                cur_token = cur_logits.argmax(-1, keepdim=True)
                prev_decoded = torch.cat([prev_decoded, cur_token], 1)

            logits = torch.stack(list_logits, 1)

        logits.batch_first = True

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["logits"]

        if log_probs:
            res = utils.log_softmax(logits.float(), dim=-1)
        else:
            res = utils.softmax(logits.float(), dim=-1)

        res.batch_first = True

        return res


@register_model("wav2vec_cif_lm")
class W2V_MIX_CIF_LM(CIFModel):
    def __init__(self, args, encoder, assigner, decoder, mixer, lm):
        super().__init__(args, encoder, assigner, decoder)
        self.mixer = mixer
        self.lm = lm
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates

    @staticmethod
    def add_args(parser):
        CIFModel.add_args(parser)
        add_mixer_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        global PAD_IDX, EOS_IDX
        # make sure all arguments are present in older models
        w2v_cif_lm_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary
        PAD_IDX = tgt_dict.pad()
        EOS_IDX = tgt_dict.eos()

        encoder = cls.build_encoder(args)
        assigner = cls.build_assigner(args, encoder.d)
        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        lm = cls.build_lm(args, task)
        # mixer = cls.build_mixer(args, encoder.d + len(tgt_dict), len(tgt_dict))
        mixer = cls.build_mixer(args, 2 * len(tgt_dict), len(tgt_dict))

        return cls(args, encoder, assigner, decoder, mixer, lm)

    @classmethod
    def build_mixer(cls, args, dim_input, dim_output):

        return nn.Linear(dim_input, dim_output, bias=True)

    @classmethod
    def build_lm(cls, args, task):
        from fairseq.models.lstm_lm import LSTMLanguageModel
        return LSTMLanguageModel.build_model(args, task)

    def forward(self, **kwargs):
        encoder_output = self.encoder(tbc=False, **kwargs)
        alphas = self.assigner(encoder_output)
        _alphas, num_output = self.resize(alphas, kwargs['target_lengths'])
        cif_outputs = self.cif(encoder_output, _alphas)

        if self.num_updates <= self.teacher_forcing_updates:
            logits = self.decode(encoded_shrunk=cif_outputs,
                                 prev_output_tokens=kwargs["prev_output_tokens"])
        else:
            logits = self.decode(encoded_shrunk=cif_outputs)

        return {'logits': logits, 'len_logits': kwargs['target_lengths'],
                'alphas': alphas, 'num_output': num_output}

    # teacher-forcing
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
                                      incremental_state=incremental_states[0])
            cur_logits = cur_logits["logits"]
            cur_logits_lm, _ = self.lm(prev_output_tokens, incremental_state=incremental_states[1])
            cur_logits = self.mixer(torch.cat([cur_logits, cur_logits_lm], -1))
            logits = cur_logits

        elif prev_output_tokens is not None: # teacher-forcing
            logits = self.decoder(prev_output_tokens=prev_output_tokens,
                                  encoder_out=encoder_output)
            logits = logits["logits"]
            ft = self.freeze_lm_finetune_updates <= self.num_updates
            with torch.no_grad() if not ft else contextlib.ExitStack():
                logits_lm, _ = self.lm(prev_output_tokens.long())
            logits = self.mixer(torch.cat([logits, logits_lm], -1))

        else: # self-decode
            B, T, V = encoded_shrunk.size()
            device = encoded_shrunk.device
            prev_decoded = torch.ones([B, 1]).to(device).long() * EOS_IDX
            list_logits = []
            incremental_state = {}
            ft = self.freeze_lm_finetune_updates <= self.num_updates
            for encoded_t in torch.unbind(encoded_shrunk, 1):
                cur_logits = self.decoder(prev_output_tokens=prev_decoded,
                                          encoder_out=encoder_output,
                                          incremental_state=incremental_state)
                with torch.no_grad() if not ft else contextlib.ExitStack():
                    cur_logits_lm, _ = self.lm(prev_decoded, incremental_state=incremental_state)
                cur_logits = self.mixer(torch.cat([cur_logits, cur_logits_lm[:, 0, :]], -1))
                list_logits.append(cur_logits)
                cur_token = cur_logits.argmax(-1, keepdim=True)
                prev_decoded = torch.cat([prev_decoded, cur_token], 1)

            logits = torch.stack(list_logits, 1)

        logits.batch_first = True

        return logits


@register_model("wav2vec_cif_rnn_lm")
class W2V_MIX_CIF_RNN_LM(W2V_MIX_CIF_LM):

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        global PAD_IDX, EOS_IDX
        # make sure all arguments are present in older models
        w2v_cif_rnn_lm_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary
        PAD_IDX = tgt_dict.pad()
        EOS_IDX = tgt_dict.eos()

        encoder = cls.build_encoder(args)
        assigner = cls.build_assigner(args, encoder.d)
        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)
        decoder = cls.build_decoder(args,
                                    embed_tokens=decoder_embed_tokens,
                                    dim_input=encoder.d,
                                    dim_output=encoder.d)
        lm = cls.build_lm(args, task)
        mixer = cls.build_mixer(args, 2 * encoder.d + len(tgt_dict), len(tgt_dict))

        return cls(args, encoder, assigner, decoder, mixer, lm)

    @classmethod
    def build_decoder(cls, args, dim_input, dim_output, embed_tokens):
        from .wav2vec2_seq2seq import CIF_RNN_Model
        return CIF_RNN_Model.build_decoder(args, dim_input, dim_output, embed_tokens)

    # teacher-forcing
    def decode(self, encoded_shrunk, prev_output_tokens=None, incremental_states=None, t=None):

        if incremental_states is not None: # step forward
            assert prev_output_tokens is not None, t is not None

            frame = self.decoder(prev_output_tokens=prev_output_tokens,
                                 encoder_out=encoded_shrunk,
                                 incremental_state=incremental_states[0])
            cur_logits_lm, _ = self.lm(prev_output_tokens, incremental_state=incremental_states[1])
            cur_logits = self.mixer(torch.cat([encoded_shrunk[:, t:t+1, :], frame, cur_logits_lm], -1))
            logits = cur_logits

        elif prev_output_tokens is not None: # teacher-forcing
            frame = self.decoder(prev_output_tokens=prev_output_tokens,
                                 encoder_out=encoded_shrunk)
            ft = self.freeze_lm_finetune_updates <= self.num_updates
            with torch.no_grad() if not ft else contextlib.ExitStack():
                logits_lm, _ = self.lm(prev_output_tokens.long())
            logits = self.mixer(torch.cat([encoded_shrunk, frame, logits_lm], -1))

        else: # self-decode
            B, T, V = encoded_shrunk.size()
            device = encoded_shrunk.device
            prev_decoded = torch.ones([B, 1]).to(device).long() * EOS_IDX
            list_logits = []
            incremental_state = {}
            ft = self.freeze_lm_finetune_updates <= self.num_updates
            for encoded_t in torch.unbind(encoded_shrunk, 1):
                cur_logits = self.decoder(prev_output_tokens=prev_decoded,
                                          encoder_out=encoded_shrunk,
                                          incremental_state=incremental_state)
                with torch.no_grad() if not ft else contextlib.ExitStack():
                    cur_logits_lm, _ = self.lm(prev_decoded, incremental_state=incremental_state)
                cur_logits = self.mixer(torch.cat([cur_logits, cur_logits_lm[:, 0, :]], -1))
                list_logits.append(cur_logits)
                cur_token = cur_logits.argmax(-1, keepdim=True)
                prev_decoded = torch.cat([prev_decoded, cur_token], 1)

            logits = torch.stack(list_logits, 1)

        logits.batch_first = True

        return logits


@register_model("wav2vec_cif_lm_rnn")
class W2V_MIX_CIF_LM_RNN(W2V_MIX_CIF_RNN_LM):

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        global PAD_IDX, EOS_IDX
        # make sure all arguments are present in older models
        w2v_cif_lm_rnn_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        tgt_dict = task.target_dictionary
        PAD_IDX = tgt_dict.pad()
        EOS_IDX = tgt_dict.eos()

        encoder = cls.build_encoder(args)
        assigner = cls.build_assigner(args, encoder.d)
        lm = cls.build_lm(args, task)
        mixer = cls.build_mixer(args,
            dim_input=encoder.d + len(tgt_dict),
            dim_hidden=args.dim_hidden_mixer,
            dim_output=len(tgt_dict))

        return cls(args, encoder, assigner, lm, mixer, lm)

    @classmethod
    def build_mixer(cls, args, dim_input, dim_hidden, dim_output):

        return LSTM_MIXER(dim_input, dim_hidden, dim_output, residuals=True)

    # teacher-forcing
    def decode(self, encoded_shrunk, prev_output_tokens=None, incremental_states=None, t=None):

        if incremental_states is not None: # step forward
            assert prev_output_tokens is not None, t is not None

            cur_logits_lm, _ = self.lm(prev_output_tokens, incremental_state=incremental_states[1])
            cur_logits = self.mixer(encoded_shrunk[:, t:t+1, :], cur_logits_lm,
                                    incremental_state=incremental_states[0])
            logits = cur_logits

        elif prev_output_tokens is not None: # teacher-forcing
            ft = self.freeze_lm_finetune_updates <= self.num_updates
            with torch.no_grad() if not ft else contextlib.ExitStack():
                logits_lm, _ = self.lm(prev_output_tokens.long())
            logits = self.mixer(encoded_shrunk, logits_lm)

        else: # self-decode
            B, T, V = encoded_shrunk.size()
            device = encoded_shrunk.device
            prev_decoded = torch.ones([B, 1]).to(device).long() * EOS_IDX
            list_logits = []
            incremental_state = {}
            ft = self.freeze_lm_finetune_updates <= self.num_updates
            for encoded_t in torch.unbind(encoded_shrunk, 1):
                cur_logits = self.decoder(prev_output_tokens=prev_decoded,
                                          encoder_out=encoded_shrunk,
                                          incremental_state=incremental_state)
                with torch.no_grad() if not ft else contextlib.ExitStack():
                    cur_logits_lm, _ = self.lm(prev_decoded, incremental_state=incremental_state)
                cur_logits = self.mixer(torch.cat([cur_logits, cur_logits_lm[:, 0, :]], -1))
                list_logits.append(cur_logits)
                cur_token = cur_logits.argmax(-1, keepdim=True)
                prev_decoded = torch.cat([prev_decoded, cur_token], 1)

            logits = torch.stack(list_logits, 1)

        logits.batch_first = True

        return logits


@register_model("wav2vec_cif_bert")
class W2V_MIX_CIF_BERT(BaseFairseqModel):

    def __init__(self, args, encoder, assigner, lm):
        super().__init__()
        self.encoder = encoder
        self.assigner = assigner
        self.lm = lm
        self.num_updates = 0
        self.args = args
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.proj = Linear(encoder.d, lm.encoder.encoder.sentence_encoder.embedding_dim)
        # self.proj = Linear(encoder.d, lm.encoder.encoder.sentence_encoder.vocab_size)

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        parser.add_argument(
            "--freeze-lm-finetune-updates", type=int, help="freeze_lm_finetune_updates"
        )
        parser.add_argument(
            "--lm-path", type=str, help="dim-hidden-mixer"
        )
        parser.add_argument(
            "--assigner-conv-layers",
            type=str,
            metavar="EXPR",
            help="convolutional feature extraction layers [(dim, kernel_size, stride), ...]",
        )
        parser.add_argument("--lambda-qua", type=float, metavar="D", help="lambda-qua")
        parser.add_argument("--lambda-alpha", type=float, metavar="D", help="lambda-alpha")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        w2v_cif_bert_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        lm = cls.build_lm(args, task.dictionary)
        # lm = cls.build_bert()
        # import pdb; pdb.set_trace()
        encoder = cls.build_encoder(args) # encoder
        assigner = cls.build_assigner(args, encoder.d)

        return cls(args, encoder, assigner, lm)

    @classmethod
    def build_encoder(cls, args, tgt_dict=None):
        return Wav2VecEncoder(args, tgt_dict=tgt_dict)

    @classmethod
    def build_assigner(cls, args, dim_input):
        return Assigner(args, dim_input)

    @staticmethod
    def resize(*args, **kwargs):
        return CIFModel.resize(*args, **kwargs)

    @staticmethod
    def cif(*args, **kwargs):
        return CIFModel.cif(*args, **kwargs)

    @classmethod
    def build_lm(cls, args, dictionary):
        from fairseq.models.masked_lm import MaskedLMModel as LM
        args.tokens_per_sample = 100
        return LM.build_model(args, task=None, dictionary=dictionary)

    @classmethod
    def build_bert(cls):
        from transformers import BertModel

        bert = BertModel.from_pretrained("hfl/rbt3")
        return bert

    def forward(self, **kwargs):
        """
        encoder_output= "encoder_out": x,
                        "encoded": encoded,
                        "encoder_padding_mask": padding_mask,  # B x T
                        "padding_mask": padding_mask,
        """
        encoder_output = self.encoder(tbc=False, **kwargs)
        alphas, alphas_pen = self.assigner(encoder_output)
        _alphas, num_output = self.resize(alphas, kwargs['target_lengths'])
        cif_outputs = self.cif(encoder_output, _alphas)

        if self.training and cif_outputs.abs().sum(-1).ne(0).sum() != kwargs['target_lengths'].sum():
            print('_alphas:\t', _alphas.sum(-1))
            print('alphas:\t', alphas.sum(-1))
            print('target:\t', kwargs['target_lengths'])
            import pdb; pdb.set_trace()

        hidden = self.proj(cif_outputs)
        padding_mask = ~utils.sequence_mask(kwargs['target_lengths']).bool()
        # logits = self.proj(cif_outputs)

        ft = self.freeze_lm_finetune_updates <= self.num_updates
        if self.freeze_lm_finetune_updates == self.num_updates: print('unfreeze LM ...')
        with torch.no_grad() if not ft else contextlib.ExitStack():
            logits = self.lm.forward_embeded(hidden, padding_mask)
            # logits = self.forward_embeded(hidden, padding_mask)

        return {'logits': logits, 'len_logits': kwargs['target_lengths'],
                'alphas': alphas, 'num_output': num_output, 'alphas_pen': alphas_pen}

    def forward_embeded(self, hidden, padding_mask):
        import pdb; pdb.set_trace()
        encoded = self.lm.encoder(hidden)
        encoded = self.lm.pooler(encoded)

        return encoded

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output["logits"]
        if log_probs:
            res = utils.log_softmax(logits.float(), dim=-1)
        else:
            res = utils.softmax(logits.float(), dim=-1)
        res.batch_first = True

        return res

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

from fairseq.models.lstm import LSTMCell
class LSTM_MIXER(LSTMDecoder):
    """LSTM decoder."""
    def __init__(
        self, dim_input, hidden_size, dim_output, num_layers=2,
        dropout_in=0.1, dropout_out=0.1, residuals=False,
    ):
        FairseqIncrementalDecoder.__init__(self, None)
        self.dropout_lm = FairseqDropout(dropout_in, module_name=self.__class__.__name__)
        self.dropout_out_module = FairseqDropout(dropout_out, module_name=self.__class__.__name__)
        self.hidden_size = hidden_size
        self.residuals = residuals
        self.num_layers = num_layers
        self.max_target_positions = 200

        self.encoder_output_units = dim_input
        if dim_input != hidden_size:
            self.encoder_hidden_proj = Linear(dim_input, hidden_size)
        else:
            self.encoder_hidden_proj = None

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])

        if hidden_size != dim_output:
            self.fc_out = Linear(hidden_size, dim_output, bias=True)
        else:
            self.fc_out = None

    def forward(self, frame_ac, logits_lm,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        # logits_lm = self.dropout_lm(logits_lm)
        frame_mix = torch.cat([frame_ac, logits_lm], -1)
        x = self.extract_features(frame_mix, incremental_state)

        return self.output_layer(x)

    def extract_features(
        self,
        encoder_out,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Similar to *forward* but only return features.
        """
        bsz, seqlen, _ = encoder_out.size()

        # B x T x C -> T x B x C
        x = encoder_out.transpose(0, 1)
        if self.encoder_hidden_proj is not None:
            x = self.encoder_hidden_proj(x)
        # initialize previous states (or get from cache during incremental generation)
        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells = self.get_cached_state(incremental_state)
        else:
            # setup recurrent cells
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]

        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = x[j]

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
                "prev_cells": prev_cells_tensor
            }
        )
        self.set_incremental_state(incremental_state, 'cached_state', cache_state)
        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen

        return x


@register_model_architecture("wav2vec_lm", "wav2vec_lm")
def w2v_lm_architecture(args):
    cif_architecture(args)
    args.freeze_lm_finetune_updates = getattr(args, "freeze_lm_finetune_updates", 1000)


@register_model_architecture("wav2vec_cif_lm", "wav2vec_cif_lm")
def w2v_cif_lm_architecture(args):
    w2v_lm_architecture(args)


@register_model_architecture("wav2vec_cif_rnn_lm", "wav2vec_cif_rnn_lm")
def w2v_cif_rnn_lm_architecture(args):
    w2v_cif_lm_architecture(args)


@register_model_architecture("wav2vec_cif_lm_rnn", "wav2vec_cif_lm_rnn")
def w2v_cif_lm_rnn_architecture(args):
    w2v_cif_lm_architecture(args)


@register_model_architecture("wav2vec_cif_bert", "wav2vec_cif_bert")
def w2v_cif_bert_architecture(args):
    w2v_lm_architecture(args)


@register_model_architecture("wav2vec_ctc_bert", "wav2vec_ctc_bert")
def w2v_ctc_bert_architecture(args):
    w2v_lm_architecture(args)
