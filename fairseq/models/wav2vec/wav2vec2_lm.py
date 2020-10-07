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

from fairseq import utils
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
    Fp32GroupNorm
)

from .wav2vec2_ctc import add_common_args, base_architecture
from .wav2vec2_seq2seq import CIFModel, Wav2VecEncoder, Linear, cif_architecture, add_decoder_args


PAD_IDX = 1
EOS_IDX = 2
BLK_IDX = -1


def add_mixer_args(parser):
    parser.add_argument(
        "--dim-hidden-mixer", type=int, help="dim-hidden-mixer"
    )
    parser.add_argument(
        "--lambda-qua", type=float, help="lambda-qua"
    )
    parser.add_argument(
        "--freeze-lm-finetune-updates", type=int, help="freeze_lm_finetune_updates"
    )
    parser.add_argument(
        "--lm-path", type=str, help="dim-hidden-mixer"
    )
    parser.add_argument(
        "--teacher-forcing-updates", type=int, help="is teacher-forcing"
    )

@register_model("wav2vec_ctc_lm")
class W2V_CTC_MIX_LM(CIFModel):
    def __init__(self, args, encoder, mixer, lm):
        super().__init__(args, encoder, mixer, lm)
        self.mixer = mixer
        self.lm = lm
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.num_updates = 0

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_decoder_args(parser)
        add_mixer_args(parser)

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
        mixer = cls.build_mixer(args, encoder.d + lm.dim_output, args.dim_hidden_mixer, 2, len(tgt_dict))

        return cls(args, encoder, mixer, lm)

    @classmethod
    def build_encoder(cls, args, tgt_dict=None):
        return Wav2VecEncoder(args, tgt_dict)

    @classmethod
    def build_mixer(cls, args, dim_input, dim_hidden, dim_output):
        return Linear(dim_input, dim_output)

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


@register_model("wav2vec_lm")
class W2V_MIX_LM(W2V_CTC_MIX_LM):
    def __init__(self, args, encoder, assigner, mixer, lm):
        super().__init__(args, encoder, assigner, lm)
        self.mixer = mixer
        self.lm = lm
        self.lm.eval()
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.num_updates = 0
        self.teacher_forcing_updates = args.teacher_forcing_updates
        # self.phone_fc = nn.Linear(encoder.d, lm.dim_output, bias=True)

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

    def forward(self, **kwargs):
        encoder_output = self.encoder(tbc=False, **kwargs)
        alphas = self.assigner(encoder_output)
        _alphas, num_output = self.resize(alphas, kwargs['target_lengths'], at_least_one=True)
        cif_outputs = self.cif(encoder_output, _alphas)

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

        return {'logits': logits, 'len_logits': kwargs['target_lengths'], 'num_output': num_output}

    # teacher-forcing
    def decode(self, encoded_shrunk, prev_output_tokens=None, incremental_states=None, t=None):
        # logits_ac = self.phone_fc(encoded_shrunk)
        # probs_ac = F.softmax(logits_ac, -1)
        if incremental_states is not None: # step forward
            assert prev_output_tokens is not None, t is not None
            # t, incremental_state = incremental_states # t, ({...}, {...})
            B, T, V = encoded_shrunk.size()
            device = encoded_shrunk.device
            with torch.no_grad():
                encoded_t = encoded_shrunk[:, t, :]
                cur_logits_lm, _ = self.lm(prev_output_tokens, incremental_state=incremental_states[1])
                cur_logits = self.mixer(torch.cat([encoded_t, cur_logits_lm[:, 0, :]], -1))
                logits = cur_logits

        elif prev_output_tokens is not None: # teacher-forcing
            ft = self.freeze_lm_finetune_updates <= self.num_updates
            with torch.no_grad() if not ft else contextlib.ExitStack():
                logits_lm, _ = self.lm(prev_output_tokens.long())

            logits = self.mixer(torch.cat([encoded_shrunk, logits_lm], -1))
            # probs_lm = F.softmax(logits_lm, -1)
            # logits = self.mixer(torch.cat([probs_ac, probs_lm], -1))

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
                cur_token = cur_logits_lm.argmax(-1, keepdim=True)
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


@register_model_architecture("wav2vec_lm", "wav2vec_lm")
def w2v_lm_architecture(args):
    cif_architecture(args)
    args.freeze_lm_finetune_updates = getattr(args, "freeze_lm_finetune_updates", 1000)
    args.lambda_qua = getattr(args, "lambda_qua", 0.1)
