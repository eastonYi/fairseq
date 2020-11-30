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

from .wav2vec2_seq2seq import (
    LSTMDecoder,
    build_embedding,
    Wav2VecEncoder,
    Linear,
    add_decoder_args
)

PAD_IDX = 1
EOS_IDX = 2


@register_model("wav2vec_ctc_lm")
class W2V_CTC_LM(BaseFairseqModel):
    def __init__(self, args, encoder, lm):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.lm = lm
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.proj = Linear(encoder.d, lm.encoder.encoder.sentence_encoder.embedding_dim)
        self.num_updates = 0

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
        w2v_ctc_bert_architecture(args)

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
        # self.proj = Linear(encoder.d, lm.encoder.encoder.sentence_encoder.embedding_dim)
        self.proj = Linear(encoder.d, lm.encoder.encoder.sentence_encoder.vocab_size)

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
        w2v_ctc_bert_architecture(args)

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
        logits_ctc = encoder_output['encoder_out']
        len_encoded = (~encoder_output['padding_mask']).sum(-1)
        encoded_shrunk, len_encoded_shrunk = utils.ctc_shrink(
            hidden=encoded, logits=logits_ctc, len_logits=len_encoded,
            blk=BLK_IDX, at_least_one=True)
        encoder_shrunk_out = {'encoded_shrunk': encoded_shrunk,
                              'len_encoded_shrunk': len_encoded_shrunk}

        logits = self.decode(encoder_shrunk_out)

        return {'logits': logits, 'len_logits': len_encoded_shrunk,
                'logits_ctc': logits_ctc, 'len_logits_ctc': len_encoded,
                'encoder_output': encoder_output}

    def decode(self, encoder_shrunk_out):
        encoded_logits = encoder_shrunk_out["encoded_shrunk"]
        padding_mask = utils.sequence_mask(encoder_shrunk_out["len_encoded_shrunk"],
                                           dtype=torch.bool, reverse=True)
        # prob = torch.softmax(encoded_logits[:, :, :-1], -1)
        ft = self.freeze_lm_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            # embedded = torch.mm(prob.view(-1, prob.size(-1)),
            #                     self.lm.encoder.encoder.sentence_encoder.embed_tokens.weight[:-1, :]
            #                     ).view(prob.size(0), prob.size(1), -1)
            # embedded = self.proj(encoded_logits)
            # logits = self.lm.forward_embeded(embedded, padding_mask)
            logits = self.proj(encoded_logits)

        logits.batch_first = True

        return logits

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits_ctc = net_output["logits_ctc"]
        logits = net_output["logits"]

        if log_probs:
            ctc_res = utils.log_softmax(logits_ctc.float(), dim=-1)
            res = utils.log_softmax(logits.float(), dim=-1)
        else:
            ctc_res = utils.softmax(logits_ctc.float(), dim=-1)
            res = utils.softmax(logits.float(), dim=-1)

        return ctc_res, res

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


@register_model_architecture("wav2vec_ctc_bert", "wav2vec_ctc_bert")
def w2v_ctc_bert_architecture(args):
    base_architecture(args)
    args.freeze_lm_finetune_updates = getattr(args, "freeze_lm_finetune_updates", 1000)
