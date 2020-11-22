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

from .wav2vec2_seq2seq import CIFModel, cif_architecture
from .wav2vec2_lm import W2V_MIX_LM, W2V_CTC_MIX_LM

PAD_IDX = 1
EOS_IDX = 2

def NonlinearLayer(in_features, out_features, bias=True, activation_fn=nn.ReLU):
    """Weight-normalized non-linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return nn.Sequential(m, activation_fn())


@register_model("wav2vec_lm_ColdFusion")
class W2V_MIX_LM2(W2V_MIX_LM):

    def __init__(self, args, encoder, assigner, lm):
        W2V_CTC_MIX_LM.__init__(self, args, encoder, assigner, lm)
        self.lm = lm
        self.lm.eval()
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.num_updates = 0
        self.teacher_forcing_updates = args.teacher_forcing_updates

        dim_phone, dim_hidden, vocab_size = encoder.d, 1024, lm.dim_output
        self.hidden_layer = NonlinearLayer(
            vocab_size, dim_hidden, bias=False, activation_fn=nn.ReLU
        )
        self.gating_network = NonlinearLayer(
            dim_hidden + dim_phone,
            dim_hidden,
            bias=True,
            activation_fn=nn.Sigmoid,
        )
        self.output_projections = NonlinearLayer(
            dim_hidden + dim_phone,
            vocab_size,
            bias=False,
            activation_fn=nn.ReLU,
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        global PAD_IDX, EOS_IDX
        # make sure all arguments are present in older models
        w2v_lm_architecture2(args)

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

        return cls(args, encoder, assigner, lm)

    def decode(self, encoded_shrunk, prev_output_tokens=None, incremental_states=None, t=None):
        if incremental_states is not None: # step forward
            assert prev_output_tokens is not None, t is not None
            # t, incremental_state = incremental_states # t, ({...}, {...})
            B, T, V = encoded_shrunk.size()
            with torch.no_grad():
                encoded_t = encoded_shrunk[:, t:t+1, :]
                cur_logits_lm, _ = self.lm(prev_output_tokens, incremental_state=incremental_states[1])
                pred_lm = F.softmax(cur_logits_lm, -1)
                h_lm = self.hidden_layer(pred_lm)
                h_am_lm = torch.cat([encoded_t, h_lm], -1)
                g = self.gating_network(h_am_lm)
                h_cf = torch.cat([encoded_t, g * h_lm], -1)
                logits = self.output_projections(h_cf)[:, 0, :]

        elif prev_output_tokens is not None: # teacher-forcing
            ft = self.freeze_lm_finetune_updates <= self.num_updates
            with torch.no_grad() if not ft else contextlib.ExitStack():
                logits_lm, _ = self.lm(prev_output_tokens.long())
                pred_lm = F.softmax(logits_lm, -1)
            h_lm = self.hidden_layer(pred_lm)
            h_am_lm = torch.cat([encoded_shrunk, h_lm], -1)
            g = self.gating_network(h_am_lm)
            h_cf = torch.cat([encoded_shrunk, g * h_lm], -1)
            logits = self.output_projections(h_cf)

        logits.batch_first = True

        return logits


@register_model_architecture("wav2vec_lm_ColdFusion", "wav2vec_lm_ColdFusion")
def w2v_lm_architecture2(args):
    cif_architecture(args)
    args.freeze_lm_finetune_updates = getattr(args, "freeze_lm_finetune_updates", 1000)
    args.lambda_qua = getattr(args, "lambda_qua", 0.1)
