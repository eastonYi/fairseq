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
from .wav2vec2_lm import W2V_MIX_LM

PAD_IDX = 1
EOS_IDX = 2
BLK_IDX = -1


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
    parser.add_argument(
        "--teacher-forcing", action='store_true', help="is teacher-forcing"
    )


@register_model("wav2vec_lm2")
class W2V_MIX_LM2(W2V_MIX_LM):

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
        mixer = cls.build_mixer(args, encoder.d, args.dim_hidden_mixer, 2, len(tgt_dict))

        return cls(args, encoder, assigner, mixer, lm)

    @classmethod
    def build_mixer(cls, args, dim_phone, dim_hidden, num_layers, dim_output):

        # return nn.Linear(dim_input, dim_output, bias=True)
        return MLP(dim_phone, dim_hidden, num_layers, dim_output)

    def forward(self, **kwargs):
        encoder_output = self.encoder(tbc=False, **kwargs)
        alphas = self.assigner(encoder_output)
        _alphas, num_output = self.resize(alphas, kwargs['target_lengths'], at_least_one=True)
        cif_outputs = self.cif(encoder_output, _alphas)
        logits = self.decode(encoded_shrunk=cif_outputs,
                             prev_output_tokens=kwargs["prev_output_tokens"])
        try:
            if self.teacher_forcing:
                logits = self.decode(encoded_shrunk=cif_outputs,
                                     prev_output_tokens=kwargs["prev_output_tokens"])
            else:
                logits = self.decode(encoded_shrunk=cif_outputs)
        except:
            print('cif_outputs: ', cif_outputs.size())
            print('prev_output_tokens: ', kwargs["prev_output_tokens"], kwargs["prev_output_tokens"].size())
            print('_alphas: ', _alphas.sum(-1))
            print('target_lengths: ', kwargs['target_lengths'])
            import pdb; pdb.set_trace()

        return {'logits': logits, 'len_logits': kwargs['target_lengths'], 'num_output': num_output}

    # teacher-forcing
    def decode(self, encoded_shrunk, prev_output_tokens=None):
        ft = self.freeze_lm_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            logits_raw, _ = self.lm(prev_output_tokens.long())
            pred_raw = F.softmax(logits_raw, -1)
        try:
            logits = self.mixer(encoded_shrunk, pred_raw)
        except:
            print('encoded_shrunk: ', encoded_shrunk.size())
            print('logits_raw: ', logits_raw.size())
            print('prev_output_tokens: ', prev_output_tokens.size())

        logits.batch_first = True

        return logits

class MLP(nn.Module):
    def __init__(self, dim_phone, dim_hidden, num_layers, dim_output):
        super().__init__()
        dim_token = dim_output
        self.module_list = nn.ModuleList([nn.Linear(dim_token + dim_phone, dim_hidden, bias=True), nn.ReLU()])
        # self.module_list = nn.ModuleList([nn.Linear(dim_input, dim_hidden, bias=True)])
        for _ in range(num_layers-1):
            self.module_list.extend([nn.Linear(dim_hidden, dim_hidden, bias=True), nn.ReLU()])
            # self.module_list.extend([nn.Linear(dim_hidden, dim_hidden, bias=True)])
        self.module_list.extend([nn.Linear(dim_hidden, dim_output, bias=False)])
        self.phoen_fc = nn.Linear(dim_phone, dim_output, bias=False)

    def forward(self, vec_phone, vec_token):
        x = torch.cat([vec_phone, vec_token], -1)
        for layer in self.module_list:
            x = layer(x)

        x += self.phoen_fc(vec_phone)

        return x


@register_model_architecture("wav2vec_lm2", "wav2vec_lm2")
def w2v_lm_architecture2(args):
    cif_architecture(args)
    args.freeze_lm_finetune_updates = getattr(args, "freeze_lm_finetune_updates", 1000)
