# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import contextlib
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from transformers import GPT2LMHeadModel

from fairseq import utils
from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    GradMultiply,
    PositionalEmbedding,
    TransformerDecoderLayer,
    TransposeLast,
    Fp32LayerNorm,
    Fp32GroupNorm,
    FairseqDropout
)

from .wav2vec2_ctc import (
    Linear,
    Wav2VecEncoder,
    add_common_args,
    base_architecture
)
from .wav2vec2_cif import (
    CIFFcModel,
    CIFFcModelV2,
    cif_architecture,
)


def pred2bert_input(pred, token_mask, cls=101, sep=102):

    pred *= token_mask
    end_index = token_mask.sum(-1).long().unsqueeze(1) + 1
    pred.scatter_(dim=-1, index=end_index, value=sep)
    pred[:, 0] = cls

    return pred


def add_lm_args(parser):
    parser.add_argument(
        "--freeze-lm-finetune-updates", type=int, default=100, help="freeze_lm_finetune_updates"
    )
    parser.add_argument(
        "--lambda-embedding", type=float, metavar="D", help="lambda-embedding"
    )
    parser.add_argument(
        "--lambda-am", type=float, default=1.0, metavar="D", help="lambda-am"
    )
    parser.add_argument(
        "--lambda-lm", type=float, default=0.2, metavar="D", help="lambda-lm"
    )
    parser.add_argument("--lambda-qua", type=float, default=0.1, metavar="D", help="lambda-qua")


@register_model("w2v_cif_gpt2")
class W2V_CIF_GPT2(BaseFairseqModel):

    def __init__(self, args, encoder, gpt2, tgt_dict):
        """
        """
        super().__init__()
        self.encoder = encoder
        self.gpt2 = gpt2
        self.dim_gpt2 = gpt2.transformer.wte.weight.size(1)
        self.proj = Linear(encoder.d-1, self.dim_gpt2)
        self.to_vocab_lm = self.gpt2.lm_head
        self.to_vocab_ac = copy.deepcopy(self.gpt2.lm_head)
        self.to_vocab_ctc = copy.deepcopy(self.gpt2.lm_head)
        self.gpt2.lm_head
        self.tgt_dict = tgt_dict
        self.num_updates = 0
        self.args = args
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates

        # for p in self.gpt2.transformer.wte.parameters():
        #     p.requires_grad = False

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_lm_args(parser)
        parser.add_argument("--lambda-ctc", type=float, metavar="D", help="lambda-ctc")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        w2v_cif_gpt2_architecture(args)
        tgt_dict = task.target_dictionary

        gpt2 = cls.build_gpt2(args, tgt_dict)
        encoder = cls.build_encoder(args) # encoder

        return cls(args, encoder, gpt2, tgt_dict)

    @classmethod
    def build_encoder(cls, args, tgt_dict=None):
        return Wav2VecEncoder(args, tgt_dict=tgt_dict)

    @classmethod
    def build_gpt2(cls, args, tgt_dict):
        model = GPT2LMHeadModel.from_pretrained(args.gpt2_name)

        return model

    def forward(self, **kwargs):
        """
        encoder_output= "encoder_out": x,
                        "encoded": encoded,
                        "encoder_padding_mask": padding_mask,  # B x T
                        "padding_mask": padding_mask,
        """
        encoder_output = self.encoder(tbc=False, **kwargs)
        hidden_encoded = encoder_output['encoder_out'][:, :, :-1]
        hidden_ctc = F.pad(hidden_encoded, [0, 1, 0, 0, 0, 0], value=0)
        logits_ctc = self.to_vocab_ctc(hidden_ctc)
        len_logits_ctc = (~encoder_output['padding_mask']).sum(-1).long()
        alphas = CIFFcModelV2.get_alphas(encoder_output)

        if self.training:
            decode_length = kwargs['target_lengths']
        else:
            decode_length = torch.round(alphas.sum(-1)).int()
        _alphas, num_output = self.resize(alphas, decode_length)
        cif_outputs = self.cif(hidden_encoded, _alphas)
        hidden_ac = self.proj(cif_outputs)
        logits_ac = self.to_vocab_ac(hidden_ac)

        # other inputs
        B, T = hidden_ac.size(0), hidden_ac.size(1)
        padding_mask = ~utils.sequence_mask(decode_length).bool()
        mask_ac = (~padding_mask)
        position_ac = torch.arange(T).repeat(B, 1).long().cuda()
        type_ac = torch.ones((B, T)).long().cuda() * 102

        if self.training:
            targets = kwargs['gpt2_input']
            token_type = torch.ones_like(targets) * 103
            text_embs = self.gpt2.transformer.wte(targets)
            ft = self.freeze_lm_finetune_updates <= self.num_updates
            with torch.no_grad() if not ft else contextlib.ExitStack():
                input_embs = torch.cat([hidden_ac, text_embs], dim=1)
                type_ids = torch.cat([type_ac, token_type], dim=1)
                position_ids = torch.cat([position_ac, position_ac], dim=1)
                attention_mask = torch.cat([mask_ac, mask_ac], dim=1)[:, None, None, :]
                outputs = self.gpt2(
                    inputs_embeds=input_embs,
                    token_type_ids=type_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )
            logits_lm = outputs[0][:, T:, :]
            logits = self.args.lambda_am * logits_ac + self.args.lambda_lm * logits_lm
            logits *= (~padding_mask).unsqueeze(-1).float()
        else:
            past = None
            list_logits = [logits_ac[:, 0:1, :]]
            token_type = torch.ones_like(type_ac) * 103
            first = torch.argmax(logits_ac, -1)[:, 0][:, None]
            text_embs = self.gpt2.transformer.wte(first)
            input_embs = torch.cat([hidden_ac, text_embs], dim=1)
            type_ids = torch.cat([type_ac, token_type[:, 0:1]], dim=1)
            position_ids = torch.cat([position_ac, position_ac[:, 0:1]], dim=1)
            for i in range(1, T):
                outputs = self.gpt2(
                    past_key_values=past,
                    inputs_embeds=input_embs,
                    token_type_ids=type_ids,
                    position_ids=position_ids,
                    use_cache=True,
                    # output_attentions=False
                    )
                logits_lm, past = outputs[0], outputs[1]
                logits_i = self.args.lambda_am * logits_ac[..., -1, :] + self.args.lambda_lm * logits_lm[..., -1, :]
                list_logits.append(logits_i.unsqueeze(1))
                preds = torch.argmax(logits_i, -1)[:, None]
                input_embs = self.gpt2.transformer.wte(preds)
                type_ids = token_type[:, i:i+1]
                position_ids = position_ac[:, i:i+1]
            logits = torch.cat(list_logits, 1)
            logits *= (~padding_mask).unsqueeze(-1).float()

        return {'logits': logits, 'len_logits': decode_length,
                'alphas': alphas, 'num_output': num_output,
                'logits_ctc': logits_ctc, 'len_logits_ctc': len_logits_ctc}

        return logits

    @staticmethod
    def resize(*args, **kwargs):
        return CIFFcModel.resize(*args, **kwargs)

    @staticmethod
    def cif(*args, **kwargs):
        return CIFFcModel.cif(*args, **kwargs)

    def get_normalized_probs(self, net_output, log_probs, retrun_ctc=False):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits_ctc = net_output["logits_ctc"]
        logits = net_output["logits"]
        if log_probs:
            res_ctc = utils.log_softmax(logits_ctc.float(), dim=-1)
            res = utils.log_softmax(logits.float(), dim=-1)
        else:
            res_ctc = utils.softmax(logits_ctc.float(), dim=-1)
            res = utils.softmax(logits.float(), dim=-1)
        res_ctc.batch_first = True
        res.batch_first = True

        if retrun_ctc:
            return res_ctc, res
        else:
            return res

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


@register_model_architecture("w2v_cif_gpt2", "w2v_cif_gpt2")
def w2v_cif_gpt2_architecture(args):
    cif_architecture(args)
