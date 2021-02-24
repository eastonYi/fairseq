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
# from transformers import GPT2LMHeadModel, GPT2LMHeadFinetuneModel

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
    parser.add_argument(
        "--gold-rate-range", type=str, help="gold-rate-range"
    )
    parser.add_argument(
        "--gold-rate-steps", type=str, help="gold-rate-steps"
    )
    parser.add_argument("--lambda-qua", type=float, default=0.1, metavar="D", help="lambda-qua")
    parser.add_argument("--position-bias", type=int, default=0, metavar="D", help="position-bias")
    parser.add_argument(
        "--no-type-id",
        action="store_true",
        help="if set, does not learn bias for conv layers",
    )


@register_model("w2v_cif")
class W2V_CIF(BaseFairseqModel):

    def __init__(self, args, encoder, gpt2, tgt_dict):
        """
        """
        super().__init__()
        self.encoder = encoder
        self.dim_gpt2 = gpt2.transformer.wte.weight.size(1)
        self.proj = Linear(encoder.d-1, self.dim_gpt2)
        self.to_vocab_ac = copy.deepcopy(gpt2.lm_head)
        self.to_vocab_ctc = copy.deepcopy(gpt2.lm_head)
        self.tgt_dict = tgt_dict
        self.num_updates = 0
        self.args = args

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        parser.add_argument("--lambda-qua", type=float, default=0.1, metavar="D", help="lambda-qua")
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
            decode_length = torch.max(decode_length, torch.ones_like(decode_length))
        padding_mask = ~utils.sequence_mask(decode_length).bool()
        _alphas, num_output = self.resize(alphas, decode_length)
        cif_outputs = self.cif(hidden_encoded, _alphas)
        hidden_ac = self.proj(cif_outputs)
        logits = self.to_vocab_ac(hidden_ac)
        logits *= (~padding_mask).unsqueeze(-1).float()
        gold_rate = 0.0

        return {'logits': logits, 'len_logits': decode_length,
                'alphas': alphas, 'num_output': num_output, 'gold_rate': gold_rate,
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


@register_model("w2v_gpt2")
class W2V_GPT2(W2V_CIF):

    def __init__(self, args, encoder, gpt2, tgt_dict, tokenizer):
        """
        """
        super().__init__(args, encoder, gpt2, tgt_dict)
        self.gpt2 = gpt2
        self.to_vocab_lm = self.gpt2.lm_head
        self.tokenizer = tokenizer
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates

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
        tokenizer = task.target_dictionary.tokenizer

        gpt2 = cls.build_gpt2(args, tgt_dict)
        encoder = cls.build_encoder(args) # encoder

        return cls(args, encoder, gpt2, tgt_dict, tokenizer)

    # def forward(self, **kwargs):
    #     # decode_length = kwargs['target_lengths']
    #     # input_ids = kwargs['prev_output_tokens']
    #     input_ids = self.tokenizer.encode("The Manhattan bridge is a major")
    #     input_ids = torch.tensor([[self.tokenizer.bos_token_id] + input_ids]).cuda()
    #     # text_embs = self.gpt2.transformer.wte(input_ids)
    #     # res = self.gpt2(inputs_embeds=text_embs)[0]
    #     # decoded_ids = torch.argmax(res, -1)[0]
    #     # import pdb; pdb.set_trace()
    #     # sequence = self.tokenizer.decode(decoded_ids)
    #
    #     self.gpt2.eval()
    #     input_embs = self.gpt2.transformer.wte(input_ids)
    #     # with torch.no_grad():
    #     outputs = self.gpt2(inputs_embeds=input_embs)[0]
    #     decoded_ids = torch.argmax(outputs, -1)
    #     sequence = [self.tokenizer.decode(i) for i in decoded_ids]
    #
    #     # return {'logits': logits, 'len_logits': decode_length}

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
            decode_length = torch.max(decode_length, torch.ones_like(decode_length))
        _alphas, num_output = self.resize(alphas, decode_length)
        cif_outputs = self.cif(hidden_encoded, _alphas)
        hidden_ac = self.proj(cif_outputs)
        logits_ac = self.to_vocab_ac(hidden_ac)

        # other inputs
        B, T = hidden_ac.size(0), hidden_ac.size(1)
        padding_mask = ~utils.sequence_mask(decode_length).bool()
        position_ac = torch.arange(T).repeat(B, 1).long().cuda()
        type_ac = torch.ones((B, T)).long().cuda() * 103
        # [batch_size, num_heads, seq_length, seq_length]
        zeros = torch.zeros([B, 1, T, T]).cuda()
        ones = torch.ones([T, T]).cuda()[None, None, :, :]
        diag = torch.diag(torch.ones([T])).cuda()[None, None, :, :]
        tril = torch.tril(torch.ones([T, T])).cuda()[None, None, :, :]
        rm_padding_mask = (~padding_mask)[:, None, None, :] * \
                          (~padding_mask)[:, None, None, :].permute(0, 1, 3, 2)
        # mask_acQac = ones * rm_padding_mask
        mask_acQac = diag * rm_padding_mask
        mask_lmQac = diag * rm_padding_mask
        mask_lmQlm = tril * rm_padding_mask
        mask_ac = torch.cat([mask_acQac, zeros], dim=-1)
        mask_lm = torch.cat([mask_lmQac, mask_lmQlm], dim=-1)
        attention_mask = torch.cat([mask_ac, mask_lm], dim=-2)
        gold_rate = 0.0

        if self.training:
            self.gpt2.eval()
            input_ids = kwargs['prev_output_tokens']
            # input_ids = self.tokenizer.encode("The Manhattan bridge is a major")
            # input_ids = torch.tensor([[self.tokenizer.bos_token_id] + input_ids + [100] * 3]).cuda()
            text_embs = self.gpt2.transformer.wte(input_ids)
            ft = self.freeze_lm_finetune_updates <= self.num_updates
            with torch.no_grad() if not ft else contextlib.ExitStack():
                input_embs = text_embs
                # input_embs = torch.cat([hidden_ac, text_embs], dim=1)
                # token_type = torch.zeros_like(input_ids)
                # type_ids = torch.cat([type_ac, token_type], dim=1)
                # position_ids = torch.cat([position_ac+self.args.position_bias, position_ac], dim=1)
                outputs = self.gpt2(
                    inputs_embeds=input_embs,
                    # token_type_ids=type_ids if not self.args.no_type_id else None,
                    # position_ids=position_ids,
                    # attention_mask=attention_mask,
                )
            logits = outputs[0]

            # print(torch.argmax(logits, -1))
            print(torch.argmax(logits, -1)[:, -T:])
            import pdb; pdb.set_trace()
        else:
            list_logits = []
            token_type = torch.zeros_like(type_ac)
            decoded = torch.ones([B, 1], device=type_ac.device, dtype=type_ac.dtype) * self.tgt_dict.bos()
            text_embs = self.gpt2.transformer.wte(decoded)
            input_embs = text_embs
            # input_embs = torch.cat([hidden_ac, text_embs], dim=1)
            # type_ids = torch.cat([type_ac, token_type], dim=1)
            # position_ids = torch.cat([position_ac+self.args.position_bias, position_ac], dim=1)
            for i in range(T):
                outputs = self.gpt2(
                    inputs_embeds=input_embs,
                    # token_type_ids=type_ids[:, :T+i+1] if not self.args.no_type_id else None,
                    # position_ids=position_ids[:, :T+i+1],
                    # attention_mask=attention_mask[:, :, :T+i+1, :T+i+1]
                )
                logits_lm = outputs[0][..., -1, :]
                # logits_i = self.args.lambda_am * logits_ac[..., i, :] + self.args.lambda_lm * logits_lm
                logits_i = logits_lm
                list_logits.append(logits_i.unsqueeze(1))
                preds = torch.argmax(logits_i, -1)[:, None]
                cur_embs = self.gpt2.transformer.wte(preds)
                input_embs = torch.cat([input_embs, cur_embs], dim=1)
            logits = torch.cat(list_logits, 1)
        logits *= (~padding_mask).unsqueeze(-1).float()

        return {'logits': logits, 'len_logits': decode_length,
                'alphas': alphas, 'num_output': num_output, 'gold_rate': gold_rate,
                'logits_ctc': logits_ctc, 'len_logits_ctc': len_logits_ctc}

        return logits


@register_model("w2v_cif_gpt2")
class W2V_CIF_GPT2(W2V_CIF):

    def __init__(self, args, encoder, gpt2, tgt_dict):
        """
        """
        super().__init__(args, encoder, gpt2, tgt_dict)
        self.gpt2 = gpt2
        self.to_vocab_lm = self.gpt2.lm_head
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.gold_rate_range = eval(args.gold_rate_range)
        self.gold_rate_steps = eval(args.gold_rate_steps)

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_lm_args(parser)
        parser.add_argument("--lambda-ctc", type=float, metavar="D", help="lambda-ctc")

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
            decode_length = torch.max(decode_length, torch.ones_like(decode_length))
        _alphas, num_output = self.resize(alphas, decode_length)
        cif_outputs = self.cif(hidden_encoded, _alphas)
        hidden_ac = self.proj(cif_outputs)
        logits_ac = self.to_vocab_ac(hidden_ac)

        # other inputs
        B, T = hidden_ac.size(0), hidden_ac.size(1)
        padding_mask = ~utils.sequence_mask(decode_length).bool()
        position_ac = torch.arange(T).repeat(B, 1).long().cuda()
        type_ac = torch.ones((B, T)).long().cuda() * 103
        # [batch_size, num_heads, seq_length, seq_length]
        zeros = torch.zeros([B, 1, T, T]).cuda()
        ones = torch.ones([T, T]).cuda()[None, None, :, :]
        diag = torch.diag(torch.ones([T])).cuda()[None, None, :, :]
        tril = torch.tril(torch.ones([T, T])).cuda()[None, None, :, :]
        rm_padding_mask = (~padding_mask)[:, None, None, :] * \
                          (~padding_mask)[:, None, None, :].permute(0, 1, 3, 2)
        # mask_acQac = ones * rm_padding_mask
        mask_acQac = diag * rm_padding_mask
        mask_lmQac = diag * rm_padding_mask
        mask_lmQlm = tril * rm_padding_mask
        mask_ac = torch.cat([mask_acQac, zeros], dim=-1)
        mask_lm = torch.cat([mask_lmQac, mask_lmQlm], dim=-1)
        attention_mask = torch.cat([mask_ac, mask_lm], dim=-2)

        if self.training:
            input_ids = kwargs['prev_output_tokens']
            gold_rate = self.set_gold_rate()
            input_ids = self.schedule_samlping(gold_rate, input_ids, logits_ac, padding_mask)
            text_embs = self.gpt2.transformer.wte(input_ids)
            ft = self.freeze_lm_finetune_updates <= self.num_updates
            with torch.no_grad() if not ft else contextlib.ExitStack():
                input_embs = torch.cat([hidden_ac, text_embs], dim=1)
                token_type = torch.zeros_like(input_ids)
                type_ids = torch.cat([type_ac, token_type], dim=1)
                position_ids = torch.cat([position_ac+self.args.position_bias, position_ac], dim=1)
                outputs = self.gpt2(
                    inputs_embeds=input_embs,
                    # token_type_ids=type_ids if not self.args.no_type_id else None,
                    # position_ids=position_ids,
                    # attention_mask=attention_mask,
                )
            logits_lm = outputs[0][:, T:, :]
            logits = self.args.lambda_am * logits_ac + self.args.lambda_lm * logits_lm
        else:
            gold_rate = 0.0
            list_logits = []
            token_type = torch.zeros_like(type_ac)
            decoded = torch.ones([B, 1], device=type_ac.device, dtype=type_ac.dtype) * self.tgt_dict.bos()
            text_embs = self.gpt2.transformer.wte(decoded)
            input_embs = torch.cat([hidden_ac, text_embs], dim=1)
            type_ids = torch.cat([type_ac, token_type], dim=1)
            position_ids = torch.cat([position_ac+self.args.position_bias, position_ac], dim=1)
            for i in range(T):
                outputs = self.gpt2(
                    inputs_embeds=input_embs,
                    # token_type_ids=type_ids[:, :T+i+1] if not self.args.no_type_id else None,
                    # position_ids=position_ids[:, :T+i+1],
                    # attention_mask=attention_mask[:, :, :T+i+1, :T+i+1]
                )
                logits_lm = outputs[0][..., -1, :]
                # logits_i = self.args.lambda_am * logits_ac[..., i, :] + self.args.lambda_lm * logits_lm
                logits_i = logits_lm
                list_logits.append(logits_i.unsqueeze(1))
                preds = torch.argmax(logits_i, -1)[:, None]
                cur_embs = self.gpt2.transformer.wte(preds)
                input_embs = torch.cat([input_embs, cur_embs], dim=1)
            logits = torch.cat(list_logits, 1)
        logits *= (~padding_mask).unsqueeze(-1).float()

        return {'logits': logits, 'len_logits': decode_length,
                'alphas': alphas, 'num_output': num_output, 'gold_rate': gold_rate,
                'logits_ctc': logits_ctc, 'len_logits_ctc': len_logits_ctc}

        return logits

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def set_gold_rate(self):
        s, e = self.gold_rate_range
        s1, s2 = self.gold_rate_steps
        gold_rate = max((1 - max((self.num_updates - s1), 0) / s2) * (s-e), 0) + e

        return gold_rate

    def schedule_samlping(self, gold_rate, input_ids, logits_ac, padding_mask):
        bos = input_ids[:, 0:1]
        preds_ac = torch.argmax(logits_ac, -1)[:, :-1]
        sample = torch.rand(preds_ac.size()).cuda() > gold_rate
        mixed_input_ids = torch.where(sample, preds_ac, input_ids[:, 1:])
        mixed_input_ids = torch.cat([bos, mixed_input_ids], dim=1) * (~padding_mask)

        return mixed_input_ids


@register_model("w2v_cif_gpt2_v2")
class W2V_CIF_GPT2_v2(W2V_CIF_GPT2):

    @classmethod
    def build_gpt2(cls, args, tgt_dict):
        model = GPT2LMHeadFinetuneModel.from_pretrained(args.gpt2_name)

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
            decode_length = torch.max(decode_length, torch.ones_like(decode_length))
        _alphas, num_output = self.resize(alphas, decode_length)
        cif_outputs = self.cif(hidden_encoded, _alphas)
        hidden_ac = self.proj(cif_outputs)
        logits_ac = self.to_vocab_ac(hidden_ac)

        # other inputs
        B, T = hidden_ac.size(0), hidden_ac.size(1)
        padding_mask = ~utils.sequence_mask(decode_length).bool()
        # [batch_size, num_heads, seq_length, seq_length]
        # zeros = torch.zeros([B, 1, T, T]).cuda()
        ones = torch.ones([B, 1, T, T]).cuda()
        diag = torch.diag(torch.ones([T])).cuda()[None, None, :, :]
        tril = torch.tril(torch.ones([T, T])).cuda()[None, None, :, :]
        rm_padding_mask = (~padding_mask)[:, None, None, :] * \
                          (~padding_mask)[:, None, None, :].permute(0, 1, 3, 2)
        # mask_acQac = ones * rm_padding_mask
        # mask_lmQac = diag * rm_padding_mask
        # mask_lmQac = zeros
        mask_lmQac = ones * rm_padding_mask
        mask_lmQlm = tril * rm_padding_mask
        mask_lm = torch.cat([mask_lmQac, mask_lmQlm], dim=-1)
        # mask_ac = torch.ones_like(mask_lm)
        attention_mask = torch.cat([mask_lm, mask_lm], dim=-2)

        if self.training:
            input_ids = kwargs['prev_output_tokens']
            gold_rate = self.set_gold_rate()
            input_ids = self.schedule_samlping(gold_rate, input_ids, logits_ac, padding_mask)
            text_embs = self.gpt2.transformer.wte(input_ids)
            ft = self.freeze_lm_finetune_updates <= self.num_updates
            with torch.no_grad() if not ft else contextlib.ExitStack():
                outputs = self.gpt2(
                    inputs_embeds=text_embs,
                    external_embeds=hidden_ac,
                    # attention_mask=attention_mask,
                )
            logits_lm = outputs[0]
            logits = self.args.lambda_am * logits_ac + self.args.lambda_lm * logits_lm
        else:
            gold_rate = 0.0
            list_logits = []
            device, dtype = kwargs['prev_output_tokens'].device, kwargs['prev_output_tokens'].dtype
            decoded = torch.ones([B, 1], device=device, dtype=dtype) * self.tgt_dict.bos()
            text_embs = self.gpt2.transformer.wte(decoded)
            for i in range(T):
                outputs = self.gpt2(
                    inputs_embeds=text_embs,
                    external_embeds=hidden_ac,
                    # attention_mask=attention_mask[:, :, :T+i+1, :T+i+1]
                )
                logits_lm = outputs[0][..., -1, :]
                logits_i = self.args.lambda_am * logits_ac[..., i, :] + self.args.lambda_lm * logits_lm
                list_logits.append(logits_i.unsqueeze(1))
                preds = torch.argmax(logits_i, -1)[:, None]
                cur_embs = self.gpt2.transformer.wte(preds)
                text_embs = torch.cat([text_embs, cur_embs], dim=1)
            logits = torch.cat(list_logits, 1)
        logits *= (~padding_mask).unsqueeze(-1).float()

        return {'logits': logits, 'len_logits': decode_length,
                'alphas': alphas, 'num_output': num_output, 'gold_rate': gold_rate,
                'logits_ctc': logits_ctc, 'len_logits_ctc': len_logits_ctc}

        return logits


@register_model_architecture("w2v_cif", "w2v_cif")
def w2v_cif_gpt2_architecture(args):
    cif_architecture(args)


@register_model_architecture("w2v_gpt2", "w2v_gpt2")
def w2v_cif_gpt2_architecture(args):
    cif_architecture(args)
    args.no_type_id = getattr(args, "no_type_id", False)


@register_model_architecture("w2v_cif_gpt2", "w2v_cif_gpt2")
def w2v_cif_gpt2_architecture(args):
    cif_architecture(args)
    args.no_type_id = getattr(args, "no_type_id", False)


@register_model_architecture("w2v_cif_gpt2_v2", "w2v_cif_gpt2_v2")
def w2v_cif_gpt2_architecture(args):
    cif_architecture(args)
    args.no_type_id = getattr(args, "no_type_id", False)
