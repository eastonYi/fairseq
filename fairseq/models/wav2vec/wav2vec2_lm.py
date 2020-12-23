# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

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
    GradMultiply,
    PositionalEmbedding,
    TransformerDecoderLayer,
    TransposeLast,
    Fp32LayerNorm,
    Fp32GroupNorm,
    FairseqDropout
)

from .wav2vec2_ctc import (
    add_common_args,
    base_architecture
)
from .wav2vec2_cif import (
    CIFFcModel,
    CIFFcModelV2,
    Assigner,
    add_cif_args,
    cif_architecture,
)
from .wav2vec2_seq2seq import (
    Wav2VecEncoder,
    Linear
)


def padding2attention_mask(padding_mask):

    mask1 = F.pad(padding_mask, [0, 1, 0, 0], value=1)
    mask2 = F.pad(padding_mask, [1, 0, 0, 0], value=0)
    mask = 1 - mask1.int() * mask2.int()

    return F.pad(mask, [1, 0, 0, 0], value=1)


def pred2bert_input(pred, token_mask, cls=101, sep=102):

    pred *= token_mask
    end_index = token_mask.sum(-1).long()
    pred.scatter_(dim=-1, index=end_index.unsqueeze(1)+1, value=sep)
    pred[:, 0] = cls

    return pred


def add_lm_args(parser):
    parser.add_argument(
        "--freeze-lm-finetune-updates", type=int, help="freeze_lm_finetune_updates"
    )
    parser.add_argument(
        "--gold-rate-range", type=str, help="gold-rate-range"
    )
    parser.add_argument(
        "--gold-updates", type=float, default=50000.0, metavar="D", help="gold-updates"
    )
    parser.add_argument(
        "--infer-threash", type=float, default=0.8, help="infer-threash"
    )
    parser.add_argument(
        "--lm-path", type=str, help="dim-hidden-mixer"
    )
    parser.add_argument(
        "--bert-name", type=str, metavar="D", help="bert_name"
    )
    parser.add_argument(
        "--lambda-embedding", type=float, metavar="D", help="lambda-embedding"
    )
    parser.add_argument(
        "--lambda-lm", type=float, default=0.2, metavar="D", help="lambda-lm"
    )


@register_model("wav2vec_cif_bert")
class W2V_MIX_CIF_BERT(BaseFairseqModel):

    def __init__(self, args, encoder, assigner, lm, tgt_dict):
        super().__init__()
        self.encoder = encoder
        self.assigner = assigner
        self.bert = lm
        self.tgt_dict = tgt_dict
        self.num_updates = 0
        self.args = args
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.proj = Linear(encoder.d, lm.embeddings.word_embeddings.weight.size(1))
        self.final_proj = Linear(lm.embeddings.word_embeddings.weight.size(1), len(tgt_dict))
        self.gold_rate_range = eval(args.gold_rate_range)

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_cif_args(parser)
        add_lm_args(parser)
        parser.add_argument("--lambda-embedding", type=float, metavar="D", help="lambda-embedding")
        parser.add_argument("--gold-updates", type=float, default=50000.0, metavar="D", help="gold-updates")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        w2v_cif_bert_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        lm = cls.build_bert(args)
        encoder = cls.build_encoder(args) # encoder
        assigner = cls.build_assigner(args, encoder.d)

        tgt_dict = task.target_dictionary

        return cls(args, encoder, assigner, lm, tgt_dict)

    @classmethod
    def build_encoder(cls, args, tgt_dict=None):
        return Wav2VecEncoder(args, tgt_dict=tgt_dict)

    @classmethod
    def build_assigner(cls, args, dim_input):
        return Assigner(args, dim_input)

    @classmethod
    def build_bert(cls, args):
        from transformers import BertModel, BertTokenizer

        bert = BertModel.from_pretrained(args.bert_name)

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
        if self.training:
            _alphas, num_output = self.resize(alphas, kwargs['target_lengths'])
            padding_mask = ~utils.sequence_mask(kwargs['target_lengths']).bool()
        else:
            _alphas, num_output = self.resize(alphas)
            padding_mask = ~utils.sequence_mask(torch.round(num_output).int()).bool()

        cif_outputs = self.cif(encoder_output, _alphas)
        hidden = self.proj(cif_outputs)

        if self.training:
            gold_rate = self.set_gold_rate()
            # gold_rate = 1.0
            input_ids = kwargs['target'].long()
        else:
            input_ids = None
            gold_rate = 0.0

        bert_output, gold_embedding, pred_mask = self.forward_embeded(
            hidden, padding_mask, input_ids, gold_rate)

        logits = self.final_proj(bert_output)

        return {'logits': logits, 'len_logits': kwargs['target_lengths'],
                'alphas': alphas, 'num_output': num_output, 'alphas_pen': alphas_pen,
                'embedding': hidden, 'gold_embedding': gold_embedding, 'pred_mask': pred_mask,
                'gold_rate': gold_rate}

    def forward_embeded(self, hidden, padding_mask, input_ids=None, gold_rate=0.0):
        device = hidden.device

        if input_ids is not None:
            input_ids, edge_mask = self.index_convert(input_ids, padding_mask)
            gold_embedding = self.bert.embeddings.word_embeddings(input_ids)
            pred_mask = (torch.rand(input_ids.size(), device=device) > gold_rate) * (~padding_mask) * (~edge_mask)
            hidden_mix = torch.where(pred_mask[:, :, None].repeat(1, 1, hidden.size(-1)),
                                     hidden,
                                     gold_embedding)
        else:
            _, edge_mask = self.index_convert(None, padding_mask)
            gold_embedding = pred_mask = None
            hidden_mix = hidden

        embeddings = self.bert.embeddings(inputs_embeds=hidden_mix)
        encoder_outputs = self.bert.encoder(
            embeddings,
            attention_mask=(~padding_mask).int()[:, None, None, :])

        sequence_output = encoder_outputs[0]

        return sequence_output, gold_embedding, pred_mask

    @staticmethod
    def resize(*args, **kwargs):
        return CIFFcModel.resize(*args, **kwargs)

    @staticmethod
    def cif(*args, **kwargs):
        return CIFFcModel.cif(*args, **kwargs)

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

    def set_gold_rate(self):
        s, e = self.gold_rate_range
        gold_rate = max((1 - self.num_updates / self.args.gold_updates) * (s-e), 0) + e

        return gold_rate


@register_model("wav2vec_cif2_bert")
class W2V_MIX_CIF2_BERT(W2V_MIX_CIF_BERT):

    def __init__(self, args, encoder, lm, tgt_dict):
        BaseFairseqModel.__init__(self)
        self.encoder = encoder
        self.bert = lm
        self.tgt_dict = tgt_dict
        self.num_updates = 0
        self.args = args
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.proj = Linear(encoder.d-1, lm.embeddings.word_embeddings.weight.size(1))
        self.final_proj = Linear(lm.embeddings.word_embeddings.weight.size(1), len(tgt_dict))
        self.gold_rate_range = eval(args.gold_rate_range)

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_cif_args(parser)
        add_lm_args(parser)
        parser.add_argument("--lambda-embedding", type=float, metavar="D", help="lambda-embedding")
        parser.add_argument("--gold-updates", type=float, default=50000.0, metavar="D", help="gold-updates")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        w2v_cif_bert_architecture(args)

        lm = cls.build_bert(args)
        encoder = cls.build_encoder(args) # encoder

        tgt_dict = task.target_dictionary

        return cls(args, encoder, lm, tgt_dict)

    def forward(self, **kwargs):
        """
        encoder_output= "encoder_out": x,
                        "encoded": encoded,
                        "encoder_padding_mask": padding_mask,  # B x T
                        "padding_mask": padding_mask,
        """
        encoder_output = self.encoder(tbc=False, **kwargs)
        alphas = CIFFcModelV2.get_alphas(encoder_output)
        if self.training:
            _alphas, num_output = self.resize(alphas, kwargs['target_lengths'])
            padding_mask = ~utils.sequence_mask(kwargs['target_lengths']).bool()
        else:
            _alphas, num_output = self.resize(alphas)
            padding_mask = ~utils.sequence_mask(torch.round(num_output).int()).bool()

        cif_outputs = self.cif(encoder_output['encoder_out'][:, :, :-1], _alphas)
        hidden = self.proj(cif_outputs)

        if self.training:
            gold_rate = self.set_gold_rate()
            input_ids = kwargs['bert_input'].long()
        else:
            input_ids = None
            gold_rate = 0.0

        bert_output, gold_embedding, pred_mask = self.forward_embeded(
            hidden, padding_mask, input_ids, gold_rate)

        logits = self.final_proj(bert_output)

        return {'logits': logits, 'len_logits': kwargs['target_lengths'],
                'alphas': alphas, 'num_output': num_output,
                'embedding': hidden, 'gold_embedding': gold_embedding, 'pred_mask': pred_mask,
                'gold_rate': gold_rate}

    def forward_embeded(self, hidden, padding_mask, input_ids=None, gold_rate=0.0):
        """
        """
        device = hidden.device

        if input_ids is not None:
            token_mask = input_ids.ne(self.tgt_dict.cls()) * \
                         input_ids.ne(self.tgt_dict.sep()) * \
                         input_ids.ne(self.tgt_dict.pad())
            gold_embedding = self.bert.embeddings.word_embeddings(input_ids)
            pred_mask = (torch.rand(input_ids.size(), device=device) > gold_rate) * token_mask
            hidden_mix = torch.where(pred_mask[:, :, None].repeat(1, 1, hidden.size(-1)),
                                     F.pad(hidden, [0, 0, 1, 1, 0, 0], value=0),
                                     gold_embedding)
        else:
            gold_embedding = pred_mask = None
            hidden_mix = F.pad(hidden, [0, 0, 1, 1, 0, 0], value=0)

        attention_mask = padding2attention_mask(padding_mask)

        embeddings = self.bert.embeddings(inputs_embeds=hidden_mix)
        encoder_outputs = self.bert.encoder(
            embeddings,
            attention_mask=attention_mask[:, None, None, :])

        import pdb; pdb.set_trace()
        sequence_output = encoder_outputs[0][:, 1:-1, :] * (~padding_mask)[:, :, None]

        return sequence_output, gold_embedding, pred_mask


@register_model("wav2vec_cif2_bert_2")
class W2V_MIX_CIF2_BERT_2(W2V_MIX_CIF2_BERT):

    def __init__(self, args, encoder, bert, to_vocab, tgt_dict):
        BaseFairseqModel.__init__(self)
        self.encoder = encoder
        self.bert = bert
        self.to_vocab = to_vocab
        self.tgt_dict = tgt_dict
        self.num_updates = 0
        self.args = args
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.proj = Linear(encoder.d-1, bert.embeddings.word_embeddings.weight.size(1))
        if args.share_final_proj:
            self.to_vocab_ac = to_vocab
        else:
            self.to_vocab_ac = copy.deepcopy(to_vocab)
        # self.proj2 = Linear(encoder.d-1, len(tgt_dict))
        self.gold_rate_range = eval(args.gold_rate_range)

        # for p in self.to_vocab.parameters():
        #     p.requires_grad = False
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_cif_args(parser)
        add_lm_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        w2v_cif_bert_architecture(args)
        tgt_dict = task.target_dictionary

        bert, to_vocab = cls.build_bert(args)
        encoder = cls.build_encoder(args) # encoder

        return cls(args, encoder, bert, to_vocab, tgt_dict)

    @classmethod
    def build_bert(cls, args):
        from transformers import BertForMaskedLM

        pretrained_model = BertForMaskedLM.from_pretrained(args.bert_name)
        bert = pretrained_model.bert
        to_vocab = pretrained_model.cls

        return bert, to_vocab

    def forward(self, **kwargs):
        """
        encoder_output= "encoder_out": x,
                        "encoded": encoded,
                        "encoder_padding_mask": padding_mask,  # B x T
                        "padding_mask": padding_mask,
        """
        encoder_output = self.encoder(tbc=False, **kwargs)
        alphas = CIFFcModelV2.get_alphas(encoder_output)
        input_ids = kwargs['bert_input'].long()
        if self.training:
            _alphas, num_output = self.resize(alphas, kwargs['target_lengths'])
            padding_mask = ~utils.sequence_mask(kwargs['target_lengths']).bool()
            gold_rate = self.set_gold_rate()
        else:
            decode_length = kwargs['decode_length']
            # _alphas, num_output = self.resize(alphas)
            # padding_mask = ~utils.sequence_mask(torch.round(num_output).int()).bool()
            _alphas, num_output = self.resize(alphas, decode_length)
            padding_mask = ~utils.sequence_mask(decode_length).bool()
            gold_rate = 0.0

        cif_outputs = self.cif(encoder_output['encoder_out'][:, :, :-1], _alphas)
        hidden = self.proj(cif_outputs)
        logits_ac = self.to_vocab_ac(hidden)

        logits, gold_embedding, pred_mask, token_mask = self.bert_forward(
            hidden, logits_ac, padding_mask, input_ids, gold_rate, threash=self.args.infer_threash)
        # logits = GradMultiply.apply(logits, 0.1)
        logits = logits_ac + 0.1 * logits

        return {'logits': logits, 'len_logits': kwargs['target_lengths'],
                'alphas': alphas, 'num_output': num_output,
                'embedding': hidden, 'gold_embedding': gold_embedding,
                'pred_mask': pred_mask, 'token_mask': token_mask,
                'gold_rate': gold_rate}

    def bert_forward(self, hidden, logits_ac, padding_mask, input_ids=None, gold_rate=0.0, threash=0.8):
        """
        """
        device = hidden.device

        if self.training:
            token_mask = input_ids.ne(self.tgt_dict.cls()) * \
                         input_ids.ne(self.tgt_dict.sep()) * \
                         input_ids.ne(self.tgt_dict.pad())
            gold_embedding = self.bert.embeddings.word_embeddings(input_ids)
            pred_mask = (torch.rand(input_ids.size(), device=device) > gold_rate) * token_mask
        else: # infer
            token_mask = F.pad(~padding_mask, [1, 1, 0, 0], value=0)
            probs = F.pad(utils.softmax(logits_ac.float(), dim=-1), [0, 0, 1, 1, 0, 0], value=0)
            confident, preds = probs.max(-1)
            preds_ids = pred2bert_input(preds, token_mask)
            # preds = torch.where(token_mask, preds, input_ids)
            gold_embedding = self.bert.embeddings.word_embeddings(preds_ids)
            pred_mask = (confident < threash) * token_mask

        hidden_mix = torch.where(pred_mask[:, :, None].repeat(1, 1, hidden.size(-1)),
                                 F.pad(hidden, [0, 0, 1, 1, 0, 0], value=0),
                                 gold_embedding)

        attention_mask = padding2attention_mask(padding_mask)

        embeddings = self.bert.embeddings(inputs_embeds=hidden_mix)
        encoder_outputs = self.bert.encoder(
            embeddings,
            attention_mask=attention_mask[:, None, None, :])

        logits = self.to_vocab(encoder_outputs[0])
        logits = logits[:, 1:-1, :]

        return logits, gold_embedding, pred_mask, token_mask


@register_model("wav2vec_ctc_cif2_bert")
class W2V_MIX_CTC_CIF2_BERT(W2V_MIX_CIF2_BERT_2):

    def __init__(self, args, encoder, bert, to_vocab, tgt_dict):
        """
        .copy_() clone to_vocab
        """
        BaseFairseqModel.__init__(self)
        self.encoder = encoder
        self.bert = bert
        self.to_vocab = to_vocab # 768 -> 21128
        self.to_vocab_ac = copy.deepcopy(to_vocab)
        self.to_vocab_ctc = copy.deepcopy(to_vocab)
        self.proj = Linear(encoder.d-1, bert.embeddings.word_embeddings.weight.size(1))
        self.tgt_dict = tgt_dict
        self.num_updates = 0
        self.args = args
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.gold_rate_range = eval(args.gold_rate_range)

        for p in self.bert.embeddings.parameters():
            p.requires_grad = False

    @staticmethod
    def add_args(parser):
        add_common_args(parser)
        add_cif_args(parser)
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
        hidden_ctc = F.pad(encoder_output['encoder_out'][:, :, :-1], [0, 1, 0, 0, 0, 0], value=0)
        logits_ctc = self.to_vocab_ctc(hidden_ctc)
        len_logits_ctc = (~encoder_output['padding_mask']).sum(-1).long()
        alphas = CIFFcModelV2.get_alphas(encoder_output)
        if self.training:
            input_ids = kwargs['bert_input'].long()
            gold_rate = self.set_gold_rate()
            decode_length = kwargs['target_lengths']
            _alphas, num_output = self.resize(alphas, decode_length, noise=0.0)
        else:
            input_ids = None
            gold_rate = 0.0
            decode_length = torch.round(alphas.sum(-1)).int()
            _alphas, num_output = self.resize(alphas, decode_length, noise=0.0)

        padding_mask = ~utils.sequence_mask(decode_length).bool()
        cif_outputs = self.cif(encoder_output['encoder_out'][:, :, :-1], _alphas)
        hidden = self.proj(cif_outputs)
        logits_ac = self.to_vocab_ac(hidden)

        logits, gold_embedding, pred_mask, token_mask = self.bert_forward(
            hidden, logits_ac, padding_mask, input_ids, gold_rate,
            threash=self.args.infer_threash)
        # logits = GradMultiply.apply(logits, 0.2)
        logits = logits_ac + self.args.lambda_lm * logits

        return {'logits': logits, 'len_logits': decode_length,
                'alphas': alphas, 'num_output': num_output,
                'logits_ctc': logits_ctc, 'len_logits_ctc': len_logits_ctc,
                'pred_mask': pred_mask, 'token_mask': token_mask,
                'gold_rate': gold_rate}

    def get_ctc_num_output(self, **kwargs):
        encoder_output = self.encoder(tbc=False, **kwargs)
        logits_ctc = self.to_vocab_ctc(encoder_output['encoder_out'])
        ctc_greedy_res = logits_ctc.argmax(-1) * (~encoder_output['padding_mask'])
        num_ctc_output = ctc_greedy_res.unique_consecutive(dim=-1).gt(1).sum(-1).long()

        return num_ctc_output

    def get_normalized_probs(self, net_output, log_probs):
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

        return res_ctc, res


@register_model_architecture("wav2vec_cif_bert", "wav2vec_cif_bert")
def w2v_cif_bert_architecture(args):
    cif_architecture(args)
    args.freeze_lm_finetune_updates = getattr(args, "freeze_lm_finetune_updates", 1000)


@register_model_architecture("wav2vec_cif2_bert", "wav2vec_cif2_bert")
def w2v_cif_bert_architecture(args):
    cif_architecture(args)
    args.freeze_lm_finetune_updates = getattr(args, "freeze_lm_finetune_updates", 1000)


@register_model_architecture("wav2vec_cif2_bert_2", "wav2vec_cif2_bert_2")
def w2v_cif_bert_architecture(args):
    cif_architecture(args)
    args.freeze_lm_finetune_updates = getattr(args, "freeze_lm_finetune_updates", 1000)
    args.share_final_proj = getattr(args, "share_final_proj", False)


@register_model_architecture("wav2vec_ctc_cif2_bert", "wav2vec_ctc_cif2_bert")
def w2v_cif_bert_architecture(args):
    cif_architecture(args)
    args.freeze_lm_finetune_updates = getattr(args, "freeze_lm_finetune_updates", 1000)
    args.share_final_proj = getattr(args, "share_final_proj", False)
