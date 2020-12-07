# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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

PAD_IDX = 1
EOS_IDX = 2


def padding2attention_mask(padding_mask):
    """
    gen_edge_mask(torch.tensor([3,4]), True)

    tensor([[ True, False, False, False,  True, False],
        [ True, False, False, False, False,  True]])
    """
    mask1 = F.pad(padding_mask, [0, 1, 0, 0], value=1)
    mask2 = F.pad(padding_mask, [1, 0, 0, 0], value=0)
    mask = 1 - mask1.int() * mask2.int()

    return F.pad(mask, [1, 0, 0, 0], value=1)


def add_lm_args(parser):
    parser.add_argument(
        "--freeze-lm-finetune-updates", type=int, help="freeze_lm_finetune_updates"
    )
    parser.add_argument(
        "--gold-rate-range", type=str, help="gold-rate-range"
    )
    parser.add_argument(
        "--lm-path", type=str, help="dim-hidden-mixer"
    )
    parser.add_argument("--bert-name", type=str, metavar="D", help="bert_name")


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
        self.final_proj = lambda x: x
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

        lm = cls.build_bert(args, task.dictionary)
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
        return CIFFcModel.resize(*args, **kwargs)

    @staticmethod
    def cif(*args, **kwargs):
        return CIFFcModel.cif(*args, **kwargs)

    @classmethod
    def build_bert(cls, args, dictionary):
        from fairseq.models.masked_lm import MaskedLMModel as LM
        args.tokens_per_sample = 512
        return LM.build_model(args, task=None, dictionary=dictionary)

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

        if cif_outputs.size(1) == 0:
            import pdb; pdb.set_trace()

        bert_output, gold_embedding, pred_mask = self.forward_embeded(
            hidden, padding_mask, input_ids, gold_rate)

        logits = self.final_proj(bert_output)

        return {'logits': logits, 'len_logits': kwargs['target_lengths'],
                'alphas': alphas, 'num_output': num_output, 'alphas_pen': alphas_pen,
                'embedding': hidden, 'gold_embedding': gold_embedding, 'pred_mask': pred_mask,
                'gold_rate': gold_rate}

    def forward_embeded(self, hidden, padding_mask, input_ids=None, gold_rate=0.0):
        """
        """
        device = hidden.device
        lm = self.lm.encoder.encoder.sentence_encoder
        lm_ = self.lm.encoder.encoder
        mask = (~padding_mask[:, :, None]).float().repeat(1, 1, hidden.size(-1))

        if input_ids is not None:
            gold_embedding = lm.embed_tokens(input_ids)
            pred_mask = (torch.rand(input_ids.size(), device=device) > gold_rate) * (~padding_mask)
            hidden_mix = torch.where(pred_mask[:, :, None].repeat(1, 1, hidden.size(-1)),
                                     hidden,
                                     gold_embedding)
            hidden_mix *= mask
            embeddings = hidden_mix
        else:
            gold_embedding = pred_mask = None
            embeddings = hidden

        x = lm.forward_embeded(embeddings, padding_mask)

        x = lm_.layer_norm(lm_.activation_fn(lm_.lm_head_transform_weight(x)))

        x = lm_.embed_out(x)

        return x, gold_embedding, pred_mask

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


@register_model("wav2vec_cif_bertEx")
class W2V_MIX_CIF_BERTEX(W2V_MIX_CIF_BERT):

    def __init__(self, args, encoder, assigner, lm, tgt_dict, tokenizer):
        BaseFairseqModel.__init__(self)
        self.encoder = encoder
        self.assigner = assigner
        self.lm = lm
        self.tgt_dict = tgt_dict
        self.tokenizer = tokenizer
        self.num_updates = 0
        self.args = args
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.proj = Linear(encoder.d, lm.embeddings.word_embeddings.weight.size(1))
        self.final_proj = Linear(lm.embeddings.word_embeddings.weight.size(1), len(tgt_dict))
        self.gold_rate_range = eval(args.gold_rate_range)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        w2v_cif_bert_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        lm, tokenizer = cls.build_bert(args)
        encoder = cls.build_encoder(args) # encoder
        assigner = cls.build_assigner(args, encoder.d)

        tgt_dict = task.target_dictionary

        return cls(args, encoder, assigner, lm, tgt_dict, tokenizer)

    @classmethod
    def build_bert(cls, args):
        from transformers import BertModel, BertTokenizer

        bert = BertModel.from_pretrained(args.bert_name)
        tokenizer = BertTokenizer.from_pretrained(args.bert_name)

        return bert, tokenizer

    def forward_embeded(self, hidden, padding_mask, input_ids=None, gold_rate=0.0):
        # import pdb; pdb.set_trace()
        device = hidden.device

        if input_ids is not None:
            input_ids, edge_mask = self.index_convert(input_ids, padding_mask)
            gold_embedding = self.lm.embeddings.word_embeddings(input_ids)
            pred_mask = (torch.rand(input_ids.size(), device=device) > gold_rate) * (~padding_mask) * (~edge_mask)
            hidden_mix = torch.where(pred_mask[:, :, None].repeat(1, 1, hidden.size(-1)),
                                     hidden,
                                     gold_embedding)
        else:
            _, edge_mask = self.index_convert(None, padding_mask)
            gold_embedding = pred_mask = None
            hidden_mix = hidden

        embeddings = self.lm.embeddings(inputs_embeds=hidden_mix)
        encoder_outputs = self.lm.encoder(
            embeddings,
            attention_mask=(~padding_mask).int()[:, None, None, :])

        sequence_output = encoder_outputs[0]

        return sequence_output, gold_embedding, pred_mask

    def index_convert(self, input_ids, padding_mask):
        """
        pad: 0, unk: 100
        """
        max_len = input_ids.size(1)
        lengths = (~padding_mask).sum(-1)
        list_sents = []
        for sent_ids, length in zip(input_ids, lengths):
            sent = self.tgt_dict.string(sent_ids[:length])
            sent = sent.replace('<unk>', '`').replace('[NOISE]', '`').replace('[VOCALIZED-NOISE]', '`').replace('[LAUGHTER]', '`')
            sent_ids_ = self.tokenizer(sent)['input_ids'][1:-1] + [0] * (max_len-length)
            list_sents.append(sent_ids_)

        return torch.tensor(list_sents).type_as(input_ids), padding_mask


@register_model("wav2vec_cif2_bert")
class W2V_MIX_CIF2_BERT(W2V_MIX_CIF_BERT):

    def __init__(self, args, encoder, lm):
        BaseFairseqModel.__init__(self)
        self.encoder = encoder
        self.lm = lm
        self.num_updates = 0
        self.args = args
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.proj = Linear(encoder.d-1, lm.encoder.encoder.sentence_encoder.embedding_dim)
        self.final_proj = lambda x: x
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

        lm = cls.build_bert(args, task.dictionary)
        encoder = cls.build_encoder(args) # encoder

        return cls(args, encoder, lm)

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
            input_ids = kwargs['target'].long()
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


@register_model("wav2vec_cif2_bertEx")
class W2V_MIX_CIF2_BERTEX(W2V_MIX_CIF2_BERT):

    def __init__(self, args, encoder, lm, tgt_dict, tokenizer):
        BaseFairseqModel.__init__(self)
        self.encoder = encoder
        self.lm = lm
        self.tgt_dict = tgt_dict
        self.tokenizer = tokenizer
        self.num_updates = 0
        self.args = args
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.proj = Linear(encoder.d-1, lm.embeddings.word_embeddings.weight.size(1))
        self.final_proj = Linear(lm.embeddings.word_embeddings.weight.size(1), len(tgt_dict))
        self.gold_rate_range = eval(args.gold_rate_range)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        w2v_cif_bert_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        lm, tokenizer = cls.build_bert(args)
        encoder = cls.build_encoder(args) # encoder

        tgt_dict = task.target_dictionary

        return cls(args, encoder, lm, tgt_dict, tokenizer)

    @classmethod
    def build_bert(cls, args):
        from transformers import BertModel, BertTokenizer

        bert = BertModel.from_pretrained(args.bert_name)
        tokenizer = BertTokenizer.from_pretrained(args.bert_name)

        return bert, tokenizer

    def forward_embeded(self, hidden, padding_mask, input_ids=None, gold_rate=0.0):
        """
        """
        device = hidden.device

        if input_ids is not None:
            input_ids_, token_mask = self.index_convert(input_ids, padding_mask)
            gold_embedding = self.lm.embeddings.word_embeddings(input_ids_)
            pred_mask = (torch.rand(input_ids_.size(), device=device) > gold_rate) * token_mask
            hidden_mix = torch.where(pred_mask[:, :, None].repeat(1, 1, hidden.size(-1)),
                                     F.pad(hidden, [0, 0, 1, 1, 0, 0], value=0),
                                     gold_embedding)
        else:
            gold_embedding = pred_mask = None
            hidden_mix = F.pad(hidden, [0, 0, 1, 1, 0, 0], value=0)

        attention_mask = padding2attention_mask(padding_mask)

        embeddings = self.lm.embeddings(inputs_embeds=hidden_mix)
        encoder_outputs = self.lm.encoder(
            embeddings,
            attention_mask=attention_mask[:, None, None, :])

        sequence_output = encoder_outputs[0][:, 1:-1, :] * (~padding_mask)[:, :, None]

        return sequence_output, gold_embedding, pred_mask

    def index_convert(self, input_ids, padding_mask):
        """
        pad: 0, unk: 100
        """
        max_len = input_ids.size(1)
        lengths = (~padding_mask).sum(-1)
        list_sents = []
        for sent_ids, length in zip(input_ids, lengths):
            sent = self.tgt_dict.string(sent_ids[:length])
            sent = sent.replace('<unk>', '`').replace('[NOISE]', '`').replace('[VOCALIZED-NOISE]', '`').replace('[LAUGHTER]', '`')
            sent_ids_ = self.tokenizer(sent)['input_ids'] + [0] * (max_len-length)
            list_sents.append(sent_ids_)

        bert_input_ids = torch.tensor(list_sents).type_as(input_ids)
        token_mask = bert_input_ids.ne(101) * bert_input_ids.ne(102) * bert_input_ids.ne(0)

        return bert_input_ids, token_mask


@register_model_architecture("wav2vec_cif_bert", "wav2vec_cif_bert")
def w2v_cif_bert_architecture(args):
    cif_architecture(args)
    args.freeze_lm_finetune_updates = getattr(args, "freeze_lm_finetune_updates", 1000)


@register_model_architecture("wav2vec_cif2_bert", "wav2vec_cif2_bert")
def w2v_cif_bert_architecture(args):
    cif_architecture(args)
    args.freeze_lm_finetune_updates = getattr(args, "freeze_lm_finetune_updates", 1000)


@register_model_architecture("wav2vec_cif_bertEx", "wav2vec_cif_bertEx")
def w2v_cif_bertEx_architecture(args):
    cif_architecture(args)
    args.freeze_lm_finetune_updates = getattr(args, "freeze_lm_finetune_updates", 1000)


@register_model_architecture("wav2vec_cif2_bertEx", "wav2vec_cif2_bertEx")
def w2v_cif_bertEx_architecture(args):
    cif_architecture(args)
    args.freeze_lm_finetune_updates = getattr(args, "freeze_lm_finetune_updates", 1000)
