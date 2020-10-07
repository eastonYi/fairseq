# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)

from .wav2vec2_seq2seq import (
    TransformerModel,
    Wav2VecEncoder,
    TransformerDecoder,
    Embedding,
    seq2seq_architecture,
    add_common_args,
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
    parser.add_argument(
        "--teacher-forcing", action='store_true', help="is teacher-forcing"
    )


@register_model("wav2vec_seq2seq_lm")
class TransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder, lm):
        super().__init__(args, encoder, decoder)
        self.decoder.lm = lm

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

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)

            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        lm = cls.build_lm(args, tgt_dict)
        return cls(args, encoder, decoder, lm)

    @classmethod
    def build_encoder(cls, args):
        return Wav2VecEncoder(args)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder_DF(args, tgt_dict, embed_tokens)

    @classmethod
    def build_lm(cls, args, dictionary):
        from fairseq.models.lstm_lm import LSTMLanguageModel
        return LSTMLanguageModel.build_model(args, task=None, dictionary=dictionary)

    def forward(self, **kwargs):
        encoder_out = self.encoder(tbc=False, **kwargs)
        decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)
        return decoder_out


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

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.freeze_lm_finetune_updates = args.freeze_lm_finetune_updates
        self.num_updates = 0
        self.output_embed_dim = args.decoder_embed_dim + len(dictionary)
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
            # self.lm.eval()
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


@register_model_architecture("wav2vec_seq2seq_lm", "wav2vec_seq2seq_lm")
def w2v_seq2seq_lm_architecture(args):
    seq2seq_architecture(args)
    args.freeze_lm_finetune_updates = getattr(args, "freeze_lm_finetune_updates", 1000)
