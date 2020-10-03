# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional, Tuple

import torch
import math
from torch import nn, Tensor
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from .transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
    Linear,
    base_architecture,
)

DEFAULT_MAX_SOURCE_POSITIONS = 2048
DEFAULT_MAX_TARGET_POSITIONS = 1024


def add_spec_args(parser):
    parser.add_argument(
        "--freq-mask-num",
        type=int,
        metavar="N",
        help="encoder input dimension per input channel",
    )
    parser.add_argument(
        "--freq-mask-width",
        type=int,
        metavar="N",
        help="encoder input dimension per input channel",
    )
    parser.add_argument(
        "--time-mask-num",
        type=int,
        metavar="N",
        help="encoder input dimension per input channel",
    )
    parser.add_argument(
        "--time-mask-width",
        type=int,
        metavar="N",
        help="encoder input dimension per input channel",
    )


def add_conv_args(parser):
    parser.add_argument(
        "--dim-conv-hidden",
        type=int,
        metavar="N",
        help="encoder input dimension per input channel",
    )
    parser.add_argument(
        "--num-conv-layers",
        type=int,
        metavar="N",
        help="encoder input dimension per input channel",
    )


@register_model("speech_transformer")
class SpeechTransformerModel(TransformerModel):

    def __init__(self, args, conv, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.args = args
        self.conv = conv

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        add_spec_args(parser)
        add_conv_args(parser)
        TransformerModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        tgt_dict = task.target_dictionary

        decoder_embed_tokens = cls.build_embedding(
            args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
        )

        convSub = cls.build_conv(args)
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, convSub, encoder, decoder)

    @classmethod
    def build_conv(cls, args):
        return Conv2dSubsample(args)

    @classmethod
    def build_encoder(cls, args):
        return SpeechTransformerEncoder(args)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(self, **kwargs):
        """
        feature: padded tensor (B, T, C * feat)
        feature_lengths: tensor of original lengths of input utterances (B,)

        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        feature = kwargs["src_tokens"]
        feature_lengths = kwargs["src_lengths"]
        prev_output_tokens = kwargs["prev_output_tokens"]

        feature, feature_lengths = self.spec_aug(feature, feature_lengths)
        x, len_x = self.conv(feature, feature_lengths)
        encoder_out = self.encoder(x, src_lengths=len_x)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=len_x
        )
        return decoder_out

    def get_encoder_output(self, net_input: Dict[str, Tensor]):
        feature = net_input["src_tokens"]
        feature_lengths = net_input["src_lengths"]
        x, len_x = self.conv(feature, feature_lengths)
        encoder_output = self.encoder(x, src_lengths=len_x)

        return encoder_output

    def spec_aug(self, features, feature_lengths):
        freq_means = torch.mean(features, dim=-1)
        time_means = (torch.sum(features, dim=1)
                /feature_lengths[:, None].float()) # Note that features are padded with zeros.

        B, T, V = features.shape
        # mask freq
        for _ in range(2):
            fs = (27 * torch.rand(size=[B],
                device=features.device, requires_grad=False)).long()
            f0s = ((V-fs).float()*torch.rand(size=[B],
                device=features.device, requires_grad=False)).long()
            for b in range(B):
                features[b, :, f0s[b]:f0s[b]+fs[b]] = freq_means[b][:, None]

        # mask time
        for _ in range(2):
            ts = (40 * torch.rand(size=[B],
                device=features.device, requires_grad=False)).long()
            t0s = ((feature_lengths-ts).float()*torch.rand(size=[B],
                device=features.device, requires_grad=False)).long()
            for b in range(B):
                features[b, t0s[b]:t0s[b]+ts[b], :] = time_means[b][None, :]

        return features, feature_lengths


@register_model_architecture("speech_transformer", "speech_transformer")
def speech_architecture(args):
    base_architecture(args)
    args.decoder_embed_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)


class Conv2dSubsample(torch.nn.Module):
    def __init__(self, args, d_input=80, d_model=512, layer_num=2):
        super().__init__()
        assert layer_num >= 1
        self.layer_num = layer_num
        list_conv = [torch.nn.Conv2d(1, 32, 3, (2, 1)),
                     torch.nn.ReLU()]
        for i in range(layer_num-1):
            list_conv.extend([torch.nn.Conv2d(32, 32, 3, (2, 1)),
                              torch.nn.ReLU()])
        self.conv = torch.nn.Sequential(*list_conv)
        self.affine = torch.nn.Linear(32 * (d_input - 2 * layer_num), d_model)

    def forward(self, feats, feat_lengths):
        outputs = feats.unsqueeze(1)  # [B, C, T, D]
        outputs = self.conv(outputs)
        B, C, T, D = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(B, T, C*D)

        outputs = self.affine(outputs)
        output_lengths = feat_lengths
        for _ in range(self.layer_num):
            output_lengths = ((output_lengths-1) / 2.0).long()

        return outputs, output_lengths


class SpeechTransformerEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        FairseqEncoder.__init__(self, None)

        self.dropout = nn.Dropout(args.dropout)

        embed_dim = args.encoder_embed_dim
        self.max_source_positions = args.max_source_positions

        self.pe = PositionalEncoding(embed_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)
        self.layer_norm = LayerNorm(embed_dim)

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
        """
        x = self.dropout(self.pe(src_tokens))
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = (1-utils.sequence_mask(src_lengths)).bool()

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
        x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_embedding=None,
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        # if self.embed_positions is None:
        #     return self.max_source_positions
        return DEFAULT_MAX_SOURCE_POSITIONS
        # return min(self.max_source_positions, self.embed_positions.max_positions)


class PositionalEncoding(nn.Module):
    """Implement the positional encoding (PE) function.

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        self.scale = d_model**0.5
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        Args:
            input: N x T x D
        """
        length = input.size(1)

        return input*(self.scale)+self.pe[:, :length]
