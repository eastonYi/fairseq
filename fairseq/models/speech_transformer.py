# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional, Tuple

import torch
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
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

    def __init__(self, args, convSub, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True
        self.convSub = convSub

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
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

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
        return TransformerEncoder(args)

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
    def forward(
        self,
        feature,
        feature_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        feature: padded tensor (B, T, C * feat)
        feature_lengths: tensor of original lengths of input utterances (B,)

        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        feature, feature_lengths = self.spec_aug(feature, feature_lengths)
        x, len_x = self.convSub(feature, feature_lengths)
        encoder_out = self.encoder(
            x, src_lengths=len_x, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=len_x,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    def spec_aug(self, features, feature_lengths):
        freq_means = torch.mean(features, dim=-1)
        time_means = (torch.sum(features, dim=1)
                /feature_lengths[:, None].float()) # Note that features are padded with zeros.

        B, T, V = features.shape
        # mask freq
        for _ in range(self.spec_aug_conf["freq_mask_num"]):
            fs = (self.spec_aug_conf["freq_mask_width"]*torch.rand(size=[B],
                device=features.device, requires_grad=False)).long()
            f0s = ((V-fs).float()*torch.rand(size=[B],
                device=features.device, requires_grad=False)).long()
            for b in range(B):
                features[b, :, f0s[b]:f0s[b]+fs[b]] = freq_means[b][:, None]

        # mask time
        for _ in range(self.spec_aug_conf["time_mask_num"]):
            ts = (self.spec_aug_conf["time_mask_width"]*torch.rand(size=[B],
                device=features.device, requires_grad=False)).long()
            t0s = ((feature_lengths-ts).float()*torch.rand(size=[B],
                device=features.device, requires_grad=False)).long()
            for b in range(B):
                features[b, t0s[b]:t0s[b]+ts[b], :] = time_means[b][None, :]

        return features, feature_lengths


@register_model_architecture("speech_transformer", "speech_transformer")
def speech_architecture(args):
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)


class Conv2dSubsample(torch.nn.Module):
    def __init__(self, d_input, d_model, layer_num=2):
        super().__init__()
        assert layer_num >= 1
        self.layer_num = layer_num
        list_conv = [("subsample/conv0", torch.nn.Conv2d(1, 32, 3, (2, 1))),
                  ("subsample/relu0", torch.nn.ReLU())]
        for i in range(layer_num-1):
            list_conv.extend([
                ("subsample/conv{}".format(i+1), torch.nn.Conv2d(32, 32, 3, (2, 1))),
                ("subsample/relu{}".format(i+1), torch.nn.ReLU())
            ])
        self.conv = torch.nn.Sequential(*list_conv)
        self.affine = torch.nn.Linear(32 * (d_input-2*layer_num), d_model)

    def forward(self, feats, feat_lengths):
        outputs = feats.unsqueeze(1)  # [B, C, T, D]
        outputs = self.conv(outputs)
        B, C, T, D = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3).contiguous().view(B, T, C*D)

        outputs = self.affine(outputs)
        output_lengths = feat_lengths
        for _ in range(self.layer_num):
            output_lengths = ((output_lengths-1) / 2).long()

        return outputs, output_lengths
