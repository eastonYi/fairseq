#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
CTC decoders.
"""
import itertools as it
import torch

from fairseq import utils
from fairseq.models.wav2vec.wav2vec2_cif import CIFFcModel, CIFFcModelV2


class W2V_CIF_BERT_Decoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        model = models[0]
        encoder_output = model.encoder(tbc=False, **sample["net_input"])
        alphas = CIFFcModelV2.get_alphas(encoder_output)
        # input_ids = sample['bert_input'].long()

        _alphas, num_output = model.resize(alphas)
        padding_mask = ~utils.sequence_mask(torch.round(num_output).int()).bool()
        # _alphas, num_output = self.resize(alphas, kwargs['target_lengths'])
        # padding_mask = ~utils.sequence_mask(kwargs['target_lengths']).bool()

        cif_outputs = model.cif(encoder_output['encoder_out'][:, :, :-1], _alphas)
        hidden = model.proj(cif_outputs)
        logits_ac = model.to_vocab_ac(hidden)

        logits, gold_embedding, pred_mask, token_mask = model.bert_forward(
            hidden, logits_ac, padding_mask, input_ids=None, gold_rate=0.0,
            threash=model.args.infer_threash)
        logits = logits_ac + model.args.lambda_lm * logits

        probs = utils.softmax(logits.float(), dim=-1)

        for distribution, length in zip(probs, num_output):
            result = distribution.argmax(-1)
            score = 0.0
            top = [{'tokens': self.get_tokens(result[:length]),
                    "score": score}]

        return [top]

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)

        return torch.LongTensor(list(idxs))
