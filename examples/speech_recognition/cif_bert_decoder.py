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


class CIF_BERT_Decoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        model = models[0]
        model_output = model(**sample["net_input"])
        logits = model_output['logits']
        len_decoded = model_output['len_logits']

        probs = utils.softmax(logits.float(), dim=-1)

        res = []
        for distribution, length in zip(probs, len_decoded):
            result = distribution.argmax(-1)
            score = 0.0
            res.append([{'tokens': result[:length],
                         "score": score}])

        return res

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)

        return torch.LongTensor(list(idxs))
