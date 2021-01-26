# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import BaseWrapperDataset
from . import data_utils


class AddTargetDataset(BaseWrapperDataset):
    def __init__(self, dataset, labels, pad, bos, eos, batch_targets, process_label=None):
        super().__init__(dataset)
        self.labels = self.del_skip(dataset.skip_ids, labels)
        self.batch_targets = batch_targets
        self.pad = pad
        self.bos = bos # defaultly reuse eos
        self.eos = eos
        self.process_label = process_label

        assert len(dataset) == len(labels), 'wav: {}; trans: {}'.format(len(dataset), len(self.labels))
        assert batch_targets

    @staticmethod
    def del_skip(list_to_skip, labels):
        if list_to_skip:
            for i in list_to_skip[::-1]:
                labels.pop(i)
        return labels

    def get_label(self, index):
        return self.labels[index] if self.process_label is None else self.process_label(self.labels[index])

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label"] = self.get_label(index)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = len(self.get_label(index))
        return (sz, own_sz)

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())

        if self.pad is None: # ali
            target = [s["label"] for s in samples if s["id"] in indices]
            min_len = min([len(tokens) for tokens in target])
            collated["target"] = data_utils.collate_tokens(target, pad_idx=1, left_pad=False)[:, :min_len]

            return collated

        elif self.bos is not None and self.eos is not None and self.pad == 0: # BERT:
            target = [s["label"] for s in samples if s["id"] in indices]
            bos = torch.ones([1]).int() * self.bos
            eos = torch.ones([1]).int() * self.eos
            bert_input = [torch.cat([bos, s["label"], eos], dim=-1) for s in samples if s["id"] in indices]
            collated["net_input"]["bert_input"] = \
                data_utils.collate_tokens(bert_input, pad_idx=self.pad, left_pad=False)

        elif self.bos is None and self.eos is None and self.pad == 100: # GPT2:
            target = [s["label"] for s in samples if s["id"] in indices]
            gpt2_input = [s["label"] for s in samples if s["id"] in indices]
            collated["net_input"]["gpt2_input"] = \
                data_utils.collate_tokens(gpt2_input, pad_idx=self.pad, left_pad=False)

        elif self.bos is not None and self.eos is not None and self.pad == 1: # seq2seq
            eos = torch.ones([1]).int() * self.eos
            target = [torch.cat([s["label"], eos], dim=-1) for s in samples if s["id"] in indices]
            bos = torch.ones([1]).int() * self.bos
            prev_output_tokens = [torch.cat([bos, s["label"]], dim=-1) for s in samples if s["id"] in indices]
            collated["net_input"]["prev_output_tokens"] = \
                data_utils.collate_tokens(prev_output_tokens, pad_idx=self.pad, left_pad=False)

        elif self.bos is not None and self.eos is None: # CIF, ctc-lm
            target = [s["label"] for s in samples if s["id"] in indices]
            bos = torch.ones([1]).int() * self.bos
            prev_output_tokens = [torch.cat([bos, s["label"][:-1]], dim=-1) for s in samples if s["id"] in indices]
            collated["net_input"]["prev_output_tokens"] = \
                data_utils.collate_tokens(prev_output_tokens, pad_idx=self.pad, left_pad=False)

        elif self.bos is None and self.eos is None: # ctc
            target = [s["label"] for s in samples if s["id"] in indices]

        else:
            raise NotImplementedError('')

        collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
        collated["net_input"]["target_lengths"] = collated["target_lengths"]
        target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
        collated["ntokens"] = collated["target_lengths"].sum().item()
        collated["target"] = target

        return collated
