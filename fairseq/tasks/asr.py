# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os

from fairseq.data import Dictionary, AddTargetDataset, FileAudioDataset
from .audio_pretraining import AudioUnsuperviseTrainingTask
from . import register_task


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@register_task("audio_ctc")
class AudioCtcTask(AudioUnsuperviseTrainingTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--labels",
            type=str,
            help="extension of the label file to load, if any",
        )
        parser.add_argument(
            "--not-add-ctc-blank",
            action="store_true",
            help="if true, does not add <ctc_blank>",
        )
        AudioUnsuperviseTrainingTask.add_args(parser)

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries)."""
        dict_path = os.path.join(args.data, f"dict.{args.labels}.txt")
        if not os.path.isfile(dict_path):
            raise FileNotFoundError("Dict not found: {}".format(dict_path))
        tgt_dict = Dictionary.load(dict_path)

        if not args.not_add_ctc_blank:
            tgt_dict.add_symbol("<ctc_blank>")

        print("| dictionary: {} types".format(len(tgt_dict)))
        return cls(args, tgt_dict)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        manifest = os.path.join(self.args.data, "{}.tsv".format(split))
        self.datasets[split] = FileAudioDataset(
            manifest,
            sample_rate=self.args.sample_rate,
            max_sample_size=self.args.max_sample_size,
            min_sample_size=self.args.max_sample_size,
            min_length=self.args.min_sample_size,
            pad=True,
            normalize=self.args.normalize,
        )

        label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")
        labels = self.load_labels(label_path)

        process_label = LabelEncoder(self.dictionary)

        self.datasets[split] = AddTargetDataset(
            self.datasets[split],
            labels,
            pad=self.dictionary.pad(),
            bos=None,
            eos=None,
            batch_targets=True,
            process_label=process_label,
        )

    @staticmethod
    def load_labels(label_path):
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                labels.append(line)

        return labels

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary


@register_task("audio_cif")
class AudioCifTask(AudioCtcTask):

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        manifest = os.path.join(self.args.data, "{}.tsv".format(split))
        self.datasets[split] = FileAudioDataset(
            manifest,
            sample_rate=self.args.sample_rate,
            max_sample_size=self.args.max_sample_size,
            min_sample_size=self.args.max_sample_size,
            min_length=self.args.min_sample_size,
            pad=True,
            normalize=self.args.normalize,
        )

        label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")
        labels = self.load_labels(label_path)

        process_label = LabelEncoder(self.dictionary)

        self.datasets[split] = AddTargetDataset(
            self.datasets[split],
            labels,
            bos=self.dictionary.eos(),
            pad=self.dictionary.pad(),
            eos=None,
            batch_targets=True,
            process_label=process_label
        )


@register_task("audio_seq2seq")
class AudioSeq2seqTask(AudioCtcTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--labels",
            type=str,
            help="extension of the label file to load, if any",
        )
        AudioUnsuperviseTrainingTask.add_args(parser)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries)."""
        dict_path = os.path.join(args.data, f"dict.{args.labels}.txt")
        if not os.path.isfile(dict_path):
            raise FileNotFoundError("Dict not found: {}".format(dict_path))
        tgt_dict = Dictionary.load(dict_path)

        print("| dictionary: {} types".format(len(tgt_dict)))
        return cls(args, tgt_dict)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        manifest = os.path.join(self.args.data, "{}.tsv".format(split))
        self.datasets[split] = FileAudioDataset(
            manifest,
            sample_rate=self.args.sample_rate,
            max_sample_size=self.args.max_sample_size,
            min_sample_size=self.args.max_sample_size,
            min_length=self.args.min_sample_size,
            pad=True,
            normalize=self.args.normalize,
        )

        label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")
        labels = self.load_labels(label_path)

        process_label = LabelEncoder(self.dictionary)

        self.datasets[split] = AddTargetDataset(
            self.datasets[split],
            labels,
            pad=self.dictionary.pad(),
            bos=self.dictionary.eos(),
            eos=self.dictionary.eos(),
            batch_targets=True,
            process_label=process_label,
        )


@register_task("audio_ctc_ce")
class AudioCtcCeTask(AudioCtcTask):

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        manifest = os.path.join(self.args.data, "{}.tsv".format(split))
        self.datasets[split] = FileAudioDataset(
            manifest,
            sample_rate=self.args.sample_rate,
            max_sample_size=self.args.max_sample_size,
            min_sample_size=self.args.max_sample_size,
            min_length=self.args.min_sample_size,
            pad=True,
            normalize=self.args.normalize,
        )

        label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")
        labels = self.load_labels(label_path)

        process_label = LabelEncoder(self.dictionary)

        self.datasets[split] = AddTargetDataset(
            self.datasets[split],
            labels,
            bos=self.dictionary.eos(),
            pad=self.dictionary.pad(),
            eos=None,
            batch_targets=True,
            process_label=process_label
        )
