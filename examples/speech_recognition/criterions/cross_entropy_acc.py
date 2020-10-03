# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss


@register_criterion("cross_entropy_acc")
class CrossEntropyWithAccCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def compute_loss(self, model, net_output, target, reduction, log_probs):
        # N, T -> N * T
        target = target.view(-1)
        lprobs = model.get_normalized_probs(net_output, log_probs=log_probs)
        if not hasattr(lprobs, "batch_first"):
            logging.warning(
                "ERROR: we need to know whether "
                "batch first for the net output; "
                "you need to set batch_first attribute for the return value of "
                "model.get_normalized_probs. Now, we assume this is true, but "
                "in the future, we will raise exception instead. "
            )
        batch_first = getattr(lprobs, "batch_first", True)
        if not batch_first:
            lprobs = lprobs.transpose(0, 1)

        # N, T, D -> N * T, D
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss = F.nll_loss(
            lprobs, target.long(), ignore_index=self.padding_idx, reduction=reduction
        )
        return lprobs, loss

    def get_logging_output(self, sample, target, lprobs, loss):
        target = target.view(-1)
        mask = target != self.padding_idx
        correct = torch.sum(
            lprobs.argmax(1).masked_select(mask) == target.masked_select(mask)
        )
        total = torch.sum(mask)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "correct": utils.item(correct.data),
            "total": utils.item(total.data),
        }

        return sample_size, logging_output

    def forward(self, model, sample, reduction="sum", log_probs=True):
        """Computes the cross entropy with accuracy metric for the given sample.

        This is similar to CrossEntropyCriterion in fairseq, but also
        computes accuracy metrics as part of logging

        Args:
            logprobs (Torch.tensor) of shape N, T, D i.e.
                batchsize, timesteps, dimensions
            targets (Torch.tensor) of shape N, T  i.e batchsize, timesteps

        Returns:
        tuple: With three elements:
            1) the loss
            2) the sample size, which is used as the denominator for the gradient
            3) logging outputs to display while training

        TODO:
            * Currently this Criterion will only work with LSTMEncoderModels or
            FairseqModels which have decoder, or Models which return TorchTensor
            as net_output.
            We need to make a change to support all FairseqEncoder models.
        """
        # decoder_out = model(**sample["net_input"])
        encoder_output = model.get_encoder_output(sample["net_input"])
        input_lengths = (~encoder_output.encoder_padding_mask).sum(-1)
        decoder_out = model.decoder(
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            encoder_out=encoder_output,
            src_lengths=input_lengths
        )
        target = sample["target"]
        lprobs, loss = self.compute_loss(
            model, decoder_out, target, reduction, log_probs
        )
        sample_size, logging_output = self.get_logging_output(
            sample, target, lprobs, loss
        )

        # if not model.training and loss/sample_size < 2.0:
        #     import editdistance
        #
        #     c_err = 0
        #     c_len = 0
        #     with torch.no_grad():
        #         decodeds = self.w2l_decoder.decode(encoder_output)
        #         import pdb; pdb.set_trace()
        #         for decoded, t, inp_l in zip(decodeds, sample["target"], input_lengths):
        #             decoded = decoded[0]
        #             if len(decoded) < 1:
        #                 decoded = None
        #             else:
        #                 decoded = decoded[0]
        #
        #             p = (t != self.task.target_dictionary.pad()) & (
        #                 t != self.task.target_dictionary.eos()
        #             )
        #             targ = t[p]
        #             targ_units_arr = targ.tolist()
        #             pred_units_arr = decoded.tolist()
        #
        #             c_err += editdistance.eval(pred_units_arr, targ_units_arr)
        #             c_len += len(targ_units_arr)
        #
        #         logging_output["c_errors"] = c_err
        #         logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
            # if args.sentence_avg, then sample_size is nsentences, then loss
            # is per-sentence loss; else sample_size is ntokens, the loss
            # becomes per-output token loss
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "acc": correct_sum * 100.0 / total_sum if total_sum > 0 else 0.0,
            "correct": correct_sum,
            "total": total_sum,
            # total is the number of validate tokens
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens

        if c_total > 0:
            agg_output["uer"] = c_errors / c_total
        # loss: per output token loss
        # nll_loss: per sentence loss
        return agg_output


@register_criterion("ce_acc")
class CrossEntropyWithAccV2Criterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(task)

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        return cls(args, task)

    def compute_loss(self, model, net_output, target, reduction, log_probs):
        # N, T -> N * T
        target = target.view(-1)
        lprobs = model.get_normalized_probs(net_output, log_probs=log_probs)
        if not hasattr(lprobs, "batch_first"):
            logging.warning(
                "ERROR: we need to know whether "
                "batch first for the net output; "
                "you need to set batch_first attribute for the return value of "
                "model.get_normalized_probs. Now, we assume this is true, but "
                "in the future, we will raise exception instead. "
            )
        batch_first = getattr(lprobs, "batch_first", True)
        if not batch_first:
            lprobs = lprobs.transpose(0, 1)

        # N, T, D -> N * T, D
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss = F.nll_loss(
            lprobs, target.long(), ignore_index=self.padding_idx, reduction=reduction
        )
        return lprobs, loss

    def get_logging_output(self, sample, target, lprobs, loss):
        target = target.view(-1)
        mask = target != self.padding_idx
        correct = torch.sum(
            lprobs.argmax(1).masked_select(mask) == target.masked_select(mask)
        )
        total = torch.sum(mask)
        sample_size = sample["ntokens"]

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "correct": utils.item(correct.data),
            "total": utils.item(total.data),
        }

        return sample_size, logging_output

    def forward(self, model, sample, reduction="sum", log_probs=True):
        encoder_output = model.get_encoder_output(sample["net_input"])
        input_lengths = (~encoder_output.encoder_padding_mask).sum(-1)
        decoder_out = model.decoder(
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            encoder_out=encoder_output,
            src_lengths=input_lengths
        )
        target = sample["target"]
        lprobs, loss = self.compute_loss(
            model, decoder_out, target, reduction, log_probs
        )
        sample_size, logging_output = self.get_logging_output(
            sample, target, lprobs, loss
        )

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
            # if args.sentence_avg, then sample_size is nsentences, then loss
            # is per-sentence loss; else sample_size is ntokens, the loss
            # becomes per-output token loss
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "acc": correct_sum * 100.0 / total_sum if total_sum > 0 else 0.0,
            "correct": correct_sum,
            "total": total_sum,
            # total is the number of validate tokens
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens
        # loss: per output token loss
        # nll_loss: per sentence loss
        return agg_output


@register_criterion("ls_ce_acc")
class LabelSmoothedCrossEntropyWithAccCriterion(CrossEntropyWithAccV2Criterion):

    def compute_loss(self, model, net_output, target, reduction, log_probs):
        # N, T -> N * T
        target = target.view(-1)
        lprobs = model.get_normalized_probs(net_output, log_probs=log_probs)
        if not hasattr(lprobs, "batch_first"):
            logging.warning(
                "ERROR: we need to know whether "
                "batch first for the net output; "
                "you need to set batch_first attribute for the return value of "
                "model.get_normalized_probs. Now, we assume this is true, but "
                "in the future, we will raise exception instead. "
            )
        batch_first = getattr(lprobs, "batch_first", True)
        if not batch_first:
            lprobs = lprobs.transpose(0, 1)

        # N, T, D -> N * T, D
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss, _ = label_smoothed_nll_loss(
            lprobs, target.long(), 0.1, ignore_index=self.padding_idx, reduce=reduction,
        )

        return lprobs, loss
