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

from .cross_entropy_acc import CrossEntropyWithAccCriterion, LabelSmoothedCrossEntropyWithAccCriterion


@register_criterion("qua_ce_acc")
class QuantityCrossEntropyWithAccCriterion(LabelSmoothedCrossEntropyWithAccCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.args = args
        self.decoder = self.build_decoder(args, task)

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        return cls(args, task)

    def build_decoder(self, args, task):
        decoder = getattr(args, "decoder", None)

        from examples.speech_recognition.cif_decoder import CIFDecoder
        if decoder == "cif_decoder":

            decoder = CIFDecoder(args, task.target_dictionary, {})
        elif decoder == "cif_lm_decoder":

            decoder = CIFDecoder(args, task.target_dictionary, ({}, {}))
        else:
            import pdb; pdb.set_trace()

        return decoder

    def compute_loss(self, model, net_output, sample, reduction, log_probs):
        # number loss
        _number = net_output["num_output"]
        number = sample["target_lengths"].float()
        diff = torch.sqrt(torch.pow(_number - number, 2)).sum()
        # alphas_pen
        alphas_pen = net_output["alphas_pen"]
        qua_loss = diff +  self.args.lambda_alpha * alphas_pen
        target = sample["target"] # no eos bos
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
        ce_loss, _ = label_smoothed_nll_loss(
            lprobs, target.long(), 0.1, ignore_index=self.padding_idx, reduce=reduction,
        )

        if torch.isnan(qua_loss.sum() + ce_loss.sum() + lprobs.sum()):
            import pdb; pdb.set_trace()
            print('here')

        return lprobs, qua_loss, ce_loss

    def get_logging_output(self, sample, lprobs, loss, qua_loss, ce_loss):
        target = sample["target"].view(-1)
        mask = target != self.padding_idx
        correct = torch.sum(
            lprobs.argmax(1).masked_select(mask) == target.masked_select(mask)
        )
        total = torch.sum(mask)
        sample_size = sample["ntokens"]

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "qua_loss": utils.item(qua_loss.data),  # * sample['ntokens'],
            "ce_loss": utils.item(ce_loss.data),  # * sample['ntokens'],
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
        net_output = model(**sample["net_input"])
        lprobs, qua_loss, ce_loss = self.compute_loss(
            model, net_output, sample, reduction, log_probs
        )

        nsentences = sample["target"].size(0) + 1.0
        ntokens = sample["ntokens"]
        loss = self.args.lambda_qua * qua_loss * ntokens / nsentences + ce_loss

        sample_size, logging_output = self.get_logging_output(
            sample, lprobs, loss, qua_loss, ce_loss
        )

        if not model.training:
            import editdistance

            c_err = 0
            c_len = 0
            self.decoder.step_forward_fn = model.decoder
            with torch.no_grad():
                decodeds = self.decoder.generate([model], sample)
                for decoded, t in zip(decodeds, sample["target"]):
                    decoded = decoded[0]['tokens']

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units_arr = targ.tolist()
                    pred_units_arr = decoded.tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ce_loss = sum(log['ce_loss'] for log in logging_outputs)
        qua_loss = sum(log['qua_loss'] for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
            "ce_loss": ce_loss / sample_size if sample_size > 0 else 0.0,
            "qua_loss": qua_loss / nsentences if nsentences > 0 else 0.0,
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
            agg_output["nll_loss"] = ce_loss / ntokens
        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        c_total = sum(log.get("c_total", 1) for log in logging_outputs)
        if c_total > 1:
            agg_output["uer"] = c_errors * 100.0 / c_total
        # loss: per output token loss
        # nll_loss: per sentence loss
        return agg_output


@register_criterion("qua_ce_acc_v2")
class QuantityCrossEntropyWithAccCriterionV2(LabelSmoothedCrossEntropyWithAccCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.args = args

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        return cls(args, task)

    def compute_loss(self, model, net_output, sample, reduction, log_probs):
        # number loss
        _number = net_output["num_output"]
        number = sample["target_lengths"].float()
        diff = torch.sqrt(torch.pow(_number - number, 2)).sum()
        # alphas_pen
        alphas_pen = net_output["alphas_pen"]
        qua_loss = diff +  self.args.lambda_alpha * alphas_pen
        target = sample["target"] # no eos bos
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
        ce_loss, _ = label_smoothed_nll_loss(
            lprobs, target.long(), 0.1, ignore_index=self.padding_idx, reduce=reduction,
        )

        return lprobs, qua_loss, ce_loss

    def get_logging_output(self, sample, lprobs, loss, qua_loss, ce_loss):
        target = sample["target"].view(-1)
        mask = target != self.padding_idx
        correct = torch.sum(
            lprobs.argmax(1).masked_select(mask) == target.masked_select(mask)
        )
        total = torch.sum(mask)
        sample_size = sample["ntokens"]

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "qua_loss": utils.item(qua_loss.data),  # * sample['ntokens'],
            "ce_loss": utils.item(ce_loss.data),  # * sample['ntokens'],
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
        net_output = model(**sample["net_input"])
        lprobs, qua_loss, ce_loss = self.compute_loss(
            model, net_output, sample, reduction, log_probs
        )

        nsentences = sample["target"].size(0) + 1.0
        ntokens = sample["ntokens"]
        loss = self.args.lambda_qua * qua_loss * ntokens / nsentences + ce_loss

        sample_size, logging_output = self.get_logging_output(
            sample, lprobs, loss, qua_loss, ce_loss
        )

        if not model.training:
            import editdistance

            c_err = 0
            c_len = 0
            with torch.no_grad():
                for lp, t in zip(lprobs, sample["target"]):
                    decoded = lp.argmax(dim=-1).unique_consecutive()

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units_arr = targ.tolist()
                    pred_units_arr = decoded.tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ce_loss = sum(log['ce_loss'] for log in logging_outputs)
        qua_loss = sum(log['qua_loss'] for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
            "ce_loss": ce_loss / sample_size if sample_size > 0 else 0.0,
            "qua_loss": qua_loss / nsentences if nsentences > 0 else 0.0,
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
            agg_output["nll_loss"] = ce_loss / ntokens
        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        c_total = sum(log.get("c_total", 1) for log in logging_outputs)
        if c_total > 1:
            agg_output["uer"] = c_errors * 100.0 / c_total
        # loss: per output token loss
        # nll_loss: per sentence loss
        return agg_output
