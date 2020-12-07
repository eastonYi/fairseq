# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.logging.meters import safe_round


@register_criterion('wav2vec')
class Wav2vecCriterion(FairseqCriterion):

    def __init__(self, task, infonce=False, loss_weights=None, log_keys=None):
        super().__init__(task)
        self.infonce = infonce
        self.loss_weights = None if loss_weights is None else eval(loss_weights)
        self.log_keys = [] if log_keys is None else eval(log_keys)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--infonce', action='store_true',
                            help='if set, uses cross entropy instead of binary cross entropy (i.e. InfoNCE loss)')
        parser.add_argument('--loss-weights', type=str, default=None,
                            help='weights for additional loss terms (not first one)')
        parser.add_argument('--log-keys', type=str, default=None,
                            help='output keys to log')

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        logits = model.get_logits(net_output).float()
        target = model.get_targets(sample, net_output)

        weights = None
        if hasattr(model, 'get_target_weights') and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        losses = []

        if self.infonce:
            loss = F.cross_entropy(logits, target, reduction="sum" if reduce else "none",)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, target.float(), weights, reduction="sum" if reduce else "none",)

        sample_size = target.numel() if self.infonce else target.long().sum().item()
        losses.append(loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(self.loss_weights), f'{len(extra_losses)}, {len(self.loss_weights)}'
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)

        logging_output = {
            'loss': loss.item() if reduce else loss,
            'ntokens': sample_size,
            'nsentences': sample['id'].numel(),
            'sample_size': sample_size,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f'loss_{i}'] = l.item()

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = max.numel()

                logging_output["correct"] = corr
                logging_output["count"] = count

        if log_pred:
            logging_output['logits'] = logits.cpu().numpy()
            logging_output['target'] = target.cpu().numpy()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('ntokens', ntokens)
        metrics.log_scalar('nsentences', nsentences)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)

        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(meters["_correct"].sum / meters["_total"].sum, 5)
                if meters["_total"].sum > 0
                else float("nan"),
            )

        builtin_keys = {'loss', 'ntokens', 'nsentences', 'sample_size', 'correct', 'count'}

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs) / len(logging_outputs)
                if k.startswith('loss'):
                    metrics.log_scalar(k, val / sample_size / math.log(2), sample_size)
                else:
                    metrics.log_scalar(k, val, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False


@register_criterion('wav2vec_semi')
class Wav2vecSemiCriterion(Wav2vecCriterion):

    def __init__(self, task, infonce=False, loss_weights=None, log_keys=None):
        super().__init__(task, infonce, loss_weights, log_keys)
        self.blank_idx = task.target_dictionary.index("<ctc_blank>")
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        logits, lprobs = model.get_logits(net_output)
        target = model.get_targets(sample, net_output)

        weights = None
        if hasattr(model, 'get_target_weights') and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        losses = []

        if self.infonce:
            cpc_loss = F.cross_entropy(logits, target, reduction="sum" if reduce else "none",)
        else:
            cpc_loss = F.binary_cross_entropy_with_logits(logits, target.float(), weights, reduction="sum" if reduce else "none",)

        sample_size = target.numel() if self.infonce else target.long().sum().item()
        losses.append(cpc_loss.detach().clone())

        # ctc loss
        non_padding_mask = ~net_output["padding_mask"]
        input_lengths = non_padding_mask.long().sum(-1)

        pad_mask = (sample["target"] != self.pad_idx) & (sample["target"] != self.eos_idx)
        targets_flat = sample["target"].masked_select(pad_mask)
        target_lengths = sample["target_lengths"]

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=True,
            )
        # ctc_loss = torch.zeros_like(cpc_loss)

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(self.loss_weights), f'{len(extra_losses)}, {len(self.loss_weights)}'
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    cpc_loss += p
                    losses.append(p)

        logging_output = {
            'loss': cpc_loss.item() if reduce else cpc_loss,
            'ctc_loss': ctc_loss.item(),
            'ntokens': target_lengths.sum().item(),
            'nsentences': sample['id'].numel(),
            'sample_size': sample_size,
        }

        loss = cpc_loss + ctc_loss * sample_size / target_lengths.sum().float()

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f'loss_{i}'] = l.item()

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = max.numel()

                logging_output["correct"] = corr
                logging_output["count"] = count

        if log_pred:
            logging_output['logits'] = logits.cpu().numpy()
            logging_output['target'] = target.cpu().numpy()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(sum(log.get('ctc_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('ctc_loss', ctc_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('ntokens', ntokens)
        metrics.log_scalar('nsentences', nsentences)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)

        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(meters["_correct"].sum / meters["_total"].sum, 5)
                if meters["_total"].sum > 0
                else float("nan"),
            )

        builtin_keys = {'loss', 'ntokens', 'nsentences', 'sample_size', 'correct', 'count'}

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs) / len(logging_outputs)
                if k.startswith('loss'):
                    metrics.log_scalar(k, val / sample_size / math.log(2), sample_size)
                else:
                    metrics.log_scalar(k, val, round=3)


@register_criterion('wav2vec_v1')
class Wav2vecV1Criterion(Wav2vecCriterion):

    def __init__(self, task, infonce=False, loss_weights=None, log_keys=None):
        super().__init__(task, infonce, loss_weights, log_keys)

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        logits = model.get_logits(net_output)
        target = model.get_targets(sample, net_output)

        weights = None
        if hasattr(model, 'get_target_weights') and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        if self.infonce:
            loss = F.cross_entropy(logits, target, reduction="sum" if reduce else "none",)
            # lprobs = utils.log_softmax(logits.float(), dim=-1)
            # loss, _ = label_smoothed_nll_loss(
            #     lprobs, target, 0.1, reduce="sum" if reduce else "none",)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, target.float(), weights, reduction="sum" if reduce else "none",)

        losses = []
        sample_size = target.numel() if self.infonce else target.long().sum().item()
        losses.append(loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(self.loss_weights), f'{len(extra_losses)}, {len(self.loss_weights)}'
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)

        logging_output = {
            'loss': loss.item(),
            'ntokens': sample_size,
            'nsentences': sample['id'].numel(),
            'sample_size': sample_size,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f'loss_{i}'] = l.item()

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    corr = (logits.argmax(1) == target).sum()
                    count = target.numel()

                logging_output["correct"] = corr.data
                logging_output["count"] = count

        if log_pred:
            logging_output['target'] = target.cpu().numpy()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('ntokens', ntokens)
        metrics.log_scalar('nsentences', nsentences)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)

        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(meters["_correct"].sum * 1.0 / meters["_total"].sum, 5)
                if meters["_total"].sum > 0
                else float("nan"),
            )

        builtin_keys = {'loss', 'ntokens', 'nsentences', 'sample_size', 'correct', 'count'}
        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs) / len(logging_outputs)
                if k.startswith('loss'):
                    metrics.log_scalar(k, val / sample_size / math.log(2), sample_size)
                else:
                    metrics.log_scalar(k, val, round=3)


@register_criterion('wav2vec_v2')
class Wav2vecV2Criterion(Wav2vecCriterion):

    def __init__(self, task, infonce=False, loss_weights=None, log_keys=None):
        super().__init__(task, infonce, loss_weights, log_keys)
        self.blank_idx = task.target_dictionary.index("<ctc_blank>")
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        lprobs = model.get_logits(net_output)
        target = model.get_targets(sample, net_output)

        weights = None
        if hasattr(model, 'get_target_weights') and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        losses = []
        sample_size = sample["ntokens"] if "ntokens" in sample else sample["target_lengths"].sum().item()

        # ctc loss
        non_padding_mask = ~net_output["padding_mask"]
        input_lengths = non_padding_mask.long().sum(-1)

        pad_mask = (sample["target"] != self.pad_idx) & (sample["target"] != self.eos_idx)
        targets_flat = sample["target"].masked_select(pad_mask)
        target_lengths = sample["target_lengths"]

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=True,
            )

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(self.loss_weights), f'{len(extra_losses)}, {len(self.loss_weights)}'
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)

        logging_output = {
            'loss': loss.item(),
            'ntokens': sample_size,
            'nsentences': sample['id'].numel(),
            'sample_size': sample_size,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f'loss_{i}'] = l.item()

        if log_pred:
            logging_output['target'] = target.cpu().numpy()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('ntokens', ntokens)
        metrics.log_scalar('nsentences', nsentences)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)

        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(meters["_correct"].sum / meters["_total"].sum, 5)
                if meters["_total"].sum > 0
                else float("nan"),
            )

        builtin_keys = {'loss', 'ntokens', 'nsentences', 'sample_size', 'correct', 'count'}

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs) / len(logging_outputs)
                if k.startswith('loss'):
                    metrics.log_scalar(k, val / sample_size / math.log(2), sample_size)
                else:
                    metrics.log_scalar(k, val, round=3)


@register_criterion('wav2vec_v3')
class Wav2vecV3Criterion(Wav2vecCriterion):

    def __init__(self, task, infonce=False, loss_weights=None, log_keys=None):
        super().__init__(task, infonce, loss_weights, log_keys)

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        logits = model.get_logits(net_output).float() # self pred
        target = model.get_targets(sample, net_output) # self target

        logits_ali = model.get_ali_logits(net_output) # ali pred
        target_ali = model.get_ali_targets(sample, net_output) # ali target

        weights = None
        if hasattr(model, 'get_target_weights') and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        if self.infonce:
            loss = F.cross_entropy(logits, target, reduction="sum" if reduce else "none",)
            lprobs = utils.log_softmax(logits_ali.float(), dim=-1)
            loss_ali, _ = label_smoothed_nll_loss(
                lprobs, target_ali, 0.0, reduce="sum" if reduce else "none",)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, target.float(), weights, reduction="sum" if reduce else "none",)

        sample_size = target.numel() if self.infonce else target.long().sum().item()
        losses = [loss.detach().clone(), loss_ali.detach().clone()]
        loss += 0.001 * loss_ali

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(self.loss_weights), f'{len(extra_losses)}, {len(self.loss_weights)}'
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)

        logging_output = {
            'loss': loss.item(),
            'ntokens': sample_size,
            'nsentences': sample['id'].numel(),
            'sample_size': sample_size,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f'loss_{i}'] = l.item()

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0

                    corr_ali = 0
                    count_ali = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = max.numel()

                    corr_ali = (logits_ali.argmax(1) == target_ali).sum()
                    count_ali = target_ali.numel()

                logging_output["correct"] = corr
                logging_output["count"] = count

                logging_output["correct_ali"] = corr_ali.data
                logging_output["count_ali"] = count_ali

        if log_pred:
            logging_output['logits'] = logits.cpu().numpy()
            logging_output['target'] = target.cpu().numpy()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('ntokens', ntokens)
        metrics.log_scalar('nsentences', nsentences)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)

        correct = sum(log.get("correct_ali", 0) for log in logging_outputs)
        metrics.log_scalar("_correct_ali", correct)

        total_ali = sum(log.get("count_ali", 0) for log in logging_outputs)
        metrics.log_scalar("_total_ali", total_ali)

        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(meters["_correct"].sum * 1.0 / meters["_total"].sum, 5)
                if meters["_total"].sum > 0
                else float("nan"),
            )

        if total_ali > 0:
            metrics.log_derived(
                "accuracy_ali",
                lambda meters: safe_round(meters["_correct_ali"].sum * 1.0 / meters["_total_ali"].sum, 5)
                if meters["_total_ali"].sum > 0
                else float("nan"),
            )

        builtin_keys = {'loss', 'ntokens', 'nsentences', 'sample_size', 'correct', 'count','correct_ali', 'count_ali'}
        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs) / len(logging_outputs)
                if k.startswith('loss'):
                    metrics.log_scalar(k, val / sample_size / math.log(2), sample_size)
                else:
                    metrics.log_scalar(k, val, round=3)
