#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
seq2seq decoders.
"""
from copy import deepcopy
import itertools as it
import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.models.fairseq_encoder import EncoderOut
from .seq2seq_decoder import Seq2seqDecoder
inf = 1e9


class CIFDecoder(Seq2seqDecoder):
    def __init__(self, args, tgt_dict, incremental_state={}):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = getattr(args, "nbest", 1)
        self.beam = getattr(args, "beam", 1)
        self.sos_idx = tgt_dict.eos()
        self.pad_idx = tgt_dict.pad()
        self.incremental_state = incremental_state

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences.
        EncoderOut(
            encoder_out=encoder_out['encoder_out'],  # T x B x C
            encoder_embedding=None,
            encoder_padding_mask=encoder_out['encoder_padding_mask'],  # B x T
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )
        """
        encoder_output = models[0].get_encoder_output(sample['net_input'])
        encoder_out = {
            "encoder_out": encoder_output.encoder_out.transpose(0,1), # B x T x C
            "padding_mask": encoder_output.encoder_padding_mask
        }
        alphas = models[0].assigner(encoder_out)
        # _alphas, num_output = self.resize(alphas, kwargs['target_lengths'], at_least_one=True)
        cif_outputs = models[0].cif(encoder_out, alphas)
        src_lengths = torch.round(alphas.sum(-1)).int()
        self.step_forward_fn = models[0].decode
        encoder_output = EncoderOut(
            encoder_out=cif_outputs.transpose(0, 1),  # T x B x C
            encoder_embedding=None,
            encoder_padding_mask=~utils.sequence_mask(src_lengths, dtype=torch.bool),  # B x T
            encoder_states=None,
            src_tokens=None,
            src_lengths=src_lengths,
        )

        return self.decode(encoder_output)

    def get_tokens(self, idxs):
        """Normalize tokens by handling ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x not in [self.sos_idx, self.pad_idx], idxs)

        return torch.LongTensor(list(idxs))

    def beam_decode(self, encoder_output):
        hypos = []
        beam_results, out_seq_len, beam_scores = self.batch_beam_decode(
            encoder_output,
            step_forward_fn=self.step_forward_fn,
            incremental_state=deepcopy(self.incremental_state),
            SOS_ID=self.sos_idx, EOS_ID=self.eos_idx, vocab_size=self.vocab_size,
            beam_size=self.beam, max_decode_len=100)

        for beam_result, scores, lengthes in zip(beam_results, beam_scores, out_seq_len):
            # beam_ids: beam x id; score: beam; length: beam
            top = []
            for result, score, length in zip(beam_result, scores, lengthes):
                top.append({'tokens': self.get_tokens(result[:length]),
                            "score": score})
            hypos.append(top)

        return hypos

    def greedy_decode(self, encoder_output):
        hypos = []

        results, out_seq_len, scores = self.batch_greedy_decode(
            encoder_output,
            step_forward_fn=self.step_forward_fn,
            incremental_state=deepcopy(self.incremental_state),
            SOS_ID=self.sos_idx, vocab_size=self.vocab_size,
            max_decode_len=100)

        for result, score, length in zip(results, scores, out_seq_len):
            res = {'tokens': self.get_tokens(result[:length]),
                   "score": score}
            hypos.append([res])

        return hypos

    @staticmethod
    def batch_beam_decode(encoder_output, step_forward_fn, incremental_state,
                          SOS_ID, vocab_size, beam_size=1, max_decode_len=100):
        """
        encoder_output:
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        """
        encoded = encoder_output.encoder_out # T x B x C
        len_encoded = (~encoder_output.encoder_padding_mask).sum(-1)
        max_decode_len = len_encoded.max()
        batch_size = len_encoded.size(0)
        device = encoded.device
        d_output = vocab_size

        # beam search Initialize
        # repeat each sample in batch along the batch axis [1,2,3,4] -> [1,1,2,2,3,3,4,4]
        encoded = encoded[:, None, :, :].repeat(1, beam_size, 1, 1) # [batch_size, beam_size, *, hidden_units]
        encoded = encoded.view(batch_size * beam_size, -1, encoded.size(-1))
        len_encoded = len_encoded[:, None].repeat(1, beam_size).view(-1) # [batch_size * beam_size]

        # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
        preds = torch.ones([batch_size * beam_size, 1]).long().to(device) * SOS_ID
        logits = torch.zeros([batch_size * beam_size, 0, d_output]).float().to(device)
        len_decoded = torch.ones_like(len_encoded)
        # the score must be [0, -inf, -inf, ...] at init, for the preds in beam is same in init!!!
        scores = torch.tensor([0.0] + [-inf] * (beam_size - 1)).float().repeat(batch_size).to(device)  # [batch_size * beam_size]

        # collect the initial states of lstms used in decoder.
        base_indices = torch.arange(batch_size)[:, None].repeat(1, beam_size).view(-1).to(device)

        for t in range(max_decode_len):
            # i, preds, scores, logits, len_decoded, finished
            cur_logits = step_forward_fn(
                prev_output_tokens=preds,
                encoder_out=encoder_output,
                incremental_states=incremental_state,
                t=t)

            logits = torch.cat([logits, cur_logits[:, None, :]], 1)  # [batch*beam, t, size_output]
            z = F.log_softmax(cur_logits, dim=-1) # [batch*beam, size_output]

            # rank the combined scores
            next_scores, next_preds = torch.topk(z, k=beam_size, sorted=True, dim=-1)

            # beamed scores & Pruning
            scores = scores[:, None] + next_scores  # [batch_size * beam_size, beam_size]
            scores = scores.view(batch_size, beam_size * beam_size)

            _, k_indices = torch.topk(scores, k=beam_size)
            k_indices = base_indices * beam_size * beam_size + k_indices.view(-1)  # [batch_size * beam_size]
            # Update scores.
            scores = scores.view(-1)[k_indices]
            # Update predictions.
            next_preds = next_preds.view(-1)[k_indices]

            # k_indices: [0~batch*beam*beam], preds: [0~batch*beam]
            # preds, cache_lm, cache_decoder: these data are shared during the beam expand among vocab
            preds = preds[k_indices // beam_size]
            preds = torch.cat([preds, next_preds[:, None]], axis=1)  # [batch_size * beam_size, i]

        preds = preds[:, 1:]
        # tf.nn.top_k is used to sort `scores`
        scores_sorted, sorted = torch.topk(scores.view(batch_size, beam_size),
                                           k=beam_size, sorted=True)
        sorted = base_indices * beam_size + sorted.view(-1)  # [batch_size * beam_size]

        # [batch_size * beam_size, ...] -> [batch_size, beam_size, ...]
        preds_sorted = preds[sorted].view(batch_size, beam_size, -1) # [batch_size, beam_size, max_length]
        len_decoded_sorted = len_decoded[sorted].view(batch_size, beam_size)
        scores_sorted = scores[sorted].view(batch_size, beam_size)

        return preds_sorted, len_decoded_sorted, scores_sorted

    @staticmethod
    def batch_greedy_decode(encoder_output, step_forward_fn, incremental_state,
                            SOS_ID, vocab_size, max_decode_len=100):
        """
        encoder_output:
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        """
        len_encoded = (~encoder_output.encoder_padding_mask).sum(-1)
        max_decode_len = len_encoded.max()
        batch_size = len_encoded.size(0)
        device = encoder_output.encoder_out.device
        d_output = vocab_size

        preds = torch.ones([batch_size, 1]).long().to(device) * SOS_ID
        logits = torch.zeros([batch_size, 0, d_output]).float().to(device)
        scores = torch.zeros([batch_size]).to(device)

        for t in range(max_decode_len):
            # cur_logits: [B, 1, V]
            cur_logits = step_forward_fn(
                prev_output_tokens=preds,
                encoded_shrunk=encoder_output.encoder_out.transpose(0,1),
                incremental_states=incremental_state,
                t=t)
            assert cur_logits.size(1) == 1
            logits = torch.cat([logits, cur_logits], 1)  # [batch, t, size_output]
            z = F.log_softmax(cur_logits[:, 0, :], dim=-1) # [batch, size_output]

            # rank the combined scores
            next_scores, next_preds = torch.topk(z, k=1, sorted=True, dim=-1)
            next_scores = next_scores.squeeze(-1)
            next_preds = next_preds.squeeze(-1)
            scores += next_scores

            preds = torch.cat([preds, next_preds[:, None]], axis=1)  # [batch_size, i]

        preds = preds[:, 1:]
        len_decoded = len_encoded

        return preds, len_decoded, scores
