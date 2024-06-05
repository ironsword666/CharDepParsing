# -*- encoding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from supar.utils.common import MIN


class LinearCRFWordSegmentation(nn.Module):
    r"""
    Linear Chain Conditional Random Field (LC-CRF) model for Word Segmentation.
    """

    def __init__(self, masked_crf=True):
        super().__init__()

        self.masked_crf = masked_crf

        self.n_tags = 4
        self.id2tag = ['B', 'M', 'E', 'S']
        self.tag2id = {tag: i for i, tag in enumerate(self.id2tag)}
        # legal start/end and transition tags
        # the beginning tag of a sequence could be `B`, `S` or `O`
        self.starts = ['B', 'S']
        self.ends = ['E', 'S']
        self.transitions = [('B', 'E'), ('B', 'M'),
                            ('M', 'E'), ('M', 'M'),
                            ('E', 'B'), ('E', 'S'),
                            ('S', 'B'), ('S', 'S')]

        self.trans_score = nn.Parameter(torch.Tensor(self.n_tags, self.n_tags))
        self.start_score = nn.Parameter(torch.Tensor(self.n_tags))
        self.end_score = nn.Parameter(torch.Tensor(self.n_tags))

        # value `0` for log p=1, value `-inf` for log p=0
        self.trans_mask = nn.Parameter(torch.Tensor(
            self.n_tags, self.n_tags), requires_grad=False)
        self.start_mask = nn.Parameter(
            torch.Tensor(self.n_tags), requires_grad=False)
        self.end_mask = nn.Parameter(
            torch.Tensor(self.n_tags), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.trans_score)
        nn.init.zeros_(self.start_score)
        nn.init.zeros_(self.end_score)

        nn.init.constant_(self.trans_mask, MIN)
        nn.init.constant_(self.start_mask, MIN)
        nn.init.constant_(self.end_mask, MIN)

        for start in self.starts:
            self.start_mask[self.tag2id[start]] = 0
        for end in self.ends:
            self.end_mask[self.tag2id[end]] = 0
        for start, end in self.transitions:
            self.trans_mask[self.tag2id[start], self.tag2id[end]] = 0

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def set_mask(self, flag=True):
        self.masked_crf = flag

    @property
    def start_scores(self):
        if self.masked_crf:
            return self.start_score + self.start_mask
        return self.start_score

    @property
    def end_scores(self):
        if self.masked_crf:
            return self.end_score + self.end_mask
        return self.end_score

    @property
    def trans_scores(self):
        if self.masked_crf:
            return self.trans_score + self.trans_mask
        return self.trans_score

    def forward(self, s_tag, tags, mask):
        batch_size = mask.shape[0]
        score = self.get_scores(s_tag, tags, mask)
        logZ = self.crf(s_tag, mask)
        loss = (logZ - score) / batch_size
        return loss

    def get_scores(self, s_tag, tags, mask):
        self.set_mask(False)
        batch_size, seq_len, n_tags = s_tag.shape
        # [seq_len, batch_size, n_tags]
        s_tag = s_tag.permute(1, 0, 2)
        # [seq_len, batch_size]
        tags = tags.permute(1, 0)
        mask = mask.permute(1, 0)
        # [seq_len, batch_size]
        scores = s_tag.new_zeros((seq_len, batch_size))
        # transition scores
        scores[1:] += self.trans_scores[tags[:-1], tags[1:]]
        # emition scores
        scores += s_tag.gather(-1, tags.unsqueeze(-1)).squeeze(-1)
        score = scores.masked_select(mask).sum()
        # the start position
        score += self.start_scores[tags[0]].sum()
        # [batch_size]
        end_pos = mask.sum(dim=0) - 1
        # the end position
        score += self.end_scores[tags.gather(0, end_pos.unsqueeze(0))].sum()

        return score

    def crf(self, s_tag, mask):
        """
        Args:
            s_tag (Tensor): emission scores of tags, [batch_size, seq_len, n_tags]
            mask (Tensor): mask of the input, [batch_size, seq_len]

        Returns:
            logZ (Tensor): log partition function, [batch_size]
        """
        self.set_mask(False)
        batch_size, seq_len, n_tags = s_tag.shape
        # [seq_len, batch_size, n_tags]
        s_tag = s_tag.permute(1, 0, 2)

        s = torch.full_like(s_tag, MIN)

        # start position
        # [batch_size, n_tags]
        s[0] = s_tag[0] + self.start_scores

        for i in range(1, seq_len):
            # [batch_size, n_tags, n_tags]
            scores = s[i-1].unsqueeze(-1) + s_tag[i].unsqueeze(1) + self.trans_scores
            s[i] = torch.logsumexp(scores, dim=1)

        end_pos = mask.sum(dim=-1) - 1
        batch_idx = end_pos.new_tensor(torch.arange(batch_size))
        # the end position
        # [batch_size, n_tags]
        scores = s[end_pos, batch_idx]

        # end position can only be `E` and `S`
        scores += self.end_scores

        return scores.logsumexp(dim=-1).sum()

    @torch.enable_grad()
    def viterbi(self, s_tag, mask):
        self.set_mask(True)
        batch_size, seq_len, n_tags = s_tag.shape
        # [seq_len, batch_size, n_tags]
        s_tag = s_tag.permute(1, 0, 2).requires_grad_()

        s = torch.zeros_like(s_tag)

        # start position
        # [batch_size, n_tags]
        s[0] = s_tag[0] + self.start_scores

        for i in range(1, seq_len):
            # [batch_size, n_tags, n_tags]
            scores = s[i-1].unsqueeze(-1) + s_tag[i].unsqueeze(1) + self.trans_scores
            s[i], _ = torch.max(scores, dim=1)

        end_pos = mask.sum(dim=-1) - 1
        batch_idx = end_pos.new_tensor(torch.arange(batch_size))
        # the end position
        # [batch_size, n_tags]
        scores = s[end_pos, batch_idx]
        # end position can only be `E` and `S`
        scores += self.end_scores

        # [batch_size]
        scores, _ = torch.max(scores, dim=-1)
        logZ = scores.sum()
        # [seq_len, batch_size, n_tags]
        gradients, = autograd.grad(logZ, s_tag)

        return [i.nonzero()[:, 1].tolist() for i in gradients.permute(1, 0, 2)]

    def greedy_decode(self, s_tag, mask):
        raise NotImplementedError


class LinearCRFSemanticRoleLabeling(nn.Module):
    r"""
    Linear Chain Conditional Random Field (LC-CRF) model for Semantic Role Labeling.
    """

    def __init__(self):
        super().__init__()

        self.n_tags = 5
        self.id2tag = ['B', 'I', 'E', 'S', 'O']
        self.tag2id = {tag: i for i, tag in enumerate(self.id2tag)}
        # legal start/end and transition tags
        # the beginning tag of a sequence could be `B`, `S` or `O`
        self.starts = ['B', 'S', 'O']
        self.ends = ['E', 'S', 'O']
        self.transitions = [('B', 'E'), ('B', 'I'),
                            ('I', 'E'), ('I', 'I'),
                            ('E', 'B'), ('E', 'S'), ('E', 'O'),
                            ('S', 'B'), ('S', 'S'), ('S', 'O'),
                            ('O', 'B'), ('O', 'S'), ('O', 'O')]

        self.trans_score = nn.Parameter(torch.Tensor(self.n_tags, self.n_tags))
        self.start_score = nn.Parameter(torch.Tensor(self.n_tags))
        self.end_score = nn.Parameter(torch.Tensor(self.n_tags))

        # value `0` for log p=1, value `-inf` for log p=0
        self.trans_mask = nn.Parameter(torch.Tensor(
            self.n_tags, self.n_tags), requires_grad=False)
        self.start_mask = nn.Parameter(
            torch.Tensor(self.n_tags), requires_grad=False)
        self.end_mask = nn.Parameter(
            torch.Tensor(self.n_tags), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.trans_score)
        nn.init.zeros_(self.start_score)
        nn.init.zeros_(self.end_score)

        nn.init.constant_(self.trans_mask, float('-inf'))
        nn.init.constant_(self.start_mask, float('-inf'))
        nn.init.constant_(self.end_mask, float('-inf'))

        for start in self.starts:
            self.start_mask[self.tag2id[start]] = 0
        for end in self.ends:
            self.end_mask[self.tag2id[end]] = 0
        for start, end in self.transitions:
            self.trans_mask[self.tag2id[start], self.tag2id[end]] = 0

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def forward(self):
        raise NotImplementedError

    def loss(self, s_arg_begin, s_arg_end, props, mask):
        prd_mask, arg_begin_mask, arg_end_mask = mask
        pad_mask = prd_mask.unsqueeze(1) & prd_mask.unsqueeze(2)
        # [batch_size, n+1, n+1]
        prob_arg_begin = s_arg_begin.sigmoid()
        # [batch_size, n+1, n+1]
        prob_arg_end = s_arg_end.sigmoid()
        # [batch_size, n+1, n+1]
        prob_B = prob_arg_begin * (1 - prob_arg_end)
        prob_E = (1 - prob_arg_begin) * prob_arg_end
        prob_S = prob_arg_begin * prob_arg_end
        prob_O = (1 - prob_arg_begin) * (1 - prob_arg_end)
        # [batch_size, n+1, n+1, n_tags]
        probs = torch.stack((prob_B, prob_E, prob_S, prob_O), dim=-1)

        gold_arg_begin, gold_arg_end = props[..., 0], props[..., 1]
        gold_B = gold_arg_begin & (gold_arg_end == False)
        gold_E = (gold_arg_begin == False) & gold_arg_end
        gold_S = gold_arg_begin & gold_arg_end
        gold_O = (gold_arg_begin == False) & (gold_arg_end == False)
        golds = torch.stack((gold_B, gold_E, gold_S, gold_O), dim=-1)
        # [batch_size, n+1, n+1, n_tags]
        gold_mask = golds & pad_mask.unsqueeze(-1)

        return F.binary_cross_entropy(probs[gold_mask], golds[gold_mask].float())

    def decode(self, s_arg_begin, s_arg_end, mask, gold_prds=None, use_viterbi=False):
        prd_mask, arg_begin_mask, arg_end_mask = mask['prd_mask'], mask['arg_begin_mask'], mask['arg_end_mask']

        # [batch_size, n+1, n+1]
        prob_arg_begin = s_arg_begin.sigmoid() * arg_begin_mask
        # [batch_size, n+1, n+1]
        prob_arg_end = s_arg_end.sigmoid() * arg_end_mask
        # [batch_size, n+1, n+1]
        prob_B = prob_arg_begin * (1 - prob_arg_end)
        prob_E = (1 - prob_arg_begin) * prob_arg_end
        prob_S = prob_arg_begin * prob_arg_end
        prob_O = prob_I = (1 - prob_arg_begin) * (1 - prob_arg_end)
        # it's illegal that a predicate is in an argument
        prob_I = prob_I.triu(1) + prob_I.tril(-1)
        # [batch_size, n+1, n+1, n_tags]
        probs = torch.stack((prob_B, prob_I, prob_E, prob_S, prob_O), dim=-1)

        # [batch_size, n+1]
        if gold_prds is not None:
            prds = gold_prds & prd_mask
        else:
            prds = ((prob_arg_begin.gt(0.5).sum(dim=-1) > 0) |
                    (prob_arg_end.gt(0.5).sum(dim=-1) > 0)) & prd_mask
            # prds = s_prd.sigmoid().gt(0.5)
            # prds &= prd_mask

        if use_viterbi:
            return self.viterbi_decode(prds, probs, prd_mask, arg_begin_mask, arg_end_mask)
        else:
            return self.greedy_decode(s_arg_begin, s_arg_end, mask, gold_prds)

    def viterbi_decode(self, prds, probs, prd_mask, arg_begin_mask, arg_end_mask):
        # TODO: check conflicts in advance
        # [size, 2]
        batch_indices, prd_indices = prds.nonzero(as_tuple=True)
        # not include <bos>
        # [size]
        lens = prd_mask[batch_indices].sum(dim=-1)
        # [size, n]
        word_mask = prd_mask[batch_indices, 1:]
        # [*, n, n_tags], s(i, t) is the log-probability of the i-th token being the t-th tag
        emit_scores = probs[batch_indices, prd_indices, 1:].log()
        size, seq_len, n_tags = emit_scores.shape
        # s(t) is the maximum log-probability of path
        # from 1-th position to the current position of t-th tag
        # p(path) = sum_i(logp(i, t))
        # [*, n_tags]
        optimal_score = emit_scores[:, 0, :] + self.start_mask
        # An additional <pad> is added to the end of the sentence.
        # bp(i, t) is the tag of (i-1)-th token which maximizes p(path)
        # from 1-th token to the i-th token which being the t-th tag.
        # [*, n+1, n_tags]
        # FIXME: use -1 as default pointer
        backpoints = emit_scores.new_zeros(
            (size, seq_len+1, n_tags), dtype=torch.long)

        for i in range(1, seq_len):
            # s(t1, t2) is the maximum log-probability of path from 1-th token to the i-th token,
            # where the tag of i-th token is t2 and the tag of (i-1)-th token is t1
            # [*, n_tags, n_tags] = [*, n_tags, 1] + [*, 1, n_tags] + [n_tag, n_tag]
            score = optimal_score.unsqueeze(-1) + \
                emit_scores[:, i].unsqueeze(-2) + self.trans_mask
            # [*, n_tags]
            max_values, max_indices = torch.max(score, dim=-2)
            # if reach the end of the sentence, not update the value
            optimal_score[word_mask[:, i]] = max_values[word_mask[:, i]]
            # the default point for <pad> tokens is 0
            backpoints[:, i, :][word_mask[:, i]] = max_indices[word_mask[:, i]]

        optimal_score += self.end_mask
        # set the 0-th tag as the pointer for <pad> token
        backpoints[range(size), lens, 0] = torch.argmax(optimal_score, dim=-1)

        # record the optimal tag for each position
        # [size, n]
        preds = emit_scores.new_zeros((size, seq_len), dtype=torch.long)
        # the tag chosen at the current position, default is 0 for <pad>
        pred_tags = emit_scores.new_zeros((size, 1), dtype=torch.long)

        # backtrack
        for i in range(seq_len, 0, -1):
            pred_tags = torch.gather(backpoints[:, i, :], 1, pred_tags)
            preds[:, i-1] = pred_tags.squeeze(-1)

        # set the edge from prd to the beginning of the argument to True
        # [batch_size, n+1, n+1]
        arg_begin = torch.full_like(arg_begin_mask, 0)
        arg_begin[batch_indices, prd_indices, 1:] = preds.eq(
            self.tag2id['B']) | preds.eq(self.tag2id['S'])
        arg_begin &= arg_begin_mask

        arg_end = torch.full_like(arg_end_mask, 0)
        arg_end[batch_indices, prd_indices, 1:] = preds.eq(
            self.tag2id['E']) | preds.eq(self.tag2id['S'])
        arg_end &= arg_end_mask

        # # TODO: recover to proposition
        # batch_size, seq_len = prds.shape
        # props = args_begin.new_zeros((batch_size, seq_len, seq_len, seq_len), dtype=torch.bool)
        # batch_indices, prd_indices, args_begin_indices = args_begin.nonzero(as_tuple=True)
        # batch_indices, prd_indices, args_end_indices = args_end.nonzero(as_tuple=True)
        # props[batch_indices, prd_indices, args_begin_indices, args_end_indices] = True

        # return props

        return torch.stack((arg_begin, arg_end), dim=-1)

    def greedy_decode(self, prds, prob_arg_begin, prob_arg_end, mask):
        raise NotImplementedError


class OneStageLinearCRFSemanticRoleLabeling(LinearCRFSemanticRoleLabeling):

    def decode(self, s_edge, s_label, mask, label2tag, gold_prds=None, use_viterbi=False):
        prd_mask, arg_begin_mask, arg_end_mask = mask['prd_mask'], mask['arg_begin_mask'], mask['arg_end_mask']
        prob_edge, prob_labels = s_edge.sigmoid(), s_label.softmax(dim=-1)

        if gold_prds is not None:
            prds = gold_prds & prd_mask
        else:
            prds = prob_edge[:, 0].gt(0.5) & prd_mask

        edge_mask = prd_mask.unsqueeze(1) & prd_mask.unsqueeze(2)
        edge_mask = edge_mask.triu(1) | edge_mask.tril(-1)
        prob_edge = prob_edge * edge_mask
        prob_B = prob_edge * prob_labels[..., label2tag['B']].sum(dim=-1)
        prob_E = prob_edge * prob_labels[..., label2tag['E']].sum(dim=-1)
        prob_S = prob_edge * prob_labels[..., label2tag['S']].sum(dim=-1)
        prob_O = prob_I = 1 - prob_edge
        # make sure that the predicate is not in the argument
        prob_I = prob_I.triu(1) + prob_I.tril(-1)
        probs = torch.stack((prob_B, prob_I, prob_E, prob_S, prob_O), dim=-1)

        if use_viterbi:
            return self.viterbi_decode(prds, probs, prd_mask, arg_begin_mask, arg_end_mask)
        else:
            return self.greedy_decode(prds, probs, prd_mask)


class TwoStageLinearCRFSemanticRoleLabeling(LinearCRFSemanticRoleLabeling):

    def decode(self, s_edge, s_tag, mask, gold_prds=None, use_viterbi=False):
        prd_mask, arg_begin_mask, arg_end_mask = mask['prd_mask'], mask['arg_begin_mask'], mask['arg_end_mask']
        prob_edge, prob_tags = s_edge.sigmoid(), s_tag.softmax(dim=-1)

        if gold_prds is not None:
            prds = gold_prds & prd_mask
        else:
            prds = prob_edge[:, 0].gt(0.5) & prd_mask

        edge_mask = prd_mask.unsqueeze(1) & prd_mask.unsqueeze(2)
        edge_mask = edge_mask.triu(1) | edge_mask.tril(-1)
        prob_edge = prob_edge * edge_mask
        prob_B = prob_edge * prob_tags[..., 0]
        prob_E = prob_edge * prob_tags[..., 1]
        prob_S = prob_edge * prob_tags[..., 2]
        prob_O = prob_I = 1 - prob_edge
        # make sure that the predicate is not in the argument
        prob_I = prob_I.triu(1) + prob_I.tril(-1)
        probs = torch.stack((prob_B, prob_I, prob_E, prob_S, prob_O), dim=-1)

        if use_viterbi:
            return self.viterbi_decode(prds, probs, prd_mask, arg_begin_mask, arg_end_mask)
        else:
            return self.greedy_decode(prds, probs, prd_mask)
