# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
from supar.utils.fn import stripe
from supar.utils.common import MIN
from supar.structs.semiring import LogSemiring, MaxSemiring
from supar.utils.logging import get_logger

logger = get_logger(__name__)


class MatrixTree(nn.Module):
    r"""
    MatrixTree for calculating partitions and marginals of directed spanning trees (a.k.a. non-projective trees)
    in :math:`O(n^3)` by an adaptation of Kirchhoff's MatrixTree Theorem :cite:`koo-etal-2007-structured`.

    Different from the original paper, marginals are computed via back-propagation rather than matrix inversion.
    """

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=False, partial=False):
        r"""
        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible dependent-head pairs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
                The first column serving as pseudo words for roots should be ``False``.
            target (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard dependent-head pairs. Default: ``None``.
            mbr (bool):
                If ``True``, marginals will be returned to perform minimum Bayes-risk (MBR) decoding. Default: ``False``.
            partial (bool):
                ``True`` indicates that the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """

        training = scores.requires_grad
        logZ = self.matrix_tree(scores.requires_grad_(), mask)
        marginals = scores
        # calculate the marginals
        if mbr:
            marginals, = autograd.grad(logZ, marginals, retain_graph=training)

        if target is None:
            return marginals
        # the second inside process is needed if using partial annotation
        if partial:
            score = self.matrix_tree(scores, mask, target)
        else:
            score = scores.gather(-1, target.unsqueeze(-1)
                                  ).squeeze(-1)[mask].sum()
        loss = (logZ - score) / mask.sum()

        return loss, marginals

    def matrix_tree(self, s_arc, mask, cands=None):
        lens = mask.sum(-1)
        batch_size, seq_len, _ = s_arc.shape
        mask = mask.index_fill(1, lens.new_tensor(0), 1)
        chart_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        s_arc = s_arc.masked_fill(~chart_mask, float('-inf'))

        # set the arcs scores excluded by cands to -inf
        if cands is not None:
            cands = cands.unsqueeze(-1).index_fill(1, lens.new_tensor(0), -1)
            cands = cands.eq(lens.new_tensor(range(seq_len))) | cands.lt(0)
            cands = cands & chart_mask
            s_arc = s_arc.masked_fill(~cands, float('-inf'))

        # A(i, j) = exp(s(i, j))
        # double precision to prevent overflows
        A = torch.exp(s_arc).double()
        # Weighted degree matrix
        # D(i, j) = sum_j(A(i, j)), if h == m
        #           0,              otherwise
        D = torch.zeros_like(A)
        D.diagonal(0, 1, 2).copy_(A.sum(-1))
        # Laplacian matrix
        # L(i, j) = D(i, j) - A(i, j)
        L = nn.init.eye_(torch.empty_like(A[0])).repeat(
            batch_size, 1, 1).masked_scatter_(mask.unsqueeze(-1), (D - A)[mask])
        # Z = L^(0, 0), which is the minor of L w.r.t row 0 and column 0
        logZ = L[:, 1:, 1:].logdet().sum().float()

        return logZ


class CRFDependency(nn.Module):
    r"""
    First-order TreeCRF for calculating partitions and marginals of projective dependency trees
    in :math:`O(n^3)` :cite:`zhang-etal-2020-efficient`.
    """

    def __init__(self, multiroot=False):
        super().__init__()

        self.multiroot = multiroot

    def __repr__(self):
        return f"{self.__class__.__name__}(multiroot={self.multiroot})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=False, partial=False):
        r"""
        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible dependent-head pairs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
                The first column serving as pseudo words for roots should be ``False``.
            target (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard dependent-head pairs.
                This should be provided for loss calculation.
                If partially annotated, the unannotated positions should be filled with -1.
                Default: ``None``.
            mbr (bool):
                If ``True``, marginals will be returned to perform minimum Bayes-risk (MBR) decoding. Default: ``False``.
            partial (bool):
                ``True`` indicates that the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """

        training = scores.requires_grad
        # always enable the gradient computation of scores in order for the computation of marginals
        logZ = self.inside(scores.requires_grad_(), mask)
        # marginals are used for decoding, and can be computed by combining the inside pass and autograd mechanism
        marginals = scores
        if mbr:
            marginals, = autograd.grad(logZ, scores, retain_graph=training)

        if target is None:
            return marginals
        # the second inside process is needed if using partial annotation
        if partial:
            score = self.inside(scores, mask, target)
        else:
            score = scores.gather(-1, target.unsqueeze(-1)
                                  ).squeeze(-1)[mask].sum()
        # print(f"logZ: {logZ}, score: {score}")
        loss = (logZ - score) / mask.sum()

        return loss, marginals

    def inside(self, s_arc, mask, cands=None):
        # the end position of each sentence in a batch
        lens = mask.sum(1)
        batch_size, seq_len, _ = s_arc.shape
        # [seq_len, seq_len, batch_size]
        s_arc = s_arc.permute(2, 1, 0)
        s_i = torch.full_like(s_arc, float('-inf')).double()
        s_c = torch.full_like(s_arc, float('-inf')).double()
        s_c.diagonal().fill_(0)

        # set the arcs scores excluded by cands to -inf
        if cands is not None:
            # set the bos position to True
            mask = mask.index_fill(1, lens.new_tensor(0), 1)
            # [seq_len, seq_len, batch_size], padding mask
            mask = (mask.unsqueeze(1) & mask.unsqueeze(-1)).permute(2, 1, 0)
            # [batch_size, seq_len, 1], set the parent of bos to -1, which may be 0 in pad()
            cands = cands.unsqueeze(-1).index_fill(1, lens.new_tensor(0), -1)
            # [batch_size, seq_len, seq_len], arc mask
            # cands.eq(lens.new_tensor(range(seq_len))) is the arcs in the gold tree
            # cands.lt(0) is that if a word has no head (cands = -1) in the gold tree, any word can be its head
            cands = cands.eq(lens.new_tensor(range(seq_len))) | cands.lt(0)
            # FIXME: it is illegal that allow any word to be the head of bos
            cands = cands.permute(2, 1, 0) & mask
            s_arc = s_arc.masked_fill(~cands, float('-inf'))

        for w in range(1, seq_len):
            # n denotes the number of spans to iterate,
            # from span (0, w) to span (n, n+w) given width w
            n = seq_len - w

            # ilr = C(i->r) + C(j->r+1)
            # [n, w, batch_size]
            ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
            if ilr.requires_grad:
                ilr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            il = ir = ilr.permute(2, 0, 1).logsumexp(-1)
            # I(j->i) = logsumexp(C(i->r) + C(j->r+1)) + s(j->i), i <= r < j
            # fill the w-th diagonal of the lower triangular part of s_i with I(j->i) of n spans
            s_i.diagonal(-w).copy_(il + s_arc.diagonal(-w))
            # I(i->j) = logsumexp(C(i->r) + C(j->r+1)) + s(i->j), i <= r < j
            # fill the w-th diagonal of the upper triangular part of s_i with I(i->j) of n spans
            s_i.diagonal(w).copy_(ir + s_arc.diagonal(w))

            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
            cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).logsumexp(-1))
            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
            cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_c.diagonal(w).copy_(cr.permute(2, 0, 1).logsumexp(-1))
            if not self.multiroot:
                s_c[0, w][lens.ne(w)] = float('-inf')

        # if torch.isnan(s_c[0].gather(0, lens.unsqueeze(0))).sum() > 0:
            # print(f"s_c[0->n]: {s_c[0].gather(0, lens.unsqueeze(0))}")

        return s_c[0].gather(0, lens.unsqueeze(0)).sum()


class CharCRFDependency(nn.Module):
    r"""
    First-order TreeCRF for calculating partitions and marginals of projective dependency trees
    in :math:`O(n^3)` :cite:`zhang-etal-2020-efficient`.
    """

    def __init__(self, multiroot=False):
        super().__init__()

        self.multiroot = multiroot

    def __repr__(self):
        return f"{self.__class__.__name__}(multiroot={self.multiroot})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, partial=False, c_span_mask=None, combine_mask=None, marginal=None, normalizer='token'):
        r"""
        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible dependent-head pairs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
                The first column serving as pseudo words for roots should be ``False``.
            target (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard dependent-head pairs.
                This should be provided for loss calculation.
                If partially annotated, the unannotated positions should be filled with -1.
                Default: ``None``.
            partial (bool):
                ``True`` indicates that the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """
        training = scores.requires_grad
        # always enable the gradient computation of scores in order for the computation of marginals
        logZ = self.inside(scores.requires_grad_(), mask)
        # the second inside process is necessary if using partial annotation
        if partial:
            score = self.inside(scores, mask, target,
                                c_span_mask, combine_mask)
        else:
            score = scores.gather(-1, target.unsqueeze(-1)
                                  ).squeeze(-1)[mask].sum()

        if marginal == 'denominator':
            marginals, = autograd.grad(logZ, scores, retain_graph=training)
        elif marginal == 'numerator' and partial:
            marginals, = autograd.grad(score, scores, retain_graph=training)
        else:
            marginals = scores

        if normalizer == 'token':
            norm = mask.sum()
        elif normalizer == 'sentence':
            norm = mask.size(0)
        elif normalizer == 'target':
            if target.dtype == torch.bool:
                norm = target.sum()
            elif target.dtype == torch.long:
                norm = target.ge(0).sum()

        loss = (logZ - score) / norm

        # print(f'logZ: {logZ}, score: {score}, loss: {loss}')

        return loss, marginals

    @torch.enable_grad()
    def labeling(self, scores, s_labels, mask, target=None, c_span_mask=None, combine_mask=None, normalizer='token'):
        """merge unlabeled and labeled loss by adding label (log) probability to unlabeled scores"""
        logZ = self.inside(scores.requires_grad_(), mask)
        # the second inside process is necessary if using partial annotation
        score = self.inside((scores + s_labels).requires_grad_(),
                            mask, target, c_span_mask, combine_mask)
        if normalizer == 'token':
            norm = mask.sum()
        elif normalizer == 'sentence':
            norm = mask.size(0)
        elif normalizer == 'target':
            if target.dtype == torch.bool:
                norm = target.sum()
            elif target.dtype == torch.long:
                norm = target.ge(0).sum()
        else:
            raise NotImplementedError
        # print(f'logZ: {logZ}, score: {score}')
        loss = (logZ - score) / norm

        return loss

    @torch.enable_grad()
    def marginalize(self, scores, mask, target=None, c_span_mask=None, combine_mask=None):
        """Get marginal probability for elements in scores."""
        # marginals are used for decoding, and can be computed by combining the inside pass and autograd mechanism
        logZ = self.inside(scores.requires_grad_(), mask,
                           target, c_span_mask, combine_mask)
        marginals, = autograd.grad(logZ, scores)
        return marginals

    def inside(self, s_arc, mask, target=None, c_span_mask=None, combine_mask=None):
        # the end position of each sentence in a batch
        lens = mask.sum(1)
        batch_size, seq_len, _ = s_arc.shape
        # [seq_len, seq_len, batch_size]
        s_arc = s_arc.permute(2, 1, 0)
        s_i = torch.full_like(s_arc, MIN).double()
        s_c = torch.full_like(s_arc, MIN).double()
        # the logsumexp score of complete spans (i, i) is logsumexp(1)
        s_c.diagonal().fill_(0)
        c_span_mask = c_span_mask.permute(
            2, 1, 0) if c_span_mask is not None else None
        combine_mask = combine_mask.permute(
            1, 2, 3, 0) if combine_mask is not None else None

        # mask illegal arcs
        if target is not None:
            # [seq_len, seq_len, batch_size]
            target = target.permute(2, 1, 0).gt(0)
            s_arc = s_arc.masked_fill(~target, MIN)

        for w in range(1, seq_len):
            # n denotes the number of spans to iterate,
            # from span (0, w) to span (n, n+w) given width w
            n = seq_len - w

            # ilr = C(i->r) + C(j->r+1)
            # [n, w, batch_size]
            ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
            if ilr.requires_grad:
                ilr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            il = ir = ilr.permute(2, 0, 1).logsumexp(-1)
            # I(j->i) = logsumexp(C(i->r) + C(j->r+1)) + s(j->i), i <= r < j
            # fill the w-th diagonal of the lower triangular part of s_i with I(j->i) of n spans
            s_i.diagonal(-w).copy_(il + s_arc.diagonal(-w))
            # I(i->j) = logsumexp(C(i->r) + C(j->r+1)) + s(i->j), i <= r < j
            # fill the w-th diagonal of the upper triangular part of s_i with I(i->j) of n spans
            s_i.diagonal(w).copy_(ir + s_arc.diagonal(w))

            w_combine_mask = combine_mask[w].clone(
            ).detach() if combine_mask is not None else None
            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            # [n, w, batch_size]
            cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
            if w_combine_mask is not None:
                cl = cl.masked_fill(
                    stripe(w_combine_mask, n, w, (0, 0), 0), MIN)
            cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            # [batch_size, n]
            cl = cl.permute(2, 0, 1).logsumexp(-1)
            if c_span_mask is not None:
                cl = cl.masked_fill(~c_span_mask.diagonal(-w), MIN)
            s_c.diagonal(-w).copy_(cl)
            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
            if w_combine_mask is not None:
                cr = cr.masked_fill(
                    stripe(w_combine_mask, n, w, (1, w), 0), MIN)
            cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            cr = cr.permute(2, 0, 1).logsumexp(-1)
            if c_span_mask is not None:
                cr = cr.masked_fill(~c_span_mask.diagonal(w), MIN)
            s_c.diagonal(w).copy_(cr)
            # prevent multiroot for ROOT
            s_c[0, w][lens.ne(w)] = MIN

        return s_c[0].gather(0, lens.unsqueeze(0)).sum()


class Coarse2FineCharCRFDependency(CharCRFDependency):
    r"""
    First-order TreeCRF for calculating partitions and marginals of projective dependency trees
    in :math:`O(n^3)` :cite:`zhang-etal-2020-efficient`.
    """
    @torch.enable_grad()
    def forward(self, scores, mask, target=None, partial=False, marginal=None, c_span_mask=None, i_span_mask=None, root_as_head=True, normalizer='token'):
        r"""
        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible dependent-head pairs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
                The first column serving as pseudo words for roots should be ``False``.
            target (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard dependent-head pairs.
                This should be provided for loss calculation.
                If partially annotated, the unannotated positions should be filled with -1.
                Default: ``None``.
            partial (bool):
                ``True`` indicates that the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """
        training = scores.requires_grad
        # always enable the gradient computation of scores in order for the computation of marginals
        # logZ = self.inside(scores.requires_grad_(), mask, LogSemiring, root_as_head=False)
        logZ = self.inside(scores.requires_grad_(), mask, LogSemiring, root_as_head=root_as_head)
        marginals = scores
        if marginal == 'denominator':
            marginals, = autograd.grad(logZ, scores, retain_graph=training)

        # the second inside process is necessary if using partial annotation
        if partial:
            scores = self.mask(scores, target)
            score = self.inside(scores, mask, LogSemiring,
                                c_span_mask, i_span_mask, root_as_head)
            if marginal == 'numerator':
                marginals, = autograd.grad(
                    score, scores, retain_graph=training)
        else:
            score = scores.gather(-1, target.unsqueeze(-1)
                                  ).squeeze(-1)[mask].sum()

        if normalizer == 'token':
            norm = mask.sum()
        elif normalizer == 'sentence':
            norm = mask.size(0)
        elif normalizer == 'target':
            if target.dtype == torch.bool:
                norm = target.sum()
            elif target.dtype == torch.long:
                norm = target.ge(0).sum()

        loss = (logZ - score) / norm

        return loss, marginals

    def mask(self, scores, target):
        """mask scores with target"""
        assert target.dtype == torch.bool and target.size() == scores.size()
        return scores.masked_fill(~target, MIN)

    @torch.enable_grad()
    def labeling(self, scores, s_labels, mask, target, c_span_mask=None, i_span_mask=None, root_as_head=True, normalizer='token'):
        """merge unlabeled and labeled loss by adding label (log) probability to unlabeled scores"""
        # logZ = self.inside(scores.requires_grad_(), mask, LogSemiring, root_as_head=False)
        logZ = self.inside(scores.requires_grad_(), mask, LogSemiring, root_as_head=root_as_head)
        scores = self.mask(scores + s_labels, target)
        # the second inside process is necessary if using partial annotation
        score = self.inside(scores.requires_grad_(),
                            mask, LogSemiring, c_span_mask, i_span_mask, root_as_head)
        if normalizer == 'token':
            norm = mask.sum()
        elif normalizer == 'sentence':
            norm = mask.size(0)
        elif normalizer == 'target':
            if target.dtype == torch.bool:
                norm = target.sum()
            elif target.dtype == torch.long:
                norm = target.ge(0).sum()
        else:
            raise NotImplementedError
        # print(f'logZ: {logZ}, score: {score}')
        loss = (logZ - score) / norm

        return loss

    @torch.enable_grad()
    def marginalize(self, scores, mask, target=None, c_span_mask=None, i_span_mask=None, root_as_head=True):
        """Get marginal probability for elements in scores."""
        # marginals are used for decoding, and can be computed by combining the inside pass and autograd mechanism
        scores = self.mask(scores, target) if target is not None else scores
        logZ = self.inside(scores.requires_grad_(), mask,
                           LogSemiring, c_span_mask, i_span_mask, root_as_head)
        marginals, = autograd.grad(logZ, scores)
        return marginals

    @torch.enable_grad()
    def decode(self, scores, mask, target=None, c_span_mask=None, i_span_mask=None, root_as_head=True):
        """
        Args:
            scores: [batch_size, seq_len, seq_len, 2]
        """
        scores = self.mask(scores, target) if target is not None else scores
        logZ = self.inside(scores.requires_grad_(), mask,
                           MaxSemiring, c_span_mask, i_span_mask, root_as_head)
        marginals, = autograd.grad(logZ, scores)

        return marginals

    def inside(self, s_arc, mask, semiring, c_span_mask=None, i_span_mask=None, root_as_head=True):
        """
        Args:
            s_arc: [batch_size, seq_len, seq_len, 2]
        """
        # only allow an intra-span link to another intra-span for intra-span linking
        intra_link_rules = mask.new_tensor([1, 0, 0, 0])
        inter_link_rules = mask.new_tensor([1, 1, 1, 1])
        intra_combine_rules = mask.new_tensor([1, 0, 0, 0])
        # allow non-root as head for logZ
        inter_combine_rules = (mask.new_tensor([0, 1, 0, 1]) if root_as_head else mask.new_tensor([0, 1, 1, 1]))
        # NOTE: the following rule is not correct
        # inter_combine_rules = mask.new_tensor([0, 0, 1, 1])
        lens = mask.sum(dim=-1)
        batch_size, _, seq_len, n_state = s_arc.shape
        # [seq_len, seq_len, batch_size, n_state]
        s_arc = s_arc.permute(2, 1, 0, 3).requires_grad_()
        # complete spans
        s_c = torch.full_like(s_arc, MIN)
        # allow complete spans of width 1,
        # and don't allow inter-word complete spans of width 1
        s_c.diagonal()[:, 0, :].fill_(0)
        # s_c.diagonal().fill_(0)
        # incomplete spans
        s_i = torch.full_like(s_arc, MIN)

        c_span_mask = c_span_mask.permute(2, 1, 0, 3) if c_span_mask is not None else None
        i_span_mask = i_span_mask.permute(2, 1, 0, 3) if i_span_mask is not None else None

        for w in range(1, seq_len):
            n = seq_len - w

            # I(i->j/i<-j) = C(i->r) + C(j->r+1) + s(i->j/i<-j)
            # rightward incomplete spans, \hat{C}(i->r), C(i->r)
            # [n, w, batch_size, n_state] -> [batch_size, n, w, n_state]
            ir = stripe(s_c, n, w).permute(2, 0, 1, 3)
            # leftward incomplete spans, \hat{C}(j->r+1), C(j->r+1)
            il = stripe(s_c, n, w, (w, 1)).permute(2, 0, 1, 3)
            # [batch_size, n, w, n_state*n_state]
            # \hat{C}(i->r) + \hat{C}(j->r+1), \hat{C}(i->r) + C(j->r+1), C(i->r) + \hat{C}(j->r+1), C(i->r) + C(j->r+1)
            lr_link = (ir.unsqueeze(-1) + il.unsqueeze(-2)).flatten(start_dim=-2, end_dim=-1)
            # accumulate values of all split points
            # [batch_size, n, n_state*n_state]
            lr_link = semiring.sum(lr_link, dim=-2)

            # only preserve legal link operations
            # for intra-word incomplete spans: \hat{C}(i->r) + \hat{C}(j->r+1)
            # [batch_size, n, n_rules]
            intra_lr_link = lr_link[..., intra_link_rules]
            # accumulate values of all rules
            # [batch_size, n, 1]
            intra_l_link = intra_r_link = semiring.sum(intra_lr_link, dim=-1, keepdim=True)
            # for inter-word incomplete spans:
            # \hat{C}(i->r) + \hat{C}(j->r+1), \hat{C}(i->r) + C(j->r+1), C(i->r) + \hat{C}(j->r+1), C(i->r) + C(j->r+1)
            # [batch_size, n, n_rules]
            inter_lr_link = lr_link[..., inter_link_rules]
            # [batch_size, n, 1]
            inter_l_link = inter_r_link = semiring.sum(inter_lr_link, dim=-1, keepdim=True)
            # left/right intra/inter-span link
            # [batch_size, n, n_state] -> [batch_size, n_state, n]
            l_link = torch.cat((intra_l_link, inter_l_link), -1).permute(0, 2, 1)
            # [batch_size, n_state, n]
            l_link = l_link + s_arc.diagonal(-w)
            # NOTE: it is no necessary to mask out illegal incomplete spans,
            # which have already been masked out by inter-arcs.
            if i_span_mask is not None:
                # mask out illegal incomplete spans
                l_link = l_link.masked_fill(~i_span_mask.diagonal(-w), MIN)
            # [batch_size, n_state, n]
            s_i.diagonal(-w).copy_(l_link)
            r_link = torch.cat((intra_r_link, inter_r_link), -1).permute(0, 2, 1)
            r_link = r_link + s_arc.diagonal(w)
            if i_span_mask is not None:
                r_link = r_link.masked_fill(~i_span_mask.diagonal(w), MIN)
            # [batch_size, n, n_state]
            s_i.diagonal(w).copy_(r_link)

            # [batch_size, n, w, n_state]
            # \hat{C}(r->i), C(r->i)
            cl = stripe(s_c, n, w, (0, 0), 0).permute(2, 0, 1, 3)
            # \hat{I}(j->r), I(j->r)
            il = stripe(s_i, n, w, (w, 0)).permute(2, 0, 1, 3)
            # [batch_size, n, w, n_state*n_state]
            # \hat{C}(r->i) + \hat{I}(j->r), \hat{C}(r->i) + I(j->r), C(r->i) + \hat{I}(j->r), C(r->i) + I(j->r)
            l_combine = (cl.unsqueeze(-1) + il.unsqueeze(-2)).flatten(start_dim=-2, end_dim=-1)
            # l_combine.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            # [batch_size, n, n_state*n_state], select maximum values from all split points
            # l_combine = l_combine.logsumexp(-2)
            l_combine = semiring.sum(l_combine, dim=-2)
            # [batch_size, n, n_rules], only preserve legal combining operations
            intra_l_combine = l_combine[..., intra_combine_rules]
            # [batch_size, n, 1], select maximum values from all rules
            # intra_l_combine = intra_l_combine.logsumexp(-1, keepdim=True)
            intra_l_combine = semiring.sum(intra_l_combine, dim=-1, keepdim=True)
            # [batch_size, n, n_rules]
            inter_l_combine = l_combine[..., inter_combine_rules]
            # [batch_size, n, 1]
            # inter_l_combine = inter_l_combine.logsumexp(-1, keepdim=True)
            inter_l_combine = semiring.sum(inter_l_combine, dim=-1, keepdim=True)
            # left intra/inter-span combine
            # [batch_size, n, n_state] -> [batch_size, n_state, n]
            l_combine = torch.cat((intra_l_combine, inter_l_combine), -1).permute(0, 2, 1)
            if c_span_mask is not None:
                # mask out illegal complete spans
                l_combine = l_combine.masked_fill(~c_span_mask.diagonal(-w), MIN)
            # [batch_size, n_state, n]
            s_c.diagonal(-w).copy_(l_combine)

            # [batch_size, n, w, n_state]
            # \hat{C}(r->j), C(r->j)
            cr = stripe(s_c, n, w, (1, w), 0).permute(2, 0, 1, 3)
            # \hat{I}(i->r), I(i->r)
            ir = stripe(s_i, n, w, (0, 1)).permute(2, 0, 1, 3)
            # [batch_size, n, w, n_state*n_state]
            r_combine = (cr.unsqueeze(-1) + ir.unsqueeze(-2)).flatten(start_dim=-2, end_dim=-1)
            # r_combine.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            # [batch_size, n, n_state*n_state], select maximum values from all split points
            r_combine = semiring.sum(r_combine, dim=-2)
            # [batch_size, n, n_rules], only preserve legal combining operations
            intra_r_combine = r_combine[..., intra_combine_rules]
            # [batch_size, n, 1], select maximum values from all rules
            intra_r_combine = semiring.sum(intra_r_combine, dim=-1, keepdim=True)
            # [batch_size, n, n_rules]
            inter_r_combine = r_combine[..., inter_combine_rules]
            # [batch_size, n, 1]
            inter_r_combine = semiring.sum(inter_r_combine, dim=-1, keepdim=True)
            # right intra/inter-span combine
            # [batch_size, n, n_state] -> [batch_size, n_state, n]
            r_combine = torch.cat((intra_r_combine, inter_r_combine), -1).permute(0, 2, 1)
            if c_span_mask is not None:
                r_combine = r_combine.masked_fill(~c_span_mask.diagonal(w), MIN)
            # [batch_size, n_state, n]
            s_c.diagonal(w).copy_(r_combine)
            # prevent multiple root
            s_c[0, w][lens.ne(w)] = MIN

        return s_c[0, :, :, 1].gather(0, lens.unsqueeze(0)).sum()


class CRF2oDependency(nn.Module):
    r"""
    Second-order TreeCRF for calculating partitions and marginals of projective dependency trees
    in :math:`O(n^3)` :cite:`zhang-etal-2020-efficient`.
    """

    def __init__(self, multiroot=False):
        super().__init__()

        self.multiroot = multiroot

    def __repr__(self):
        return f"{self.__class__.__name__}(multiroot={self.multiroot})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=True, partial=False):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of two tensors `s_arc` and `s_sib`.
                `s_arc` (``[batch_size, seq_len, seq_len]``) holds Scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
                The first column serving as pseudo words for roots should be ``False``.
            target (~torch.LongTensor): ``[batch_size, seq_len]``.
                Tensors of gold-standard dependent-head pairs and dependent-head-sibling triples.
                If partially annotated, the unannotated positions should be filled with -1.
                Default: ``None``.
            mbr (bool):
                If ``True``, marginals will be returned to perform minimum Bayes-risk (MBR) decoding. Default: ``False``.
            partial (bool):
                ``True`` indicates that the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """

        s_arc, s_sib = scores
        training = s_arc.requires_grad
        # always enable the gradient computation of scores in order for the computation of marginals
        logZ = self.inside(*(s.requires_grad_() for s in scores), mask)
        # marginals are used for decoding, and can be computed by combining the inside pass and autograd mechanism
        marginals = s_arc
        if mbr:
            marginals, = autograd.grad(logZ, s_arc, retain_graph=training)

        if target is None:
            return marginals
        arcs, sibs = target
        # the second inside process is needed if using partial annotation
        if partial:
            score = self.inside(s_arc, s_sib, mask, arcs)
        else:
            s_arc = s_arc.gather(-1, arcs.unsqueeze(-1))[mask]
            s_sib = s_sib.gather(-1, sibs.unsqueeze(-1))[sibs.gt(0)]
            score = s_arc.sum() + s_sib.sum()
        loss = (logZ - score) / mask.sum()

        return loss, marginals

    def inside(self, s_arc, s_sib, mask, cands=None):
        # the end position of each sentence in a batch
        lens = mask.sum(1)
        batch_size, seq_len, _ = s_arc.shape
        # [seq_len, seq_len, batch_size]
        s_arc = s_arc.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size]
        s_sib = s_sib.permute(2, 1, 3, 0)
        s_i = torch.full_like(s_arc, float('-inf'))
        s_s = torch.full_like(s_arc, float('-inf'))
        s_c = torch.full_like(s_arc, float('-inf'))
        s_c.diagonal().fill_(0)

        # set the arcs scores excluded by cands to -inf
        if cands is not None:
            mask = mask.index_fill(1, lens.new_tensor(0), 1)
            mask = (mask.unsqueeze(1) & mask.unsqueeze(-1)).permute(2, 1, 0)
            cands = cands.unsqueeze(-1).index_fill(1, lens.new_tensor(0), -1)
            cands = cands.eq(lens.new_tensor(range(seq_len))) | cands.lt(0)
            cands = cands.permute(2, 1, 0) & mask
            s_arc = s_arc.masked_fill(~cands, float('-inf'))

        for w in range(1, seq_len):
            # n denotes the number of spans to iterate,
            # from span (0, w) to span (n, n+w) given width w
            n = seq_len - w
            # I(j->i) = logsum(exp(I(j->r) + S(j->r, i)) +, i < r < j
            #                  exp(C(j->j) + C(i->j-1)))
            #           + s(j->i)
            # [n, w, batch_size]
            il = stripe(s_i, n, w, (w, 1)) + stripe(s_s, n, w, (1, 0), 0)
            il += stripe(s_sib[range(w, n+w), range(n)], n, w, (0, 1))
            # [n, 1, batch_size]
            il0 = stripe(s_c, n, 1, (w, w)) + stripe(s_c, n, 1, (0, w - 1))
            # il0[0] are set to zeros since the scores of the complete spans starting from 0 are always -inf
            il[:, -1] = il0.index_fill_(0, lens.new_tensor(0), 0).squeeze(1)
            if il.requires_grad:
                il.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            il = il.permute(2, 0, 1).logsumexp(-1)
            s_i.diagonal(-w).copy_(il + s_arc.diagonal(-w))
            # I(i->j) = logsum(exp(I(i->r) + S(i->r, j)) +, i < r < j
            #                  exp(C(i->i) + C(j->i+1)))
            #           + s(i->j)
            # [n, w, batch_size]
            ir = stripe(s_i, n, w) + stripe(s_s, n, w, (0, w), 0)
            ir += stripe(s_sib[range(n), range(w, n+w)], n, w)
            ir[0] = float('-inf')
            # [n, 1, batch_size]
            ir0 = stripe(s_c, n, 1) + stripe(s_c, n, 1, (w, 1))
            ir[:, 0] = ir0.squeeze(1)
            if ir.requires_grad:
                ir.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            ir = ir.permute(2, 0, 1).logsumexp(-1)
            s_i.diagonal(w).copy_(ir + s_arc.diagonal(w))

            # [n, w, batch_size]
            slr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
            if slr.requires_grad:
                slr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            slr = slr.permute(2, 0, 1).logsumexp(-1)
            # S(j, i) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
            s_s.diagonal(-w).copy_(slr)
            # S(i, j) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
            s_s.diagonal(w).copy_(slr)

            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
            cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).logsumexp(-1))
            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
            cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_c.diagonal(w).copy_(cr.permute(2, 0, 1).logsumexp(-1))
            if not self.multiroot:
                s_c[0, w][lens.ne(w)] = float('-inf')

        return s_c[0].gather(0, lens.unsqueeze(0)).sum()


class Const2DepCRF(nn.Module):
    def __init__(self, multiroot=False):
        super().__init__()

        self.multiroot = multiroot

    def __repr__(self):
        return f"{self.__class__.__name__}(multiroot={self.multiroot})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, c_span_mask=None, combine_mask=None, partial=True, normalizer='token'):
        r"""
        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible dependent-head pairs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
                The first column serving as pseudo words for roots should be ``False``.
            target (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard dependent-head pairs.
                This should be provided for loss calculation.
                If partially annotated, the unannotated positions should be filled with -1.
                Default: ``None``.
            partial (bool):
                ``True`` indicates that the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """
        # always enable the gradient computation of scores in order for the computation of marginals
        logZ = self.inside(scores.requires_grad_(), mask, LogSemiring)
        marginals = scores
        # the second inside process is necessary if using partial annotation
        if partial:
            # TODO: mask scores with target
            score = self.inside(scores, mask, LogSemiring,
                                c_span_mask, combine_mask)
        else:
            raise NotImplementedError

        if normalizer == 'token':
            norm = mask.sum()
        elif normalizer == 'sentence':
            norm = mask.size(0)
        else:
            raise ValueError
        loss = (logZ - score) / norm

        # logger.info(f'logZ: {logZ}, score: {score}, loss: {loss}')
        # print(f'logZ: {logZ}, score: {score}, loss: {loss}, mask: {mask.shape}')

        return loss, marginals

    @torch.enable_grad()
    def decode(self, scores, mask, target=None, c_span_mask=None, combine_mask=None):
        """
        Args:
            scores: [batch_size, seq_len, seq_len]
        """
        lens = mask.sum(1)
        batch_size, seq_len, _ = scores.shape
        scores = self.mask(scores, target) if target is not None else scores
        logZ = self.inside(scores.requires_grad_(), mask,
                           MaxSemiring, c_span_mask, combine_mask)
        marginals, = autograd.grad(logZ, scores)
        preds = lens.new_zeros(batch_size, seq_len).masked_scatter_(mask, marginals.nonzero()[:, 2])
        return preds

    def inside(self, s_arc, mask, semiring, c_span_mask=None, combine_mask=None):
        # the end position of each sentence in a batch
        lens = mask.sum(1)
        batch_size, seq_len, _ = s_arc.shape
        # [seq_len, seq_len, batch_size]
        s_arc = s_arc.permute(2, 1, 0)
        s_i = torch.full_like(s_arc, MIN).double()
        s_c = torch.full_like(s_arc, MIN).double()
        # the logsumexp score of complete spans (i, i) is logsumexp(1)
        s_c.diagonal().fill_(0)
        c_span_mask = c_span_mask.permute(
            2, 1, 0) if c_span_mask is not None else None
        combine_mask = combine_mask.permute(
            1, 2, 3, 0) if combine_mask is not None else None

        for w in range(1, seq_len):
            # n denotes the number of spans to iterate,
            # from span (0, w) to span (n, n+w) given width w
            n = seq_len - w

            # ilr = C(i->r) + C(j->r+1)
            # [n, w, batch_size]
            ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
            if ilr.requires_grad:
                ilr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            il = ir = semiring.sum(ilr.permute(2, 0, 1), dim=-1)
            # I(j->i) = logsumexp(C(i->r) + C(j->r+1)) + s(j->i), i <= r < j
            # fill the w-th diagonal of the lower triangular part of s_i with I(j->i) of n spans
            s_i.diagonal(-w).copy_(il + s_arc.diagonal(-w))
            # I(i->j) = logsumexp(C(i->r) + C(j->r+1)) + s(i->j), i <= r < j
            # fill the w-th diagonal of the upper triangular part of s_i with I(i->j) of n spans
            s_i.diagonal(w).copy_(ir + s_arc.diagonal(w))

            w_combine_mask = combine_mask[w].clone(
            ).detach() if combine_mask is not None else None
            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            # [n, w, batch_size]
            cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
            if w_combine_mask is not None:
                cl = cl.masked_fill(
                    stripe(w_combine_mask, n, w, (0, 0), 0), MIN)
            cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            # [batch_size, n]
            cl = semiring.sum(cl.permute(2, 0, 1), dim=-1)
            if c_span_mask is not None:
                cl = cl.masked_fill(~c_span_mask.diagonal(-w), MIN)
            s_c.diagonal(-w).copy_(cl)
            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
            if w_combine_mask is not None:
                cr = cr.masked_fill(
                    stripe(w_combine_mask, n, w, (1, w), 0), MIN)
            cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            cr = semiring.sum(cr.permute(2, 0, 1), dim=-1)
            if c_span_mask is not None:
                cr = cr.masked_fill(~c_span_mask.diagonal(w), MIN)
            s_c.diagonal(w).copy_(cr)
            # prevent multiroot for ROOT
            s_c[0, w][lens.ne(w)] = MIN

        # logger.info(f"{s_c[0].gather(0, lens.unsqueeze(0))}")
        # print(f"{s_c[0].gather(0, lens.unsqueeze(0))}")

        return s_c[0].gather(0, lens.unsqueeze(0)).sum()


class CRFConstituency(nn.Module):
    r"""
    TreeCRF for calculating partitions and marginals of constituency trees in :math:`O(n^3)` :cite:`zhang-etal-2020-fast`.
    """

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=False):
        r"""
        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible constituents.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid parsing over padding tokens.
                For each square matrix in a batch, the positions except upper triangular part should be masked out.
            target (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard constituents. ``True`` if a constituent exists. Default: ``None``.
            mbr (bool):
                If ``True``, marginals will be returned to perform minimum Bayes-risk (MBR) decoding. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """

        training = scores.requires_grad
        # always enable the gradient computation of scores in order for the computation of marginals
        logZ = self.inside(scores.requires_grad_(), mask)
        # marginals are used for decoding, and can be computed by combining the inside pass and autograd mechanism
        marginals = scores
        if mbr:
            marginals, = autograd.grad(logZ, scores, retain_graph=training)
        if target is None:
            return marginals
        loss = (logZ - scores[mask & target].sum()) / mask[:, 0].sum()

        return loss, marginals

    def inside(self, scores, mask):
        lens = mask[:, 0].sum(-1)
        batch_size, seq_len, _ = scores.shape
        # [seq_len, seq_len, batch_size]
        scores, mask = scores.permute(1, 2, 0), mask.permute(1, 2, 0)
        s = torch.full_like(scores, float('-inf'))

        for w in range(1, seq_len):
            # n denotes the number of constituents to iterate,
            # from constituent (0, w) to constituent (n, n+w) given width w
            n = seq_len - w

            if w == 1:
                s.diagonal(w).copy_(scores.diagonal(w))
                continue
            # [n, w, batch_size]
            s_s = stripe(s, n, w-1, (0, 1)) + stripe(s, n, w-1, (1, w), 0)
            # [batch_size, n, w]
            s_s = s_s.permute(2, 0, 1)
            if s_s.requires_grad:
                s_s.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_s = s_s.logsumexp(-1)
            s.diagonal(w).copy_(s_s + scores.diagonal(w))

        return s[0].gather(0, lens.unsqueeze(0)).sum()


# class CRFCharConstituency(CRFConstituency):

#     def __init__(self):
#         super().__init__()

#         self.mask_inside = False
#         self.start_mask = None
#         self.trans_mask = None
#         self.rule_mask = None
#         self.rule_index = None

#     @torch.enable_grad()
#     def forward(self, scores, mask, target=None, marg=False):
#         """[summary]

#         Args:
#             scores (Tensor(B, seq_len, seq_len, n_labels))
#             mask (Tensor(B, seq_len, seq_len))
#             target (Tensor(B, seq_len, seq_len)): Defaults to None.
#             marg (bool, optional): Defaults to False.

#         Returns:
#             crf-loss, marginal probability for spans
#         """
#         # (B)
#         lens = mask[:, 0].sum(-1)
#         total = lens.sum()
#         batch_size, seq_len, seq_len, n_labels = scores.shape
#         # in eval(), it's false; and in train(), it's true
#         training = scores.requires_grad
#         # always enable the gradient computation of scores
#         # in order for the computation of marginal probs.
#         # requires_grad_(requires_grad=True):
#         # Change if autograd should record operations on scores:
#         #   sets scores’s requires_grad attribute in-place. Returns this tensor.
#         if mask_inside:
#             # (seq_len, seq_len, B)
#             s = inside_mask(scores.requires_grad_(),
#                             trans_mask, start_mask, mask)
#             # (1, n_labels, batch_size)
#             lens = lens.view(1, 1, batch_size).expand([-1, n_labels, -1])
#             # get alpha(0, length, l) for each sentence
#             # (seq_len, n_labels, B).gather(0, Tensor(1, B)) -> Tensor(1, B)
#             logZ = s[0].gather(0, lens).logsumexp(-2).sum()
#         else:
#             s = inside_simple(scores.requires_grad_(), mask)
#             # get alpha(0, length, l) for each sentence
#             # (seq_len, B).gather(0, Tensor(1, B)) -> Tensor(1, B)
#             logZ = s[0].gather(0, lens.unsqueeze(0)).sum()

#         probs = scores
#         if marg:
#             # Computes and returns the sum of gradients of outputs w.r.t. the inputs.
#             # retain_graph: If False, the graph used to compute the grad will be freed.
#             # Tensor(B, seq_len, seq_len, n_labels)
#             probs, = autograd.grad(logZ, scores, retain_graph=training)
#         if target is None:
#             return probs

#         # TODO: target -> (B, seq_len, seq_len, 3)
#         # FIXME: simple inside
#         # (B, seq_len, seq_len, n_coarse_labels)
#         scores = scores.clone().masked_fill(~target, MIN)
#         s = inside_mask(scores.requires_grad_(), trans_mask, start_mask, mask)
#         # for span(0, n), only score of `SYN` label is allowed,
#         # the logsumexp will accumulate scores of other labels,
#         # which are `-inf` and thus will be discard, for exp(-inf) = 0.
#         s = s[0].gather(0, lens).logsumexp(-2).sum()

#         total = mask.sum()
#         loss = (logZ - s) / total
#         return loss, probs

#     def labeled_inside(self, scores, mask):
#         """Inside algorithm for labeled dependency tree.

#         Args:
#             scores (Tensor(B, seq_len, seq_len, n_labels))
#             trans_mask (Tensor(n_labels, n_labels, n_labels)): boolen value
#                 (i, j, k) == 0 indicates k->ij is impossible
#                 (i, j, k) == 1 indicates k->ij is possible
#             mask (Tensor(B, seq_len, seq_len))

#         Returns:
#             Tensor: [seq_len, seq_len, batch_size]
#         """
#         # [batch_size, seq_len, seq_len]
#         scores = scores.logsumexp(-1)
#         return self.unlabeled_inside(scores, mask)

#     def unlabeled_inside(self, scores, mask):
#         """Inside algorithm for unlabeled dependency tree.

#         Args:
#             scores (Tensor(B, seq_len, seq_len))
#             mask (Tensor(B, seq_len, seq_len))

#         Returns:
#             Tensor: [seq_len, seq_len, batch_size]
#         """
#         batch_size, seq_len, seq_len = scores.shape
#         # permute is convenient for diagonal which acts on dim1=0 and dim2=1
#         # [seq_len, seq_len, batch_size]
#         scores, mask = scores.permute(1, 2, 0), mask.permute(1, 2, 0)
#         # s[i, j]: sub-tree spanning from i to j
#         # [seq_len, seq_len, batch_size]
#         s = torch.full_like(scores, MIN)

#         for w in range(1, seq_len):
#             # n denotes the number of spans to iterate,
#             # from span (0, w) to span (n, n+w) given width w
#             n = seq_len - w
#             # diag_mask is used for ignoring the excess of each sentence
#             # [batch_size, n]
#             # diag_mask = mask.diagonal(w)

#             if w == 1:
#                 # scores.diagonal(w): [n_labels, batch_size, n]
#                 # scores.diagonal(w).permute(1, 2, 0)[diag_mask]: (T, n_labels)
#                 # s.diagonal(w).permute(1, 2, 0)[diag_mask] = scores.diagonal(w).permute(1, 2, 0)[diag_mask]
#                 # no need  diag_mask
#                 # [n_labels, batch_size]
#                 s.diagonal(w).copy_(scores.diagonal(w))
#                 continue

#             # scores for sub-tree spanning from `i to k` and `k+1 to j`, considering all labels
#             # NOTE: stripe considering all split points and spans with same width
#             # stripe: [n, w-1, batch_size]
#             s_span = stripe(s, n, w-1, (0, 1)) + stripe(s, n, w-1, (1, w), 0)
#             # [batch_size, n, w-1]
#             s_span = s_span.permute(2, 0, 1)
#             # [batch_size, n]
#             s_span = s_span.logsumexp(-1)
#             # [batch_size, n] = [batch_size, n] +  [batch_size, n]
#             s.diagonal(w).copy_(s_span + scores.diagonal(w))

#         # [seq_len, seq_len, batch_size]
#         return s

#     def constrained_inside(self, scores, mask):
#         """
#         trans_mask (Tensor(n_labels, n_labels, n_labels)): boolen value
#                 (i, j, k) == 0 indicates k->ij is impossible
#                 (i, j, k) == 1 indicates k->ij is possible
#         """
#         batch_size, seq_len, seq_len, n_labels = scores.shape
#         # [seq_len, seq_len, n_labels, batch_size]
#         scores = scores.permute(1, 2, 3, 0)
#         # [seq_len, seq_len, n_labels, batch_size]
#         mask = mask.permute(1, 2, 0)

#         s_chart = torch.full_like(scores, MIN)

#         start_mask = self.start_mask.view(1, 1, n_labels)

#         for w in range(1, seq_len):
#             n = seq_len - w
#             # (B, n, n_labels)
#             emit_scores = scores.diagonal(w).permute(1, 2, 0)
#             # (B, n, n_labels)
#             diag_s = s_chart.diagonal(w).permute(1, 2, 0)

#             if w == 1:
#                 # 考虑长度为1的span是否可以获得这些标签，如不应该是SYN/SYN*
#                 # (n_labels, B, n)
#                 diag_s.copy_(emit_scores + start_mask)
#                 continue

#             # stripe: [n, w-1, n_labels, batch_size]
#             # [n, w-1, n_labels, 1, batch_size]
#             s_left = stripe(s_chart, n, w-1, (0, 1)).unsqueeze(-2)
#             # [n, w-1, 1, n_labels, batch_size]
#             s_right = stripe(s_chart, n, w-1, (1, w), 0).unsqueeze(-3)
#             # sum: [n, w-1, n_labels, n_labels, batch_size]
#             # [batch_size, n, w-1, n_labels, n_labels, 1]
#             s_span = (s_left + s_right).permute(4, 0, 1, 2, 3).unsqueeze(-1)
#             # emit_scores.permute(1 ,2 ,0)[..., None, None, None, :]: [batch_size, n, 1, 1, 1, n_labels]
#             # [batch_size, n, w-1, n_labels, n_labels, n_labels]
#             s_span = s_span + self.trans_mask + \
#                 emit_scores[..., None, None, None, :]

#             if s_span.requires_grad:
#                 s_span.register_hook(
#                     lambda x: x.masked_fill_(torch.isnan(x), 0))

#             # [batch_size, n, n_labels]
#             s_span = s_span.logsumexp([2, 3, 4])

#             # or mask, can't implement for [batch_size, n, n_labels], can only get
#             # s_span[label_mask] -> (B, n, w-1, T)
#             # s_span = s_span[label_mask].logsumexp([])

#             s_chart.diagonal(w).copy_(s_span)

#         return s_chart

#     def constrained_inside2(self, scores, mask):
#         """
#         Args:
#             scores (torch.Tensor): scores for each labeled span
#                 [batch_size, seq_len, seq_len, n_labels]
#             mask (torch.Tensor)
#                 [batch_size, seq_len, seq_len]
#         """
#         batch_size, seq_len, seq_len, n_labels = scores.shape
#         # [seq_len, seq_len, batch_size, n_labels]
#         scores = scores.permute(2, 1, 0, 3)
#         # [seq_len, seq_len, batch_size]
#         mask = mask.permute(1, 2, 0)

#         s_chart = torch.full_like(scores, MIN)

#         for w in range(1, seq_len):
#             n = seq_len - w
#             # [batch_size, n_labels, n]
#             span_scores = scores.diagonal(w)

#             if w == 1:
#                 # for span of length 1，can't be labeled as SYN/SYN*
#                 span_scores.masked_fill_(~self.start_mask.unsqueeze(-1), MIN)
#                 s_chart.diagonal(w).copy_(span_scores)
#                 continue

#             # [n, w-1, batch_size, n_labels] -> [batch_size, n, w-1, n_labels]
#             s_left = stripe(s_chart, n, w-1, (0, 1)).permute(2, 0, 1, 3)
#             # [n, w-1, batch_size, n_labels] -> [batch_size, n, w-1, n_labels]
#             s_right = stripe(s_chart, n, w-1, (1, w), 0).permute(2, 0, 1, 3)
#             # [batch_size, n, w-1, n_labels, n_labels]
#             s_span = (s_left.unsqueeze(-1) + s_right.unsqueeze(-2))
#             # TODO: check correction, pre-sum each split point
#             # [batch_size, n, n_labels, n_labels]
#             s_span = s_span.logsumexp(dim=2)
#             # [batch_size, n, n_labels*n_labels]
#             s_span = s_span.flatten(start_dim=-2, end_dim=-1)

#             # exclude illegal rules
#             # [batch_size, n, n_rules]
#             s_span = s_span[..., self.rule_mask]

#             # select legal rules for each labeled span
#             # [batch_size, n, n_labels, n_rules]
#             s_span = s_span[..., self.rule_index]
#             # add current labeled span score
#             s_span = s_span + span_scores.permute(0, 2, 1).unsqueeze(-1)
#             # mask where rule index is `-1` for padding rules
#             s_span.masked_fill_(self.rule_index.eq(-1), MIN)

#             # [batch_size, n, n_labels]
#             # TODO: semiring
#             s_span = s_span.logsumexp(-1).permute(0, 2, 1)

#             s_chart.diagonal(w).copy_(s_span)

#         return s_chart

#     def constrained_inside3(self, scores, mask):
#         """
#         Args:
#             scores (torch.Tensor): scores for each labeled span
#                 [batch_size, seq_len, seq_len, n_labels]
#             mask (torch.Tensor)
#                 [batch_size, seq_len, seq_len]
#         """
#         batch_size, seq_len, seq_len, n_labels = scores.shape
#         # [seq_len, seq_len, batch_size, n_labels]
#         scores = scores.permute(2, 1, 0, 3)
#         # [seq_len, seq_len, batch_size]
#         mask = mask.permute(1, 2, 0)

#         s_chart = torch.full_like(scores, MIN)

#         for w in range(1, seq_len):
#             n = seq_len - w
#             # [batch_size, n_labels, n]
#             span_scores = scores.diagonal(w)

#             if w == 1:
#                 # for span of length 1，can't be labeled as SYN/SYN*
#                 span_scores.masked_fill_(~self.start_mask.unsqueeze(-1), MIN)
#                 s_chart.diagonal(w).copy_(span_scores)
#                 continue

#             # [n, w-1, batch_size, n_labels] -> [batch_size, n, w-1, n_labels]
#             s_left = stripe(s_chart, n, w-1, (0, 1)).permute(2, 0, 1, 3)
#             # [n, w-1, batch_size, n_labels] -> [batch_size, n, w-1, n_labels]
#             s_right = stripe(s_chart, n, w-1, (1, w), 0).permute(2, 0, 1, 3)
#             # [batch_size, n, w-1, 2*n_labels]
#             s_span = torch.cat((s_left, s_right), dim=-1)
#             # [batch_size, n, w-1, n_rules]
#             s_span = torch.matmul(s_span, self.rule_matrix)
#             # TODO: check correction, pre-sum each split point
#             # [batch_size, n, n_rules]
#             s_span = s_span.logsumexp(dim=2)

#             # [batch_size, n, n_labels, n_rules]
#             s_span = s_span[..., self.rule_index]
#             # add current labeled span score
#             s_span = s_span + span_scores.permute(0, 2, 1).unsqueeze(-1)
#             # mask where rule index is `-1` for padding rules
#             s_span.masked_fill_(self.rule_index.eq(-1), MIN)

#             # [batch_size, n, n_labels]
#             # TODO: semiring
#             s_span = s_span.logsumexp(-1).permute(0, 2, 1)

#             s_chart.diagonal(w).copy_(s_span)

#         return s_chart
