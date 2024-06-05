# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from supar.utils.logging import get_logger

logger = get_logger(__name__)


class MFVIDependency(nn.Module):
    r"""
    Mean Field Variational Inference for approximately calculating marginals
    of dependency trees :cite:`wang-tu-2020-second`.
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of three tensors `s_arc` and `s_sib`.
                `s_arc` (``[batch_size, seq_len, seq_len]``) holds scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
            target (~torch.LongTensor): ``[batch_size, seq_len]``.
                A Tensor of gold-standard dependent-head pairs. Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        logits = self.mfvi(*scores, mask)
        marginals = logits.softmax(-1)

        if target is None:
            return marginals
        loss = F.cross_entropy(logits[mask], target[mask])

        return loss, marginals

    def mfvi(self, s_arc, s_sib, mask):
        batch_size, seq_len = mask.shape
        ls, rs = torch.stack(torch.where(mask.new_ones(seq_len, seq_len))).view(-1, seq_len, seq_len).sort(0)[0]
        mask = mask.index_fill(1, ls.new_tensor(0), 1)
        # [seq_len, seq_len, batch_size], (h->m)
        mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask2o = mask2o & ls.unsqueeze(-1).ne(ls.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & rs.unsqueeze(-1).ne(rs.new_tensor(range(seq_len))).unsqueeze(-1)
        # [seq_len, seq_len, batch_size], (h->m)
        s_arc = s_arc.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        s_sib = s_sib.permute(2, 1, 3, 0) * mask2o

        # posterior distributions
        # [seq_len, seq_len, batch_size], (h->m)
        q = s_arc

        for _ in range(self.max_iter):
            q = q.softmax(0)
            # q(ij) = s(ij) + sum(q(ik)s^sib(ij,ik)), k != i,j
            q = s_arc + (q.unsqueeze(1) * s_sib).sum(2)

        return q.permute(2, 1, 0)


class LBPDependency(nn.Module):
    r"""
    Loopy Belief Propagation for approximately calculating marginals
    of dependency trees :cite:`smith-eisner-2008-dependency`.
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of three tensors `s_arc` and `s_sib`.
                `s_arc` (``[batch_size, seq_len, seq_len]``) holds scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
            target (~torch.LongTensor): ``[batch_size, seq_len]``.
                A Tensor of gold-standard dependent-head pairs. Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        logits = self.lbp(*scores, mask)
        marginals = logits.softmax(-1)

        if target is None:
            return marginals
        loss = F.cross_entropy(logits[mask], target[mask])

        return loss, marginals

    def lbp(self, s_arc, s_sib, mask):
        batch_size, seq_len = mask.shape
        ls, rs = torch.stack(torch.where(mask.new_ones(seq_len, seq_len))).view(-1, seq_len, seq_len).sort(0)[0]
        mask = mask.index_fill(1, ls.new_tensor(0), 1)
        # [seq_len, seq_len, batch_size], (h->m)
        mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask2o = mask2o & ls.unsqueeze(-1).ne(ls.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & rs.unsqueeze(-1).ne(rs.new_tensor(range(seq_len))).unsqueeze(-1)
        # [seq_len, seq_len, batch_size], (h->m)
        s_arc = s_arc.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        s_sib = s_sib.permute(2, 1, 3, 0).masked_fill_(~mask2o, float('-inf'))

        # log beliefs
        # [seq_len, seq_len, batch_size], (h->m)
        q = s_arc
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        m_sib = s_sib.new_zeros(seq_len, seq_len, seq_len, batch_size)

        for _ in range(self.max_iter):
            q = q.log_softmax(0)
            # m(ik->ij) = logsumexp(q(ik) - m(ij->ik) + s(ij->ik))
            m = q.unsqueeze(2) - m_sib
            # TODO: better solution for OOM
            m_sib = torch.logaddexp(m.logsumexp(0), m + s_sib).transpose(1, 2).log_softmax(0)
            # q(ij) = s(ij) + sum(m(ik->ij)), k != i,j
            q = s_arc + (m_sib * mask2o).sum(2)

        return q.permute(2, 1, 0)


class MFVIConstituency(nn.Module):
    r"""
    Mean Field Variational Inference for approximately calculating marginals of constituent trees.
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of two tensors `s_span` and `s_pair`.
                `s_span` (``[batch_size, seq_len, seq_len]``) holds scores of all possible spans.
                `s_pair` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of second-order triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid aggregation on padding tokens.
            target (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                A Tensor of gold-standard dependent-head pairs. Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        logits = self.mfvi(*scores, mask)
        marginals = logits.sigmoid()

        if target is None:
            return marginals
        loss = F.binary_cross_entropy_with_logits(logits[mask], target[mask].float())

        return loss, marginals

    def mfvi(self, s_span, s_pair, mask):

        # print(s_span.shape, s_pair.shape, mask.shape)
        batch_size, seq_len, _ = mask.shape
        # left index and right index
        ls, rs = torch.stack(torch.where(torch.ones_like(mask[0]))).view(-1, seq_len, seq_len).sort(0)[0]
        # print(ls.shape, rs.shape)
        # print(ls, rs)
        # [seq_len, seq_len, batch_size], (l->r)
        mask = mask.permute(1, 2, 0)
        # print('mask:\n', mask[..., 0])
        # [seq_len, seq_len, seq_len, batch_size], (l->r->b)
        mask2o = mask.unsqueeze(2).repeat(1, 1, seq_len, 1)
        mask2o = mask2o & ls.unsqueeze(-1).ne(ls.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & rs.unsqueeze(-1).ne(rs.new_tensor(range(seq_len))).unsqueeze(-1)
        # print('mask2o:\n', mask2o[..., 0])
        # [seq_len, seq_len, batch_size], (l->r)
        span_mask = mask.transpose(0, 1) | mask
        mask2o &= span_mask
        s_span = s_span.permute(1, 2, 0)
        # print(s_span[..., 0])
        # print(s_pair.permute(2, 1, 3, 0)[..., 0])
        # [seq_len, seq_len, seq_len, batch_size], (l->r->b)
        s_pair = s_pair.permute(1, 2, 3, 0) * mask2o
        # print(s_pair[..., 0])

        # posterior distributions
        # [seq_len, seq_len, batch_size], (l->r)
        q = s_span

        for _ in range(self.max_iter):
            q = q.sigmoid()
            # print(q[..., 0])
            # q(ij) = s(ij) + sum(q(jk)*s^pair(ij,jk), k != i,j
            q = s_span + (q.unsqueeze(1) * s_pair).sum(2)

        return q.permute(2, 0, 1)


class MFVIWordBasedConstituency(nn.Module):
    r"""
    Mean Field Variational Inference for approximately calculating marginals of constituent trees.
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of two tensors `s_span` and `s_pair`.
                `s_span` (``[batch_size, seq_len, seq_len]``) holds scores of all possible spans.
                `s_pair` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of second-order triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid aggregation on padding tokens.
        """

        return self.mfvi(scores['s_const'], scores['s_split'], mask)

    def mfvi(self, s_span, s_pair, mask):
        pad_mask = (mask[:, 0].unsqueeze(1) & mask[:, 0].unsqueeze(2)).permute(1, 2, 0)
        batch_size, seq_len, _ = mask.shape
        # left index and right index
        ls, rs = torch.stack(torch.where(torch.ones_like(mask[0]))).view(-1, seq_len, seq_len).sort(0)[0]
        # [seq_len, seq_len, batch_size], (l->r)
        mask = mask.permute(1, 2, 0)
        # [seq_len, seq_len, seq_len, batch_size], (l->r->b)
        mask2o = mask.unsqueeze(2).repeat(1, 1, seq_len, 1)
        # mask2o = mask2o & ls.unsqueeze(-1).ne(ls.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & rs.unsqueeze(-1).ne(rs.new_tensor(range(seq_len))).unsqueeze(-1)
        # [seq_len, seq_len, batch_size], (l->r)
        mask2o &= pad_mask
        s_span = s_span.permute(1, 2, 0)
        # [seq_len, seq_len, seq_len, batch_size], (l->r->b)
        s_pair = s_pair.permute(1, 2, 3, 0) * mask2o

        # posterior distributions
        # [seq_len, seq_len, batch_size], (l->r)
        q = s_span

        for _ in range(self.max_iter):
            q = q.sigmoid()
            # q(ij) = s(ij) + sum(q(jk)*s^pair(ij,jk), k != i,j
            q = s_span + (q.unsqueeze(1) * s_pair).sum(2)

        return q.permute(2, 0, 1)


class LBPConstituency(nn.Module):
    r"""
    Loopy Belief Propagation for approximately calculating marginals of constituent trees.
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of four tensors `s_edge`, `s_sib`, `s_cop` and `s_grd`.
                `s_span` (``[batch_size, seq_len, seq_len]``) holds scores of all possible spans.
                `s_pair` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of second-order triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid aggregation on padding tokens.
            target (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                A Tensor of gold-standard dependent-head pairs. Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        logits = self.lbp(*scores, mask)
        marginals = logits.softmax(-1)[..., 1]

        if target is None:
            return marginals
        loss = F.cross_entropy(logits[mask], target[mask].long())

        return loss, marginals

    def lbp(self, s_span, s_pair, mask):
        batch_size, seq_len, _ = mask.shape
        ls, rs = torch.stack(torch.where(torch.ones_like(mask[0]))).view(-1, seq_len, seq_len).sort(0)[0]
        # [seq_len, seq_len, batch_size], (l->r)
        mask = mask.permute(1, 2, 0)
        # [seq_len, seq_len, seq_len, batch_size], (l->r->b)
        mask2o = mask.unsqueeze(2).repeat(1, 1, seq_len, 1)
        mask2o = mask2o & ls.unsqueeze(-1).ne(ls.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & rs.unsqueeze(-1).ne(rs.new_tensor(range(seq_len))).unsqueeze(-1)
        # [2, seq_len, seq_len, batch_size], (l->r)
        s_span = torch.stack((torch.zeros_like(s_span), s_span)).permute(0, 3, 2, 1)
        # [seq_len, seq_len, seq_len, batch_size], (l->r->p)
        s_pair = s_pair.permute(2, 1, 3, 0)

        # log beliefs
        # [2, seq_len, seq_len, batch_size], (h->m)
        q = s_span
        # [2, seq_len, seq_len, seq_len, batch_size], (h->m->s)
        m_pair = s_pair.new_zeros(2, seq_len, seq_len, seq_len, batch_size)

        for _ in range(self.max_iter):
            q = q.log_softmax(0)
            # m(ik->ij) = logsumexp(q(ik) - m(ij->ik) + s(ij->ik))
            m = q.unsqueeze(3) - m_pair
            m_pair = torch.stack((m.logsumexp(0), torch.stack((m[0], m[1] + s_pair)).logsumexp(0))).log_softmax(0)
            # q(ij) = s(ij) + sum(m(ik->ij)), k != i,j
            q = s_span + (m_pair.transpose(2, 3) * mask2o).sum(3)

        return q.permute(3, 2, 1, 0)


class MFVISemanticDependency(nn.Module):
    r"""
    Mean Field Variational Inference for approximately calculating marginals
    of semantic dependency trees :cite:`wang-etal-2019-second`.
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of four tensors `s_edge`, `s_sib`, `s_cop` and `s_grd`.
                `s_edge` (``[batch_size, seq_len, seq_len]``) holds scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
                `s_cop` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-coparent triples.
                `s_grd` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-grandparent triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid aggregation on padding tokens.
            target (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                A Tensor of gold-standard dependent-head pairs. Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        logits = self.mfvi(*scores, mask)
        marginals = logits.sigmoid()

        if target is None:
            return marginals
        loss = F.binary_cross_entropy_with_logits(logits[mask], target[mask].float())

        return loss, marginals

    def mfvi(self, s_edge, s_sib, s_cop, s_grd, mask):
        batch_size, seq_len, _ = mask.shape
        hs, ms = torch.stack(torch.where(torch.ones_like(mask[0]))).view(-1, seq_len, seq_len)
        # [seq_len, seq_len, batch_size], (h->m)
        mask = mask.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask2o = mask2o & hs.unsqueeze(-1).ne(hs.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & ms.unsqueeze(-1).ne(ms.new_tensor(range(seq_len))).unsqueeze(-1)
        # [seq_len, seq_len, batch_size], (h->m)
        s_edge = s_edge.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        s_sib = s_sib.permute(2, 1, 3, 0) * mask2o
        # [seq_len, seq_len, seq_len, batch_size], (h->m->c)
        s_cop = s_cop.permute(2, 1, 3, 0) * mask2o
        # [seq_len, seq_len, seq_len, batch_size], (h->m->g)
        s_grd = s_grd.permute(2, 1, 3, 0) * mask2o

        # posterior distributions
        # [seq_len, seq_len, batch_size], (h->m)
        q = s_edge

        for _ in range(self.max_iter):
            q = q.sigmoid()
            # q(ij) = s(ij) + sum(q(ik)s^sib(ij,ik) + q(kj)s^cop(ij,kj) + q(jk)s^grd(ij,jk)), k != i,j
            q = s_edge + (q.unsqueeze(1) * s_sib + q.transpose(0, 1).unsqueeze(0) * s_cop + q.unsqueeze(0) * s_grd).sum(2)

        return q.permute(2, 1, 0)


class LBPSemanticDependency(nn.Module):
    r"""
    Loopy Belief Propagation for approximately calculating marginals
    of semantic dependency trees :cite:`wang-etal-2019-second`.
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of four tensors `s_edge`, `s_sib`, `s_cop` and `s_grd`.
                `s_edge` (``[batch_size, seq_len, seq_len]``) holds scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
                `s_cop` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-coparent triples.
                `s_grd` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-grandparent triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid aggregation on padding tokens.
            target (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                A Tensor of gold-standard dependent-head pairs. Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        logits = self.lbp(*scores, mask)
        marginals = logits.softmax(-1)[..., 1]

        if target is None:
            return marginals
        loss = F.cross_entropy(logits[mask], target[mask])

        return loss, marginals

    def lbp(self, s_edge, s_sib, s_cop, s_grd, mask):
        batch_size, seq_len, _ = mask.shape
        hs, ms = torch.stack(torch.where(torch.ones_like(mask[0]))).view(-1, seq_len, seq_len)
        # [seq_len, seq_len, batch_size], (h->m)
        mask = mask.permute(2, 1, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask2o = mask2o & hs.unsqueeze(-1).ne(hs.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & ms.unsqueeze(-1).ne(ms.new_tensor(range(seq_len))).unsqueeze(-1)
        # [2, seq_len, seq_len, batch_size], (h->m)
        s_edge = torch.stack((torch.zeros_like(s_edge), s_edge)).permute(0, 3, 2, 1)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        s_sib = s_sib.permute(2, 1, 3, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->c)
        s_cop = s_cop.permute(2, 1, 3, 0)
        # [seq_len, seq_len, seq_len, batch_size], (h->m->g)
        s_grd = s_grd.permute(2, 1, 3, 0)

        # log beliefs
        # [2, seq_len, seq_len, batch_size], (h->m)
        q = s_edge
        # log messages of siblings
        # [2, seq_len, seq_len, seq_len, batch_size], (h->m->s)
        m_sib = s_sib.new_zeros(2, seq_len, seq_len, seq_len, batch_size)
        # log messages of co-parents
        # [2, seq_len, seq_len, seq_len, batch_size], (h->m->c)
        m_cop = s_cop.new_zeros(2, seq_len, seq_len, seq_len, batch_size)
        # log messages of grand-parents
        # [2, seq_len, seq_len, seq_len, batch_size], (h->m->g)
        m_grd = s_grd.new_zeros(2, seq_len, seq_len, seq_len, batch_size)

        for _ in range(self.max_iter):
            q = q.log_softmax(0)
            # m(ik->ij) = logsumexp(q(ik) - m(ij->ik) + s(ij->ik))
            m = q.unsqueeze(3) - m_sib
            m_sib = torch.stack((m.logsumexp(0), torch.stack((m[0], m[1] + s_sib)).logsumexp(0))).log_softmax(0)
            m = q.unsqueeze(3) - m_cop
            m_cop = torch.stack((m.logsumexp(0), torch.stack((m[0], m[1] + s_cop)).logsumexp(0))).log_softmax(0)
            m = q.unsqueeze(3) - m_grd
            m_grd = torch.stack((m.logsumexp(0), torch.stack((m[0], m[1] + s_grd)).logsumexp(0))).log_softmax(0)
            # q(ij) = s(ij) + sum(m(ik->ij)), k != i,j
            q = s_edge + ((m_sib + m_cop + m_grd).transpose(2, 3) * mask2o).sum(3)

        return q.permute(3, 2, 1, 0)


class MFVISemanticRoleLabeling(nn.Module):
    r"""
    Mean Field Variational Inference for approximately calculating marginals
    of semantic role labeling :cite:`wang-etal-2019-second`.
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, second_order_structures):
        r"""
        Args:
            scores (tuple(~torch.Tensor)):
                s_prd (tensor): [batch_size, seq_len]
                s_arg_begin (tensor): [batch_size, seq_len, seq_len]
                s_arg_end (tensor): [batch_size, seq_len, seq_len]
                s_sib (tensor): [batch_size, seq_len, seq_len, seq_len]
                s_cop_b (tensor): [batch_size, seq_len, seq_len, seq_len]
                s_cop_e (tensor): [batch_size, seq_len, seq_len, seq_len]
                s_grd_b (tensor): [batch_size, seq_len, seq_len]
                s_grd_e (tensor): [batch_size, seq_len, seq_len]
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid aggregation on illegal positions.             
            second_order_structures (List[str]):

        Returns:
            ~torch.Tensor: Scores mixed with second-order structures.
        """
        return self.mfvi(scores, mask, second_order_structures)

    # def mfvi(self, s_arg_begin, s_arg_end, s_sib_be, s_sib_bb, s_sib_ee, s_cop_b, s_cop_e, mask):
    def mfvi(self, scores, mask, second_order_structures):
        """
        Args:
            s_prd (tensor): [batch_size, seq_len]
            s_arg_begin (tensor): [batch_size, seq_len, seq_len]
            s_arg_end (tensor): [batch_size, seq_len, seq_len]
            s_sib (tensor): [batch_size, seq_len, seq_len, seq_len]
            s_cop_b (tensor): [batch_size, seq_len, seq_len, seq_len]
            s_cop_e (tensor): [batch_size, seq_len, seq_len, seq_len]
            s_grd_b (tensor): [batch_size, seq_len, seq_len]
            s_grd_e (tensor): [batch_size, seq_len, seq_len]
            mask (tensor): [batch_size, seq_len, seq_len]
        Returns:
            tuple(tensor):
                q_h (tensor): [batch_size, seq_len]
                q_b (tensor): [batch_size, seq_len, seq_len]
                q_e (tensor): [batch_size, seq_len, seq_len]
        """
        s_arg_begin, s_arg_end = scores['s_arg_begin'], scores['s_arg_end']

        # [batch_size, seq_len, seq_len], (h->b/e)
        edge_mask = mask['arg_begin_mask']
        _, seq_len, _ = edge_mask.shape
        hs, ms = torch.stack(torch.where(torch.ones_like(edge_mask[0]))).view(-1, seq_len, seq_len)
        # [batch_size, seq_len, seq_len, seq_len]
        mask2o = edge_mask.unsqueeze(2) & edge_mask.unsqueeze(3)
        # mask head index
        mask2o = mask2o & hs.unsqueeze(-1).ne(hs.new_tensor(range(seq_len)))
        sib_mask2o = mask2o
        # mask modifier index, i.e. the begin/end index
        mask2o = mask2o & ms.unsqueeze(-1).ne(ms.new_tensor(range(seq_len)))

        if 'sib_be' in second_order_structures:
            s_sib_be, s_sib_eb = scores['s_sib_be'], scores['s_sib_eb']
            s_sib_be = s_sib_be * sib_mask2o
            s_sib_eb = s_sib_eb * sib_mask2o.transpose(-2, -1)
        # [batch_size, seq_len, seq_len, seq_len]
        if 'sib_be_strict' in second_order_structures:
            ls, rs = torch.stack(torch.where(torch.ones_like(edge_mask[0]))).view(-1, seq_len, seq_len).sort(0)[0]
            lhs, rhs = ls.unsqueeze(0).ge(hs.unsqueeze(-1)), rs.unsqueeze(0).le(hs.unsqueeze(-1))
            # for (h, i), only h <= j <= i is legal
            lrhs = lhs | rhs
            sib_mask2o = sib_mask2o.triu() & lrhs
            s_sib_be, s_sib_eb = scores['s_sib_be'], scores['s_sib_eb']
            # include j for mask2o[i, j]
            # (h->b->e)
            s_sib_be = s_sib_be * sib_mask2o
            # (h->e->b)
            s_sib_eb = s_sib_eb * sib_mask2o.transpose(-2, -1)
            # s_sib_eb = s_sib_eb * sib_mask2o
        if 'sib_bb' in second_order_structures:
            s_sib_bb, s_sib_ee = scores['s_sib_bb'], scores['s_sib_ee']
            # (h->b->b)
            s_sib_bb = s_sib_bb * mask2o
            # (h->e->e)
            s_sib_ee = s_sib_ee * mask2o
        if 'cop' in second_order_structures:
            s_cop_b, s_cop_e = scores['s_cop_b'], scores['s_cop_e']
            # (h->b->h)
            s_cop_b = s_cop_b * mask2o
            # (h->e->h)
            s_cop_e = s_cop_e * mask2o

        q_b, q_e = s_arg_begin, s_arg_end

        for _ in range(self.max_iter):
            q_b, q_e = q_b.sigmoid(), q_e.sigmoid()
            # q_b(ij) = s_b(ij) + sum(q_e(ik)*s_sib(ij,ik)) {k >=j, k != i} 
            #                   + sum(q_b(kj)*s_cop_b(ik, kj)) {k != i, k != j} 
            tmp_q_b, tmp_q_e = s_arg_begin, s_arg_end
            if 'sib_be' in second_order_structures or 'sib_be_strict' in second_order_structures:
                tmp_q_b += (q_e.unsqueeze(2) * s_sib_be).sum(3)
                tmp_q_e += (q_b.unsqueeze(2) * s_sib_eb).sum(3)
            if 'sib_bb' in second_order_structures:
                tmp_q_b += (q_b.unsqueeze(2) * s_sib_bb).sum(3)
                tmp_q_e += (q_e.unsqueeze(2) * s_sib_ee).sum(3)
            if 'cop' in second_order_structures:
                tmp_q_b += (q_b.transpose(-2, -1).unsqueeze(1) * s_cop_b).sum(3)
                tmp_q_e += (q_e.transpose(-2, -1).unsqueeze(1) * s_cop_e).sum(3)

            q_b, q_e = tmp_q_b, tmp_q_e

        scores.update({'q_arg_begin': q_b, 'q_arg_end': q_e})

        return scores


class MFVIWordBasedJointSYNAndSpanSRL(nn.Module):
    r"""
    Mean Field Variational Inference for approximately calculating marginals
    of semantic role labeling and constituency parsing :cite:`wang-etal-2019-second`.
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, second_order_structures):
        r"""
        Args:
            scores (tuple(~torch.Tensor)):
                s_prd (tensor): [batch_size, seq_len]
                s_arg_begin (tensor): [batch_size, seq_len, seq_len]
                s_arg_end (tensor): [batch_size, seq_len, seq_len]
                s_sib (tensor): [batch_size, seq_len, seq_len, seq_len]
                s_cop_b (tensor): [batch_size, seq_len, seq_len, seq_len]
                s_cop_e (tensor): [batch_size, seq_len, seq_len, seq_len]
                s_grd_b (tensor): [batch_size, seq_len, seq_len]
                s_grd_e (tensor): [batch_size, seq_len, seq_len]
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid aggregation on illegal positions.             
            second_order_structures (List[str]):

        Returns:
            ~torch.Tensor: Scores mixed with second-order structures.
        """
        s_arg_begin, s_arg_end, s_const = self.mfvi(scores, mask, second_order_structures)
        scores.update({'s_arg_begin': s_arg_begin, 's_arg_end': s_arg_end, 's_const': s_const})

        return scores

    # def mfvi(self, s_arg_begin, s_arg_end, s_sib_be, s_sib_bb, s_sib_ee, s_cop_b, s_cop_e, mask):
    def mfvi(self, scores, mask, second_order_structures):
        """
        Args:
            s_prd (tensor): [batch_size, seq_len]
            s_arg_begin (tensor): [batch_size, seq_len, seq_len]
            s_arg_end (tensor): [batch_size, seq_len, seq_len]
            s_sib (tensor): [batch_size, seq_len, seq_len, seq_len]
            s_cop_b (tensor): [batch_size, seq_len, seq_len, seq_len]
            s_cop_e (tensor): [batch_size, seq_len, seq_len, seq_len]
            s_grd_b (tensor): [batch_size, seq_len, seq_len]
            s_grd_e (tensor): [batch_size, seq_len, seq_len]
            mask (tensor): [batch_size, seq_len, seq_len]
        Returns:
            tuple(tensor):
                q_h (tensor): [batch_size, seq_len]
                q_b (tensor): [batch_size, seq_len, seq_len]
                q_e (tensor): [batch_size, seq_len, seq_len]
        """
        s_span, s_arg_begin, s_arg_end = scores['s_const'], scores['s_arg_begin'], scores['s_arg_end']

        # [batch_size, seq_len, seq_len], (h->b/e)
        edge_mask, chart_mask = mask['arg_begin_mask'], mask['chart_mask']
        pad_mask = chart_mask[:, 0].unsqueeze(1) & chart_mask[:, 0].unsqueeze(2)
        chart_mask = torch.cat((chart_mask[:, -1:], chart_mask[:, :-1]), dim=1)
        _, seq_len, _ = edge_mask.shape
        # [seq_len, seq_len]
        hs, ms = torch.stack(torch.where(torch.ones_like(edge_mask[0]))).view(-1, seq_len, seq_len)
        ls, rs = torch.stack(torch.where(torch.ones_like(edge_mask[0]))).view(-1, seq_len, seq_len).sort(0)[0]

        # [batch_size, seq_len, seq_len, seq_len]
        mask2o = edge_mask.unsqueeze(2) & edge_mask.unsqueeze(3)
        # mask head index
        mask2o = mask2o & hs.unsqueeze(-1).ne(hs.new_tensor(range(seq_len)))
        mask2o_with_modifier = mask2o
        # mask modifier index, i.e. the begin/end index
        mask2o = mask2o & ms.unsqueeze(-1).ne(ms.new_tensor(range(seq_len)))

        lhs, rhs = ls.unsqueeze(0).ge(hs.unsqueeze(-1)), rs.unsqueeze(0).le(hs.unsqueeze(-1))
        # for (h, i) and h is the index of head word
        #   if i < h, then j > h is not allowed
        #   if i > h, then j < h is not allowed
        lrhs = lhs | rhs
        # for (h, b):
        #   if h < b, only j >= b is legal
        #   if h > b, only j >= b and j < h is legal
        # for (h, e):
        #   if h < e, only j <= e and j > h is legal
        #   if h > e, only j <= e is legal
        noncross_mask2o = mask2o_with_modifier.triu() & lrhs

        # [batch_size, seq_len, seq_len, seq_len]
        span_mask2o = chart_mask.unsqueeze(3).repeat(1, 1, 1, seq_len)
        # span_mask2o &= ls.unsqueeze(-1).ne(ls.new_tensor(range(seq_len)))
        span_mask2o &= rs.unsqueeze(-1).ne(rs.new_tensor(range(seq_len)))
        # for (i, j, k), (i, j) is a span, k != j
        span_mask2o &= pad_mask.unsqueeze(1)

        # index k >= l
        left_boundary = hs.unsqueeze(-1).le(hs.new_tensor(range(seq_len)))
        # index k <= r
        right_boundary = ms.unsqueeze(-1).ge(ms.new_tensor(range(seq_len)))
        # for (i, j, k), (i, j) is a span, k < i and k > i
        outside_mask2o = ~(left_boundary & right_boundary) & span_mask2o

        # [batch_size, seq_len, seq_len, seq_len]
        if 'sib_be_strict' in second_order_structures:
            s_sib_be, s_sib_eb = scores['s_sib_be'], scores['s_sib_eb']
            # include j for mask2o[i, j]
            # (h->b->e)
            s_sib_be = s_sib_be * noncross_mask2o
            # (h->e->b)
            s_sib_eb = s_sib_eb * noncross_mask2o.transpose(-2, -1)
        if 'sib_be' in second_order_structures:
            s_sib_be, s_sib_eb = scores['s_sib_be'], scores['s_sib_eb']
            # include j for mask2o[i, j]
            # (h->b->e)
            s_sib_be = s_sib_be * mask2o_with_modifier
            # (h->e->b)
            s_sib_eb = s_sib_eb * mask2o_with_modifier.transpose(-2, -1)
        if 'sib_bb' in second_order_structures:
            s_sib_bb, s_sib_ee = scores['s_sib_bb'], scores['s_sib_ee']
            # (h->b->b)
            s_sib_bb = s_sib_bb * mask2o
            # (h->e->e)
            s_sib_ee = s_sib_ee * mask2o
        if 'cop' in second_order_structures:
            s_cop_b, s_cop_e = scores['s_cop_b'], scores['s_cop_e']
            # (h->b->h)
            s_cop_b = s_cop_b * mask2o
            # (h->e->h)
            s_cop_e = s_cop_e * mask2o
        if 'grd' in second_order_structures:
            # (h->b->e) or (h->e->b)
            s_grd_b, s_grd_e = scores['s_grd_b'], scores['s_grd_e']
            # (h->b->e)
            s_edge_grd_b = s_grd_b * noncross_mask2o
            # (h->e->b)
            s_edge_grd_e = s_grd_e * noncross_mask2o.transpose(-2, -1)
            # (b->e->h)
            s_span_grd_b = s_grd_b.permute(0, 2, 3, 1) * outside_mask2o
            # (b->e->h)
            s_span_grd_e = s_grd_e.permute(0, 3, 2, 1) * outside_mask2o
        if 'split' in second_order_structures:
            s_split = scores['s_split']
            s_split = s_split * span_mask2o

        # [batch_size, seq_len, seq_len]
        q_b, q_e, q_span = s_arg_begin, s_arg_end, s_span

        for _ in range(self.max_iter):
            q_b, q_e, q_span = q_b.sigmoid(), q_e.sigmoid(), q_span.sigmoid()
            # q_b(ij) = s_b(ij) + sum(q_e(ik)*s_sib(ij,ik)) {k >=j, k != i} 
            #                   + sum(q_b(kj)*s_cop_b(ik, kj)) {k != i, k != j} 
            tmp_q_b, tmp_q_e, tmp_q_span = s_arg_begin, s_arg_end, s_span
            if 'sib_be' in second_order_structures or 'sib_be_strict' in second_order_structures:
                tmp_q_b += (q_e.unsqueeze(2) * s_sib_be).sum(3)
                tmp_q_e += (q_b.unsqueeze(2) * s_sib_eb).sum(3)
            if 'sib_bb' in second_order_structures:
                tmp_q_b += (q_b.unsqueeze(2) * s_sib_bb).sum(3)
                tmp_q_e += (q_e.unsqueeze(2) * s_sib_ee).sum(3)
            if 'cop' in second_order_structures:
                tmp_q_b += (q_b.transpose(-2, -1).unsqueeze(1) * s_cop_b).sum(3)
                tmp_q_e += (q_e.transpose(-2, -1).unsqueeze(1) * s_cop_e).sum(3)
            if 'grd' in second_order_structures:
                tmp_q_b += (q_span.unsqueeze(1) * s_edge_grd_b).sum(3)
                tmp_q_e += (q_span.transpose(-2, -1).unsqueeze(1) * s_edge_grd_e).sum(3)
                tmp_q_span += (q_b.transpose(-2, -1).unsqueeze(2) * s_span_grd_b + q_e.transpose(-2, -1).unsqueeze(1) * s_span_grd_e).sum(3)
            if 'split' in second_order_structures:
                tmp_q_span += (q_span.unsqueeze(2) * s_split).sum(3)

            q_b, q_e, q_span = tmp_q_b, tmp_q_e, tmp_q_span
            # logger.info(q_span[0])

        return q_b, q_e, q_span


class MFVIJointConstAndOneStagePruningSRL(nn.Module):
    r"""
    Mean Field Variational Inference for approximately calculating marginals
    of semantic role labeling and constituency parsing :cite:`wang-etal-2019-second`.
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask):
        r"""
        Args:
            scores (tuple(~torch.Tensor)):

            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid aggregation on illegal positions.             
            second_order_structures (List[str]):

        Returns:
            ~torch.Tensor: Scores mixed with second-order structures.
        """
        return self.mfvi(scores, mask)

    def mfvi(self, scores, mask):
        """
        Args:
            s_prd (tensor): [batch_size, seq_len]
        Returns:
            tuple(tensor):
                q_h (tensor): [batch_size, seq_len]
                q_b (tensor): [batch_size, seq_len, seq_len]
                q_e (tensor): [batch_size, seq_len, seq_len]
        """
        s_arg, s_const, s_co_span = scores['s_arg'], scores['s_const'], scores['s_co_span']
        chart_mask = mask['chart_mask']
        # mask spans with length 1
        chart_mask = chart_mask.triu(2) | chart_mask.tril(0)

        # [batch_size, seq_len, seq_len]
        s_co_span = s_co_span * chart_mask

        # [batch_size, seq_len, seq_len]
        q_arg, q_const = s_arg, s_const

        for _ in range(self.max_iter):
            q_arg, q_const = q_arg.sigmoid(), q_const.sigmoid()

            tmp_q_arg = s_arg + (q_const * s_co_span)
            tmp_q_const = s_const + (q_arg * s_co_span)

            q_arg, q_const = tmp_q_arg, tmp_q_const

        scores.update({'s_arg': q_arg, 's_const': q_const})

        return scores


class MFVIJointConstAndOneStageLinearCRFSRL(nn.Module):
    r"""
    Mean Field Variational Inference for approximately calculating marginals
    of semantic role labeling and constituency parsing :cite:`wang-etal-2019-second`.
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, second_order_structures):
        scores = self.mfvi(scores, mask, second_order_structures)

        return scores

    def mfvi(self, scores, mask, second_order_structures):
        """
        Args:
            mask (tensor): [batch_size, seq_len, seq_len]
        Returns:
            tuple(tensor):
        """
        s_edge, s_span = scores['s_edge'], scores['s_const']
        s_span = s_span.triu() + s_span.triu(1).transpose(-2, -1)
        # [batch_size, seq_len, seq_len]
        edge_mask, chart_mask = mask['edge_mask'], mask['chart_mask']
        _, seq_len, _ = edge_mask.shape

        # [seq_len, seq_len], for edge's head index and modifier index
        hs, ms = torch.stack(torch.where(torch.ones_like(edge_mask[0]))).view(-1, seq_len, seq_len)
        # [seq_len, seq_len], for span's left index and right index
        ls, rs = torch.stack(torch.where(torch.ones_like(edge_mask[0]))).view(-1, seq_len, seq_len).sort(0)[0]

        # [batch_size, seq_len, seq_len, seq_len]
        edge_mask2o = edge_mask.unsqueeze(2) & edge_mask.unsqueeze(3)
        # force the length of spans is at least 2, i.e. for (i, j, k), j != k
        edge_mask2o = edge_mask2o & ms.unsqueeze(-1).ne(ms.new_tensor(range(seq_len)))

        lhs, rhs = ls.unsqueeze(0).ge(hs.unsqueeze(-1)), rs.unsqueeze(0).le(hs.unsqueeze(-1))
        # for (i, j, k), (i, j) is an edge and i is the head word, (j, k)/(k, j) is a span
        #   if j < i, then k > i is not allowed
        #   if j > i, then k < i is not allowed
        lrhs = lhs | rhs
        noncross_mask2o = edge_mask2o & lrhs

        # mask head index, i.e. for (i, j, k), i != k
        edge_mask2o = edge_mask2o & hs.unsqueeze(-1).ne(hs.new_tensor(range(seq_len)))

        # # mask pad positions (<bos> is seen as a pad position)
        pad_mask = chart_mask[:, 0].unsqueeze(1) & chart_mask[:, 0].unsqueeze(2)
        # convert the fence-based chart to word-based chart
        chart_mask = torch.cat((chart_mask[:, -1:], chart_mask[:, :-1]), dim=1)
        # [batch_size, seq_len, seq_len, seq_len]
        span_mask2o = chart_mask.unsqueeze(3).repeat(1, 1, 1, seq_len) & pad_mask.unsqueeze(1)
        # for (i, j, k), (i, j) is a span, k != j
        span_mask2o &= rs.unsqueeze(-1).ne(rs.new_tensor(range(seq_len)))
        # index k >= l
        left_boundary = hs.unsqueeze(-1).le(hs.new_tensor(range(seq_len)))
        # index k <= r
        right_boundary = ms.unsqueeze(-1).ge(ms.new_tensor(range(seq_len)))
        # for (i, j, k), (i, j) is a span, k < i and k > i
        outside_mask2o = ~(left_boundary & right_boundary) & span_mask2o
        # only consider spans whose length is at least 2
        outside_mask2o = (pad_mask.triu(1) | pad_mask.tril(-1)).unsqueeze(-1) & outside_mask2o

        if 'sib' in second_order_structures:
            s_sib = scores['s_sib']
            # (h->d->sib)
            s_sib = s_sib * edge_mask2o
        if 'cop' in second_order_structures:
            s_cop = scores['s_cop']
            # (h->d->cop)
            s_cop = s_cop * edge_mask2o
        if 'grd' in second_order_structures:
            s_grd = scores['s_grd']
            # (h->d->grd)
            s_grd = s_grd * edge_mask2o
        if 'end_point' in second_order_structures:
            # (h->m->p)
            s_end_point = scores['s_end_point']
            # (h->m->p)
            s_end_point = s_end_point * noncross_mask2o
        if 'split' in second_order_structures:
            s_split = scores['s_split']
            s_split = s_split * span_mask2o
        if 'head_span' in second_order_structures:
            # FIXME: not use s_grd, use a separated score
            s_head_span_b, s_head_span_e = scores['s_head_span_b'], scores['s_head_span_e']
            # (b->e->h)
            s_head_span_b = s_head_span_b * outside_mask2o
            # (e->b->h) -> (b->e->h)
            s_head_span_e = s_head_span_e.permute(0, 2, 1, 3) * outside_mask2o

        # [batch_size, seq_len, seq_len]
        q_edge, q_span = s_edge, s_span

        for _ in range(3):
            q_edge, q_span = q_edge.sigmoid(), q_span.sigmoid()
            tmp_q_edge, tmp_q_span = s_edge, s_span
            if 'sib' in second_order_structures:
                tmp_q_edge += (q_edge.unsqueeze(2) * s_sib).sum(dim=-1)
            if 'cop' in second_order_structures:
                tmp_q_edge += (q_edge.transpose(-2, -1).unsqueeze(1) * s_cop).sum(dim=-1)
            if 'grd' in second_order_structures:
                tmp_q_edge += (q_span.unsqueeze(1) * s_grd).sum(dim=-1)
            if 'end_point' in second_order_structures:
                tmp_q_edge += (q_span.unsqueeze(1) * s_end_point).sum(dim=-1)
            if 'split' in second_order_structures:
                tmp_q_span += (q_span.unsqueeze(2) * s_split).sum(dim=-2)
            if 'head_span' in second_order_structures:
                tmp_q_span += (q_edge.transpose(-2, -1).unsqueeze(2) * s_head_span_b).sum(dim=-1)
                tmp_q_span += (q_edge.transpose(-2, -1).unsqueeze(1) * s_head_span_e).sum(dim=-1)

            q_edge, q_span = tmp_q_edge, tmp_q_span

        scores.update({'q_edge': q_edge, 'q_const': q_span})
        return scores
