# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
from supar.utils.fn import pad, stripe
from supar.utils.common import MIN


def kmeans(x, k, max_it=32):
    r"""
    KMeans algorithm for clustering the sentences by length.

    Args:
        x (list[int]):
            The list of sentence lengths.
        k (int):
            The number of clusters.
            This is an approximate value. The final number of clusters can be less or equal to `k`.
        max_it (int):
            Maximum number of iterations.
            If centroids does not converge after several iterations, the algorithm will be early stopped.

    Returns:
        list[float], list[list[int]]:
            The first list contains average lengths of sentences in each cluster.
            The second is the list of clusters holding indices of data points.

    Examples:
        >>> x = torch.randint(10,20,(10,)).tolist()
        >>> x
        [15, 10, 17, 11, 18, 13, 17, 19, 18, 14]
        >>> centroids, clusters = kmeans(x, 3)
        >>> centroids
        [10.5, 14.0, 17.799999237060547]
        >>> clusters
        [[1, 3], [0, 5, 9], [2, 4, 6, 7, 8]]
    """

    # the number of clusters must not be greater than the number of datapoints
    x, k = torch.tensor(x, dtype=torch.float), min(len(x), k)
    # collect unique datapoints
    d = x.unique()
    # initialize k centroids randomly
    c = d[torch.randperm(len(d))[:k]]
    # assign each datapoint to the cluster with the closest centroid
    dists, y = torch.abs_(x.unsqueeze(-1) - c).min(-1)

    for _ in range(max_it):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster and move that the empty one
        mask = torch.arange(k).unsqueeze(-1).eq(y)
        none = torch.where(~mask.any(-1))[0].tolist()
        while len(none) > 0:
            for i in none:
                # the biggest cluster
                b = torch.where(mask[mask.sum(-1).argmax()])[0]
                # the datapoint farthest from the centroid of cluster b
                f = dists[b].argmax()
                # update the assigned cluster of f
                y[b[f]] = i
                # re-calculate the mask
                mask = torch.arange(k).unsqueeze(-1).eq(y)
            none = torch.where(~mask.any(-1))[0].tolist()
        # update the centroids
        c, old = (x * mask).sum(-1) / mask.sum(-1), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(x.unsqueeze(-1) - c).min(-1)
        # stop iteration early if the centroids converge
        if c.equal(old):
            break
    # assign all datapoints to the new-generated clusters
    # the empty ones are discarded
    assigned = y.unique().tolist()
    # get the centroids of the assigned clusters
    centroids = c[assigned].tolist()
    # map all values of datapoints to buckets
    clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]

    return centroids, clusters


def tarjan(sequence):
    r"""
    Tarjan algorithm for finding Strongly Connected Components (SCCs) of a graph.

    Args:
        sequence (list):
            List of head indices.

    Yields:
        A list of indices making up a SCC. All self-loops are ignored.

    Examples:
        >>> next(tarjan([2, 5, 0, 3, 1]))  # (1 -> 5 -> 2 -> 1) is a cycle
        [2, 5, 1]
    """

    sequence = [-1] + sequence
    # record the search order, i.e., the timestep
    dfn = [-1] * len(sequence)
    # record the the smallest timestep in a SCC
    low = [-1] * len(sequence)
    # push the visited into the stack
    stack, onstack = [], [False] * len(sequence)

    def connect(i, timestep):
        dfn[i] = low[i] = timestep[0]
        timestep[0] += 1
        stack.append(i)
        onstack[i] = True

        for j, head in enumerate(sequence):
            if head != i:
                continue
            if dfn[j] == -1:
                yield from connect(j, timestep)
                low[i] = min(low[i], low[j])
            elif onstack[j]:
                low[i] = min(low[i], dfn[j])

        # a SCC is completed
        if low[i] == dfn[i]:
            cycle = [stack.pop()]
            while cycle[-1] != i:
                onstack[cycle[-1]] = False
                cycle.append(stack.pop())
            onstack[i] = False
            # ignore the self-loop
            if len(cycle) > 1:
                yield cycle

    timestep = [0]
    for i in range(len(sequence)):
        if dfn[i] == -1:
            yield from connect(i, timestep)


def chuliu_edmonds(s):
    r"""
    ChuLiu/Edmonds algorithm for non-projective decoding :cite:`mcdonald-etal-2005-non`.

    Some code is borrowed from `tdozat's implementation`_.
    Descriptions of notations and formulas can be found in :cite:`mcdonald-etal-2005-non`.

    Notes:
        The algorithm does not guarantee to parse a single-root tree.

    Args:
        s (~torch.Tensor): ``[seq_len, seq_len]``.
            Scores of all dependent-head pairs.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[seq_len]`` for the resulting non-projective parse tree.

    .. _tdozat's implementation:
        https://github.com/tdozat/Parser-v3
    """

    s[0, 1:] = float('-inf')
    # prevent self-loops
    s.diagonal()[1:].fill_(float('-inf'))
    # select heads with highest scores
    tree = s.argmax(-1)
    # return the cycle finded by tarjan algorithm lazily
    cycle = next(tarjan(tree.tolist()[1:]), None)
    # if the tree has no cycles, then it is a MST
    if not cycle:
        return tree
    # indices of cycle in the original tree
    cycle = torch.tensor(cycle)
    # indices of noncycle in the original tree
    noncycle = torch.ones(len(s)).index_fill_(0, cycle, 0)
    noncycle = torch.where(noncycle.gt(0))[0]

    def contract(s):
        # heads of cycle in original tree
        cycle_heads = tree[cycle]
        # scores of cycle in original tree
        s_cycle = s[cycle, cycle_heads]

        # calculate the scores of cycle's potential dependents
        # s(c->x) = max(s(x'->x)), x in noncycle and x' in cycle
        s_dep = s[noncycle][:, cycle]
        # find the best cycle head for each noncycle dependent
        deps = s_dep.argmax(1)
        # calculate the scores of cycle's potential heads
        # s(x->c) = max(s(x'->x) - s(a(x')->x') + s(cycle)), x in noncycle and x' in cycle
        #                                                    a(v) is the predecessor of v in cycle
        #                                                    s(cycle) = sum(s(a(v)->v))
        s_head = s[cycle][:, noncycle] - s_cycle.view(-1, 1) + s_cycle.sum()
        # find the best noncycle head for each cycle dependent
        heads = s_head.argmax(0)

        contracted = torch.cat((noncycle, torch.tensor([-1])))
        # calculate the scores of contracted graph
        s = s[contracted][:, contracted]
        # set the contracted graph scores of cycle's potential dependents
        s[:-1, -1] = s_dep[range(len(deps)), deps]
        # set the contracted graph scores of cycle's potential heads
        s[-1, :-1] = s_head[heads, range(len(heads))]

        return s, heads, deps

    # keep track of the endpoints of the edges into and out of cycle for reconstruction later
    s, heads, deps = contract(s)

    # y is the contracted tree
    y = chuliu_edmonds(s)
    # exclude head of cycle from y
    y, cycle_head = y[:-1], y[-1]

    # fix the subtree with no heads coming from the cycle
    # len(y) denotes heads coming from the cycle
    subtree = y < len(y)
    # add the nodes to the new tree
    tree[noncycle[subtree]] = noncycle[y[subtree]]
    # fix the subtree with heads coming from the cycle
    subtree = ~subtree
    # add the nodes to the tree
    tree[noncycle[subtree]] = cycle[deps[subtree]]
    # fix the root of the cycle
    cycle_root = heads[cycle_head]
    # break the cycle and add the root of the cycle to the tree
    tree[cycle[cycle_root]] = noncycle[cycle_head]

    return tree


def mst(scores, mask, multiroot=False):
    r"""
    MST algorithm for decoding non-projective trees.
    This is a wrapper for ChuLiu/Edmonds algorithm.

    The algorithm first runs ChuLiu/Edmonds to parse a tree and then have a check of multi-roots,
    If ``multiroot=True`` and there indeed exist multi-roots, the algorithm seeks to find
    best single-root trees by iterating all possible single-root trees parsed by ChuLiu/Edmonds.
    Otherwise the resulting trees are directly taken as the final outputs.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all dependent-head pairs.
        mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
            The mask to avoid parsing over padding tokens.
            The first column serving as pseudo words for roots should be ``False``.
        multiroot (bool):
            Ensures to parse a single-root tree If ``False``.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[batch_size, seq_len]`` for the resulting non-projective parse trees.

    Examples:
        >>> scores = torch.tensor([[[-11.9436, -13.1464,  -6.4789, -13.8917],
                                    [-60.6957, -60.2866, -48.6457, -63.8125],
                                    [-38.1747, -49.9296, -45.2733, -49.5571],
                                    [-19.7504, -23.9066,  -9.9139, -16.2088]]])
        >>> scores[:, 0, 1:] = float('-inf')
        >>> scores.diagonal(0, 1, 2)[1:].fill_(float('-inf'))
        >>> mask = torch.tensor([[False,  True,  True,  True]])
        >>> mst(scores, mask)
        tensor([[0, 2, 0, 2]])
    """

    batch_size, seq_len, _ = scores.shape
    scores = scores.cpu().unbind()

    preds = []
    for i, length in enumerate(mask.sum(1).tolist()):
        s = scores[i][:length+1, :length+1]
        tree = chuliu_edmonds(s)
        roots = torch.where(tree[1:].eq(0))[0] + 1
        if not multiroot and len(roots) > 1:
            s_root = s[:, 0]
            s_best = float('-inf')
            s = s.index_fill(1, torch.tensor(0), float('-inf'))
            for root in roots:
                s[:, 0] = float('-inf')
                s[root, 0] = s_root[root]
                t = chuliu_edmonds(s)
                s_tree = s[1:].gather(1, t[1:].unsqueeze(-1)).sum()
                if s_tree > s_best:
                    s_best, tree = s_tree, t
        preds.append(tree)

    return pad(preds, total_length=seq_len).to(mask.device)


@torch.enable_grad()
def eisner(scores, mask, multiroot=False):
    r"""
    First-order Eisner algorithm for projective decoding :cite:`mcdonald-etal-2005-online`.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all dependent-head pairs.
        mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
            The mask to avoid parsing over padding tokens.
            The first column serving as pseudo words for roots should be ``False``.
        multiroot (bool):
            Ensures to parse a single-root tree If ``False``.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[batch_size, seq_len]`` for the resulting projective parse trees.

    Examples:
        >>> scores = torch.tensor([[[-13.5026, -18.3700, -13.0033, -16.6809],
                                    [-36.5235, -28.6344, -28.4696, -31.6750],
                                    [ -2.9084,  -7.4825,  -1.4861,  -6.8709],
                                    [-29.4880, -27.6905, -26.1498, -27.0233]]])
        >>> mask = torch.tensor([[False,  True,  True,  True]])
        >>> eisner(scores, mask)
        tensor([[0, 2, 0, 2]])
    """

    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    scores = scores.permute(2, 1, 0).requires_grad_()
    s_i = torch.full_like(scores, float('-inf'))
    s_c = torch.full_like(scores, float('-inf'))
    s_c.diagonal().fill_(0)

    for w in range(1, seq_len):
        n = seq_len - w
        # ilr = C(i->r) + C(j->r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # [batch_size, n, w]
        il = ir = ilr.permute(2, 0, 1)
        # I(j->i) = max(C(i->r) + C(j->r+1) + s(j->i)), i <= r < j
        il_span, _ = il.max(-1)
        s_i.diagonal(-w).copy_(il_span + scores.diagonal(-w))
        # I(i->j) = max(C(i->r) + C(j->r+1) + s(i->j)), i <= r < j
        ir_span, _ = ir.max(-1)
        s_i.diagonal(w).copy_(ir_span + scores.diagonal(w))

        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl_span, _ = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, _ = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        if not multiroot:
            s_c[0, w][lens.ne(w)] = float('-inf')

    logZ = s_c[0].gather(0, lens.unsqueeze(0)).sum()
    marginals, = autograd.grad(logZ, scores)
    preds = lens.new_zeros(batch_size, seq_len).masked_scatter_(mask, marginals.permute(2, 1, 0).nonzero()[:, 2])

    return preds


@torch.enable_grad()
def span_constrained_eisner(s_arc, mask, c_span_mask=None, combine_mask=None, target=None):
    lens = mask.sum(1)
    batch_size, seq_len, _ = s_arc.shape
    s_arc = s_arc.permute(2, 1, 0).requires_grad_()
    s_i = torch.full_like(s_arc, MIN)
    s_c = torch.full_like(s_arc, MIN)
    s_c.diagonal().fill_(0)

    c_span_mask = c_span_mask.permute(2, 1, 0) if c_span_mask is not None else None
    combine_mask = combine_mask.permute(1, 2, 3, 0) if combine_mask is not None else None

    if target is not None:
        target = target.permute(2, 1, 0).gt(0)
        s_arc = s_arc.masked_fill(~target, float('-inf'))

    for w in range(1, seq_len):
        n = seq_len - w
        # ilr = C(i->r) + C(j->r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # [batch_size, n, w]
        il = ir = ilr.permute(2, 0, 1)
        # I(j->i) = max(C(i->r) + C(j->r+1) + s(j->i)), i <= r < j
        il_span, _ = il.max(-1)
        s_i.diagonal(-w).copy_(il_span + s_arc.diagonal(-w))
        # I(i->j) = max(C(i->r) + C(j->r+1) + s(i->j)), i <= r < j
        ir_span, _ = ir.max(-1)
        s_i.diagonal(w).copy_(ir_span + s_arc.diagonal(w))

        w_combine_mask = combine_mask[w].clone().detach() if combine_mask is not None else None
        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        if w_combine_mask is not None:
            cl = cl.masked_fill(stripe(w_combine_mask, n, w, (0, 0), 0), MIN)
        cl_span, _ = cl.permute(2, 0, 1).max(-1)
        if c_span_mask is not None:
            cl_span = cl_span.masked_fill(~c_span_mask.diagonal(-w), MIN)
        s_c.diagonal(-w).copy_(cl_span)
        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        if w_combine_mask is not None:
            cr = cr.masked_fill(stripe(w_combine_mask, n, w, (1, w), 0), MIN)
        cr_span, _ = cr.permute(2, 0, 1).max(-1)
        if c_span_mask is not None:
            cr_span = cr_span.masked_fill(~c_span_mask.diagonal(w), MIN)
        s_c.diagonal(w).copy_(cr_span)
        s_c[0, w][lens.ne(w)] = MIN

    logZ = s_c[0].gather(0, lens.unsqueeze(0)).sum()
    marginals, = autograd.grad(logZ, s_arc)
    # print(f"mask: {mask.sum()}, marginals: {marginals.sum()}")
    preds = lens.new_zeros(batch_size, seq_len).masked_scatter_(mask, marginals.permute(2, 1, 0).nonzero()[:, 2])

    return preds


@torch.enable_grad()
def constraint_eisner(s_arc, mask):
    """
    Args:
        s_arc: [batch_size, 2, seq_len, seq_len]
    """
    intra_link_rules = mask.new_tensor([1, 0, 0, 0])
    inter_link_rules = mask.new_tensor([1, 1, 1, 1])
    intra_combine_rules = mask.new_tensor([1, 0, 0, 0])
    inter_combine_rules = mask.new_tensor([0, 0, 1, 1])

    lens = mask.sum(dim=-1)
    batch_size, _, seq_len, n_state = s_arc.shape
    # [seq_len, seq_len, batch_size, n_state]
    s_arc = s_arc.permute(2, 1, 0, 3).requires_grad_()
    # complete spans
    s_c = torch.full_like(s_arc, MIN)
    s_c.diagonal().fill_(0)
    # incomplete spans
    s_i = torch.full_like(s_arc, MIN)

    for w in range(1, seq_len):
        n = seq_len - w
        # print(f'w: {w}, n: {n}')

        # ilr = C(i->r) + C(j->r+1)
        # [n, w, batch_size, n_state] -> [batch_size, n, w, n_state]
        ir = stripe(s_c, n, w).permute(2, 0, 1, 3)
        il = stripe(s_c, n, w, (w, 1)).permute(2, 0, 1, 3)
        # [batch_size, n, w, n_state*n_state]
        lr_link = (ir.unsqueeze(-1) + il.unsqueeze(-2)).flatten(start_dim=-2, end_dim=-1)
        # [batch_size, n, n_state*n_state], select maximum values from all split points
        lr_link, _ = lr_link.max(-2)

        # [batch_size, n, n_rules], only preserve legal combining operations
        intra_lr_link = lr_link[..., intra_link_rules]
        # [batch_size, n, 1], select maximum values from all rules
        intra_l_link, _ = intra_r_link, _ = intra_lr_link.max(-1, keepdim=True)
        # [batch_size, n, n_rules]
        inter_lr_link = lr_link[..., inter_link_rules]
        # [batch_size, n, 1]
        inter_l_link, _ = inter_r_link, _ = inter_lr_link.max(-1, keepdim=True)
        # left/right intra/inter-span link
        # [batch_size, n, n_state] -> [batch_size, n_state, n]
        l_link = torch.cat((intra_l_link, inter_l_link), -1).permute(0, 2, 1)
        # [batch_size, n_state, n]
        s_i.diagonal(-w).copy_(l_link + s_arc.diagonal(-w))
        r_link = torch.cat((intra_r_link, inter_r_link), -1).permute(0, 2, 1)
        # [batch_size, n, n_state]
        s_i.diagonal(w).copy_(r_link + s_arc.diagonal(w))

        # [batch_size, n, w, n_state]
        cl = stripe(s_c, n, w, (0, 0), 0).permute(2, 0, 1, 3)
        il = stripe(s_i, n, w, (w, 0)).permute(2, 0, 1, 3)
        # [batch_size, n, w, n_state*n_state]
        l_combine = (cl.unsqueeze(-1) + il.unsqueeze(-2)).flatten(start_dim=-2, end_dim=-1)
        # [batch_size, n, n_state*n_state], select maximum values from all split points
        l_combine, _ = l_combine.max(-2)
        # [batch_size, n, n_rules], only preserve legal combining operations
        intra_l_combine = l_combine[..., intra_combine_rules]
        # [batch_size, n, 1], select maximum values from all rules
        intra_l_combine, _ = intra_l_combine.max(-1, keepdim=True)
        # [batch_size, n, n_rules]
        inter_l_combine = l_combine[..., inter_combine_rules]
        # [batch_size, n, 1]
        inter_l_combine, _ = inter_l_combine.max(-1, keepdim=True)
        # left intra/inter-span combine
        # [batch_size, n, n_state] -> [batch_size, n_state, n]
        l_combine = torch.cat((intra_l_combine, inter_l_combine), -1).permute(0, 2, 1)
        # [batch_size, n_state, n]
        s_c.diagonal(-w).copy_(l_combine)

        # [batch_size, n, w, n_state]
        cr = stripe(s_c, n, w, (1, w), 0).permute(2, 0, 1, 3)
        ir = stripe(s_i, n, w, (0, 1)).permute(2, 0, 1, 3)
        # [batch_size, n, w, n_state*n_state]
        r_combine = (cr.unsqueeze(-1) + ir.unsqueeze(-2)).flatten(start_dim=-2, end_dim=-1)
        # [batch_size, n, n_state*n_state], select maximum values from all split points
        r_combine, _ = r_combine.max(-2)
        # [batch_size, n, n_rules], only preserve legal combining operations
        intra_r_combine = r_combine[..., intra_combine_rules]
        # [batch_size, n, 1], select maximum values from all rules
        intra_r_combine, _ = intra_r_combine.max(-1, keepdim=True)
        # [batch_size, n, n_rules]
        inter_r_combine = r_combine[..., inter_combine_rules]
        # [batch_size, n, 1]
        inter_r_combine, _ = inter_r_combine.max(-1, keepdim=True)
        # right intra/inter-span combine
        # [batch_size, n, n_state] -> [batch_size, n_state, n]
        r_combine = torch.cat((intra_r_combine, inter_r_combine), -1).permute(0, 2, 1)
        # [batch_size, n_state, n]
        s_c.diagonal(w).copy_(r_combine)
        # prevent multiple root
        s_c[0, w][lens.ne(w)] = MIN

    logZ = s_c[0, :, :, 1].gather(0, lens.unsqueeze(0)).sum()
    marginals, = autograd.grad(logZ, s_arc)
    # preds = lens.new_zeros(batch_size, seq_len, n_state).masked_scatter_(mask, marginals.permute(2, 1, 0, 3).nonzero()[:, 2])

    return marginals.permute(2, 1, 0, 3)


def eisner2o(scores, mask, multiroot=False):
    r"""
    Second-order Eisner algorithm for projective decoding :cite:`mcdonald-pereira-2006-online`.
    This is an extension of the first-order one that further incorporates sibling scores into tree scoring.

    Args:
        scores (~torch.Tensor, ~torch.Tensor):
            A tuple of two tensors representing the first-order and second-order scores respectively.
            The first (``[batch_size, seq_len, seq_len]``) holds scores of all dependent-head pairs.
            The second (``[batch_size, seq_len, seq_len, seq_len]``) holds scores of all dependent-head-sibling triples.
        mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
            The mask to avoid parsing over padding tokens.
            The first column serving as pseudo words for roots should be ``False``.
        multiroot (bool):
            Ensures to parse a single-root tree If ``False``.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[batch_size, seq_len]`` for the resulting projective parse trees.

    Examples:
        >>> s_arc = torch.tensor([[[ -2.8092,  -7.9104,  -0.9414,  -5.4360],
                                   [-10.3494,  -7.9298,  -3.6929,  -7.3985],
                                   [  1.1815,  -3.8291,   2.3166,  -2.7183],
                                   [ -3.9776,  -3.9063,  -1.6762,  -3.1861]]])
        >>> s_sib = torch.tensor([[[[ 0.4719,  0.4154,  1.1333,  0.6946],
                                    [ 1.1252,  1.3043,  2.1128,  1.4621],
                                    [ 0.5974,  0.5635,  1.0115,  0.7550],
                                    [ 1.1174,  1.3794,  2.2567,  1.4043]],
                                   [[-2.1480, -4.1830, -2.5519, -1.8020],
                                    [-1.2496, -1.7859, -0.0665, -0.4938],
                                    [-2.6171, -4.0142, -2.9428, -2.2121],
                                    [-0.5166, -1.0925,  0.5190,  0.1371]],
                                   [[ 0.5827, -1.2499, -0.0648, -0.0497],
                                    [ 1.4695,  0.3522,  1.5614,  1.0236],
                                    [ 0.4647, -0.7996, -0.3801,  0.0046],
                                    [ 1.5611,  0.3875,  1.8285,  1.0766]],
                                   [[-1.3053, -2.9423, -1.5779, -1.2142],
                                    [-0.1908, -0.9699,  0.3085,  0.1061],
                                    [-1.6783, -2.8199, -1.8853, -1.5653],
                                    [ 0.3629, -0.3488,  0.9011,  0.5674]]]])
        >>> mask = torch.tensor([[False,  True,  True,  True]])
        >>> eisner2o((s_arc, s_sib), mask)
        tensor([[0, 2, 0, 2]])
    """

    # the end position of each sentence in a batch
    lens = mask.sum(1)
    s_arc, s_sib = scores
    batch_size, seq_len, _ = s_arc.shape
    # [seq_len, seq_len, batch_size]
    s_arc = s_arc.permute(2, 1, 0)
    # [seq_len, seq_len, seq_len, batch_size]
    s_sib = s_sib.permute(2, 1, 3, 0)
    s_i = torch.full_like(s_arc, float('-inf'))
    s_s = torch.full_like(s_arc, float('-inf'))
    s_c = torch.full_like(s_arc, float('-inf'))
    p_i = s_arc.new_zeros(seq_len, seq_len, batch_size).long()
    p_s = s_arc.new_zeros(seq_len, seq_len, batch_size).long()
    p_c = s_arc.new_zeros(seq_len, seq_len, batch_size).long()
    s_c.diagonal().fill_(0)

    for w in range(1, seq_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = seq_len - w
        starts = p_i.new_tensor(range(n)).unsqueeze(0)
        # I(j->i) = max(I(j->r) + S(j->r, i)), i < r < j |
        #               C(j->j) + C(i->j-1))
        #           + s(j->i)
        # [n, w, batch_size]
        il = stripe(s_i, n, w, (w, 1)) + stripe(s_s, n, w, (1, 0), 0)
        il += stripe(s_sib[range(w, n+w), range(n)], n, w, (0, 1))
        # [n, 1, batch_size]
        il0 = stripe(s_c, n, 1, (w, w)) + stripe(s_c, n, 1, (0, w - 1))
        # il0[0] are set to zeros since the scores of the complete spans starting from 0 are always -inf
        il[:, -1] = il0.index_fill_(0, lens.new_tensor(0), 0).squeeze(1)
        il_span, il_path = il.permute(2, 0, 1).max(-1)
        s_i.diagonal(-w).copy_(il_span + s_arc.diagonal(-w))
        p_i.diagonal(-w).copy_(il_path + starts + 1)
        # I(i->j) = max(I(i->r) + S(i->r, j), i < r < j |
        #               C(i->i) + C(j->i+1))
        #           + s(i->j)
        # [n, w, batch_size]
        ir = stripe(s_i, n, w) + stripe(s_s, n, w, (0, w), 0)
        ir += stripe(s_sib[range(n), range(w, n+w)], n, w)
        ir[0] = float('-inf')
        # [n, 1, batch_size]
        ir0 = stripe(s_c, n, 1) + stripe(s_c, n, 1, (w, 1))
        ir[:, 0] = ir0.squeeze(1)
        ir_span, ir_path = ir.permute(2, 0, 1).max(-1)
        s_i.diagonal(w).copy_(ir_span + s_arc.diagonal(w))
        p_i.diagonal(w).copy_(ir_path + starts)

        # [n, w, batch_size]
        slr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        slr_span, slr_path = slr.permute(2, 0, 1).max(-1)
        # S(j, i) = max(C(i->r) + C(j->r+1)), i <= r < j
        s_s.diagonal(-w).copy_(slr_span)
        p_s.diagonal(-w).copy_(slr_path + starts)
        # S(i, j) = max(C(i->r) + C(j->r+1)), i <= r < j
        s_s.diagonal(w).copy_(slr_span)
        p_s.diagonal(w).copy_(slr_path + starts)

        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl_span, cl_path = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        p_c.diagonal(-w).copy_(cl_path + starts)
        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, cr_path = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        if not multiroot:
            s_c[0, w][lens.ne(w)] = float('-inf')
        p_c.diagonal(w).copy_(cr_path + starts + 1)

    def backtrack(p_i, p_s, p_c, heads, i, j, flag):
        if i == j:
            return
        if flag == 'c':
            r = p_c[i, j]
            backtrack(p_i, p_s, p_c, heads, i, r, 'i')
            backtrack(p_i, p_s, p_c, heads, r, j, 'c')
        elif flag == 's':
            r = p_s[i, j]
            i, j = sorted((i, j))
            backtrack(p_i, p_s, p_c, heads, i, r, 'c')
            backtrack(p_i, p_s, p_c, heads, j, r + 1, 'c')
        elif flag == 'i':
            r, heads[j] = p_i[i, j], i
            if r == i:
                r = i + 1 if i < j else i - 1
                backtrack(p_i, p_s, p_c, heads, j, r, 'c')
            else:
                backtrack(p_i, p_s, p_c, heads, i, r, 'i')
                backtrack(p_i, p_s, p_c, heads, r, j, 's')

    preds = []
    p_i = p_i.permute(2, 0, 1).cpu()
    p_s = p_s.permute(2, 0, 1).cpu()
    p_c = p_c.permute(2, 0, 1).cpu()
    for i, length in enumerate(lens.tolist()):
        heads = p_c.new_zeros(length + 1, dtype=torch.long)
        backtrack(p_i[i], p_s[i], p_c[i], heads, 0, length, 'c')
        preds.append(heads.to(mask.device))

    return pad(preds, total_length=seq_len).to(mask.device)


@torch.enable_grad()
def cky(scores, mask, grad=False):
    r"""
    The implementation of `Cocke-Kasami-Younger`_ (CKY) algorithm to parse constituency trees :cite:`zhang-etal-2020-fast`.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
            Scores of all candidate constituents.
        mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
            The mask to avoid parsing over padding tokens.
            For each square matrix in a batch, the positions except upper triangular part should be masked out.

    Returns:
        Sequences of factorized predicted bracketed trees that are traversed in pre-order.

    Examples:
        >>> scores = torch.tensor([[[ 2.5659,  1.4253, -2.5272,  3.3011],
                                    [ 1.3687, -0.5869,  1.0011,  3.3020],
                                    [ 1.2297,  0.4862,  1.1975,  2.5387],
                                    [-0.0511, -1.2541, -0.7577,  0.2659]]])
        >>> mask = torch.tensor([[[False,  True,  True,  True],
                                  [False, False,  True,  True],
                                  [False, False, False,  True],
                                  [False, False, False, False]]])
        >>> cky(scores, mask)
        [[(0, 3), (0, 1), (1, 3), (1, 2), (2, 3)]]

    .. _Cocke-Kasami-Younger:
        https://en.wikipedia.org/wiki/CYK_algorithm
    """

    lens = mask[:, 0].sum(-1)
    scores = scores.permute(1, 2, 3, 0).requires_grad_()
    seq_len, seq_len, n_labels, batch_size = scores.shape
    s = scores.new_zeros(seq_len, seq_len, batch_size)
    p_l = scores.new_zeros(seq_len, seq_len, batch_size).long()

    for w in range(1, seq_len):
        n = seq_len - w
        s_l, p = scores.diagonal(w).max(0)
        p_l.diagonal(w).copy_(p)

        if w == 1:
            s.diagonal(w).copy_(s_l)
            continue
        # [n, w, batch_size]
        s_s = stripe(s, n, w-1, (0, 1)) + stripe(s, n, w-1, (1, w), 0)
        # [batch_size, n, w]
        s_s = s_s.permute(2, 0, 1)
        # [batch_size, n]
        s_s, _ = s_s.max(-1)
        s.diagonal(w).copy_(s_s + s_l)
        # s_s, p = s_s.max(-1)
        # s.diagonal(w).copy_(s_s + s_l)
        # p_s.diagonal(w).copy_(p + starts + 1)

    # def backtrack(p_s, p_l, i, j):
    #     if j == i + 1:
    #         return [(i, j, p_l[i][j])]
    #     split, label = p_s[i][j], p_l[i][j]
    #     ltree = backtrack(p_s, p_l, i, split)
    #     rtree = backtrack(p_s, p_l, split, j)
    #     return [(i, j, label)] + ltree + rtree

    # p_s = p_s.permute(2, 0, 1).tolist()
    # p_l = p_l.permute(2, 0, 1).tolist()
    # trees = [backtrack(p_s[i], p_l[i], 0, length) for i, length in enumerate(lens.tolist())]

    logZ = s[0].gather(0, lens.unsqueeze(0)).sum()
    # [seq_len, seq_len, n_labels, batch_size]
    marginals, = autograd.grad(logZ, scores)

    if grad:
        return marginals.permute(3, 0, 1, 2)
    return [sorted(i.nonzero().tolist(), key=lambda x:(x[0], -x[1])) for i in marginals.permute(3, 0, 1, 2)]

