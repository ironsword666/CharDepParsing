# -*- coding: utf-8 -*-

import torch

from supar.models import CharModel
from supar.modules import MLP, Biaffine
from supar.utils import Config
from supar.structs import CharCRFDependency, Coarse2FineCharCRFDependency
from supar.utils.alg import eisner, span_constrained_eisner, constraint_eisner
from supar.utils.transform.conll import CoNLL
from supar.utils.transform.conll import CharCoNLL
from supar.utils.logging import get_logger
from supar.utils.common import MIN

import torch.nn.functional as F

logger = get_logger(__name__)


class CharCRFDependencyModel(CharModel):
    r"""
    The implementation of CRF Dependency Parser.

    Args:
        n_chars (int):
            The size of the word vocabulary.
        n_rels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (list[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [``'char'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        freeze (bool):
            If ``True``, freezes BERT parameters, required if using BERT features. Default: ``True``.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            The size of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_arc_mlp (int):
            Arc MLP size. Default: 500.
        n_rel_mlp  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        scale (float):
            Scaling factor for affine scores. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_chars,
                 n_rels,
                 n_bichars=None,
                 n_trichars=None,
                 encoder='lstm',
                 feat=['bichar'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_arc_mlp=500,
                 n_rel_mlp=100,
                 mlp_dropout=.33,
                 scale=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.arc_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.arc_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.rel_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)
        self.rel_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_rel_mlp, dropout=mlp_dropout)

        self.arc_attn = Biaffine(n_in=n_arc_mlp, scale=scale, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=n_rel_mlp, n_out=n_rels, bias_x=True, bias_y=True)

        self.crf = CharCRFDependency()

    def forward(self, chars, feats=None):
        r"""
        Args:
            chars (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible arcs.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each arc.
        """
        x = self.encode(chars, feats)
        mask = chars.ne(self.args.pad_index) if len(chars.shape) < 3 else chars.ne(self.args.pad_index).any(-1)

        arc_d = self.arc_mlp_d(x)
        arc_h = self.arc_mlp_h(x)
        rel_d = self.rel_mlp_d(x)
        rel_h = self.rel_mlp_h(x)

        # [batch_size, seq_len, seq_len], s(i,j) means the score of arc j->i, mask the padding
        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), MIN)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return s_arc, s_rel

    def loss(self, s_arc, s_rel, arcs, rels, mask, mbr=False, partial=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            mbr (bool):
                Whether to use MBR decoding. Default: ``False``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor:
                The training loss.
        """
        raise NotImplementedError

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.
        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """
        raise NotImplementedError


class ExplicitCharCRFDependencyModel(CharCRFDependencyModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.crf = CharCRFDependency()

    def loss(self, s_arc, s_rel, arcs, rels, mask, mbr=False, partial=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            mbr (bool):
                If ``True``, returns marginals for MBR decoding. Default: ``True``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and
                original arc scores of shape ``[batch_size, seq_len, seq_len]`` if ``mbr=False``, or marginals otherwise.
        """
        assert not partial, 'Partial annotation is not supported in Explicit.'
        arc_loss, arc_probs = self.crf(s_arc, mask, target=arcs, partial=partial, normalizer=self.args.struct_normalizer)
        # -1 denotes un-annotated arcs
        if partial:
            mask = mask & arcs.ge(0)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = F.cross_entropy(s_rel, rels)
        # logger.info(f"arc loss: {arc_loss}, rel loss: {rel_loss}")
        # loss = (1 - self.args.loss_weight) * arc_loss + self.args.loss_weight * rel_loss
        loss = arc_loss + rel_loss
        return loss, arc_probs

    def decode(self, s_arc, s_rel, mask, nul_mask=None, orientation='leftward', tree=True, proj=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """
        batch_size, seq_len = s_rel.shape[:2]
        offset = 1 if orientation == 'leftward' else -1
        s_rel[:, :, 0, 0] = MIN
        # gold segmentation is provided
        if nul_mask is not None:
            # [batch_size, seq_len]
            nul_mask = nul_mask & mask
            # [batch_size, seq_len]
            rel_mask = ~nul_mask & mask
            # word head -> word head
            # [batch_size, seq_len, seq_len]
            inter_mask = rel_mask.unsqueeze(1) & rel_mask.unsqueeze(2)
            # root -> word head
            inter_mask[:, :, 0] = rel_mask
            nul_mask = nul_mask[:, :-1] if orientation == 'leftward' else nul_mask[:, 1:]
            intra_mask = inter_mask.new_zeros((batch_size, seq_len, seq_len))
            intra_mask.diagonal(offset=offset, dim1=-2, dim2=-1).copy_(nul_mask)
            arc_mask = intra_mask | inter_mask
            arc_mask.diagonal(offset=0, dim1=-2, dim2=-1).fill_(0)
            # mask illegal arcs
            s_arc.masked_fill_(~arc_mask, MIN)
            # mask illegal labels
            s_rel[..., :self.args.nul_index].masked_fill_(intra_mask.unsqueeze(-1), MIN)
            s_rel[..., self.args.nul_index+1:].masked_fill_(intra_mask.unsqueeze(-1), MIN)
            # for head chars i, j, not allowing j -> i labeled as `unk`
            s_rel[..., self.args.nul_index].masked_fill_(inter_mask, MIN)
        else:
            arc_mask = s_arc.new_zeros((batch_size, seq_len, seq_len)).bool()
            arc_mask.diagonal(offset=offset, dim1=-2, dim2=-1).fill_(1)
            # only allow adjacent arcs to be labeled as `unk`
            s_rel[..., self.args.nul_index].masked_fill_(~arc_mask, MIN)

        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i+1], proj) for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            arc_preds[bad] = eisner(s_arc[bad], mask[bad])
        # arc_preds = eisner(s_arc, mask)

        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds


class LatentCharCRFDependencyModel(CharCRFDependencyModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.crf = CharCRFDependency()

    def loss(self, s_arc, s_rel, arcs, rels, mask, c_span_mask=None, combine_mask=None):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            c_span_mask (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                The mask for complete spans.
            combine_mask (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                The mask for combination operation.
            mbr (bool):
                If ``True``, returns the MBR loss. Default: ``False``.
            one_best (bool):
                If ``True``, returns the label loss of the one-best tree. Default: ``False``.

        Returns:
            ~torch.Tensor:
                The training loss.
        """
        if self.args.label_loss == 'crf':
            # add the label probability to the arc scores
            # [batch_size, seq_len, seq_len, n_labels]
            p_rel = s_rel.log_softmax(-1)
            # illegal labels are masked to -inf
            # [batch_size, seq_len, seq_len]
            s_rel = torch.full_like(p_rel[..., 0], MIN)
            # [batch_size, seq_len, seq_len]
            s_rel[arcs] = p_rel[arcs].gather(-1, rels[arcs].unsqueeze(-1)).squeeze(-1)
            loss = self.crf.labeling(s_arc, s_rel, mask, target=arcs, c_span_mask=c_span_mask, combine_mask=combine_mask, normalizer=self.args.struct_normalizer)
            logger.info(f"crf loss: {loss}")
            return loss

        arc_loss, marginals = self.crf(s_arc, mask, target=arcs, partial=True, normalizer=self.args.struct_normalizer, c_span_mask=c_span_mask, combine_mask=combine_mask, marginal=self.args.marginal)

        # marginals = self.crf.marginalize(s_arc, mask, arcs, c_span_mask=c_span_mask, combine_mask=combine_mask)
        batch_size, seq_len = mask.shape
        lens = mask.sum(-1)
        # [batch_size, seq_len, seq_len]
        arc_mask = mask.index_fill(1, lens.new_tensor(0), 1).unsqueeze(1) & mask.unsqueeze(2)
        gold_arcs = rels.ge(0) & arc_mask
        weights = None
        if self.args.label_loss == 'marginal':
            assert self.args.marginal is not None, "None is specified for calculating marginals,"
            # use marginal probability of arcs (or using constraint inside) to weight the label loss
            weights = marginals[gold_arcs].detach()
        elif self.args.label_loss == 'best':
            # use arcs in the one-best tree to calculate the label loss
            # [batch_size, seq_len]
            arc_preds = span_constrained_eisner(s_arc, mask, c_span_mask=c_span_mask, combine_mask=combine_mask, target=arcs)
            # [batch_size, seq_len, seq_len]
            arc_preds = arc_preds.unsqueeze(-1).eq(lens.new_tensor(range(seq_len)))
            gold_arcs = arc_preds & arc_mask
        s_rel, rels = s_rel[gold_arcs], rels[gold_arcs]
        if weights is not None:
            rel_loss = torch.mean(F.cross_entropy(s_rel, rels, reduction='none') * weights)
        else:
            rel_loss = F.cross_entropy(s_rel, rels)
        # logger.info(f"arc loss: {arc_loss}, rel loss: {rel_loss}")
        # loss = (1 - self.args.loss_weight) * arc_loss + self.args.loss_weight * rel_loss
        loss = arc_loss + rel_loss
        return loss

    def decode(self, s_arc, s_rel, mask, intra_word_rel_indexes=[0], arcs=None, rels=None, c_span_mask=None, combine_mask=None, intra_mask=None):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            arcs (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels.
            c_span_mask (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                The mask for complete spans.
            combine_mask (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                The mask for combination operation.
            intra_mask (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                The mask for intra-word arcs.
        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """
        # TODO: local decode then dp decode
        # [batch_size, seq_len]
        arc_preds = span_constrained_eisner(s_arc, mask, c_span_mask=c_span_mask, combine_mask=combine_mask, target=arcs)
        arc_mask = mask.index_fill(1, mask.new_tensor(0).long(), 1).unsqueeze(1) & mask.unsqueeze(2)
        # the label of root -> i can't be any intra-word rel, rather than just `nul`
        s_rel[:, :, 0, list(intra_word_rel_indexes)] = MIN
        if intra_mask is not None:
            # not allow `nul` label for inter-word arcs
            s_rel[..., self.args.nul_index].masked_fill_(~intra_mask & arc_mask, MIN)
            # only allow `nul` label for intra-word arcs
            s_rel[..., :self.args.nul_index].masked_fill_((intra_mask & arc_mask).unsqueeze(-1), MIN)
            s_rel[..., self.args.nul_index+1:].masked_fill_((intra_mask & arc_mask).unsqueeze(-1), MIN)
        if rels is not None:
            rels = rels.masked_fill(rels.lt(0), 0).long()
            s_rel.scatter_(-1, rels.unsqueeze(-1), 1e30)
        # [batch_size, seq_len]
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        return arc_preds, rel_preds

    def constraint_decode(self, s_arc, s_rel, mask, intra_word_rel_indexes=[0]):
        batch_size, seq_len = mask.shape
        # [batch_size, seq_len, seq_len]
        marginals = self.crf.marginalize(s_arc, mask)
        # [batch_size, seq_len, seq_len, n_labels]
        p_rel = s_rel.softmax(-1)
        # the arc root -> i shouldn't be labeled as `nul`
        p_rel[:, :, 0, list(intra_word_rel_indexes)] = 0
        # [batch_size, seq_len, seq_len]
        p_nul = p_rel[..., list(intra_word_rel_indexes)].sum(-1)
        # [batch_size, seq_len, seq_len, n_states]
        s_arc = torch.stack([p_nul, 1 - p_nul], dim=-1) * marginals.unsqueeze(-1)
        # [batch_size, seq_len, seq_len, n_states]
        arc_preds = constraint_eisner(s_arc, mask)

        # def check_conflict(arc_preds, mask):
        #     lens = mask.sum(-1)
        #     rel_preds = lens.new_zeros(batch_size, seq_len, seq_len)
        #     rel_preds[arc_preds[..., 1] == 1] = 1
        #     arc_preds = lens.new_zeros(batch_size, seq_len).masked_scatter_(mask, arc_preds.nonzero()[:, 2])
        #     rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        #     arc_preds = [seq.tolist() for seq in arc_preds[mask].split(lens.tolist())]
        #     rel_preds = [seq.tolist() for seq in rel_preds[mask].split(lens.tolist())]
        #     preds = [CharCoNLL.recover_words(arc_pred, rel_pred) for arc_pred, rel_pred in zip(arc_preds, rel_preds)]
        # check_conflict(arc_preds, mask)

        # [batch_size, seq_len, seq_len, n_labels]
        inter_s_rel = s_rel.clone().detach()
        inter_s_rel[..., list(intra_word_rel_indexes)] = MIN
        inter_rel_preds = inter_s_rel.argmax(-1)
        intra_s_rel = torch.full_like(s_rel, MIN)
        intra_s_rel[..., list(intra_word_rel_indexes)] = s_rel[..., list(intra_word_rel_indexes)]
        intra_rel_preds = intra_s_rel.argmax(-1)
        inter_rel_preds[arc_preds[..., 0] == 1] = intra_rel_preds[arc_preds[..., 0] == 1]
        rel_preds = inter_rel_preds

        # # [batch_size, seq_len, seq_len]
        # rel_preds = s_rel.argmax(-1)
        # rel_preds[arc_preds[..., 0] == 1] = self.args.nul_index

        # # intra_arc = arc_preds[..., 0] == 1
        # # inter_arc = arc_preds[..., 1] == 1
        # s_rel[inter_arc] = s_rel[inter_arc].index_fill(-1, torch.tensor(intra_word_rel_indexes), MIN)

        arc_preds = rel_preds.new_zeros(batch_size, seq_len).masked_scatter_(mask, arc_preds.nonzero()[:, 2])
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        return arc_preds, rel_preds


class ClusterCharCRFDependencyModel(LatentCharCRFDependencyModel):

    def __init__(self,
                 n_chars,
                 n_rels,
                 n_clusters,
                 n_bichars=None,
                 n_trichars=None,
                 encoder='lstm',
                 feat=['bichar'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_arc_mlp=500,
                 n_rel_mlp=100,
                 n_cluster_mlp=100,
                 mlp_dropout=.33,
                 scale=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.cluster_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_cluster_mlp, dropout=mlp_dropout)
        self.cluster_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_cluster_mlp, dropout=mlp_dropout)
        self.cluster_attn = Biaffine(n_in=n_cluster_mlp, n_out=n_clusters, bias_x=True, bias_y=True)

    def forward(self, chars, feats=None):
        x = self.encode(chars, feats)
        mask = chars.ne(self.args.pad_index) if len(chars.shape) < 3 else chars.ne(self.args.pad_index).any(-1)

        arc_d = self.arc_mlp_d(x)
        arc_h = self.arc_mlp_h(x)
        rel_d = self.rel_mlp_d(x)
        rel_h = self.rel_mlp_h(x)
        cluster_d = self.cluster_mlp_d(x)
        cluster_h = self.cluster_mlp_h(x)

        # [batch_size, seq_len, seq_len], s(i,j) means the score of arc j->i, mask the padding
        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), MIN)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # [batch_size, seq_len, seq_len, n_clusters]
        s_cluster = self.cluster_attn(cluster_d, cluster_h).permute(0, 2, 3, 1)
        s_rel = self.fuse(s_rel, s_cluster, self.args.rel2cluster)

        return s_arc, s_rel

    def fuse(self, s_rel, s_cluster, rel2cluster):
        rel2cluster = s_rel.new_tensor(rel2cluster, dtype=torch.long)
        # [batch_size, seq_len, seq_len, n_rels]
        s_cluster = s_cluster.index_select(-1, rel2cluster)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = s_rel + s_cluster

        return s_rel


class Coarse2FineCharCRFDependencyModel(LatentCharCRFDependencyModel):

    def __init__(self,
                 n_chars,
                 n_rels,
                 n_bichars=None,
                 n_trichars=None,
                 encoder='lstm',
                 feat=['bichar'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_arc_mlp=500,
                 n_rel_mlp=100,
                 mlp_dropout=.33,
                 scale=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.arc_attn = Biaffine(n_in=n_arc_mlp, n_out=2, scale=scale, bias_x=True, bias_y=False)

        self.crf = Coarse2FineCharCRFDependency()

        if getattr(self.args, 'root_as_head', None) is None:
            self.args.root_as_head = True

    def forward(self, chars, feats=None):
        x = self.encode(chars, feats)
        mask = chars.ne(self.args.pad_index) if len(chars.shape) < 3 else chars.ne(self.args.pad_index).any(-1)

        arc_d = self.arc_mlp_d(x)
        arc_h = self.arc_mlp_h(x)
        rel_d = self.rel_mlp_d(x)
        rel_h = self.rel_mlp_h(x)

        # [batch_size, seq_len, seq_len, 2], s(i,j) means the score of arc j->i, mask the padding
        s_arc = self.arc_attn(arc_d, arc_h).permute(0, 2, 3, 1).masked_fill_(~mask[:, None, :, None], MIN)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return s_arc, s_rel

    def loss(self, s_arc, s_rel, arcs, rels, mask, c_span_mask=None, i_span_mask=None):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len, 2]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            c_span_mask (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                The mask for complete spans.
            combine_mask (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                The mask for combination operation.
            mbr (bool):
                If ``True``, returns the MBR loss. Default: ``False``.
            one_best (bool):
                If ``True``, returns the label loss of the one-best tree. Default: ``False``.

        Returns:
            ~torch.Tensor:
                The training loss.
        """
        # the arc root -> i shouldn't be labeled as `nul` (intra-word label)
        s_arc[:, :, 0, 0] = MIN
        batch_size, seq_len = mask.shape
        lens = mask.sum(-1)
        # [batch_size, seq_len, seq_len]
        # allow <bos>
        arc_mask = mask.index_fill(1, lens.new_tensor(0), 1).unsqueeze(1) & mask.unsqueeze(2)
        gold_intra_arcs = arcs[..., 0] & arc_mask
        gold_inter_arcs = arcs[..., 1] & arc_mask

        if self.args.label_loss == 'crf':
            # add the label probability to the arc scores
            # [batch_size, seq_len, seq_len, n_labels]
            p_rel = s_rel.log_softmax(-1)
            # [batch_size, seq_len, seq_len, 2]
            # label probability of intra-word arcs are 1 -> log 1 = 0
            # label probability of illegal arcs are 0 -> log 0 = -inf, which is masked by inside labeling function
            s_rel = torch.full_like(s_arc, MIN)
            # s_rel[..., 0].masked_fill_(gold_intra_arcs, 0)
            s_rel[..., 0][gold_intra_arcs] = p_rel[gold_intra_arcs].gather(-1, rels[gold_intra_arcs].unsqueeze(-1)).squeeze(-1)
            s_rel[..., 1][gold_inter_arcs] = p_rel[gold_inter_arcs].gather(-1, rels[gold_inter_arcs].unsqueeze(-1)).squeeze(-1)
            loss = self.crf.labeling(s_arc, s_rel, mask, target=arcs, c_span_mask=c_span_mask, i_span_mask=i_span_mask, root_as_head=self.args.root_as_head, normalizer=self.args.struct_normalizer)
            return loss

        arc_loss, marginals = self.crf(s_arc, mask, target=arcs, partial=True, marginal=self.args.marginal, c_span_mask=c_span_mask, i_span_mask=i_span_mask, root_as_head=self.args.root_as_head, normalizer=self.args.struct_normalizer)

        weights = None
        if self.args.label_loss == 'marginal':
            # use marginal probability of arcs (or using constraint inside) to weight the label loss
            weights = marginals[..., 1][gold_inter_arcs].detach()
        elif self.args.label_loss == 'best':
            raise NotImplementedError
            # use arcs in the one-best tree to calculate the label loss
            # [batch_size, seq_len]
            arc_preds = self.crf.decode(s_arc, mask, target=arcs, c_span_mask=c_span_mask, i_span_mask=i_span_mask)
            # [batch_size, seq_len, seq_len]
            arc_preds = arc_preds.unsqueeze(-1).eq(lens.new_tensor(range(seq_len)))
            gold_arcs = arc_preds & arc_mask
        s_rel, rels = s_rel[gold_inter_arcs], rels[gold_inter_arcs]
        if weights is not None:
            rel_loss = torch.mean(F.cross_entropy(s_rel, rels, reduction='none') * weights)
        else:
            rel_loss = F.cross_entropy(s_rel, rels)
        logger.info(f"arc loss: {arc_loss}, rel loss: {rel_loss}")
        # loss = (1 - self.args.loss_weight) * arc_loss + self.args.loss_weight * rel_loss
        loss = arc_loss + rel_loss
        return loss

    def decode(self, s_arc, s_rel, mask, target=None, c_span_mask=None, i_span_mask=None):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            c_span_mask (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                The mask for complete spans.
            combine_mask (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                The mask for combination operation.
            intra_mask (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                The mask for intra-word arcs.
        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """
        batch_size, seq_len = mask.shape
        # the arc root -> i shouldn't be labeled as `nul`
        s_arc[:, :, 0, 0] = MIN
        # [batch_size, seq_len, seq_len, n_states]
        arc_preds = self.crf.decode(s_arc, mask, target=target, c_span_mask=c_span_mask, i_span_mask=i_span_mask, root_as_head=self.args.root_as_head)

        s_rel[..., self.args.nul_index] = MIN
        # [batch_size, seq_len, seq_len]
        rel_preds = s_rel.argmax(-1)
        rel_preds[arc_preds[..., 0] == 1] = self.args.nul_index
        arc_preds = rel_preds.new_zeros(batch_size, seq_len).masked_scatter_(mask, arc_preds.nonzero()[:, 2])
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        return arc_preds, rel_preds

    def subword_decode(self, s_arc, s_rel, mask, sents, c_span_mask=None, i_span_mask=None):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            c_span_mask (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                The mask for complete spans.
            combine_mask (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                The mask for combination operation.
            intra_mask (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                The mask for intra-word arcs.
        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """
        batch_size, seq_len = mask.shape
        # the arc root -> i shouldn't be labeled as `nul`
        s_arc[:, :, 0, 0] = MIN
        # get marginals of arcs
        marginals = self.crf.marginalize(s_arc, mask, c_span_mask=c_span_mask, i_span_mask=i_span_mask, root_as_head=self.args.root_as_head)
        # only inter-word arcs are used to construct marginals of word-level arcs
        marginals = marginals[..., 1]
        max_len = max(len(spans) for spans in sents)
        s_rel[..., self.args.nul_index] = MIN
        s_rel = s_rel.softmax(-1)
        # construct new scores and mask for word-level arcs
        s_arc = s_arc.new_zeros(batch_size, max_len, max_len)
        s_word_rel = s_rel.new_zeros(batch_size, max_len, max_len, s_rel.shape[-1])
        word_mask = s_arc.new_zeros(batch_size, max_len).bool()
        # word boundaries
        for batch_idx, word_boundaries in enumerate(sents):
            for i, (b, e) in enumerate(word_boundaries):
                for j, (b_, e_) in enumerate(word_boundaries):
                    s_arc[batch_idx, i, j] = marginals[batch_idx, b:e, b_:e_].sum()
                    s_word_rel[batch_idx, i, j, :] = s_rel[batch_idx, b:e, b_:e_, :].sum((0, 1))
            word_mask[batch_idx, :len(word_boundaries)] = 1

        print(sents[0])
        print(s_arc[0])
        s_arc.masked_fill_(~word_mask.unsqueeze(1), MIN)
        word_mask[:, 0] = 0
        lens = word_mask.sum(1).tolist()

        # arc_preds = eisner(s_arc, word_mask)

        # [batch_size, seq_len, seq_len]
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i+1], True) for i, seq in zip(lens, arc_preds.tolist())]
        if any(bad):
            arc_preds[bad] = eisner(s_arc[bad], word_mask[bad])
        # arc_preds = rel_preds.new_zeros(batch_size, max_len).masked_scatter_(word_mask, arc_preds.nonzero()[:, 2])
        rel_preds = s_word_rel.argmax(-1)
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        arc_preds = [seq.tolist() for seq in arc_preds[word_mask].split(lens)]
        rel_preds = [seq.tolist() for seq in rel_preds[word_mask].split(lens)]
        return arc_preds, rel_preds


class SplitCoarse2FineCharCRFDependencyModel(Coarse2FineCharCRFDependencyModel):

    def __init__(self,
                 n_chars,
                 n_rels,
                 n_bichars=None,
                 n_trichars=None,
                 encoder='lstm',
                 feat=['bichar'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_arc_mlp=500,
                 n_rel_mlp=100,
                 mlp_dropout=.33,
                 scale=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.orth_loss_fn = torch.nn.CosineEmbeddingLoss()

        # self.arc_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp+100, dropout=mlp_dropout)
        # self.arc_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp+100, dropout=mlp_dropout)
        self.intra_arc_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.intra_arc_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.inter_arc_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.inter_arc_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)

        self.intra_arc_attn = Biaffine(n_in=n_arc_mlp, n_out=1, scale=scale, bias_x=True, bias_y=False)
        self.inter_arc_attn = Biaffine(n_in=n_arc_mlp, n_out=1, scale=scale, bias_x=True, bias_y=False)

        # if self.args.shared_arc_info == 'repr' and self.args.arc_combination == 'cat':
        #     self.intra_arc_attn = Biaffine(n_in=n_arc_mlp*2, n_out=1, scale=scale, bias_x=True, bias_y=False)
        #     self.inter_arc_attn = Biaffine(n_in=n_arc_mlp*2, n_out=1, scale=scale, bias_x=True, bias_y=False)
        # else:
        #     self.intra_arc_attn = Biaffine(n_in=n_arc_mlp//2, n_out=1, scale=scale, bias_x=True, bias_y=False)
        #     self.inter_arc_attn = Biaffine(n_in=n_arc_mlp, n_out=1, scale=scale, bias_x=True, bias_y=False)

        # if self.args.shared_arc_info == 'one_biaffine' or self.args.shared_arc_info == 'two_biaffine' or self.args.shared_arc_info == 'repr':
        #     self.share_arc_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        #     self.share_arc_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        #     if self.args.shared_arc_info == 'one_biaffine':
        #         self.share_arc_attn = Biaffine(n_in=n_arc_mlp, n_out=1, scale=scale, bias_x=True, bias_y=False)
        #     elif self.args.shared_arc_info == 'two_biaffine':
        #         self.share_arc_attn = Biaffine(n_in=n_arc_mlp, n_out=2, scale=scale, bias_x=True, bias_y=False)
        #     elif self.args.shared_arc_info == 'repr':
        #         pass

    def forward(self, chars, feats=None):
        x = self.encode(chars, feats)
        mask = chars.ne(self.args.pad_index) if len(chars.shape) < 3 else chars.ne(self.args.pad_index).any(-1)

        # if self.args.shared_arc_info == 'one_biaffine' or self.args.shared_arc_info == 'two_biaffine' or self.args.shared_arc_info == 'repr':
        #     share_arc_d = self.share_arc_mlp_d(x)
        #     share_arc_h = self.share_arc_mlp_h(x)
        # if self.args.shared_arc_info == 'one_biaffine':
        #     s_share_arc = self.share_arc_attn(share_arc_d, share_arc_h).unsqueeze(-1)
        # elif self.args.shared_arc_info == 'two_biaffine':
        #     s_share_arc = self.share_arc_attn(share_arc_d, share_arc_h).permute(0, 2, 3, 1)

        if self.args.shared_arc_info == 'arc':
            arc_d = self.arc_mlp_d(x)
            arc_h = self.arc_mlp_h(x)
            intra_arc_d = self.intra_arc_mlp_d(arc_d)
            intra_arc_h = self.intra_arc_mlp_h(arc_h)
            inter_arc_d = self.inter_arc_mlp_d(arc_d)
            inter_arc_h = self.inter_arc_mlp_h(arc_h)
        elif self.args.shared_arc_info == 'inter_arc':
            inter_arc_d = self.inter_arc_mlp_d(x)
            inter_arc_h = self.inter_arc_mlp_h(x)
            intra_arc_d = self.intra_arc_mlp_d(inter_arc_d)
            intra_arc_h = self.intra_arc_mlp_h(inter_arc_h)
        else:
            intra_arc_d = self.intra_arc_mlp_d(x)
            intra_arc_h = self.intra_arc_mlp_h(x)
            inter_arc_d = self.inter_arc_mlp_d(x)
            inter_arc_h = self.inter_arc_mlp_h(x)

        rel_d = self.rel_mlp_d(x)
        rel_h = self.rel_mlp_h(x)

        orth_loss = 0
        if self.args.use_orth_loss and self.training:
            if 'd' in self.args.orth_hiddens:
                orth_loss += F.cosine_embedding_loss(intra_arc_d[mask], inter_arc_d[mask], target=torch.full_like(mask[mask], -1, dtype=torch.long), margin=self.args.cos_margin)
            if 'h' in self.args.orth_hiddens:
                orth_loss += F.cosine_embedding_loss(intra_arc_h[mask], inter_arc_h[mask], target=torch.full_like(mask[mask], -1, dtype=torch.long), margin=self.args.cos_margin)
            if 'dh' in self.args.orth_hiddens:
                orth_loss += F.cosine_embedding_loss(intra_arc_h[mask], inter_arc_d[mask], target=torch.full_like(mask[mask], -1, dtype=torch.long), margin=-1)
            if 'hd' in self.args.orth_hiddens:
                orth_loss += F.cosine_embedding_loss(intra_arc_d[mask], inter_arc_h[mask], target=torch.full_like(mask[mask], -1, dtype=torch.long), margin=-1)
        # if self.args.shared_arc_info == 'repr':
        #     if self.args.use_orth_loss and self.training:
        #         diff_intra_arc_d_loss = torch.mul(intra_arc_d, share_arc_d).pow(2)[mask].sum()
        #         diff_intra_arc_h_loss = torch.mul(intra_arc_h, share_arc_h).pow(2)[mask].sum()
        #         diff_inter_arc_d_loss = torch.mul(inter_arc_d, share_arc_d).pow(2)[mask].sum()
        #         diff_inter_arc_h_loss = torch.mul(inter_arc_h, share_arc_h).pow(2)[mask].sum()
        #         orth_loss = (diff_intra_arc_d_loss + diff_intra_arc_h_loss + diff_inter_arc_d_loss + diff_inter_arc_h_loss) / mask.sum()
        #     if self.args.arc_combination == 'cat':
        #         intra_arc_d = torch.cat((intra_arc_d, share_arc_d), dim=-1)
        #         intra_arc_h = torch.cat((intra_arc_h, share_arc_h), dim=-1)
        #         inter_arc_d = torch.cat((inter_arc_d, share_arc_d), dim=-1)
        #         inter_arc_h = torch.cat((inter_arc_h, share_arc_h), dim=-1)
        #     elif self.args.arc_combination == 'sum':
        #         intra_arc_d += share_arc_d
        #         intra_arc_h += share_arc_h
        #         inter_arc_d += share_arc_d
        #         inter_arc_h += share_arc_h
        #     else:
        #         raise ValueError('Unknown arc combination method: %s' % self.args.arc_combination)

        # [batch_size, seq_len, seq_len], s(i,j) means the score of arc j->i
        s_intra_arc = self.intra_arc_attn(intra_arc_d, intra_arc_h)
        s_inter_arc = self.inter_arc_attn(inter_arc_d, inter_arc_h)
        # [batch_size, seq_len, seq_len, 2]
        s_arc = torch.stack((s_intra_arc, s_inter_arc), dim=-1)

        # if self.args.shared_arc_info == 'one_biaffine' or self.args.shared_arc_info == 'two_biaffine':
        #     s_arc += s_share_arc
        # mask the padding
        s_arc.masked_fill_(~mask[:, None, :, None], MIN)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return s_arc, s_rel, orth_loss
