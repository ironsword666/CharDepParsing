# -*- coding=utf-8 -*-

from supar.models import Model
import torch
import torch.nn as nn
from supar.models.model import Model, EmbeddingModel
from supar.modules import MLP, Biaffine
from supar.modules.dropout import IndependentDropout
from supar.structs.treecrf import Const2DepCRF
from supar.utils import Config


class Const2DepModel(Model):
    r"""
    The implementation of first-order CRF Dependency Parser (:cite:`zhang-etal-2020-efficient`,
    :cite:`ma-hovy-2017-neural`, :cite:`koo-etal-2007-structured`).

    Args:
        n_words (int):
            The size of the word vocabulary.
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
        proj (bool):
            If ``True``, takes :class:`CRFDependency` as inference layer, :class:`MatrixTree` otherwise.
            Default: ``True``.
    """

    def __init__(self,
                 n_words,
                 n_tags=None,
                 n_chars=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
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
                 mlp_dropout=.33,
                 scale=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))

        self.arc_mlp_d = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.arc_mlp_h = MLP(n_in=self.args.n_hidden, n_out=n_arc_mlp, dropout=mlp_dropout)

        self.arc_attn = Biaffine(n_in=n_arc_mlp, scale=scale, bias_x=True, bias_y=False)
        self.crf = Const2DepCRF()

    def forward(self, words, feats=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
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

        x = self.encode(words, feats)
        mask = words.ne(self.args.pad_index) if len(words.shape) < 3 else words.ne(self.args.pad_index).any(-1)

        arc_d = self.arc_mlp_d(x)
        arc_h = self.arc_mlp_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc

    def loss(self, s_arc, mask, c_span_mask=None, combine_mask=None, mbr=False, partial=True):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
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
        arc_loss, arc_probs = self.crf(s_arc, mask, c_span_mask=c_span_mask, combine_mask=combine_mask, normalizer=self.args.struct_normalizer)
        loss = arc_loss
        return loss, arc_probs

    def decode(self, s_arc, mask, c_span_mask=None, combine_mask=None):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            mbr (bool):
                If ``True``, returns marginals for MBR decoding. Default: ``True``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The predicted arcs and
                original arc scores of shape ``[batch_size, seq_len, seq_len]`` if ``mbr=False``, or marginals otherwise.
        """
        arc_preds = self.crf.decode(s_arc, mask, c_span_mask=c_span_mask, combine_mask=combine_mask)
        return arc_preds


class Const2DepEmbeddingModel(EmbeddingModel):

    def __init__(self,
                 n_tags=None,
                 feat=['tag'],
                 bert=None,
                 stride=256,
                 pooling='mean',
                 freeze=True,
                 use_token_type=False,
                 embed_dropout=.33,
                 n_arc_mlp=500,
                 mlp_dropout=.33,
                 scale=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__(**Config().update(locals()))
        mlp_dim = self.n_embed
        if feat is not None and 'tag' in feat:
            self.tag_embed = nn.Embedding(num_embeddings=n_tags,
                                          embedding_dim=self.args.n_feat_embed)
            mlp_dim += self.args.n_feat_embed
        self.embed_dropout = IndependentDropout(p=embed_dropout)
        self.arc_mlp_d = MLP(n_in=mlp_dim, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.arc_mlp_h = MLP(n_in=mlp_dim, n_out=n_arc_mlp, dropout=mlp_dropout)

        self.arc_attn = Biaffine(n_in=n_arc_mlp, scale=scale, bias_x=True, bias_y=False)
        self.crf = Const2DepCRF()

    def forward(self, words, feats=None):
        x = self.encode(words, feats)

        mask = words.ne(self.args.pad_index) if len(words.shape) < 3 else words.ne(self.args.pad_index).any(-1)

        arc_d = self.arc_mlp_d(x)
        arc_h = self.arc_mlp_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc

    def embed(self, words, feats=None):
        word_embed = super().embed(words, feats)
        feat_embeds = []
        if self.args.feat is not None and 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats[-1]))
        if len(feat_embeds) > 0:
            feat_embed = torch.cat(feat_embeds, -1)
            embed_items = word_embed, feat_embed
        else:
            embed_items = word_embed,
        embed_items = self.embed_dropout(*embed_items)
        embeddings = torch.cat(embed_items, dim=-1)
        return embeddings

    def loss(self, s_arc, mask, c_span_mask=None, combine_mask=None, mbr=False, partial=True):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
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
        arc_loss, arc_probs = self.crf(s_arc, mask, c_span_mask=c_span_mask, combine_mask=combine_mask, normalizer=self.args.struct_normalizer)
        loss = arc_loss
        return loss, arc_probs

    def decode(self, s_arc, mask, c_span_mask=None, combine_mask=None):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            mbr (bool):
                If ``True``, returns marginals for MBR decoding. Default: ``True``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The predicted arcs and
                original arc scores of shape ``[batch_size, seq_len, seq_len]`` if ``mbr=False``, or marginals otherwise.
        """
        arc_preds = self.crf.decode(s_arc, mask, c_span_mask=c_span_mask, combine_mask=combine_mask)
        return arc_preds
