# -*- coding: utf-8 -*-

from supar.models import CharModel
from supar.modules import MLP
from supar.utils import Config
from supar.structs.linear_crf import LinearCRFWordSegmentation
from supar.utils.logging import get_logger

import torch.nn.functional as F

logger = get_logger(__name__)


class WordSegmentationModel(CharModel):
    r"""
    Args:
        n_chars (int):
            The size of the word vocabulary.
        n_tags (int):
            The number of `bmes` tags
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
                 n_tags,
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

        self.pad_index = pad_index
        self.unk_index = unk_index

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
        raise NotImplementedError

    def loss(self, s_tag, mask):
        r"""
        Args:
            s_tag (~torch.Tensor): ``[batch_size, seq_len, n_tags]``.

        Returns:
            ~torch.Tensor:
                The training loss.
        """
        raise NotImplementedError

    def decode(self, s_tag, mask):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, n_tags]``.
                Scores of all possible arcs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """
        raise NotImplementedError


class TagWordSegmentationModel(WordSegmentationModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n_tags = kwargs.pop('n_tags', 4)
        self.tag_mlp = MLP(n_in=self.args.n_hidden, n_out=n_tags, activation=False)

        self.crf = LinearCRFWordSegmentation()

    def forward(self, chars, feats=None):
        x = self.encode(chars, feats)
        s_tag = self.tag_mlp(x)
        return s_tag

    def loss(self, s_tag, tags, mask):
        r"""
        Args:
            s_tag (~torch.Tensor): ``[batch_size, seq_len, n_tags]``.
                Scores of all possible arcs.
            tag (~torch.Tensor): ``[batch_size, seq_len]``.
                Scores of all possible arcs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss
        """
        return F.cross_entropy(s_tag[mask], tags[mask])

    def decode(self, s_tag, mask):
        r"""
        Args:
            s_tag (~torch.Tensor): ``[batch_size, seq_len, n_tags]``.
                Scores of all possible arcs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """
        # FIXME:
        pred = s_tag.argmax(-1)
        return pred


class CRFWordSegmentationModel(TagWordSegmentationModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.crf = LinearCRFWordSegmentation()

    def loss(self, s_tag, tags, mask):
        r"""
        Args:
            s_tag (~torch.Tensor): ``[batch_size, seq_len, n_tags]``.
                Scores of all possible arcs.
            tag (~torch.Tensor): ``[batch_size, seq_len]``.
                Scores of all possible arcs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss
        """
        return self.crf(s_tag, tags, mask)

    def decode(self, s_tag, mask):
        r"""
        Args:
            s_tag (~torch.Tensor): ``[batch_size, seq_len, n_tags]``.
                Scores of all possible arcs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.LongTensor, ~torch.LongTensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """
        return self.crf.viterbi(s_tag, mask)
