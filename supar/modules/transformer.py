# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
from packaging import version
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from supar.modules.scalar_mix import ScalarMix
from supar.utils.fn import pad
from transformers.models.bert import modeling_bert


class TransformerEmbedding(nn.Module):
    r"""
    A module that directly utilizes the pretrained models in `transformers`_ to produce BERT representations.
    While mainly tailored to provide input preparation and post-processing for the BERT model,
    it is also compatible with other pretrained language models like XLNet, RoBERTa and ELECTRA, etc.

    Args:
        model (str):
            Path or name of the pretrained models registered in `transformers`_, e.g., ``'bert-base-cased'``.
        n_layers (int):
            The number of BERT layers to use. If 0, uses all layers.
        n_out (int):
            The requested size of the embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        stride (int):
            A sequence longer than max length will be splitted into several small pieces
            with a window size of ``stride``. Default: 10.
        pooling (str):
            Pooling way to get from token piece embeddings to token embedding.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        pad_index (int):
            The index of the padding token in BERT vocabulary. Default: 0.
        dropout (float):
            The dropout ratio of BERT layers. Default: 0. This value will be passed into the :class:`ScalarMix` layer.
        requires_grad (bool):
            If ``True``, the model parameters will be updated together with the downstream task. Default: ``False``.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, model, n_layers, n_out=0, stride=256, pooling='mean', pad_index=0, dropout=0, requires_grad=False):
        super().__init__()

        from transformers import AutoConfig, AutoModel, AutoTokenizer
        self.bert = AutoModel.from_pretrained(model, config=AutoConfig.from_pretrained(model, output_hidden_states=True))
        self.bert = self.bert.requires_grad_(requires_grad)

        self.model = model
        self.n_layers = n_layers or self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.n_out = n_out or self.hidden_size
        self.stride = stride
        self.pooling = pooling
        self.pad_index = pad_index
        self.dropout = dropout
        self.requires_grad = requires_grad
        self.max_len = int(max(0, self.bert.config.max_position_embeddings) or 1e12) - 2

        self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.scalar_mix = ScalarMix(self.n_layers, dropout)
        self.projection = nn.Linear(self.hidden_size, self.n_out, False) if self.hidden_size != n_out else nn.Identity()

    def __repr__(self):
        s = f"{self.model}, n_layers={self.n_layers}, n_out={self.n_out}, "
        s += f"stride={self.stride}, pooling={self.pooling}, pad_index={self.pad_index}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, subwords):
        r"""
        Args:
            subwords (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.
        Returns:
            ~torch.Tensor:
                BERT embeddings of shape ``[batch_size, seq_len, n_out]``.
        """
        mask = subwords.ne(self.pad_index)
        lens = mask.sum((1, 2))
        # [batch_size, n_subwords]
        subwords = pad(subwords[mask].split(lens.tolist()), self.pad_index, padding_side=self.tokenizer.padding_side)
        bert_mask = pad(mask[mask].split(lens.tolist()), 0, padding_side=self.tokenizer.padding_side)

        # return the hidden states of all layers
        bert = self.bert(subwords[:, :self.max_len], attention_mask=bert_mask[:, :self.max_len].float())[-1]
        # [n_layers, batch_size, max_len, hidden_size]
        bert = bert[-self.n_layers:]
        # [batch_size, max_len, hidden_size]
        bert = self.scalar_mix(bert)
        # [batch_size, n_subwords, hidden_size]
        for i in range(self.stride, (subwords.shape[1]-self.max_len+self.stride-1)//self.stride*self.stride+1, self.stride):
            part = self.bert(subwords[:, i:i+self.max_len], attention_mask=bert_mask[:, i:i+self.max_len].float())[-1]
            bert = torch.cat((bert, self.scalar_mix(part[-self.n_layers:])[:, self.max_len-self.stride:]), 1)

        # [batch_size, seq_len]
        bert_lens = mask.sum(-1)
        bert_lens = bert_lens.masked_fill_(bert_lens.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        embed = bert.new_zeros(*mask.shape, self.hidden_size).masked_scatter_(mask.unsqueeze(-1), bert[bert_mask])
        # [batch_size, seq_len, hidden_size]
        if self.pooling == 'first':
            embed = embed[:, :, 0]
        elif self.pooling == 'last':
            embed = embed.gather(2, (bert_lens-1).unsqueeze(-1).repeat(1, 1, self.hidden_size).unsqueeze(2)).squeeze(2)
        else:
            embed = embed.sum(2) / bert_lens.unsqueeze(-1)
        embed = self.projection(embed)

        return embed


class CharTransformerEmbedding(nn.Module):
    r"""
    A module that directly utilizes the pretrained models in `transformers`_ to produce BERT representations.
    While mainly tailored to provide input preparation and post-processing for the BERT model,
    it is also compatiable with other pretrained language models like XLNet, RoBERTa and ELECTRA, etc.

    Args:
        model (str):
            Path or name of the pretrained models registered in `transformers`_, e.g., ``'bert-base-cased'``.
        n_layers (int):
            The number of layers from the model to use.
            If 0, uses all layers.
        n_out (int):
            The requested size of the embeddings.
            If 0, uses the size of the pretrained embedding model.
        stride (int):
            A sequence longer than the limited max length will be splitted into several small pieces
            with a window size of ``stride``. Default: 5.
        pad_index (int):
            The index of the padding token in the BERT vocabulary. Default: 0.
        dropout (float):
            The dropout ratio of BERT layers. Default: 0.
            This value will be passed into the :class:`ScalarMix` layer.
        requires_grad (bool):
            If ``True``, the model parameters will be updated together with the downstream task.
            Default: ``False``.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self, model, n_layers, n_out=0, stride=256, pad_index=0, dropout=0, requires_grad=False):
        super().__init__()

        from transformers import AutoConfig, AutoModel
        self.bert = AutoModel.from_pretrained(model, config=AutoConfig.from_pretrained(model, output_hidden_states=True))
        self.bert = self.bert.requires_grad_(requires_grad)

        self.model = model
        self.n_layers = n_layers or self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.n_out = n_out or self.hidden_size
        self.stride = stride
        self.pad_index = pad_index
        self.dropout = dropout
        self.requires_grad = requires_grad
        self.max_len = int(max(0, self.bert.config.max_position_embeddings) or 1e12) - 2

        self.scalar_mix = ScalarMix(self.n_layers, dropout)
        self.projection = nn.Linear(self.hidden_size, self.n_out, False) if self.hidden_size != n_out else nn.Identity()

    def __repr__(self):
        s = f"{self.model}, n_layers={self.n_layers}, n_out={self.n_out}, "
        s += f"stride={self.stride}, pad_index={self.pad_index}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, chars):
        r"""
        Args:
            chars (~torch.Tensor): ``[batch_size, seq_len]``.
        Returns:
            ~torch.Tensor:
                BERT embeddings of shape ``[batch_size, seq_len, n_out]``.
        """
        # [batch_size, seq_len]
        bert_mask = chars.ne(self.pad_index)
        # return the hidden states of all layers
        bert = self.bert(chars[:, :self.max_len], attention_mask=bert_mask[:, :self.max_len].float())[-1]
        # [n_layers, batch_size, seq_len, hidden_size]
        bert = bert[-self.n_layers:]
        # [batch_size, max_len, hidden_size]
        bert = self.scalar_mix(bert)
        # [batch_size, seq_len, hidden_size]
        for i in range(self.stride, (chars.shape[1]-self.max_len+self.stride-1)//self.stride*self.stride+1, self.stride):
            part = self.bert(chars[:, i:i+self.max_len], attention_mask=bert_mask[:, i:i+self.max_len].float())[-1]
            bert = torch.cat((bert, self.scalar_mix(part[-self.n_layers:])[:, self.max_len-self.stride:]), 1)

        return self.projection(bert)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) /
                        emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size)).cuda()
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = pos_embedding
        # self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        # token_embedding [batch_size, seq_len, dim]
        # pdb.set_trace()
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(1), :])


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        # if token_type_ids is None:
        #     if hasattr(self, "token_type_ids"):
        #         buffered_token_type_ids = self.token_type_ids[:, :seq_length]
        #         buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
        #         token_type_ids = buffered_token_type_ids_expanded
        #     else:
        #         token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings = inputs_embeds + token_type_embeddings

        if token_type_ids is None:
            embeddings = inputs_embeds
        else:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    @classmethod
    def from_pretrained(cls, model):
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(model)
        bert = AutoModel.from_pretrained(model)
        embedding = cls(config)
        # embedding.word_embeddings.weight.copy_(bert.embeddings.word_embeddings.weight)
        # embedding.position_embeddings.weight.copy_(bert.embeddings.position_embeddings.weight)
        # embedding.token_type_embeddings.weight.copy_(bert.embeddings.token_type_embeddings.weight)
        embedding.load_state_dict(bert.embeddings.state_dict(), strict=True)

        return embedding


class SelfAttentionEncoder(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 emb_size: int,
                 num_heads: int = 8,
                 dim_feedforward: int = 800,
                 dropout: float = 0.1):
        super(SelfAttentionEncoder, self).__init__()
        self.num_layers = num_encoder_layers
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        encoder_layer = TransformerEncoderLayer(
            d_model=emb_size, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)
        self.positional_encoding = PositionalEncoding(emb_size,
                                                      dropout=dropout)

    def forward(self, src, pad_mask):
        '''
        src: [batch_size, seq_len, dim]
        pad_mask: [batch_size, seq_len] for pad
        '''
        src_emb = self.positional_encoding(src)
        memory = self.transformer_encoder(src_emb.transpose(0, 1),
                                          mask=None,
                                          src_key_padding_mask=pad_mask)
        memory = memory.transpose(0, 1)
        return memory

    def __repr__(self):
        s = f"{self.emb_size}, num_layers={self.num_layers}, num_heads={self.num_heads}, dim_ffn={self.dim_feedforward}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"

        return f"{self.__class__.__name__}({s})"