# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoConfig, AutoModel, AutoTokenizer

from supar.modules import (CharLSTM, CharTransformerEmbedding,
                           IndependentDropout, SharedDropout,
                           TransformerEmbedding, VariationalLSTM)
from supar.modules.transformer import BertEmbeddings
from supar.utils import Config
from supar.utils.fn import pad


class Model(nn.Module):

    def __init__(self,
                 n_words,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 encoder='lstm',
                 feat=['char'],
                 n_embed=100,
                 n_pretrained=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 char_dropout=0,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=False,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 pad_index=0,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())

        if encoder != 'bert':
            self.word_embed = nn.Embedding(num_embeddings=n_words,
                                           embedding_dim=n_embed)

            n_input = n_embed
            if n_pretrained != n_embed:
                n_input += n_pretrained
            if 'tag' in feat:
                self.tag_embed = nn.Embedding(num_embeddings=n_tags,
                                              embedding_dim=n_feat_embed)
                n_input += n_feat_embed
            if 'char' in feat:
                self.char_embed = CharLSTM(n_chars=n_chars,
                                           n_embed=n_char_embed,
                                           n_hidden=n_char_hidden,
                                           n_out=n_feat_embed,
                                           pad_index=char_pad_index,
                                           dropout=char_dropout)
                n_input += n_feat_embed
            if 'lemma' in feat:
                self.lemma_embed = nn.Embedding(num_embeddings=n_lemmas,
                                                embedding_dim=n_feat_embed)
                n_input += n_feat_embed
            if 'bert' in feat:
                self.bert_embed = TransformerEmbedding(model=bert,
                                                       n_layers=n_bert_layers,
                                                       n_out=n_feat_embed,
                                                       pooling=bert_pooling,
                                                       pad_index=bert_pad_index,
                                                       dropout=mix_dropout,
                                                       requires_grad=(not freeze))
                n_input += self.bert_embed.n_out
            self.embed_dropout = IndependentDropout(p=embed_dropout)
        if encoder == 'lstm':
            self.encoder = VariationalLSTM(input_size=n_input,
                                           hidden_size=n_lstm_hidden,
                                           num_layers=n_lstm_layers,
                                           bidirectional=True,
                                           dropout=encoder_dropout)
            self.encoder_dropout = SharedDropout(p=encoder_dropout)
            self.args.n_hidden = n_lstm_hidden * 2
        else:
            self.encoder = TransformerEmbedding(model=bert,
                                                n_layers=n_bert_layers,
                                                pooling=bert_pooling,
                                                pad_index=pad_index,
                                                dropout=mix_dropout,
                                                requires_grad=True)
            self.encoder_dropout = nn.Dropout(p=encoder_dropout)
            self.args.n_hidden = self.encoder.n_out

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed.to(self.args.device))
            if embed.shape[1] != self.args.n_pretrained:
                self.embed_proj = nn.Linear(embed.shape[1], self.args.n_pretrained).to(self.args.device)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def embed(self, words, feats):
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.args.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            pretrained = self.pretrained(words)
            if hasattr(self, 'embed_proj'):
                pretrained = self.embed_proj(pretrained)
            if self.args.n_embed == self.args.n_pretrained:
                word_embed += pretrained
            else:
                word_embed = torch.cat((word_embed, self.embed_proj(pretrained)), -1)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed, torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        return embed

    def encode(self, words, feats=None):
        if self.args.encoder == 'lstm':
            x = pack_padded_sequence(self.embed(words, feats), words.ne(self.args.pad_index).sum(1).tolist(), True, False)
            x, _ = self.encoder(x)
            x, _ = pad_packed_sequence(x, True, total_length=words.shape[1])
        else:
            x = self.encoder(words)
        return self.encoder_dropout(x)

    def decode(self):
        raise NotImplementedError


class CharModel(Model):

    def __init__(self,
                 n_chars,
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
                 bert_pad_index=0,
                 freeze=False,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 pad_index=0,
                 **kwargs):
        nn.Module.__init__(self)

        self.args = Config().update(locals())

        if encoder != 'bert':
            self.char_embed = nn.Embedding(num_embeddings=n_chars,
                                           embedding_dim=n_embed)
            n_input = n_embed
            if n_pretrained != n_embed:
                n_input += n_pretrained
            if 'bichar' in feat:
                self.bichar_embed = nn.Embedding(num_embeddings=n_bichars,
                                                 embedding_dim=n_feat_embed)
                n_input += n_feat_embed
                if n_pretrained != n_feat_embed:
                    n_input += n_pretrained
            if 'trichar' in feat:
                self.trichar_embed = nn.Embedding(num_embeddings=n_trichars,
                                                  embedding_dim=n_feat_embed)
                n_input += n_feat_embed
                if n_pretrained != n_feat_embed:
                    n_input += n_pretrained
            if 'bert' in feat:
                self.bert_embed = CharTransformerEmbedding(model=bert,
                                                           n_layers=n_bert_layers,
                                                           n_out=n_feat_embed,
                                                           pad_index=bert_pad_index,
                                                           dropout=mix_dropout,
                                                           requires_grad=(not freeze))
                n_input += self.bert_embed.n_out
            self.embed_dropout = IndependentDropout(p=embed_dropout)
        if encoder == 'lstm':
            self.encoder = VariationalLSTM(input_size=n_input,
                                           hidden_size=n_lstm_hidden,
                                           num_layers=n_lstm_layers,
                                           bidirectional=True,
                                           dropout=encoder_dropout)
            self.encoder_dropout = SharedDropout(p=encoder_dropout)
            self.args.n_hidden = n_lstm_hidden * 2
        else:
            self.encoder = CharTransformerEmbedding(model=bert,
                                                    n_layers=n_bert_layers,
                                                    pad_index=pad_index,
                                                    dropout=mix_dropout,
                                                    requires_grad=True)
            self.encoder_dropout = nn.Dropout(p=encoder_dropout)
            self.args.n_hidden = self.encoder.n_out

    def load_pretrained(self, pretrained_embed_dict=None):
        if pretrained_embed_dict is not None:
            pretrained_char_embed = pretrained_embed_dict.get('pretrained_char_embed', None)
            if pretrained_char_embed is not None:
                self.pretrained_char_embed = nn.Embedding.from_pretrained(pretrained_char_embed.to(self.args.device))
                if pretrained_char_embed.shape[1] != self.args.n_pretrained:
                    self.char_embed_proj = nn.Linear(pretrained_char_embed.shape[1], self.args.n_pretrained).to(self.args.device)
                nn.init.zeros_(self.char_embed.weight)

            pretrained_bichar_embed = pretrained_embed_dict.get('pretrained_bichar_embed', None)
            if pretrained_bichar_embed is not None:
                self.pretrained_bichar_embed = nn.Embedding.from_pretrained(pretrained_bichar_embed.to(self.args.device))
                if pretrained_bichar_embed.shape[1] != self.args.n_pretrained:
                    self.bichar_embed_proj = nn.Linear(pretrained_bichar_embed.shape[1], self.args.n_pretrained).to(self.args.device)
                nn.init.zeros_(self.bichar_embed.weight)

            pretrained_trichar_embed = pretrained_embed_dict.get('pretrained_trichar_embed', None)
            if pretrained_trichar_embed is not None:
                self.pretrained_trichar_embed = nn.Embedding.from_pretrained(pretrained_trichar_embed.to(self.args.device))
                if pretrained_trichar_embed.shape[1] != self.args.n_pretrained:
                    self.trichar_embed_proj = nn.Linear(pretrained_trichar_embed.shape[1], self.args.n_pretrained).to(self.args.device)
                nn.init.zeros_(self.trichar_embed.weight)

        return self

    def forward(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def embed(self, chars, feats):
        ext_chars = chars
        # set the indices larger than num_embeddings to unk_index
        # FIXME: Load pretrained embeddings for char-level model
        if hasattr(self, 'pretrained_char_embed'):
            ext_mask = chars.ge(self.char_embed.num_embeddings)
            ext_chars = chars.masked_fill(ext_mask, self.args.unk_index)

        # get outputs from embedding layers
        char_embed = self.char_embed(ext_chars)
        if hasattr(self, 'pretrained_char_embed'):
            pretrained_char_embed = self.pretrained_char_embed(chars)
            if hasattr(self, 'char_embed_proj'):
                pretrained_char_embed = self.char_embed_proj(pretrained_char_embed)
            if self.args.n_embed == self.args.n_pretrained:
                char_embed += pretrained_char_embed
            else:
                char_embed = torch.cat((char_embed, pretrained_char_embed), -1)

        feat_embeds = []
        if 'bichar' in self.args.feat:
            ext_bichar = feats.pop(0)
            if hasattr(self, 'pretrained_bichar_embed'):
                ext_mask = ext_bichar.ge(self.bichar_embed.num_embeddings)
                ext_bichar = ext_bichar.masked_fill(ext_mask, self.args.unk_index)
            bichar_embed = self.bichar_embed(ext_bichar)
            if hasattr(self, 'pretrained_bichar_embed'):
                pretrained_bichar_embed = self.pretrained_bichar_embed(ext_bichar)
                if hasattr(self, 'bichar_embed_proj'):
                    pretrained_bichar_embed = self.bichar_embed_proj(pretrained_bichar_embed)
                if self.args.n_feat_embed == self.args.n_pretrained:
                    bichar_embed += pretrained_bichar_embed
                else:
                    bichar_embed = torch.cat((bichar_embed, pretrained_bichar_embed), -1)
            feat_embeds.append(bichar_embed)
        if 'trichar' in self.args.feat:
            ext_trichar = feats.pop(0)
            if hasattr(self, 'pretrained_trichar_embed'):
                ext_mask = ext_trichar.ge(self.trichar_embed.num_embeddings)
                ext_trichar = ext_trichar.masked_fill(ext_mask, self.args.unk_index)
            trichar_embed = self.trichar_embed(ext_trichar)
            if hasattr(self, 'pretrained_trichar_embed'):
                pretrained_trichar_embed = self.pretrained_trichar_embed(ext_trichar)
                if hasattr(self, 'trichar_embed_proj'):
                    pretrained_trichar_embed = self.trichar_embed_proj(pretrained_trichar_embed)
                if self.args.n_feat_embed == self.args.n_pretrained:
                    trichar_embed += pretrained_trichar_embed
                else:
                    trichar_embed = torch.cat((trichar_embed, pretrained_trichar_embed), -1)
            feat_embeds.append(trichar_embed)
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        # no extra features provided
        if feat_embeds:
            char_embed, feat_embed = self.embed_dropout(char_embed, torch.cat(feat_embeds, -1))
            # concatenate the char and feat representations
            embed = torch.cat((char_embed, feat_embed), -1)
        else:
            embed = self.embed_dropout(char_embed)[0]

        return embed

    def encode(self, chars, feats=None):
        if self.args.encoder == 'lstm':
            x = pack_padded_sequence(self.embed(chars, feats), chars.ne(self.args.pad_index).sum(1).tolist(), True, False)
            x, _ = self.encoder(x)
            x, _ = pad_packed_sequence(x, True, total_length=chars.shape[1])
        else:
            x = self.encoder(chars)
        return self.encoder_dropout(x)

    def decode(self):
        raise NotImplementedError


class EmbeddingModel(nn.Module):

    def __init__(self,
                 bert=None,
                 stride=256,
                 pooling='mean',
                 freeze=False,
                 use_token_type=False,
                 pad_index=0,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        assert bert is not None, 'BERT model is required for embedding-based models.'
        self.embeddings = BertEmbeddings.from_pretrained(bert)
        self.embeddings = self.embeddings.requires_grad_(not freeze)

        self.bert_config = AutoConfig.from_pretrained(bert)
        self.tokenizer = AutoTokenizer.from_pretrained(bert)

        self.stride = stride
        self.pooling = pooling
        self.use_token_type = use_token_type
        self.hidden_size = self.bert_config.hidden_size
        self.pad_index = pad_index
        self.max_len = int(max(0, self.bert_config.max_position_embeddings) or 1e12) - 2

        self.embed_proj = nn.Linear(self.hidden_size, self.args.n_embed) if self.args.n_embed != self.hidden_size else nn.Identity()
        self.n_embed = self.args.n_embed

    def forward(self):
        raise NotImplementedError

    def embed(self, words, feats=None):
        # [batch_size, seq_len, fix_len]
        mask = words.ne(self.pad_index)
        lens = mask.sum((1, 2))
        # [batch_size, total_subwords]
        input_ids = pad(words[mask].split(lens.tolist()), self.pad_index, padding_side=self.tokenizer.padding_side)
        bert_mask = pad(mask[mask].split(lens.tolist()), 0, padding_side=self.tokenizer.padding_side)

        input_shape = input_ids.size()
        if self.use_token_type:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        else:
            token_type_ids = None

        # [batch_size, total_subwords, hidden_size]
        embeddings = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        # [batch_size, seq_len]
        bert_lens = mask.sum(-1)
        bert_lens = bert_lens.masked_fill_(bert_lens.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        embeddings = embeddings.new_zeros(*mask.shape, self.hidden_size).masked_scatter_(mask.unsqueeze(-1), embeddings[bert_mask])

        # [batch_size, seq_len, hidden_size]
        if self.pooling == 'first':
            embeddings = embeddings[:, :, 0]
        elif self.pooling == 'last':
            embeddings = embeddings.gather(2, (bert_lens-1).unsqueeze(-1).repeat(1, 1, self.hidden_size).unsqueeze(2)).squeeze(2)
        else:
            embeddings = embeddings.sum(2) / bert_lens.unsqueeze(-1)

        embeddings = self.embed_proj(embeddings)

        return embeddings

    def encode(self, words, feats=None):
        embeddings = self.embed(words, feats)
        return embeddings

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed.to(self.args.device))
            if embed.shape[1] != self.args.n_pretrained:
                self.embed_proj = nn.Linear(embed.shape[1], self.args.n_pretrained).to(self.args.device)
            nn.init.zeros_(self.word_embed.weight)
        return self
