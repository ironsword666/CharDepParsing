# -*- coding: utf-8 -*-

import torch


class Embedding(object):

    def __init__(self, tokens, vectors, unk=None):
        self.tokens = tokens
        self.vectors = torch.tensor(vectors)
        self.pretrained = {w: v for w, v in zip(tokens, vectors)}
        self.unk = unk

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, token):
        return token in self.pretrained

    @property
    def dim(self):
        return self.vectors.size(1)

    @property
    def unk_index(self):
        if self.unk is not None:
            return self.tokens.index(self.unk)
        else:
            raise AttributeError

    @classmethod
    def load(cls, path, unk=None):
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines]
        if len(splits[0]) == 2:
            # first line is the number of tokens and the dimension of vectors
            n_tokens, dim = map(int, splits[0])
            # handle fastNLP embedding file missing tokens
            tokens, vectors = zip(*[(s[0], list(map(float, s[1:])))
                                    for s in splits[1:] if len(s) == dim + 1])

        else:
            tokens, vectors = zip(*[(s[0], list(map(float, s[1:])))
                                    for s in splits])

        return cls(tokens, vectors, unk=unk)


class TailoredEmbedding(Embedding):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, path, vocab, unk=None):
        tokens, vectors = [], []
        dim = -1
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    # first line is the number of tokens and the dimension of vectors
                    n_tokens, dim = map(int, parts)
                    continue
                # handle fastNLP embedding file missing tokens
                if dim != - 1 and len(parts) != dim + 1:
                    continue
                token, vector = parts[0], list(map(float, parts[1:]))
                if token not in vocab:
                    continue
                tokens.append(token)
                vectors.append(vector)

        return cls(tokens, vectors, unk=unk)
