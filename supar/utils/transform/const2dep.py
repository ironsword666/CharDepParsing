import os
from typing import List

import nltk
import torch

from supar.utils.logging import progress_bar
from supar.utils.tokenizer import Tokenizer
from supar.utils.transform.conll import CoNLL, CoNLLSentence
from supar.utils.transform.tree import Tree, TreeSentence


class UnlabeledDepTreeSentence(TreeSentence):

    def __init__(self, transform, tree, nary=True):
        super().__init__(transform, tree, nary)

        words, tags = zip(*tree.pos())
        consts = UnlabeledDepTree.factorize(UnlabeledDepTree.preprocess_tree(tree, leaf_padding=False, binarization=False, unary_chain=True)[0])
        self.values = [words, tags, tree, consts]

    def __repr__(self):
        words = self.values[0]
        heads = self.arcs
        assert len(words) == len(heads)
        n_words = len(words)
        ids = list(range(1, n_words + 1))
        lemmas = ['_'] * n_words
        cpos = pos = feats = rels = pheads = prels = lemmas
        values = [ids, words, lemmas, cpos, pos, feats, heads, rels, pheads, prels]

        return '\n'.join('\t'.join(map(str, line)) for line in zip(*values)) + '\n'


class UnlabeledDepTree(Tree):

    root = ''
    fields = ['WORD', 'POS', 'TREE', 'CONST']

    def __init__(self, WORD=None, POS=None, TREE=None, CONST=None):
        super().__init__()

        self.WORD = WORD
        self.POS = POS
        self.TREE = TREE
        self.CONST = CONST

    @property
    def Sentence(self):
        return UnlabeledDepTreeSentence

    @property
    def src(self):
        return self.WORD, self.POS, self.TREE, self.CONST

    @property
    def tgt(self):
        return ()

    @classmethod
    def preprocess_tree(cls, tree, leaf_padding=True, binarization=True, unary_chain=True):
        tree = tree.copy(True)
        if leaf_padding:
            tree = cls.pad_leaves(tree)
        if binarization:
            tree.chomsky_normal_form('left', 0, 0)
        if unary_chain:
            tree.collapse_unary(joinChar='::')

        return tree

    @classmethod
    def pad_leaves(cls, tree):
        if len(tree) == 1 and not isinstance(tree[0][0], nltk.Tree):
            tree[0] = nltk.Tree(f"{tree.label()}|<>", [tree[0]])
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, nltk.Tree):
                nodes.extend([child for child in node])
                if len(node) > 1:
                    for i, child in enumerate(node):
                        if not isinstance(child[0], nltk.Tree):
                            node[i] = nltk.Tree(f"{node.label()}|<>", [child])
        return tree

    @classmethod
    def is_single_root(cls, heads, spans):
        """
        Check whether each word is headed by a root char.

        Args:
            heads (list[int]):
                A list of dependency heads.
            spans (list[tuple]): A list of spans.
                (i, j) is a span spanning from word i+1 to j.
        """
        for span in spans:
            n_root = len(cls.trace_root(heads, span))
            assert n_root >= 1
            if n_root > 1:
                return False

        return True

    @classmethod
    def is_root_as_head(cls, heads, spans):
        """
        Check whether each word is headed by a root char.
        """
        dependents = [[] for _ in heads]
        for i, head in enumerate(heads, start=1):
            if head != 0:
                dependents[head-1].append(i)
        # print(dependents)

        # # check whether span is rooted by a single word
        # if not cls.is_single_root(heads, spans):
        #     pass

        # find the root of each span
        span_roots = [cls.trace_root(heads, span) for span in spans]
        for (b, e), roots in zip(spans, span_roots):
            for h in range(b+1, e+1):
                if h in roots:
                    continue
                for d in dependents[h-1]:
                    if d < b+1 or d > e:
                        return False

        return True

    @classmethod
    def trace_root(cls, heads, span):
        """
        Trace the root of each span.
        """
        b, e = span
        roots = [d+1 for d in range(b, e) if not b < heads[d] <= e]
        return roots

    @classmethod
    def find_span_parent(cls, spans):
        parents = []
        for i in range(len(spans)):
            if i == 0:
                parents.append(spans[0])
                continue
            b, e = spans[i]
            for pb, pe in spans[i-1::-1]:
                if b >= pb and e <= pe:
                    parents.append((pb, pe))
                    break
        return parents

    @classmethod
    def get_c_span_mask(cls, consts: List[tuple]):
        '''When the word segmentation is provided, get all possible complete spans to prevent multi-root'''
        # [(0, 37, 'S'), (0, 36, 'S|<>'), (0, 9, 'NP'), ...]
        # (i, j) is a span spanning from word i+1 to j
        seq_len = consts[0][1] + 1
        # print(seq_len)
        # if seq_len <= 15:
        #     print(consts)
        # span[i][j] is True if j->i is a complete span and head word is at j
        mask = torch.zeros((seq_len, seq_len), dtype=torch.int8)
        # (i, j) is a span spanning from word i to j-1
        spans = [(b+1, e+1) for b, e, label in consts]
        # spans = [(b+1, e+1) for b, e, label in consts if not label.endswith('|<>')]
        parents = cls.find_span_parent(spans)
        # print(spans)
        # print(parents)

        for (b, e), (pb, pe) in zip(spans, parents):
            # print("spans:")
            # print(b, e)
            # print("mask:")
            # print(mask)
            # pre_mask = mask.clone()
            ib = b+1 if b == pb else b
            ie = e-1 if e == pe else e
            mask[ib:ie, :] = 0
            # chars in the same word
            mask[b:e, b:e] = 1
            # forbid diagonal
            mask[b:e, b:e].fill_diagonal_(0)
            mask[e-1, pb:b] = 1
            mask[b, e:pe] = 1
            # print("final:")
            # print(mask)
            # print("changes:")
            # print(mask - pre_mask)
        mask[-1, 0] = 1
        # print(mask)
        return mask

    @classmethod
    def get_combine_mask(cls, consts: List[str]):
        '''Get the mask for combination operation.'''
        seq_len = consts[0][1] + 1
        # chart[i][j] == `l` if j is the head of i and j->i is labeled as `l`
        mask = [[[0] * seq_len for _ in range(seq_len)] for _ in range(seq_len)]
        spans = [(b+1, e+1) for b, e, label in consts]
        # FIXME: deal with nested spans
        for b, e in spans:
            # right incomplete span combine with left complete span
            for i in range(b, e):
                for k in range(i+1, e):
                    for j in range(e, seq_len):
                        # the wide of the span is j-i, and the complete span is (k, j)
                        mask[j-i][k][j] = 1
            # left incomplete span combine with right complete span
            for i in range(e-1, b-1, -1):
                for k in range(i-1, b-1, -1):
                    for j in range(0, b):
                        # the wide of the span is i-j, and the complete span is (j, k)
                        mask[i-j][k][j] = 1

        return mask

    def load(self, data, lang=None, max_len=None, **kwargs):
        r"""
        Args:
            data (list[list] or str):
                A list of instances or a filename.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            max_len (int):
                Sentences exceeding the length will be discarded. Default: ``None``.

        Returns:
            A list of :class:`TreeSentence` instances.
        """
        if isinstance(data, str) and os.path.exists(data):
            with open(data, 'r') as f:
                trees = [nltk.Tree.fromstring(s) for s in f]
            self.root = trees[0].label()
        else:
            if lang is not None:
                tokenizer = Tokenizer(lang)
                data = [tokenizer(i) for i in ([data] if isinstance(data, str) else data)]
            else:
                data = [data] if isinstance(data[0], str) else data
            trees = [self.totree(i, self.root) for i in data]

        i, sentences = 0, []
        nary = kwargs.get('nary', False)
        for tree in progress_bar(trees):
            sentences.append(self.Sentence(self, tree, nary=nary))
            i += 1
        if max_len is not None:
            sentences = [i for i in sentences if len(i) < max_len]

        return sentences
