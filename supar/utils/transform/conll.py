from typing import Iterable, List

import torch
from supar.utils.common import bos, nul
from supar.utils.logging import progress_bar
from supar.utils.tokenizer import Tokenizer
from supar.utils.transform.transform import Sentence, Transform


import os


class CoNLLSentence(Sentence):
    r"""
    Sencence in CoNLL-X format.

    Args:
        transform (CoNLL):
            A :class:`~supar.utils.transform.CoNLL` object.
        lines (list[str]):
            A list of strings composing a sentence in CoNLL-X format.
            Comments and non-integer IDs are permitted.

    Examples:
        >>> lines = ['# text = But I found the location wonderful and the neighbors very kind.',
                     '1\tBut\t_\t_\t_\t_\t_\t_\t_\t_',
                     '2\tI\t_\t_\t_\t_\t_\t_\t_\t_',
                     '3\tfound\t_\t_\t_\t_\t_\t_\t_\t_',
                     '4\tthe\t_\t_\t_\t_\t_\t_\t_\t_',
                     '5\tlocation\t_\t_\t_\t_\t_\t_\t_\t_',
                     '6\twonderful\t_\t_\t_\t_\t_\t_\t_\t_',
                     '7\tand\t_\t_\t_\t_\t_\t_\t_\t_',
                     '7.1\tfound\t_\t_\t_\t_\t_\t_\t_\t_',
                     '8\tthe\t_\t_\t_\t_\t_\t_\t_\t_',
                     '9\tneighbors\t_\t_\t_\t_\t_\t_\t_\t_',
                     '10\tvery\t_\t_\t_\t_\t_\t_\t_\t_',
                     '11\tkind\t_\t_\t_\t_\t_\t_\t_\t_',
                     '12\t.\t_\t_\t_\t_\t_\t_\t_\t_']
        >>> sentence = CoNLLSentence(transform, lines)  # fields in transform are built from ptb.
        >>> sentence.arcs = [3, 3, 0, 5, 6, 3, 6, 9, 11, 11, 6, 3]
        >>> sentence.rels = ['cc', 'nsubj', 'root', 'det', 'nsubj', 'xcomp',
                             'cc', 'det', 'dep', 'advmod', 'conj', 'punct']
        >>> sentence
        # text = But I found the location wonderful and the neighbors very kind.
        1       But     _       _       _       _       3       cc      _       _
        2       I       _       _       _       _       3       nsubj   _       _
        3       found   _       _       _       _       0       root    _       _
        4       the     _       _       _       _       5       det     _       _
        5       location        _       _       _       _       6       nsubj   _       _
        6       wonderful       _       _       _       _       3       xcomp   _       _
        7       and     _       _       _       _       6       cc      _       _
        7.1     found   _       _       _       _       _       _       _       _
        8       the     _       _       _       _       9       det     _       _
        9       neighbors       _       _       _       _       11      dep     _       _
        10      very    _       _       _       _       11      advmod  _       _
        11      kind    _       _       _       _       6       conj    _       _
        12      .       _       _       _       _       3       punct   _       _
    """

    def __init__(self, transform, lines):
        Sentence.__init__(self, transform)

        self.values = []
        # record annotations for post-recovery
        self.annotations = dict()

        for i, line in enumerate(lines):
            value = line.split('\t')
            if value[0].startswith('#') or not value[0].isdigit():
                self.annotations[-i-1] = line
            else:
                self.annotations[len(self.values)] = line
                self.values.append(value)
        self.values = list(zip(*self.values))

    def __repr__(self):
        # cover the raw lines
        merged = {**self.annotations,
                  **{i: '\t'.join(map(str, line))
                     for i, line in enumerate(zip(*self.values))}}
        return '\n'.join(merged.values()) + '\n'


class CoNLL(Transform):
    r"""
    The CoNLL object holds ten fields required for CoNLL-X data format :cite:`buchholz-marsi-2006-conll`.
    Each field can be binded with one or more :class:`~supar.utils.field.Field` objects. For example,
    ``FORM`` can contain both :class:`~supar.utils.field.Field` and :class:`~supar.utils.field.SubwordField`
    to produce tensors for words and subwords.

    Attributes:
        ID:
            Token counter, starting at 1.
        FORM:
            Words in the sentence.
        LEMMA:
            Lemmas or stems (depending on the particular treebank) of words, or underscores if not available.
        CPOS:
            Coarse-grained part-of-speech tags, where the tagset depends on the treebank.
        POS:
            Fine-grained part-of-speech tags, where the tagset depends on the treebank.
        FEATS:
            Unordered set of syntactic and/or morphological features (depending on the particular treebank),
            or underscores if not available.
        HEAD:
            Heads of the tokens, which are either values of ID or zeros.
        DEPREL:
            Dependency relations to the HEAD.
        PHEAD:
            Projective heads of tokens, which are either values of ID or zeros, or underscores if not available.
        PDEPREL:
            Dependency relations to the PHEAD, or underscores if not available.
    """

    fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

    def __init__(self,
                 ID=None, FORM=None, LEMMA=None, CPOS=None, POS=None,
                 FEATS=None, HEAD=None, DEPREL=None, PHEAD=None, PDEPREL=None):
        super().__init__()

        self.ID = ID
        self.FORM = FORM
        self.LEMMA = LEMMA
        self.CPOS = CPOS
        self.POS = POS
        self.FEATS = FEATS
        self.HEAD = HEAD
        self.DEPREL = DEPREL
        self.PHEAD = PHEAD
        self.PDEPREL = PDEPREL

    @property
    def src(self):
        return self.FORM, self.LEMMA, self.CPOS, self.POS, self.FEATS

    @property
    def tgt(self):
        return self.HEAD, self.DEPREL, self.PHEAD, self.PDEPREL

    @classmethod
    def get_arcs(cls, sequence, placeholder='_'):
        return [-1 if i == placeholder else int(i) for i in sequence]

    @classmethod
    def get_sibs(cls, sequence, placeholder='_'):
        sibs = [[0] * (len(sequence) + 1) for _ in range(len(sequence) + 1)]
        heads = [0] + [-1 if i == placeholder else int(i) for i in sequence]

        for i, hi in enumerate(heads[1:], 1):
            for j, hj in enumerate(heads[i+1:], i + 1):
                di, dj = hi - i, hj - j
                if hi >= 0 and hj >= 0 and hi == hj and di * dj > 0:
                    if abs(di) > abs(dj):
                        sibs[i][hi] = j
                    else:
                        sibs[j][hj] = i
                    break
        return sibs[1:]

    @classmethod
    def get_edges(cls, sequence):
        # edges = [[0]*(len(sequence)+1) for _ in range(len(sequence)+1)]
        edges = [[0]*(len(sequence)+2) for _ in range(len(sequence)+2)]
        for i, s in enumerate(sequence, 1):
            if s != '_':
                for pair in s.split('|'):
                    edges[i][int(pair.split(':')[0])] = 1
        return edges

    @classmethod
    def get_labels(cls, sequence):
        labels = [[None]*(len(sequence)+2) for _ in range(len(sequence)+2)]
        for i, s in enumerate(sequence, 1):
            if s != '_':
                for pair in s.split('|'):
                    edge, label = pair.split(':')
                    labels[i][int(edge)] = label
        return labels

    @classmethod
    def build_relations(cls, chart):
        sequence = ['_'] * len(chart)
        for i, row in enumerate(chart):
            pairs = [(j, label) for j, label in enumerate(row) if label is not None]
            if len(pairs) > 0:
                sequence[i] = '|'.join(f"{head}:{label}" for head, label in pairs)
        return sequence

    @classmethod
    def toconll(cls, tokens):
        r"""
        Converts a list of tokens to a string in CoNLL-X format.
        Missing fields are filled with underscores.

        Args:
            tokens (list[str] or list[tuple]):
                This can be either a list of words, word/pos pairs or word/lemma/pos triples.

        Returns:
            A string in CoNLL-X format.

        Examples:
            >>> print(CoNLL.toconll(['She', 'enjoys', 'playing', 'tennis', '.']))
            1       She     _       _       _       _       _       _       _       _
            2       enjoys  _       _       _       _       _       _       _       _
            3       playing _       _       _       _       _       _       _       _
            4       tennis  _       _       _       _       _       _       _       _
            5       .       _       _       _       _       _       _       _       _

            >>> print(CoNLL.toconll([('She',     'she',    'PRP'),
                                     ('enjoys',  'enjoy',  'VBZ'),
                                     ('playing', 'play',   'VBG'),
                                     ('tennis',  'tennis', 'NN'),
                                     ('.',       '_',      '.')]))
            1       She     she     PRP     _       _       _       _       _       _
            2       enjoys  enjoy   VBZ     _       _       _       _       _       _
            3       playing play    VBG     _       _       _       _       _       _
            4       tennis  tennis  NN      _       _       _       _       _       _
            5       .       _       .       _       _       _       _       _       _

        """

        if isinstance(tokens[0], str):
            s = '\n'.join([f"{i}\t{word}\t" + '\t'.join(['_']*8)
                           for i, word in enumerate(tokens, 1)])
        elif len(tokens[0]) == 2:
            s = '\n'.join([f"{i}\t{word}\t_\t{tag}\t" + '\t'.join(['_']*6)
                           for i, (word, tag) in enumerate(tokens, 1)])
        elif len(tokens[0]) == 3:
            s = '\n'.join([f"{i}\t{word}\t{lemma}\t{tag}\t" + '\t'.join(['_']*6)
                           for i, (word, lemma, tag) in enumerate(tokens, 1)])
        else:
            raise RuntimeError(f"Invalid sequence {tokens}. Only list of str or list of word/pos/lemma tuples are support.")
        return s + '\n'

    @classmethod
    def isprojective(cls, sequence):
        r"""
        Checks if a dependency tree is projective.
        This also works for partial annotation.

        Besides the obvious crossing arcs, the examples below illustrate two non-projective cases
        which are hard to detect in the scenario of partial annotation.

        Args:
            sequence (list[int]):
                A list of head indices.

        Returns:
            ``True`` if the tree is projective, ``False`` otherwise.

        Examples:
            >>> CoNLL.isprojective([2, -1, 1])  # -1 denotes un-annotated cases
            False
            >>> CoNLL.isprojective([3, -1, 2])
            False
        """

        pairs = [(h, d) for d, h in enumerate(sequence, 1) if h >= 0]
        for i, (hi, di) in enumerate(pairs):
            for hj, dj in pairs[i+1:]:
                (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
                if li <= hj <= ri and hi == dj:
                    return False
                if lj <= hi <= rj and hj == di:
                    return False
                if (li < lj < ri or li < rj < ri) and (li - lj)*(ri - rj) > 0:
                    return False
        return True

    @classmethod
    def istree(cls, sequence, proj=False, multiroot=False):
        r"""
        Checks if the arcs form an valid dependency tree.

        Args:
            sequence (list[int]):
                A list of head indices.
            proj (bool):
                If ``True``, requires the tree to be projective. Default: ``False``.
            multiroot (bool):
                If ``False``, requires the tree to contain only a single root. Default: ``True``.

        Returns:
            ``True`` if the arcs form an valid tree, ``False`` otherwise.

        Examples:
            >>> CoNLL.istree([3, 0, 0, 3], multiroot=True)
            True
            >>> CoNLL.istree([3, 0, 0, 3], proj=True)
            False
        """

        from supar.utils.alg import tarjan
        if proj and not cls.isprojective(sequence):
            return False
        n_roots = sum(head == 0 for head in sequence)
        if n_roots == 0:
            return False
        if not multiroot and n_roots > 1:
            return False
        if any(i == head for i, head in enumerate(sequence, 1)):
            return False
        return next(tarjan(sequence), None) is None

    def load(self, data, lang=None, proj=False, max_len=None, **kwargs):
        r"""
        Loads the data in CoNLL-X format.
        Also supports for loading data from CoNLL-U file with comments and non-integer IDs.

        Args:
            data (list[list] or str):
                A list of instances or a filename.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            proj (bool):
                If ``True``, discards all non-projective sentences. Default: ``False``.
            max_len (int):
                Sentences exceeding the length will be discarded. Default: ``None``.

        Returns:
            A list of :class:`CoNLLSentence` instances.
        """

        if isinstance(data, str) and os.path.exists(data):
            with open(data, 'r') as f:
                lines = [line.strip() for line in f]
        else:
            if lang is not None:
                tokenizer = Tokenizer(lang)
                data = [tokenizer(i) for i in ([data] if isinstance(data, str) else data)]
            else:
                data = [data] if isinstance(data[0], str) else data
            lines = '\n'.join([self.toconll(i) for i in data]).split('\n')

        i, start, sentences = 0, 0, []
        for line in progress_bar(lines):
            if not line:
                sentences.append(CoNLLSentence(self, lines[start:i]))
                start = i + 1
            i += 1
        if proj:
            sentences = [i for i in sentences if self.isprojective(list(map(int, i.arcs)))]
        if max_len is not None:
            sentences = [i for i in sentences if len(i) < max_len]

        return sentences


class CharCoNLLSentence(CoNLLSentence):

    def __init__(self, transform, lines):
        super().__init__(transform, lines)

    def __repr__(self):
        id = getattr(self, 'id')
        word = getattr(self, 'word')
        head = getattr(self, 'head')
        dep = getattr(self, 'dep')
        intra_word_struct = getattr(self, 'intra_word_struct')
        intra_word_labels = getattr(self, 'intra_word_labels')
        underlines = ['_' for i in range(len(id))]

        columns = [id, word, underlines, underlines, underlines, underlines, head, dep, intra_word_struct, intra_word_labels]
        lines = ['\t'.join(map(str, line)) for line in zip(*columns)]
        # cover the raw lines
        return '\n'.join(lines) + '\n'


class CharCoNLL(CoNLL):

    fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def Sentence(self):
        return CharCoNLLSentence

    @classmethod
    def word2span(cls, words):
        """
        Get the span boundaries of each word in the sentence.
        """
        sents = []
        for sent in words:
            i = 1
            spans = []
            for word in sent:
                spans.append((i, i + len(word)))
                i += len(word)
            spans = [(0, 1)] + spans
            sents.append(spans)
        return sents

    @classmethod
    def word2char(cls, words: Iterable, word_heads: Iterable, rels: Iterable):
        raise NotImplementedError

    @classmethod
    def check_conflict(cls, arc_pred, rel_pred, intra_word_rel_indexes=[0]):
        """Check whether there is an intra-word dependency below an inter-word dependency."""
        intra_word_rel_indexes = list(intra_word_rel_indexes)
        n_conflict = 0
        dependencies = dict()
        # extract all dependencies
        for modifier, (head, rel) in enumerate(zip(arc_pred, rel_pred), start=1):
            dependencies[(head, modifier)] = rel
        for (head, modifier), rel in dependencies.items():
            # select an intra-word dependency
            if rel not in intra_word_rel_indexes:
                continue
            begin, end = head, modifier
            if head > modifier:
                begin, end = modifier, head
            # find an inter-word dependency which is below the intra-word dependency
            for i in range(begin, end+1):
                for j in range(begin, end+1):
                    if dependencies.get((i, j), intra_word_rel_indexes[0]) not in intra_word_rel_indexes:
                        # print(f"Conflict: an inter-word dependency {i} -> {j} ({dependencies[(i, j)]}) below an intra-word dependency {head} -> {modifier} ({rel})")
                        n_conflict += 1

        # if n_conflict:
        #     print('arc_pred: ', arc_pred)
        #     print('rel_pred: ', rel_pred)
        #     print('---------------------')

        return n_conflict

    @classmethod
    def check_multi_root(cls, arc_pred, rel_pred, intra_word_rel_indexes=[0]):
        '''
        Check whether each word is headed by a root char.

        Return:
            [((begin of modifier, end of modifier), (begin of head, end of head), relation), ...]
        '''
        roots = cls.root_trace(arc_pred, rel_pred, intra_word_rel_indexes)
        # chunk the chars based on the root indices
        headed_spans = cls.chunk(roots)
        # add $root
        headed_spans = [(0, 1, 0)] + headed_spans

        # map each char to the corresponding word
        char2word = [0] + [0 for _ in arc_pred]
        for i, (begin, end, _) in enumerate(headed_spans):
            for j in range(begin, end):
                char2word[j] = i
        # find the head word of each word
        for (begin, end, root) in headed_spans[1:]:
            # head char index of root
            head_char = arc_pred[root-1]
            # the word index of the parent char
            head_word_id = char2word[head_char]
            head_word_begin, head_word_end, head_word_root = headed_spans[head_word_id]
            if head_char != head_word_root:
                print((begin, end, root))
                print(head_char)
                print((head_word_begin, head_word_end, head_word_root))
                # print(headed_spans)

    @classmethod
    def check_root_as_head(cls, arc_pred, rel_pred, intra_word_rel_indexes=[0]):
        '''
        Check whether each word is headed by a root char.

        Return:
            [((begin of modifier, end of modifier), (begin of head, end of head), relation), ...]
        '''
        roots = cls.root_trace(arc_pred, rel_pred, intra_word_rel_indexes)
        # chunk the chars based on the root indices
        headed_spans = cls.chunk(roots)
        # add $root
        headed_spans = [(0, 1, 0)] + headed_spans

        # map each char to the corresponding word
        char2word = [0] + [0 for _ in arc_pred]
        for i, (begin, end, _) in enumerate(headed_spans):
            for j in range(begin, end):
                char2word[j] = i
        # find the head word of each word
        for (begin, end, root) in headed_spans[1:]:
            # head char index of root
            head_char = arc_pred[root-1]
            # the word index of the parent char
            head_word_id = char2word[head_char]
            head_word_begin, head_word_end, head_word_root = headed_spans[head_word_id]
            if head_char != head_word_root:
                print((begin, end, root))
                print(head_char)
                print((head_word_begin, head_word_end, head_word_root))
                # print(headed_spans)

    @classmethod
    def recover_words(cls, arc_pred, rel_pred, intra_word_rel_indexes=[0], headness=False):
        '''
        Recover the char-level tree to word-level tree

        Args:
            headness: only find the root char of each word, no need to find the head word

        Return:
            [((begin of modifier, end of modifier), (begin of head, end of head), relation), ...]
        '''
        # check_conflict()
        roots = cls.root_trace(arc_pred, rel_pred, intra_word_rel_indexes)
        # chunk the chars based on the root indices
        headed_spans = cls.chunk(roots)
        if headness:
            return [((begin, end, root), [arc_pred[i-1][0] for i in range(begin, end)]) for (begin, end, root) in headed_spans]
            # return headed_spans
        # add $root
        headed_spans = [(0, 1, 0)] + headed_spans

        # map each char to the corresponding word
        char2word = [0] + [0 for _ in arc_pred]
        for i, (begin, end, _) in enumerate(headed_spans):
            for j in range(begin, end):
                char2word[j] = i
        # find the head word of each word
        dependencies = []
        for (begin, end, root) in headed_spans[1:]:
            # head char index of root
            head_char = arc_pred[root-1]
            rel = rel_pred[root-1]
            # the word index of the parent char
            head_word = char2word[head_char]
            word = headed_spans[head_word][:2]
            # modifier (b, e) <- head word w(i, j) with relation rel
            dependencies.append(((begin, end), word, rel))

        return dependencies

    @classmethod
    def root_trace(cls, arc_pred, rel_pred, intra_word_rel_indexes=[0]):
        """
        Find the root char (of the word) for each char

        Args:
            arc_pred (list): _description_
            rel_pred (list): _description_
            intra_word_rel_indexes (list, optional): _description_. Defaults to [0].

        Returns:
            _type_: _description_
        """
        roots = []
        pairs = list(zip(arc_pred, rel_pred))
        # find the root of the subtree
        for i, (arc, rel) in enumerate(pairs, start=1):
            # any arc to the root shouldn't be labeled as intra-word relation
            if arc == 0 and rel in intra_word_rel_indexes:
                raise ValueError(f"Invalid ! arc_pred: {arc_pred}\n rel_pred: {rel_pred}\n intra_word_rel: {intra_word_rel_indexes}")
            head = i
            # until find the inter-word dependency
            while rel in intra_word_rel_indexes:
                head = arc
                arc, rel = pairs[arc-1]
            roots.append(head)
        return roots

    @classmethod
    def chunk(cls, seq: list) -> list:
        '''Chunk the sequence by the head index.
        Returns:
            list[tuple]: (begin, end, head)
        '''
        if not seq:
            raise ValueError('empty sequence')
        spans = []
        prev = seq[0]
        b = 1
        for i, item in enumerate(seq, 1):
            if item != prev:
                spans.append((b, i, prev))
                b = i
                prev = item
        spans.append((b, len(seq)+1, item))

        return spans

    @classmethod
    def get_arcs(cls, sequence: List[str]):
        raise NotImplementedError

    @classmethod
    def get_rels(cls, sequence: List[str]):
        raise NotImplementedError

    @classmethod
    def toconll(cls, tokens):
        raise NotImplementedError

    def load(self, data, lang=None, proj=False, max_len=None, **kwargs):
        r"""
        Loads the data in CoNLL-X format.
        Also supports for loading data from CoNLL-U file with comments and non-integer IDs.

        Args:
            data (list[list] or str):
                A list of instances or a filename.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            proj (bool):
                If ``True``, discards all non-projective sentences. Default: ``False``.
            max_len (int):
                Sentences exceeding the length will be discarded. Default: ``None``.

        Returns:
            A list of :class:`CoNLLSentence` instances.
        """
        if isinstance(data, str) and os.path.exists(data):
            with open(data, 'r') as f:
                lines = [line.strip() for line in f]
        else:
            if lang is not None:
                tokenizer = Tokenizer(lang)
                data = [tokenizer(i) for i in ([data] if isinstance(data, str) else data)]
            else:
                data = [data] if isinstance(data[0], str) else data
            lines = '\n'.join([self.toconll(i) for i in data]).split('\n')

        i, start, sentences = 0, 0, []
        for line in progress_bar(lines):
            if not line:
                sentences.append(self.Sentence(self, lines[start:i]))
                start = i + 1
            i += 1
        if proj:
            sentences = [i for i in sentences if self.isprojective(list(map(int, i.heads)))]
        if max_len is not None:
            sentences = [i for i in sentences if len(i) < max_len]

        return sentences


class ExplicitCharCoNLLSentence(CoNLLSentence):

    def __init__(self, transform, lines):
        super().__init__(transform, lines)

        chars, heads, rels = transform.word2char(self.values[1], self.values[6], self.values[7], transform.orientation)
        # store words to `LEMMA` field
        self.values[2] = self.values[1]
        # store chars to `FORM` field
        self.values[1] = chars
        # store char heads to `PHAED` field
        self.values[-2] = heads
        # store char rels to `PDEPREL` field
        self.values[-1] = rels
    
    def __repr__(self):
        id = getattr(self, 'id')
        word = getattr(self, 'word')
        head = getattr(self, 'head')
        dep = getattr(self, 'dep')
        intra_word_struct = getattr(self, 'intra_word_struct')
        intra_word_labels = getattr(self, 'intra_word_labels')
        underlines = ['_' for i in range(len(id))]

        columns = [id, word, underlines, underlines, underlines, underlines, head, dep, intra_word_struct, intra_word_labels]
        lines = ['\t'.join(map(str, line)) for line in zip(*columns)]
        # cover the raw lines
        return '\n'.join(lines) + '\n'


class ExplicitCharCoNLL(CharCoNLL):

    fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

    def __init__(self, orientation, **kwargs):
        super().__init__(**kwargs)
        self._orientation = orientation

    @property
    def orientation(self):
        return self._orientation

    @property
    def Sentence(self):
        return ExplicitCharCoNLLSentence

    @classmethod
    def word2char(cls, words: Iterable, word_heads: Iterable, rels: Iterable, orientation='leftward'):
        """
        Transform a word-level dep tree to char-level dep tree.
        Args:
            words: ['戴相龙', '说', '中国', '经济', '发展', '为', '亚洲', '作出', '积极', '贡献']
            words_heads: ['2', '0', '5', '5', '8', '8', '6', '2', '10', '8']
            rels: ['VMOD', 'ROOT', 'NMOD', 'NMOD', 'VMOD', 'VMOD', 'PMOD', 'VMOD', 'NMOD', 'VMOD']
            orientation (str): 'leftward' or 'rightward'
                leftward: A<-B<-C
                rightward: A->B->C
        Returns:
            chars: ['戴', '相', '龙', '说', '中', '国', '经', '济', '发', '展', '为', '亚', '洲', '作', '出', '积', '极', '贡', '献']
            char_heads: [2, 3, 4, 0, 6, 10, 8, 10, 10, 15, 15, 13, 11, 15, 4, 17, 19, 19, 15]
            char_rels: ['<nul>', '<nul>', 'VMOD', 'ROOT', '<nul>', 'NMOD', '<nul>', 'NMOD', '<nul>', 'VMOD', 'VMOD', '<nul>', 'PMOD', '<nul>', 'VMOD', '<nul>', 'NMOD', '<nul>', 'VMOD']
        """
        # add the root
        words = [[bos]] + [list(word) for word in words]
        word_heads = [-1] + [int(i) for i in word_heads]
        rels = [nul] + [rel for rel in rels]
        # get chars
        chars = [char for word in words for char in word]
        # transform the word to a char span
        spans, p = [], 0
        for word in words:
            spans.append((p, p+len(word)))
            p += len(word)

        # extract head and rel for each char
        char_heads, char_rels = [], []
        for i, (b, e) in enumerate(spans):
            # head word and rel of the current word
            word_head, rel = word_heads[i], rels[i]
            # word span corresponding to the head word
            head_span = spans[word_head]
            if orientation == 'leftward':
                char_head = head_span[1]-1
                #  A<-B
                for j in range(b, e-1):
                    char_heads.append(j+1)
                    char_rels.append(nul)
                char_heads.append(char_head)
                char_rels.append(rel)
            elif orientation == 'rightward':
                char_head = head_span[0]
                char_heads.append(char_head)
                char_rels.append(rel)
                # A->B
                for j in range(b+1, e):
                    char_heads.append(j-1)
                    char_rels.append(nul)

        return chars[1:], char_heads[1:], char_rels[1:]

    @classmethod
    def get_arcs(cls, sequence: List[str]) -> List[int]:
        return [int(i) for i in sequence]


class LatentCharCoNLLSentence(CharCoNLLSentence):

    def __init__(self, transform, lines):
        super().__init__(transform, lines)

        chars, labels = transform.word2char(self.values[1], self.values[6], self.values[7])
        # store words to `LEMMA` field
        self.values[2] = self.values[1]
        # store chars to `FORM` field
        self.values[1] = chars
        # store labels to `PHAED` field
        self.values[-2] = labels


class LatentCharCoNLL(CharCoNLL):

    fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def Sentence(self):
        return LatentCharCoNLLSentence

    @classmethod
    def build_char_level_relations(cls, words, heads, rels):
        """
        Build spans for each word.
        Args:
            words: ['戴相龙', '说', '中国', '经济', '发展', '为', '亚洲', '作出', '积极', '贡献']
            heads: ['2', '0', '5', '5', '8', '8', '6', '2', '10', '8']
            rels: a matrix whose nrow and ncol are character numbers
        Returns:
            spans: [(0, 3), (3, 4), (4, 6), (6, 8), (8, 10), (10, 11), (11, 13), (13, 15), (15, 17), (17, 19)]
        """
        heads = [-1] + [int(i) for i in heads]
        rels = [-1] + [int(i) for i in rels]
        spans, p = [(0, 1)], 1
        for word in words:
            spans.append((p, p+len(word)))
            p += len(word)
        relations = []
        for i, (ml, mr) in enumerate(spans):
            word_head = heads[i]
            (hl, hr) = spans[word_head]
            rel = rels[i]
            # relations.append(((ml, mr), (hl, hr), rels[ml][hl]))
            relations.append(((ml, mr), (hl, hr), rel))

        flag = False
        for (b, e), (hb, he), rel in relations[1:]:
            if rel == -1:
                flag = True
        if flag:
            print(words)
            print(heads)
            for rel in rels:
                print(rel)
            print(relations[1:])

        return relations[1:]


    @classmethod
    def build_relations(cls, words, heads, rels):
        """
        Build spans for each word.
        Args:
            words: ['戴相龙', '说', '中国', '经济', '发展', '为', '亚洲', '作出', '积极', '贡献']
            heads: ['2', '0', '5', '5', '8', '8', '6', '2', '10', '8']
            rels: a matrix whose nrow and ncol are character numbers
        Returns:
            spans: [(0, 3), (3, 4), (4, 6), (6, 8), (8, 10), (10, 11), (11, 13), (13, 15), (15, 17), (17, 19)]
        """
        heads = [-1] + [int(i) for i in heads]
        spans, p = [(0, 1)], 1
        for word in words:
            spans.append((p, p+len(word)))
            p += len(word)
        relations = []
        for i, (ml, mr) in enumerate(spans):
            word_head = heads[i]
            (hl, hr) = spans[word_head]
            if i == 0:
                rel = -1
            # for mixed coarse2fine where rels[ml][hl] can't find the correct rel
            for m in range(ml, mr):
                for h in range(hl, hr):
                    if rels[m][h] != -1:
                        rel = rels[m][h]
            # relations.append(((ml, mr), (hl, hr), rels[ml][hl]))
            relations.append(((ml, mr), (hl, hr), rel))

        flag = False
        for (b, e), (hb, he), rel in relations[1:]:
            if rel == -1:
                flag = True
        if flag:
            print(words)
            print(heads)
            for rel in rels:
                print(rel)
            print(relations[1:])

        return relations[1:]

    @classmethod
    def word2char(cls, words: Iterable, word_heads: Iterable, rels: Iterable):
        """
        Transform a word-level dep tree to char-level dep tree.
        Args:
            words: ['戴相龙', '说', '中国', '经济', '发展', '为', '亚洲', '作出', '积极', '贡献']
            words_heads: ['2', '0', '5', '5', '8', '8', '6', '2', '10', '8']
            rels: ['VMOD', 'ROOT', 'NMOD', 'NMOD', 'VMOD', 'VMOD', 'PMOD', 'VMOD', 'NMOD', 'VMOD']
        Returns:
            chars: ['戴', '相', '龙', '说', '中', '国', '经', '济', '发', '展', '为', '亚', '洲', '作', '出', '积', '极', '贡', '献']
            labels: ['4:B-VMOD', '4:I-VMOD', '4:I-VMOD', '0:B-ROOT', '9:B-NMOD', '9:I-NMOD', '9:B-NMOD', '9:I-NMOD', '14:B-VMOD', '14:I-VMOD', '14:B-VMOD', '11:B-PMOD', '11:I-PMOD', '4:B-VMOD', '4:I-VMOD', '18:B-NMOD', '18:I-NMOD', '14:B-VMOD', '14:I-VMOD']
        """
        words = [[bos]] + [list(word) for word in words]
        word_heads = [-1] + [int(i) for i in word_heads]
        rels = [nul] + [rel for rel in rels]
        # chars = [f'B-{char}' if i == 0 else f'I-{char}' for word in words for i, char in enumerate(word)]
        chars = [char for word in words for char in word]
        spans, p = [], 0
        for word in words:
            spans.append((p, p+len(word)))
            p += len(word)

        labels = []
        for i, (b, e) in enumerate(spans):
            word_head, rel = word_heads[i], rels[i]
            head = spans[word_head][0]
            labels.append(f'{head}:B-{rel}')
            for j in range(b+1, e):
                labels.append(f'{head}:I-{rel}')

        return chars[1:], labels[1:]

    @classmethod
    def factorize(cls, tags):
        """
        Args:
            tags (list[str]): A tag is like `0:B-OBJ` or `3:I-OBJ`.

        Returns:
            spans (list[list[int]]): A span is like `[0, 3, 2, 'OBJ']`.
        """
        spans = []
        for i, tag in enumerate(tags, 1):
            head, tag = tag.split(':')
            if tag.startswith('B'):
                spans.append([i, i+1, int(head), tag[2:]])
            else:
                spans[-1][1] += 1
        return spans

    @classmethod
    def get_spans(cls, tags):
        spans = []
        for i, tag in enumerate(tags, 1):
            head, tag = tag.split(':')
            if tag.startswith('B'):
                spans.append([i, i+1])
            else:
                spans[-1][1] += 1
        return spans

    @classmethod
    def get_arcs(cls, sequence: List[str]) -> List[List[bool]]:
        '''Get the mask chart for all possible arcs.
        Args:
            sequence (list[str]): An element is like `0:B-OBJ` or `3:I-OBJ`.
        '''
        seq_len = len(sequence) + 1
        # chart[i][j] is True if j is the head of i
        chart = [[0] * seq_len for _ in range(seq_len)]
        spans = [[0, 1, -1, nul]] + cls.factorize(sequence)
        # add the root and map the char index to word index
        char2word = [i for i, span in enumerate(spans) for _ in range(span[0], span[1])]
        for start, end, head, rel in spans[1:]:
            parent_start, parent_end, _, _ = spans[char2word[head]]
            for i in range(start, end):
                # chars in the same word
                for j in range(start, end):
                    if i == j:
                        continue
                    chart[i][j] = 1
                # chars in the parent word
                for j in range(parent_start, parent_end):
                    chart[i][j] = 1

        return chart

    @classmethod
    def get_rels(cls, sequence: List[str]) -> List[List[int]]:
        '''Get the label chart for all possible arcs.'''
        seq_len = len(sequence) + 1
        # chart[i][j] == `l` if j is the head of i and j->i is labeled as `l`
        chart = [[None] * seq_len for _ in range(seq_len)]
        spans = [[0, 1, -1, nul]] + cls.factorize(sequence)
        # map the char index to word index
        char2word = [i for i, span in enumerate(spans) for _ in range(span[0], span[1])]
        for start, end, head, rel in spans[1:]:
            parent_start, parent_end, _, _ = spans[char2word[head]]
            for i in range(start, end):
                # chars in the same word
                for j in range(start, end):
                    if i == j:
                        continue
                    chart[i][j] = nul
                # chars in the parent word
                for j in range(parent_start, parent_end):
                    chart[i][j] = rel

        return chart

    # @classmethod
    # def get_span_constraint(cls, sequence: List[str]):
    #     '''When the word segmentation is provided, get all possible complete spans to prevent multi-root'''
    #     seq_len = len(sequence) + 1
    #     # span[i][j] is True if j->i is a complete span and head word is at j
    #     # intra-words
    #     intra_span = [[0] * seq_len for _ in range(seq_len)]
    #     # inter-words
    #     inter_span = [[0] * seq_len for _ in range(seq_len)]
    #     spans = [[0, 1]] + cls.get_spans(sequence)
    #     for start, end in spans[1:]:
    #         for i in range(start, end):
    #             # chars in the same word
    #             for j in range(start, end):
    #                 if i == j:
    #                     continue
    #                 intra_span[i][j] = 1
    #         for i in range(1, start):
    #             inter_span[end-1][i] = 1
    #         for i in range(end, seq_len):
    #             inter_span[start][i] = 1

    #     # 0->n is a from the root to the last token
    #     inter_span[-1][0] = 1

    #     return [intra_span, inter_span]

    @classmethod
    def get_span_constraint(cls, sequence: List[str]):
        '''When the word segmentation is provided, get all possible complete spans to prevent multi-root'''
        seq_len = len(sequence) + 1
        # span[i][j] is True if j->i is a complete span and head word is at j
        # intra-words
        intra_span = torch.zeros((seq_len, seq_len), dtype=torch.int8)
        # inter-words
        inter_span = torch.zeros((seq_len, seq_len), dtype=torch.int8)
        spans = [[0, 1]] + cls.get_spans(sequence)
        intra_span[0, 0] = 1
        for start, end in spans[1:]:
            # chars in the same word
            intra_span[start:end, start:end] = 1
            # forbid diagonal
            # # NOTE: complete span (i,i) is allowed in inside, and scores are set to 0
            # intra_span[start:end, start:end].fill_diagonal_(0)
            inter_span[end-1, 1:start] = 1
            inter_span[start, end:seq_len] = 1

        # 0->n is a span from the root to the last token
        inter_span[-1, 0] = 1

        # return [intra_span.tolist(), inter_span.tolist()]
        return [intra_span, inter_span]

    @classmethod
    def get_combine_mask(cls, sequence: List[str]):
        '''Get the mask for combination operation.'''
        seq_len = len(sequence) + 1
        # chart[i][j] == `l` if j is the head of i and j->i is labeled as `l`
        mask = [[[0] * seq_len for _ in range(seq_len)] for _ in range(seq_len)]
        spans = [[0, 1]] + cls.get_spans(sequence)
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


class MixedCharCoNLLSentence(CharCoNLLSentence):

    def __init__(self, transform, lines):
        super().__init__(transform, lines)

        chars = transform.word2char(self.values[1])
        labels = transform.build_char_dependency(self.values[6], self.values[7], self.values[8], self.values[9])
        # store words to `LEMMA` field
        self.values[2] = self.values[1]
        # store chars to `FORM` field
        self.values[1] = chars
        # store labels to `PHAED` field
        self.values[8] = labels


class MixedCharCoNLL(LatentCharCoNLL):

    fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def Sentence(self):
        return MixedCharCoNLLSentence

    @classmethod
    def word2char(cls, words: Iterable):
        """
        Transform a word sequence to a char sequence.
        Args:
            words: ['戴相龙', '说', '中国', '经济', '发展', '为', '亚洲', '作出', '积极', '贡献']
        Returns:
            chars: ['戴', '相', '龙', '说', '中', '国', '经', '济', '发', '展', '为', '亚', '洲', '作', '出', '积', '极', '贡', '献']
        """
        return [char for word in words for char in list(word)]

    @classmethod
    def build_char_dependency(cls, word_heads: Iterable, word_rels: Iterable, char_heads: Iterable, char_rels: Iterable):
        """
        Transform a word-level dep tree to char-level dep tree.
        Args:
            words_heads: ['2', '0', '5', '5', '8', '8', '6', '2', '10', '8']
            word_rels: ['VMOD', 'ROOT', 'NMOD', 'NMOD', 'VMOD', 'VMOD', 'PMOD', 'VMOD', 'NMOD', 'VMOD']
            char_heads: if the intra-word structure is not provided, the head of each char is `-1`
                ['-1_-1_-1', '0', '1_0', '1_0', '1_0', '0', '1_0', '1_0', '1_0', '1_0']
            char_rels: if the intra-word structure is not provided, the rel of each char is `nul`
                ['nul_nul_nul', ...]
        Returns:
            labels: char-head:B-0/1-rel, char-head:I-rel, use B-0/B-1 indicate whether word has word structure
                ['2:B-0-VMOD', '2:I-VMOD', ...]
        """
        word_heads = [-1] + [int(i) for i in word_heads]
        word_rels = [nul] + [rel for rel in word_rels]
        char_heads = [[-1]] + [[int(i) for i in heads.split('_')] for heads in char_heads]
        char_rels = [[nul]] + [['c-'+rel for rel in rels.split('_')] for rels in char_rels]
        spans, p = [], 0
        for heads in char_heads:
            spans.append((p, p+len(heads)))
            p += len(heads)

        labels = []
        for i, (b, e) in enumerate(spans):
            word_head, word_rel = word_heads[i], word_rels[i]
            char_head, char_rel = char_heads[i], char_rels[i]
            specified = 0 if -1 in char_head else 1

            # find the head char of parent word
            parent_start = spans[word_head][0]
            parent_head = parent_start
            for j, h in enumerate(char_heads[word_head]):
                if h == 0:
                    parent_head = parent_start + j

            for j in range(b, e):
                h = char_head[j-b]
                if h == -1 or h == 0:
                    head = parent_head
                    rel = word_rel
                else:
                    head = b+h-1
                    rel = char_rel[j-b]
                if j == b:
                    labels.append(f'{head}:B-{specified}-{rel}')
                else:
                    labels.append(f'{head}:I-{rel}')

        return labels[1:]

    @classmethod
    def build_relations(cls, words, heads, rels):
        """
        Build spans for each word.
        Args:
            words: ['戴相龙', '说', '中国', '经济', '发展', '为', '亚洲', '作出', '积极', '贡献']
            heads: ['2', '0', '5', '5', '8', '8', '6', '2', '10', '8']
            rels: ['VMOD', 'ROOT', 'NMOD', 'NMOD', 'VMOD', 'VMOD', 'PMOD', 'VMOD', 'NMOD', 'VMOD']
        Returns:
            spans: [(0, 3), (3, 4), (4, 6), (6, 8), (8, 10), (10, 11), (11, 13), (13, 15), (15, 17), (17, 19)]
        """
        # # NOTE: used for leftward or rightward where rels is a list rather than Mixed where rels is a chart
        # raise NotImplementedError
        heads = [-1] + [int(i) for i in heads]
        rels = [-1] + rels
        spans, p = [(0, 1)], 1
        for word in words:
            spans.append((p, p+len(word)))
            p += len(word)
        relations = []
        for i, (ml, mr) in enumerate(spans):
            word_head = heads[i]
            (hl, hr) = spans[word_head]
            relations.append(((ml, mr), (hl, hr), rels[i]))

        return relations[1:]

    @classmethod
    def factorize(cls, tags):
        """
        Args:
            tags (list[str]): A tag is like '2:B-0-VMOD', '2:I-VMOD'.

        Returns:
            spans (list[list[int]]): A span is like `[0, 3, [2, 2, 2], ['OBJ', 'OBJ', 'OBJ'], [-1, -1, -1], [nul, nul, nul]]`.
        """
        spans = []
        for i, tag in enumerate(tags, 1):
            head, rel = tag.split(':')
            head = int(head)
            if rel.startswith('B'):
                spans.append([i, i+1, [head], [rel[4:]], int(rel[2])])
            else:
                spans[-1][1] += 1
                spans[-1][2].append(head)
                spans[-1][3].append(rel[2:])
        return spans

    @classmethod
    def get_arcs(cls, sequence: List[str]) -> List[List[bool]]:
        '''Get the mask chart for all possible arcs.
        Args:
            sequence (list[str]): An element is like `0:B-OBJ` or `3:I-OBJ`.
        '''
        seq_len = len(sequence) + 1
        # chart[i][j] is True if j is the head of i
        chart = [[0] * seq_len for _ in range(seq_len)]
        spans = [[0, 1, [-1], [nul], 0]] + cls.factorize(sequence)
        char2word = [i for i, span in enumerate(spans) for _ in range(span[0], span[1])]
        for start, end, heads, rels, specified in spans[1:]:
            # find the parent char of the current word
            for h in heads:
                if h < start or h >= end:
                    parent = h
            parent_start, parent_end, parent_heads, parent_rels, parent_specified = spans[char2word[parent]]
            for i in range(start, end):
                head = heads[i-start]
                chart[i][head] = 1
                # word structure not provided for current word, other chars in the same word can also be the head
                if not specified:
                    for j in range(start, end):
                        if i == j:
                            continue
                        chart[i][j] = 1
                # word structure not provided for parent word, other chars in the same word can also be the head
                if head == parent and not parent_specified:
                    for j in range(parent_start, parent_end):
                        chart[i][j] = 1

        return chart

    @classmethod
    def get_rels(cls, sequence: List[str], use_intra_rels: bool = False) -> List[List[int]]:
        '''Get the label chart for all possible arcs.'''
        seq_len = len(sequence) + 1
        # chart[i][j] == `l` if j is the head of i and j->i is labeled as `l`
        chart = [[None] * seq_len for _ in range(seq_len)]
        spans = [[0, 1, [-1], [nul], 0]] + cls.factorize(sequence)
        char2word = [i for i, span in enumerate(spans) for _ in range(span[0], span[1])]
        for start, end, heads, rels, specified in spans[1:]:
            # find the parent char of the current word
            for h in heads:
                if h < start or h >= end:
                    parent = h

            parent_start, parent_end, parent_heads, parent_rels, parent_specified = spans[char2word[parent]]
            for i in range(start, end):
                head, rel = heads[i-start], rels[i-start]
                # chart[i][head] = rel if use_intra_rels else nul
                if use_intra_rels:
                    chart[i][head] = rel
                else:
                    chart[i][head] = rel if head == parent else nul
                # word structure not provided for current word, other chars in the same word can also be the head
                if not specified:
                    for j in range(start, end):
                        if i == j:
                            continue
                        chart[i][j] = nul
                # word structure not provided for parent word, other chars in the same word can also be the head
                if head == parent and not parent_specified:
                    for j in range(parent_start, parent_end):
                        chart[i][j] = rel

        return chart

    @classmethod
    def get_intra_rels(cls, sequence: List[str]) -> List[List[int]]:
        '''Get the label chart for all possible arcs.'''
        return cls.get_rels(sequence, use_intra_rels=True)


class Coarse2FineCharCoNLLSentence(LatentCharCoNLLSentence):

    def __init__(self, transform, lines):
        super().__init__(transform, lines)


class Coarse2FineCharCoNLL(LatentCharCoNLL):

    fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def Sentence(self):
        return Coarse2FineCharCoNLLSentence


class MixedCoarse2FineCharCoNLLSentence(MixedCharCoNLLSentence):

    def __init__(self, transform, lines):
        super().__init__(transform, lines)


class MixedCoarse2FineCharCoNLL(MixedCharCoNLL):

    fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def Sentence(self):
        return MixedCoarse2FineCharCoNLLSentence
