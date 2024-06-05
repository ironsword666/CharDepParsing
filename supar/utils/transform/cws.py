from supar.utils.logging import progress_bar
from supar.utils.transform.transform import Sentence, Transform
from supar.utils.transform.conll import CoNLL


import os


class CWSCoNLLSentence(Sentence):

    def __init__(self, transform, lines):
        super().__init__(transform)

        self.values = []
        # record annotations for post-recovery
        self.annotations = dict()
        # print(lines)
        for i, line in enumerate(lines):
            value = line.split('\t')
            self.annotations[len(self.values)] = line
            self.values.append(value)
        # print(self.values)
        self.values = list(zip(*self.values))

    def __repr__(self):
        chars = self.values[0]
        if getattr(self, 'segs', None) is not None:
            words = []
            for b, e in self.segs:
                words.append(''.join(chars[b:e]))
            return CoNLL.toconll(words)
        else:
            lines = ['\t'.join(value) for value in zip(*self.values)]
        return '\n'.join(lines) + '\n'


class CWSCoNLL(Transform):

    fields = ['FORM', 'TAG']

    def __init__(self,
                 FORM=None, TAG=None):
        super().__init__()

        self.FORM = FORM
        self.TAG = TAG

    @property
    def src(self):
        return self.FORM,

    @property
    def tgt(self):
        return self.TAG,

    @classmethod
    def recover_words(cls, tags):
        """ Transform a `bmes` tag sequence to a span-based sequence.
        NOTE: the tag sequence should be legal, i.e., forbid transitions like `b` -> `s`

        Args:
            tags (list[str]): [b, m, e, s, s, ...]

        Returns:
            [(0, 1), (2, 3), ...]
        """
        spans = []
        for i, t in enumerate(tags):
            if t == 'b' or t == 's':
                spans.append([i, i+1])
            elif t == 'm' or t == 'e':
                spans[-1][1] += 1

        return [tuple(span) for span in spans]

    def load(self, data, lang=None, max_len=None, **kwargs):
        if isinstance(data, str) and os.path.exists(data):
            with open(data, 'r') as f:
                lines = [line.strip() for line in f]

        i, start, sentences = 0, 0, []
        for line in progress_bar(lines):
            if not line:
                sentences.append(CWSCoNLLSentence(self, lines[start:i]))
                start = i + 1
            i += 1
        if max_len is not None:
            sentences = [i for i in sentences if len(i) < max_len]

        return sentences