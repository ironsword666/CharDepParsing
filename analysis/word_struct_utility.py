# -*- coding=utf-8 -*-

from collections import Counter, defaultdict
from typing import Tuple

NUL = '<nul>'


class Dependency(object):

    def __init__(self) -> None:
        pass


class WordStruct(object):

    def __init__(self, word: str, heads: Tuple[int], rels: Tuple[str]) -> None:
        """
        Args:
            word (str): 词
            heads (List[int]): 词内部结构的heads
            rels (List[str]): 词内部结构的rels
        """
        self.word = word
        self.heads = heads
        self.rels = rels

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WordStruct):
            return False
        return self.word == other.word and self.heads == other.heads and self.rels == other.rels

    def __hash__(self) -> int:
        return hash((self.word, self.heads, self.rels))

    def __repr__(self) -> str:
        s = f"{self.word} {'_'.join(map(str, self.heads))} {'_'.join(self.rels)}"
        return s

    def __len__(self) -> int:
        return len(self.heads)

    @classmethod
    def load_struct_from_word(cls, file: str):
        """从一个单词中读取词内部结构.

        1	纺	3	att
        2	织	1	coo
        3	业	0	root

        Returns:
            dict: {word: (heads, rels)}, e.g. {'中国': ('0_1', 'root_null')}
        """
        with open(file, 'r') as fr:
            structs, struct = dict(), []
            for line in fr:
                line = line.strip()
                if line:
                    struct.append(line.split())
                else:
                    ids, chars, heads, rels = list(zip(*struct))
                    word = ''.join(chars)
                    structs[word] = cls(word, tuple(int(h) for h in heads), rels)
                    struct = []

        return structs

    @classmethod
    def load_struct_from_sent(cls, file: str, ignore_repeat: bool = True):
        """从一个完整的句子中读取词内部结构.

        1	新华社	_	_	_	_	7	dep	2_3_0	<nul>_<nul>_root
        2	福冈	_	_	_	_	7	dep	2_0	<nul>_root
        3	５月	_	_	_	_	7	dep	2_0	<nul>_root
        4	１１日	_	_	_	_	7	dep	2_3_0	<nul>_<nul>_root
        5	电	_	_	_	_	7	dep	0	root
        6	（	_	_	_	_	7	punct	0	root
        7	记者	_	_	_	_	0	root	2_0	<nul>_root
        8	乐绍延	_	_	_	_	7	dep	2_3_0	<nul>_<nul>_root
        9	）	_	_	_	_	7	punct	0	root

        Returns:
            dict: {word: [(heads, rels), ...]}, e.g. {'中国': [('0_1', 'root_null'), ...]}
                a word may have multiple structures
        """
        structs = defaultdict(list)
        with open(file, 'r') as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                cols = line.split('\t')
                word = cols[1]
                # skip single character
                if len(word) == 1:
                    continue
                heads = tuple(int(head) for head in cols[8].split('_'))
                # replace '<nul>' with 'null'
                rels = tuple(rel if rel != NUL else 'null' for rel in cols[9].split('_'))
                struct = cls(word, heads, rels)
                # if word not in structs or struct != structs[word][0]:
                #     structs[word].append(struct)
                structs[word].append(struct)

            if ignore_repeat:
                # remove repeat structs
                for k, v in structs.items():
                    # print(Counter(v).most_common(1)[0])
                    structs[k] = Counter(v).most_common(1)[0][0]

        return structs


class UnlabeledWordStruct(WordStruct):

    def __init__(self, word: str, heads: Tuple[int], rels=None) -> None:

        self.word = word
        self.heads = heads

    def __eq__(self, other: object) -> bool:
        return self.word == other.word and self.heads == other.heads

    def __hash__(self) -> int:
        return hash((self.word, self.heads))

    def __repr__(self) -> str:
        s = f"{self.word} {'_'.join(map(str, self.heads))}"
        return s


def is_all_word_has_struct(pred_structs, gold_structs):
    """
    Whether all words in pred_structs have struct in gold_structs.
    """
    words = []
    for word, struct in pred_structs.items():
        if word not in gold_structs:
            words.append(word)
    print(f"words without struct: {len(words)}")


def compare_struct(pred_structs, gold_structs):
    """比较一个词预测的结构和标准结构."""
    for word, structs in pred_structs.items():
        if word not in gold_structs:
            continue
        gold_struct = gold_structs[word]
        for struct in structs:
            if struct != gold_struct:
                print(f"{struct}\n{gold_struct}")
                print()


def show_struct(s):
    structs = []
    print('----')
    for line in s.split('\n'):
        if not line.strip():
            continue
        # print(line)
        length, struct, size = line.split()
        structs.append((length, struct, size))
    structs.sort(key=lambda x: int(x[0]))
    for length, struct, size in structs:
        if int(length) >= 5:
            continue
        print(length, struct, size)


def arc_direction_distribution(file):
    """统计一个词级别的依存书库中，头部在左边的词的个数和头部在右边的词的个数。"""
    with open(file, 'r') as fr:
        leftward, rightward = 0, 0
        for line in fr:
            line = line.strip()
            if not line:
                continue
            cols = line.split('\t')
            idx = int(cols[0])
            head = int(cols[6])
            if idx < head:
                leftward += 1
            else:
                rightward += 1

    print('leftward', leftward)
    print('rightward', rightward)
    print('leftward ratio', leftward / (leftward + rightward))
