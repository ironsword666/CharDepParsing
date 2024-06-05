# -*- coding: utf-8 -*-


def count(file):
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
    print('left / right', leftward / (leftward + rightward))


# file = 'data/ctb5-big/dep-sd-conj/train.conllx'
# file = 'data/ctb5/dep-sd/train.conllx'
file = 'data/ctb5/dep-malt/train.conllx'
count(file)

file = 'data/ctb5/dep-malt/train.conllx'
file = "const2dep/results/ptb/const.const2dep.c_span+combine.bert.roberta-base.seed0.pred"
count(file)
