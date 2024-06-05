# -*- coding=utf-8 -*-

import argparse
import os
from collections import Counter, defaultdict

from analysis.word_struct_utility import UnlabeledWordStruct


def load_words(file) -> dict:
    """从一个树库中读取所有的词及其词性."""
    words = defaultdict(list)
    with open(file) as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            cols = line.split('\t')
            word, pos = cols[1], cols[3]
            words[word].append(pos)
    return words





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the specified parser and dataset.')
    parser.add_argument('--pred', default='arr/results/ctb7/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=10.seed0.pred.test', help='path to predicted result')
    parser.add_argument('--gold', default='data/ctb7/dep-sd/test.conllx', help='path to gold file')
    parser.add_argument('--refer_struct', default='data/wist-main/WIST.conll', help='path to gold file')

    args = parser.parse_args()

    pred_structs = UnlabeledWordStruct.load_struct_from_sent(args.pred, ignore_repeat=False)
    refer_structs = UnlabeledWordStruct.load_struct_from_word(args.refer_struct)

    words = load_words(args.gold)

    # 检查是否存在多词性的词
    for word, structs in pred_structs.items():
        if len(word) < 4 and len(set(structs)) > 1 and word in words and len(set(words[word])) > 1:
            print(word)
            print(set(words[word]))
            print(set([struct.heads for struct in structs]))
    # 检查latent依存弧预测对了而leftward错了的情况
