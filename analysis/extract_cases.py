# -*- coding=utf-8 -*-

"""
比较预测的字符级别的依存关系与标准的依存关系是否相同
"""


def read(file):
    sentences, sentence = [], []
    with open(file, 'r') as fr:
        for line in fr:
            if not line.strip():
                sentences.append(sentence)
                sentence = []
            else:
                line = line.strip().split('\t')
                # only keep intra-and inter-word dependencies
                line[2:6] = []
                sentence.append(line)
    return sentences


def write(fo, sentence):
    for line in sentence:
        fo.write('\t'.join(line) + '\n')
    fo.write('\n')


def equal(pred, gold):
    """判断预测的字符级别的依存关系是否与标准的相同"""
    flag = True
    if len(pred) != len(gold):
        return False
    for pred_line, gold_line in zip(pred, gold):
        # no gold intra-word dependency
        if all(head == '-1' for head in gold_line[4].split('_')):
            # TODO: set the prediction to be no intra-word dependency
            n_char = len(gold_line[1])
            pred_line[4] = '_'.join(['-1'] * n_char)
            pred_line[5] = '_'.join(['null'] * n_char)
            continue
        # segment
        if pred_line[1] != gold_line[1]:
            flag = False
        # inter-word dependency
        if pred_line[2] != gold_line[2] or pred_line[3] != gold_line[3]:
            flag = False
        # intra-word dependency
        if pred_line[4] != gold_line[4]:
            flag = False
        if pred_line[5] != gold_line[5]:
            flag = False

    return flag


def compare(pred_file, gold_file):
    preds = read(pred_file)
    golds = read(gold_file)
    assert len(preds) == len(golds)
    with open(pred_file+'.case', 'w') as fp, open(gold_file+'.case', 'w') as fg:
        for i in range(len(preds)):
            if not equal(preds[i], golds[i]):
                write(fp, preds[i])
                write(fg, golds[i])


def main():
    pred_file = 'results/ctb5-big/dep-sd-with-iwdp3-reversal.mixed-char-crf-dep.bert.use_intra_rels.label-loss=crf.struct-norm=token.seed1.pred'
    gold_file = 'data/ctb5-big/dep-sd-with-iwdp3-reversal/dev.conllx'
    compare(pred_file, gold_file)


if __name__ == '__main__':
    main()
