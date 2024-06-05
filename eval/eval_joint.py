# -*- coding=utf-8 -*-

"""
Eval joint results with gold data using char-level dependency metric.
"""


from supar.utils.metric import DetailedCharAttachmentMetric, MetricByLength
from supar.utils.fn import ispunct


def read(file, fn=lambda x, y, z: (x, y, z)):
    """
    Read a word-level conllx file, then return the char-level dependency result.

    Returns:
        sents (list): [sentence1, sentence2, ...]
            sentence (list): [((begin of modifier word, end of modifier word), (begin of head word, end of head word), relation, ispunct), ...]
    """
    n_total_words = 0
    total_word_len = 0
    with open(file, 'r') as f:
        sents = []
        words, heads, rels = [], [], []
        for line in f:
            line = line.strip()
            if not line:
                sent = fn(words, heads, rels)
                sents.append(sent)
                words, heads, rels = [], [], []
                continue
            line = line.split('\t')
            words.append(line[1])
            n_total_words += 1
            total_word_len += len(line[1])
            heads.append(line[6])
            rels.append(line[7])
    # print(f"Total word number: {n_total_words}")
    # # 1.63/1.69/1.71 for ctb5
    # # 1.63/1.62/1.62 for ctb5-big
    # # 1.64/1.67/1.64 for ctb6
    # # 1.616/1.636/1.626 for ctb7
    # print(f"Average word length: {total_word_len / n_total_words}")
    return sents


def word2char(words, heads, rels):
    """
    Convert word-level dependency result to char-level dependency result.

    Args:
        words (list[str]): [word1, word2, ...]
        heads (list): [head of word1, head of word2, ...]
        rels (list): [relation of word1, relation of word2, ...]

    Returns:
        sent (list): [((begin of modifier word, end of modifier word), (begin of head word, end of head word), relation, ispunct), ...]
    """
    dependencies = []
    heads = [int(i) for i in heads]

    spans, p = [(0, 1)], 1
    for word in words:
        spans.append((p, p+len(word)))
        p += len(word)

    for i, (ml, mr) in enumerate(spans[1:]):
        head_word = heads[i]
        (hl, hr) = spans[head_word]
        rel = rels[i]
        # relations.append(((ml, mr), (hl, hr), rels[ml][hl]))
        dependencies.append(((ml, mr), (hl, hr), rel, ispunct(words[i])))

    return dependencies


def word2char_with_word(words, heads, rels):
    """
    Convert word-level dependency result to char-level dependency result.
    Also return the word form.
    """
    dependencies = []
    heads = [int(i) for i in heads]

    spans, p = [(0, 1)], 1
    for word in words:
        spans.append((p, p+len(word)))
        p += len(word)

    for i, (ml, mr) in enumerate(spans[1:]):
        head_word = heads[i]
        (hl, hr) = spans[head_word]
        rel = rels[i]
        # relations.append(((ml, mr), (hl, hr), rels[ml][hl]))
        dependencies.append(((ml, mr), (hl, hr), rel, ispunct(words[i]), words[i]))

    return dependencies


def eval(pred_file, gold_file):
    preds = read(pred_file, fn=word2char)
    golds = read(gold_file, fn=word2char)
    assert len(preds) == len(golds)
    # metric = DetailedCharAttachmentMetric()
    metric = MetricByLength(DetailedCharAttachmentMetric)
    metric(preds, golds)
    print(metric)


def eval_granular(pred_file, gold_file):
    preds = read(pred_file, fn=word2char_with_word)
    golds = read(gold_file, fn=word2char_with_word)
    assert len(preds) == len(golds)
    for pred, gold in zip(preds, golds):
        pred = [(word_bound, word) for word_bound, head_bound, rel, punct, word in pred]
        gold = [(word_bound, word) for word_bound, head_bound, rel, punct, word in gold]
        pred_word_bounds, pred_words = list(zip(*pred))
        left_pred_boundaries = {left: word for (left, right), word in zip(pred_word_bounds, pred_words)}
        right_pred_boundaries = {right: word for (left, right), word in zip(pred_word_bounds, pred_words)}
        gold_word_bounds, gold_words = list(zip(*gold))
        left_gold_boundaries = {left: word for (left, right), word in zip(gold_word_bounds, gold_words)}
        right_gold_boundaries = {right: word for (left, right), word in zip(gold_word_bounds, gold_words)}
        for (left, right), pred_word in zip(pred_word_bounds, pred_words):
            if (left, right) not in gold_word_bounds:
                if left in left_gold_boundaries and right in right_gold_boundaries:
                    print('------------------')
                    print(f"Word boundary match: {pred_word} ({left}, {right})")
                    print(f"Gold word: {left_gold_boundaries[left]}, {right_gold_boundaries[right]}")

        for (left, right), gold_word in zip(gold_word_bounds, gold_words):
            if (left, right) not in pred_word_bounds:
                if left in left_pred_boundaries and right in right_pred_boundaries:
                    print('------------------')
                    print(f"Word boundary match: {gold_word} ({left}, {right})")
                    print(f"Pred word: {left_pred_boundaries[left]}, {right_pred_boundaries[right]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', help='path to the predicted file')
    parser.add_argument('--gold_file', help='path to the gold file')
    args = parser.parse_args()
    eval(args.pred_file, args.gold_file)
    # read("data/ctb5-big/dep-malt/train.conllx")
    # read("data/ctb5-big/dep-malt/dev.conllx")
    # read("data/ctb5-big/dep-malt/test.conllx")


"""
Two eval with the method of Wu for ctb7
SegP: 97.19% SegR: 96.24% SegF: 96.71%
UCM: 37.16% LCM: 33.82% UAS: 86.09% LAS: 84.78%
UP: 87.11% UR: 86.09% UF: 86.60%
LP: 85.79% LR: 84.78% LF: 85.28% 
Seg Wrong:  4.96% Head Wrong:  8.87% All Right: 86.17%

SegP: 97.19% SegR: 96.24% SegF: 96.71%
UCM: 37.16% LCM: 33.82% UAS: 85.16% LAS: 83.94%
UP: 86.17% UR: 85.16% UF: 85.66%
LP: 84.94% LR: 83.94% LF: 84.44% 
Seg Wrong:  4.96% Head Wrong:  8.87% All Right: 86.17%
"""

"""
python -m eval.eval_joint \
    --pred_file arr/results/ctb7/dep-malt.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=10.seed0.pred.gold-seg.test \
    --gold_file data/ctb7/dep-malt/test.conllx

python -m supar.eval.eval_joint \
    --pred_file arr/results/ctb7/dep-malt.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=5.seed0.pred \
    --gold_file data/ctb7/dep-malt/dev.conllx

python -m eval.eval_joint \
    --pred_file arr/results/ctb7/dep-malt.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=5.seed0.pred \
    --gold_file data/ctb7/dep-malt/dev.conllx
"""
