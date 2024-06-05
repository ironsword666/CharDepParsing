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


def eval(pred_file, gold_file, seeds):
    metric = DetailedCharAttachmentMetric()
    # metric = MetricByLength(DetailedCharAttachmentMetric)
    golds = read(gold_file, fn=word2char)
    for seed in seeds:
        pred_file = pred_file.format(seed)
    preds = read(pred_file, fn=word2char)
    assert len(preds) == len(golds)
    # metric = DetailedCharAttachmentMetric()
    metric(preds, golds)
    print(metric)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', help='path to the predicted file')
    parser.add_argument('--gold_file', help='path to the gold file')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3], help='random seeds')
    args = parser.parse_args()

    # metric = DetailedCharAttachmentMetric()
    metric = MetricByLength(DetailedCharAttachmentMetric)
    golds = read(args.gold_file, fn=word2char)
    for seed in args.seeds:
        pred_file = args.pred_file.format(seed)
        preds = read(pred_file, fn=word2char)
        assert len(preds) == len(golds)
        metric(preds, golds)
    print(metric)

"""
python -m eval.eval_joint_word_parsing_of_diff_len \
    --pred_file arr/results/ctb7/dep-sd.explicit-char-crf-dep.leftward.bert.epc=10.seed{}.pred.gold-seg.test \
    --gold_file data/ctb7/dep-sd/test.conllx


python -m eval.eval_joint_word_parsing_of_diff_len \
    --pred_file arr/results/ctb7/dep-sd.latent-char-crf-dep.label-loss=crf.bert.epc=10.seed{}.pred.gold-seg.test \
    --gold_file data/ctb7/dep-sd/test.conllx

python -m eval.eval_joint_word_parsing_of_diff_len \
    --pred_file arr/results/ctb7/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=10.seed{}.pred.gold-seg.test \
    --gold_file data/ctb7/dep-sd/test.conllx
"""