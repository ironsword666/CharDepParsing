# -*- coding=utf-8 -*-

"""
Eval pipeline results with gold data using char-level dependency metric.
"""


from supar.utils.metric import CharAttachmentMetric
from supar.utils.fn import ispunct


def read(file):
    """
    Read a word-level conllx file, then return the char-level dependency result.

    Returns:
        sents (list): [sentence1, sentence2, ...]
            sentence (list): [((begin of modifier word, end of modifier word), (begin of head word, end of head word), relation, ispunct), ...]
    """
    with open(file, 'r') as f:
        sents = []
        words, heads, rels = [], [], []
        for line in f:
            line = line.strip()
            if not line:
                sent = word2char(words, heads, rels)
                sents.append(sent)
                words, heads, rels = [], [], []
                continue
            line = line.split('\t')
            words.append(line[1])
            heads.append(line[6])
            rels.append(line[7])
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


def eval(pred_file, gold_file):
    preds = read(pred_file)
    golds = read(gold_file)
    assert len(preds) == len(golds)
    metric = CharAttachmentMetric()
    metric(preds, golds)
    print(metric)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', help='path to the predicted file')
    parser.add_argument('--gold_file', help='path to the gold file')
    args = parser.parse_args()
    eval(args.pred_file, args.gold_file)

"""
python -m supar.eval.eval_pipeline \
    --pred_file arr/results/ctb5-big/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.seed1.pred \
    --gold_file data/ctb5-big/dep-sd/dev.conllx
"""
