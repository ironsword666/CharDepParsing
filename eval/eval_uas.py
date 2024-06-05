# -*- coding=utf-8 -*-


from supar.utils.metric import UnlabeledAttachmentMetric
from supar.utils.fn import ispunct


def read_conllx(file):
    """
    Read a word-level conllx file, then return the char-level dependency result.

    Returns:
        sents (list): [sentence1, sentence2, ...]
            sentence (list): [((begin of modifier word, end of modifier word), (begin of head word, end of head word), relation, ispunct), ...]
    """
    with open(file, 'r') as f:
        sents = []
        words, heads = [], []
        for line in f:
            line = line.strip()
            if not line:
                sent = list(zip(words, heads))
                # sent = heads
                sents.append(sent)
                words, heads = [], []
                continue
            line = line.split('\t')
            words.append(line[1])
            heads.append(int(line[6]))
    return sents


def eval(pred_file, gold_file):
    preds = read_conllx(pred_file)
    golds = read_conllx(gold_file)
    metric = UnlabeledAttachmentMetric()
    for pred, gold in zip(preds, golds):
        metric(pred, gold)
    print(metric)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', help='path to the dep file')
    parser.add_argument('--gold_file', help='path to the const file')
    args = parser.parse_args()
    eval(args.pred_file, args.gold_file)

"""
python -m supar.eval.eval_uas \
    --pred_file const2dep/results/ptb/const.const2dep.embed.c_span+combine.lstm.seed0.pred  \
    --gold_file data/ptb/test.conllx

python -m supar.eval.eval_uas \
    --pred_file const2dep/results/ptb/const.const2dep.c_span+combine.bert.roberta-base.seed0.pred  \
    --gold_file data/ptb/test.conllx

python -m supar.eval.eval_uas \
    --pred_file const2dep/results/ptb/const.const2dep.embed.c_span+combine.lstm.freeze.pos.embed_proj=1024.roberta-base.seed0.pred  \
    --gold_file data/ptb/test.conllx

"""
