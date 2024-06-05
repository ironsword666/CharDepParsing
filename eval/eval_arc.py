# -*- coding=utf-8 -*-

from collections import Counter

from nltk.tree import Tree
from supar.utils.transform.const2dep import UnlabeledDepTree
from supar.utils.transform.conll import CoNLL


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
                # sent = list(zip(words, heads))
                sent = heads
                sents.append(sent)
                words, heads = [], []
                continue
            line = line.split('\t')
            words.append(line[1])
            heads.append(int(line[6]))
    return sents


def load_tree(file):
    """Load constituency trees from file.

    Args:
        file (str): path to the file.
    """
    with open(file, 'r') as f:
        trees = []
        for line in f:
            tree = Tree.fromstring(line)
            consts = UnlabeledDepTree.factorize(UnlabeledDepTree.preprocess_tree(tree, leaf_padding=False, binarization=False, unary_chain=True)[0])
            spans = [(b, e) for b, e, _ in consts]
            trees.append(spans)

    return trees


def eval_dep_direction(sents):
    """Evaluate the direction of dependency arcs.

    Args:
        sents (list[list[int]]): list of sentences, each sentence is a list of dependency heads.

    Returns:
        float: the percentage of correct dependency directions.
    """
    left = 0
    total = 0
    for sent in sents:
        for modifier, head in enumerate(sent, 1):
            if modifier < head:
                left += 1
            total += 1
    return left / total


def clear_struct(heads, spans):
    structs = []
    for b, e in spans:
        if b + 1 == e:
            continue
        struct = []
        hs = heads[b:e]
        for h in hs:
            if h < b+1 or h > e:
                struct.append(0)
            else:
                struct.append(h - b)
        structs.append('_'.join(map(str, struct)))

    return Counter(structs)


def eval_struct(sents, trees):
    span_struct = Counter()
    for sent, tree in zip(sents, trees):
        span_struct.update(clear_struct(sent, tree))

    for struct, freq in span_struct.most_common(20):
        print(f"{struct}\t{freq}")


def eval_span_constraint(sents, trees):
    for sent, tree in zip(sents, trees):
        if not CoNLL.isprojective(sent):
            print('Non-projective sentence: ', sent)
        if not UnlabeledDepTree.is_single_root(sent, tree):
            print('Multiple roots: ', sent, tree)
        if not UnlabeledDepTree.is_root_as_head(sent, tree):
            print('Non-Root as head: ', sent, tree)

    print('Done!')


def main(dep_file, const_file):
    sents = read_conllx(dep_file)
    left_rate = eval_dep_direction(sents=sents)
    print(f"Left rate: {left_rate}")
    trees = load_tree(const_file)
    eval_span_constraint(sents=sents, trees=trees)
    eval_struct(sents=sents, trees=trees)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dep_file', help='path to the dep file')
    parser.add_argument('--const_file', help='path to the const file')
    args = parser.parse_args()
    main(args.dep_file, args.const_file)

"""
python -m supar.eval.eval_arc \
    --dep_file const2dep/results/ptb/const.const2dep.c_span+combine.bert.seed0.pred \
    --const_file data/ptb/const/test.pid

python -m supar.eval.eval_arc \
    --dep_file data/ptb/test.conllx \
    --const_file data/ptb/const/test.pid

python -m supar.eval.eval_arc \
    --dep_file const2dep/results/ptb/const.const2dep.embed.c_span+combine.lstm.freeze.pos.seed0.pred \
    --const_file data/ptb/const/test.pid

python -m supar.eval.eval_arc \
    --dep_file const2dep/results/ptb/const.const2dep.c_span+combine.bert.roberta-base.seed0.pred \
    --const_file data/ptb/const/test.pid

"""