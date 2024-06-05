# -*- coding=utf-8 -*-

from .io import load_struct_from_word, load_structure_from_tree


def compare_struct_generalization(tree_file, gold_file, gold_file_all, output_file):
    tree_structs = load_structure_from_tree(tree_file)
    print(f'word size: {len(tree_structs)}')
    gold_structs = load_struct_from_word(gold_file)
    print(f'training gold structure size: {len(gold_structs)}')
    gold_structs_all = load_struct_from_word(gold_file_all)
    print(f'all gold structure size: {len(gold_structs_all)}')
    n_own_struct = 0
    n_in_training = 0
    n_seperate = 0
    n_same = 0
    with open(output_file, 'w') as fw:
        for word, struct in tree_structs.items():
            if word not in gold_structs_all:
                continue
            n_own_struct += 1
            if word in gold_structs and word not in gold_structs_all:
                raise ValueError(f'{word} in gold_structs but not in gold_structs_all')
            if word in gold_structs:
                n_in_training += 1
                continue
            n_seperate += 1
            gold_struct = list(gold_structs_all[word][0])
            if struct == gold_struct:
                n_same += 1
                continue
            # s = f'# {word}\n'
            s = ''
            for i, (c, h, g) in enumerate(zip(word, struct, gold_struct), start=1):
                s += f'{i}\t{c}\t{h}\t{g}\n'
            fw.write(s+'\n')
    print(f'total in training structure size: {n_in_training}')
    print(f'total not in traing structure size: {n_seperate}')
    print(f'total having structure size: {n_own_struct}')
    print(f'total same structure size: {n_same}')


def compare_struct_with_gold(tree_file, gold_file, output_file):
    tree_structs = load_structure_from_tree(tree_file)
    print(f'word size: {len(tree_structs)}')
    gold_structs = load_struct_from_word(gold_file)
    print(f'training gold structure size: {len(gold_structs)}')
    n_same = 0
    n_own_struct = 0
    with open(output_file, 'w') as fw:
        for word, struct in tree_structs.items():
            if word not in gold_structs:
                continue
            n_own_struct += 1
            gold_struct = list(gold_structs[word][0])
            if struct == gold_struct:
                n_same += 1
                continue
            s = f'# {word}\n'
            for i, (c, h, g) in enumerate(zip(word, struct, gold_struct), start=1):
                s += f'{i}\t{c}\t{h}\t{g}\n'
            fw.write(s+'\n')
    print(f'total own structure size: {n_own_struct}')
    print(f'total same structure size: {n_same}')
    print(f"uas: {n_same/n_own_struct}")


if __name__ == '__main__':

    # tree_file = 'tacl/results/ctb5-big/dep-sd-iwdp.mixed-coarse2fine-char-crf-dep.label-loss=crf.struct_normalizer=sentence.bert.seed0.pred'
    # tree_file = 'tacl/results/ctb5-big/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.seed1.test.pred'
    # tree_file = "tacl/results/ctb5-big/dep-sd-conj.split-coarse2fine-char-crf-dep.label-loss=crf.struct_normalizer=token.bert.seed0.test.pred"
    # tree_file = "tacl/results/ctb5-big/dep-sd-conj-iwdp.mixed-coarse2fine-char-crf-dep.label-loss=crf.struct_normalizer=token.bert.seed0.pred"
    tree_file = "arr/results/ctb5-big/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.seed1.pred"
    # tree_file = "tacl/results/ctb5-big/dep-sd.word-crf-dep.bert.seed0.pred"
    # gold_file_all = 'data/iwdp/ctb5_iwdp_v2.conllx.all'
    # gold_file = 'data/iwdp/ctb5_iwdp_v4.conllx.strip'
    # gold_file = 'data/iwdp/ctb5_iwdp_v2.conllx.strip'
    gold_file = 'data/wist-main/WIST.conll'
    # gold_file = "data/iwdp/ctb5_iwdp_all.conllx"
    output_file = tree_file + '.compare'
    # compare_struct_generalization(tree_file, gold_file, gold_file_all, output_file)
    compare_struct_with_gold(tree_file, gold_file, output_file)
