# -*- coding=utf-8 -*-

from collections import defaultdict, Counter


def load_struct_from_word(file):
    """
    Load annotated word-internal structures in the format of conllx.
    For example:
        1	对	2	adv
        2	接	3	att
        3	点	0	root

    Args:
        file (str): path to the file

    Returns:
        structs (dict): a dict of word-internal structures
            key: word, for example, '对接点'
            value: [heads, labels]
                heads: a tuple of head indices, for example, (2, 3, 0)
                labels: a tuple of dependency labels, for example, ('adv', 'att', 'root')
    """
    structs = {}
    struct = []
    with open(file, 'r') as fr:
        for line in fr:
            line = line.strip()
            if not line:
                struct = list(zip(*struct))
                word = ''.join(struct[0])
                structs[word] = struct[1:]
                struct = []
                continue
            cols = line.split('\t')
            char = cols[1]
            head = cols[2]
            label = cols[3]
            struct.append((char, head, label))
    return structs


def load_structure_from_tree(file):
    word_dict = defaultdict(list)
    res = dict()
    with open(file, 'r') as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            cols = line.split('\t')
            word = cols[1]
            if len(word) == 1:
                continue
            struct = cols[8]
            word_dict[word].append(struct)
        # remove repeat structs
        for k, v in word_dict.items():
            res[k] = Counter(v).most_common(1)[0][0].split('_')
    return res
