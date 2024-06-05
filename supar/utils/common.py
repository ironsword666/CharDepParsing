# -*- coding: utf-8 -*-

pad = PAD = '<pad>'
unk = UNK = '<unk>'
bos = BOS = '<bos>'
eos = EOS = '<eos>'
nul = NUL = '<nul>'
ROOT = 'root'

INF = float('inf')
MIN = -1e32
# MIN = float('-inf')
MAX = 1e32


def get_rel2cluster(rels, clusters, rel_cluster_map):
    index_rel_map = {v: k for k, v in rels.items()}
    rel2cluster = []
    for i in range(len(rels)):
        rel = index_rel_map[i]
        cluster = rel_cluster_map[rel]
        index = clusters[cluster]
        rel2cluster.append(index)
    return rel2cluster


# rels = {'<nul>': 0,
#         'advmod': 1,
#         'amod': 2,
#         'asp': 3,
#         'assm': 4,
#         'assmod': 5,
#         'attr': 6,
#         'ba': 7,
#         'c-adjct': 8,
#         'c-adv': 9,
#         'c-att': 10,
#         'c-cmp': 11,
#         'c-coo': 12,
#         'c-frag': 13,
#         'c-obj': 14,
#         'c-pobj': 15,
#         'c-repet': 16,
#         'c-subj': 17,
#         'cc': 18,
#         'ccomp': 19,
#         'clf': 20,
#         'comod': 21,
#         'conj': 22,
#         'cop': 23,
#         'cpm': 24,
#         'dep': 25,
#         'det': 26,
#         'dobj': 27,
#         'dvpm': 28,
#         'dvpmod': 29,
#         'etc': 30,
#         'lccomp': 31,
#         'lobj': 32,
#         'loc': 33,
#         'mmod': 34,
#         'neg': 35,
#         'nn': 36,
#         'nsubj': 37,
#         'nsubjpass': 38,
#         'nummod': 39,
#         'ordmod': 40,
#         'pass': 41,
#         'pccomp': 42,
#         'plmod': 43,
#         'pobj': 44,
#         'prep': 45,
#         'prnmod': 46,
#         'prtmod': 47,
#         'punct': 48,
#         'range': 49,
#         'rcmod': 50,
#         'rcomp': 51,
#         'root': 52,
#         'tmod': 53,
#         'top': 54,
#         'vmod': 55,
#         'xsubj': 56}

clusters = {'subj': 0,
            'obj': 1,
            'attr': 2,
            'adv': 3,
            'complement': 4,
            'adjunct': 5,
            'compound': 6,
            'coord': 7,
            'other': 8}

rel_cluster_map = {'<nul>': 'other',
                   'advmod': 'adv',
                   'amod': 'attr',
                   'asp': 'adjunct',
                   'assm': 'adjunct',
                   'assmod': 'attr',
                   'attr': 'other',
                   'ba': 'adjunct',
                   'c-adjct': 'adjunct',
                   'c-adv': 'adv',
                   'c-att': 'attr',
                   'c-cmp': 'complement',
                   'c-coo': 'coord',
                   'c-frag': 'compound',
                   'c-obj': 'obj',
                   'c-pobj': 'obj',
                   'c-repet': 'other',
                   'c-subj': 'subj',
                   'c-de': 'adjunct',
                   'cc': 'other',
                   'ccomp': 'complement',
                   'clf': 'attr',
                   'comod': 'coord',
                   'conj': 'coord',
                   'cop': 'other',
                   'cpm': 'adjunct',
                   'dep': 'other',
                   'det': 'attr',
                   'dobj': 'obj',
                   'dvpm': 'adjunct',
                   'dvpmod': 'adv',
                   'etc': 'adjunct',
                   'lccomp': 'complement',
                   'lobj': 'obj',
                   'loc': 'adv',
                   'mmod': 'adjunct',
                   'neg': 'adv',
                   'nn': 'compound',
                   'nsubj': 'subj',
                   'nsubjpass': 'subj',
                   'nummod': 'attr',
                   'ordmod': 'attr',
                   'pass': 'adjunct',
                   'pccomp': 'complement',
                   'plmod': 'adv',
                   'pobj': 'obj',
                   'prep': 'adv',
                   'prnmod': 'adjunct',
                   'prtmod': 'adjunct',
                   'punct': 'other',
                   'range': 'obj',
                   'rcmod': 'attr',
                   'rcomp': 'complement',
                   'root': 'other',
                   'tmod': 'adv',
                   'top': 'subj',
                   'vmod': 'attr',
                   'xsubj': 'subj'}
