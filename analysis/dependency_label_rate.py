# -*- coding=utf-8 -*-

from collections import defaultdict


def count_dependency_label(file):
    inter_word_label_count = defaultdict(int)
    inter_word_label_direction_count = defaultdict(int)
    intra_word_label_count = defaultdict(int)
    intra_word_label_direction_count = defaultdict(int)
    n = 0
    with open(file, 'r') as fr:
        for line in fr:
            line = line.strip()
            if not line:
                n = 0
                continue
            cols = line.split('\t')

            n += 1
            word_head = int(cols[6])
            char_heads = [int(i) for i in cols[8].split('_')]
            intra_word_labels = cols[9].split('_')
            inter_word_label = cols[7]
            # decide inter-word arc direction
            arc_direction = 'leftward' if n < word_head else 'rightward'
            # count inter-word label
            inter_word_label_count[inter_word_label] += 1
            inter_word_label_direction_count[inter_word_label + '-' + arc_direction] += 1
            # decide intra-word arc direction and count intra-word label
            for i, (char_head, intra_word_label) in enumerate(zip(char_heads, intra_word_labels), 1):
                if 'null' in intra_word_label:
                    continue
                intra_word_label_count[intra_word_label] += 1
                # decide intra-word arc direction
                if char_head == 0:
                    intra_word_label_direction_count[intra_word_label + '-' + arc_direction] += 1
                elif i < char_head:
                    intra_word_label_direction_count[intra_word_label + '-leftward'] += 1
                else:
                    intra_word_label_direction_count[intra_word_label + '-rightward'] += 1

    print('--- inter-word label ---')
    total_inter_word_label = sum(inter_word_label_count.values())
    inter_word_label_pairs = sorted(inter_word_label_count.items(), key=lambda x: x[1], reverse=True)
    for k, v in inter_word_label_pairs:
        print(k, v, v / total_inter_word_label)
    print('---inter-word label leftward direction ---')
    for k, _ in inter_word_label_pairs:
        if k + '-leftward' in inter_word_label_direction_count:
            v = inter_word_label_direction_count[k + '-leftward']
        else:
            v = 0
        print(k, v, v / total_inter_word_label)
    print('--- inter-word label rightward direction ---')
    for k, _ in inter_word_label_pairs:
        if k + '-rightward' in inter_word_label_direction_count:
            v = inter_word_label_direction_count[k + '-rightward']
        else:
            v = 0
        print(k, v, v / total_inter_word_label)

    print('--- intra-word label ---')
    total_intra_word_label = sum(intra_word_label_count.values())
    intra_word_label_pairs = sorted(intra_word_label_count.items(), key=lambda x: x[1], reverse=True)
    for k, v in intra_word_label_pairs:
        print(k, v, v / total_intra_word_label)
    print('---intra-word label leftward direction ---')
    for k, _ in intra_word_label_pairs:
        if k + '-leftward' in intra_word_label_direction_count:
            v = intra_word_label_direction_count[k + '-leftward']
        else:
            v = 0
        print(k, v, v / total_intra_word_label)
    print('--- intra-word label rightward direction ---')
    for k, _ in intra_word_label_pairs:
        if k + '-rightward' in intra_word_label_direction_count:
            v = intra_word_label_direction_count[k + '-rightward']
        else:
            v = 0
        print(k, v, v / total_intra_word_label)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='dependency label rate')
    parser.add_argument('--file', type=str, default='data/ctb5-big/dep-sd-with-iwdp-all/train.conllx', help='file')
    args = parser.parse_args()

    count_dependency_label(args.file)


if __name__ == '__main__':
    main()
