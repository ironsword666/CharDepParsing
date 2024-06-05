# -*- coding=utf-8 -*-


from collections import defaultdict


def load_struct_distribution(file):
    structs = defaultdict(int)
    with open(file, 'r') as fr:
        for line in fr:
            line = line.strip()
            if line:
                cols = line.split()
                struct, freq = cols[0], cols[1]
                structs[struct] += int(freq)

    return structs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution_file", type=str, default=None, help="distribution file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3], help="seeds")
    parser.add_argument("--topk", type=int, default=10000, help="topk")
    args = parser.parse_args()

    final_structs = defaultdict(int)
    length_dict = defaultdict(int)
    for seed in args.seeds:
        distribution_file = args.distribution_file.format(seed)
        structs = load_struct_distribution(distribution_file)
        for struct, count in structs.items():
            final_structs[struct] += count
            length_dict[len(struct.split('_'))] += count

    for struct, count in sorted(final_structs.items(), key=lambda x: x[1], reverse=True)[:args.topk]:
        print(struct, count, f"{count/length_dict[len(struct.split('_'))]:.2%}")
