# -*- encoding: utf-8 -*-

import re


log = """
2022-12-05 06:02:47 INFO dev:  
SegP: 98.27% SegR: 98.50% SegF: 98.39%
UCM: 53.41% LCM: 48.58% UAS: 90.07% LAS: 89.00%
UP: 89.90% UR: 90.07% UF: 89.98%
LP: 88.83% LR: 89.00% LF: 88.92% 
2022-12-05 06:02:47 INFO test: 
SegP: 97.99% SegR: 98.71% SegF: 98.35%
UCM: 46.26% LCM: 43.10% UAS: 89.42% LAS: 88.37%
UP: 88.70% UR: 89.42% UF: 89.06%
LP: 87.65% LR: 88.37% LF: 88.01% 
"""


def strip_log(log):
    """Strip log of all the useless stuff"""
    nums = re.findall(r"\s([^\s]*?)%", log)
    n = len(nums)
    res = []
    for i in range(0, n, n//2):
        usefull_nums = nums[i:i+n//2]
        usefull_nums.pop(5)
        usefull_nums.pop(5)
        res.extend(usefull_nums)
    print(" ".join(res))
    # print(" ".join(nums))


def get_metrics(log_file):
    """Strip log of all the useless stuff"""
    with open(log_file, "r") as f:
        lines = f.readlines()
    log = ''.join(lines[-5:])
    nums = re.findall(r"\s([^\s]*?)%", log)
    n = len(nums)
    res = []
    n_dataset = 1
    for i in range(0, n, n // n_dataset):
        usefull_nums = nums[i:i+n//n_dataset]
        # usefull_nums.pop(5)
        # usefull_nums.pop(5)
        res.extend(usefull_nums)
    # UCM, LCM, UAS, LAS
    return " ".join(res[3:7])


def main():
    # for seed in [0, 2]:
    data = 'ctb7'
    encoder = 'bert'
    for seed in [0, 1, 2, 3]:
        # log_file = f"tacl/log/{data}/dep-sd.explicit-char-crf-dep.leftward.{encoder}.seed{seed}.train"
        # log_file = f"tacl/log/{data}/dep-sd.explicit-char-crf-dep.rightward.{encoder}.seed{seed}.train"
        # log_file = f"tacl/log/{data}/dep-sd.latent-char-crf-dep.label-loss=crf.{encoder}.seed{seed}.train"
        # log_file = f"tacl/log/{data}/dep-sd.latent-char-crf-dep.label-loss=marginal.marginal=denominator.{encoder}.seed{seed}.train"
        # log_file = f"tacl/log/{data}/dep-sd.latent-char-crf-dep.label-loss=marginal.marginal=numerator.{encoder}.seed{seed}.train"
        # log_file = f"tacl/log/{data}/dep-sd.coarse2fine-char-crf-dep.label-loss=marginal.marginal=numerator.{encoder}.seed{seed}.test.train"
        # log_file = f"tacl/log/{data}/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.{encoder}.seed{seed}.test.train"
        # log_file = f"arr/log/{data}/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.seed{seed}.train"

        # log_file = f"arr/log/{data}/dep-sd.explicit-char-crf-dep.leftward.bert.seed{seed}.train"
        # log_file = f"arr/log/{data}/dep-sd.explicit-char-crf-dep.rightward.bert.seed{seed}.train"
        # log_file = f"arr/log/{data}/dep-sd.latent-char-crf-dep.label-loss=crf.bert.seed{seed}.train"
        # log_file = f"arr/log/{data}/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.seed{seed}.train"

        # penn2malt
        log_file = f"arr/log/{data}/dep-malt.coarse2fine-char-crf-dep.label-loss=crf.bert.seed{seed}.train"

        get_metrics(log_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str)
    # parser.add_argument("--corpus", type=str, required=True)
    # parser.add_argument("--data", type=str, required=True)
    # parser.add_argument("--encoder", type=str, default="bert")
    # parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])

    args = parser.parse_args()

    res = []
    for ext in ["dev", "test"]:
        res.append(get_metrics(args.log_file + "." + ext))
    print(" ".join(res))
