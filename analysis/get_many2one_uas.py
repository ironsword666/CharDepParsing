# -*- coding=utf-8 -*-

from collections import defaultdict

from word_struct_utility import UnlabeledWordStruct


def cal_many2one_cm(pred_file, gold_file, seeds):
    gold_structs = UnlabeledWordStruct.load_struct_from_word(gold_file)
    structs = defaultdict(list)
    for seed in seeds:
        pred_structs = UnlabeledWordStruct.load_struct_from_sent(pred_file.format(seed), ignore_repeat=False)
        update_structs(structs, pred_structs)

    print(pred_file)
    print(len(structs))
    total = 0
    correct = 0
    for word, struct in structs.items():
        if word not in gold_structs:
            continue
        if len(word) == 1:
            continue
        gold_struct = gold_structs[word]
        total += 1
        for s in struct:
            if s == gold_struct:
                correct += 1
                break

    print(correct, total, f'{correct/total:.2%}')


def cal_cm(pred_file, gold_file, seeds):
    metrics = []
    gold_structs = UnlabeledWordStruct.load_struct_from_word(gold_file)
    for seed in seeds:
        pred_structs = UnlabeledWordStruct.load_struct_from_sent(pred_file.format(seed), ignore_repeat=False)
        total = 0
        correct = 0
        for word, struct in pred_structs.items():
            if word not in gold_structs:
                continue
            if len(word) == 1:
                continue
            gold_struct = gold_structs[word]
            total += 1
            for s in struct:
                if s == gold_struct:
                    correct += 1
                    break

        metrics.append(correct/total)
        
        print(correct, total, f'{correct/total:.2%}')
    
    mean = sum(metrics) / len(metrics)
    variance = sum((v - mean) ** 2 for v in metrics) / len(metrics)

    print(f"Average: {mean:.2%}")
    print(f'Variance: {variance:.2%}')


def update_structs(structs, pred_structs):
    for word, struct in pred_structs.items():
        for s in struct:
            if s not in structs[word]:
                structs[word].append(s)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, default="arr/results/ctb6/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=10.seed{}.pred.gold-seg.test", help='pred file')
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3], help="seeds")
    parser.add_argument('--gold_file', default="data/wist-main/WIST.conll", help='pred file')
    args = parser.parse_args()

    print(args.pred_file)
    cal_cm(args.pred_file, args.gold_file, args.seeds)