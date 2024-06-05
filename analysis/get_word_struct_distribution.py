# -*- coding=utf-8 -*-

from collections import defaultdict

from word_struct_utility import UnlabeledWordStruct


def get_struct_distribution(structs, topk=10000):
    """词内部结构分布."""
    """词内部结构分布."""
    struct_dict = defaultdict(int)
    length_dict = defaultdict(int)

    for word, struct in structs.items():
        # print(word, struct)
        struct_dict[struct.heads] += 1
        length_dict[len(struct)] += 1

    for struct, count in sorted(struct_dict.items(), key=lambda x: x[1], reverse=True)[:topk]:
        print('_'.join(map(str, struct)), count, f"{count/length_dict[len(struct)]:.2%}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, default=None, help='pred file')
    parser.add_argument('--from_word', action='store_true', help='pred file')
    args = parser.parse_args()

    pred_file = args.pred_file

    # pred_file = 'results/ctb5-big/ctb5-big.latent-char-crf-dep.bert.label-loss=crf.struct-norm=token.seed3.pred'
    # pred_file= "tacl/results/ctb5-big/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.seed1.test.pred"
    # pred_file = "arr-231228/results-231228/ctb5-big/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.seed1.pred"

    # pred_file = "arr/results/ctb5/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.test"

    # pred_file = "arr/results/ctb7/dep-sd.latent-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.test"
    # pred_file = "arr/results/ctb7/dep-malt.latent-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.test"
    # pred_file = "arr/results/ctb7/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.test"
    # pred_file = "arr/results/ctb7/dep-malt.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.test"

    # pred_file = "arr/results/ctb7/dep-sd.latent-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.gold-seg.test"
    # pred_file = "arr/results/ctb7/dep-malt.latent-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.gold-seg.test"
    # pred_file = "arr/results/ctb7/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.gold-seg.test"
    # pred_file = "arr/results/ctb7/dep-malt.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=10.seed0.pred.gold-seg.test"

    if args.from_word:
        # UnlabeledWordStruct.load_struct_from_word('data/iwdp/ctb5_iwdp_all.conllx')
        pred_structs = UnlabeledWordStruct.load_struct_from_word(pred_file)
    else:
        pred_structs = UnlabeledWordStruct.load_struct_from_sent(pred_file)

    get_struct_distribution(pred_structs)


"""
# pred_file = 'results/ctb5-big/ctb5-big.latent-char-crf-dep.bert.label-loss=crf.struct-norm=token.seed3.pred'
    # pred_file= "tacl/results/ctb5-big/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.seed1.test.pred"
    # pred_file = "arr-231228/results-231228/ctb5-big/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.seed1.pred"

    # pred_file = "arr/results/ctb5/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.test"

    # pred_file = "arr/results/ctb7/dep-sd.latent-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.test"
    # pred_file = "arr/results/ctb7/dep-malt.latent-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.test"
    # pred_file = "arr/results/ctb7/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.test"
    # pred_file = "arr/results/ctb7/dep-malt.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.test"

    # pred_file = "arr/results/ctb7/dep-sd.latent-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.gold-seg.test"
    # pred_file = "arr/results/ctb7/dep-malt.latent-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.gold-seg.test"
    # pred_file = "arr/results/ctb7/dep-sd.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=10.seed1.pred.gold-seg.test"
    # pred_file = "arr/results/ctb7/dep-malt.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=10.seed0.pred.gold-seg.test"
"""
