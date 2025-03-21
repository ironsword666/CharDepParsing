# -*- coding: utf-8 -*-

import argparse

from supar import SplitCoarse2FineCharCRFDependencyParser
from supar.cmds.cmd import parse


def main():
    parser = argparse.ArgumentParser(description='Create first-order CRF Dependency Parser.')
    parser.set_defaults(Parser=SplitCoarse2FineCharCRFDependencyParser)
    parser.add_argument('--mbr', action='store_true', help='whether to use MBR decoding')
    parser.add_argument('--tree', action='store_true', help='whether to ensure well-formedness')
    parser.add_argument('--proj', action='store_true', help='whether to projectivize the data')
    parser.add_argument('--partial', action='store_true', help='whether partial annotation is included')
    parser.add_argument('--use_gold_seg', action='store_true', help='whether to use gold segmentation')
    parser.add_argument('--use_gold_tree', action='store_true', help='whether to use gold tree')
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', choices=['bichar', 'trichar', 'bert'], nargs='+', help='features to use')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--encoder', choices=['lstm', 'bert'], default='lstm', help='encoder to use')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default='data/ptb/train.conllx', help='path to train file')
    subparser.add_argument('--dev', default='data/ptb/dev.conllx', help='path to dev file')
    subparser.add_argument('--test', default='data/ptb/test.conllx', help='path to test file')
    subparser.add_argument('--char_embed', help='path to pretrained embeddings')
    subparser.add_argument('--bichar_embed', help='path to pretrained embeddings')
    subparser.add_argument('--trichar_embed',  help='path to pretrained embeddings')
    subparser.add_argument('--unk', help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed', default=100, type=int, help='dimension of embeddings')
    subparser.add_argument('--n_pretrained', default=100, type=int, help='dimension of pretrained embeddings')
    subparser.add_argument('--bert', default='bert-base-chinese', help='which BERT model to use')
    subparser.add_argument('--c_span_mask', action='store_true', help='whether to use complete span mask')
    subparser.add_argument('--i_span_mask', action='store_true', help='whether to combine mask')
    subparser.add_argument('--root_as_head', action='store_false', help='whether to use root as head')
    subparser.add_argument('--struct_normalizer', default='token', choices=['token', 'sentence', 'target'], help='which struct normalizer to use')
    subparser.add_argument('--label_loss', default='vanilla', choices=['vanilla', 'marginal', 'best', 'crf'], help='use which label loss')
    subparser.add_argument('--marginal', default=None, choices=['numerator', 'denominator'], help='use which part to cal marginals')
    subparser.add_argument('--loss_weight', default=0.7, type=float, help='weight of the label loss')
    subparser.add_argument('--shared_arc_info', default=None, choices=[None, 'arc', 'inter_arc', 'repr', 'one_biaffine', 'two_biaffine'], help='whether to share information between intra and inter arc')
    subparser.add_argument('--arc_combination', default='sum', choices=['sum', 'cat'], help='whether to share information between intra and inter arc')
    subparser.add_argument('--use_orth_loss', action='store_true', help='whether to share information between intra and inter arc')
    subparser.add_argument('--cos_margin', type=float, default=0, help='margin for cos loss')
    subparser.add_argument('--orth_hiddens', choices=['d', 'h', 'dh', 'hd'], nargs='+', help='what type of hiddens to orth')
    subparser.add_argument('--orth_loss_weight', default=0, type=float, help='whether to share information between intra and inter arc')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx', help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx', help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllx', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    # test
    subparser = subparsers.add_parser('test', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx', help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllx', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    parse(parser)


if __name__ == "__main__":
    main()
