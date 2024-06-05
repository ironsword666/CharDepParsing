# -*- coding: utf-8 -*-

import argparse

from supar import Const2DepEmbeddingParser
from supar.cmds.cmd import parse


def main():
    parser = argparse.ArgumentParser(description='Create first-order CRF Dependency Parser.')
    parser.set_defaults(Parser=Const2DepEmbeddingParser)
    parser.add_argument('--mbr', action='store_true', help='whether to use MBR decoding')
    parser.add_argument('--tree', action='store_true', help='whether to ensure well-formedness')
    parser.add_argument('--proj', action='store_true', help='whether to projectivize the data')
    parser.add_argument('--partial', action='store_true', help='whether partial annotation is included')
    parser.add_argument('--use_gold_tree', action='store_true', help='whether to use gold segmentation')
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', default=[], choices=['tag'], nargs='+', help='features to use')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--encoder', choices=['lstm', 'bert'], default='bert', help='encoder to use')
    subparser.add_argument('--punct', action='store_true', help='whether to include punctuation')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default='data/ptb/train.conllx', help='path to train file')
    subparser.add_argument('--dev', default='data/ptb/dev.conllx', help='path to dev file')
    subparser.add_argument('--test', default='data/ptb/test.conllx', help='path to test file')
    subparser.add_argument('--embed', help='path to pretrained embeddings')
    subparser.add_argument('--unk', help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed', default=100, type=int, help='dimension of embeddings')
    subparser.add_argument('--n_pretrained', default=100, type=int, help='dimension of pretrained embeddings')
    subparser.add_argument('--bert', default='bert-base-chinese', help='which BERT model to use')
    subparser.add_argument('--c_span_mask', action='store_true', help='whether to use complete span mask')
    subparser.add_argument('--combine_mask', action='store_true', help='whether to combine mask')
    subparser.add_argument('--struct_normalizer', default='token', choices=['token', 'sentence', 'target'], help='which struct normalizer to use')
    subparser.add_argument('--marginal', default=None, choices=['numerator', 'denominator'], help='use which part to cal marginals')
    subparser.add_argument('--freeze', action='store_true', help='whether freeze embeddings')
    subparser.add_argument('--use_token_type', action='store_true', help='whether use token type embeddings')
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
