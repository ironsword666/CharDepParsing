# -*- coding: utf-8 -*-

import argparse

from supar import CRFWordSegmenter
from supar.cmds.cmd import parse


def main():
    parser = argparse.ArgumentParser(description='Create CRF-based Chinese Word Segmentation.')
    parser.set_defaults(Parser=CRFWordSegmenter)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', choices=['bichar', 'trichar', 'bert'], nargs='+', help='features to use')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--encoder', choices=['lstm', 'bert'], default='lstm', help='encoder to use')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default='data/ptb/train.conllx', help='path to train file')
    subparser.add_argument('--dev', default='data/ptb/dev.conllx', help='path to dev file')
    subparser.add_argument('--test', default='data/ptb/test.conllx', help='path to test file')
    subparser.add_argument('--char_embed', help='path to pretrained embeddings')
    subparser.add_argument('--bichar_embed', help='path to pretrained embeddings')
    subparser.add_argument('--trichar_embed', help='path to pretrained embeddings')
    subparser.add_argument('--unk', help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed', default=100, type=int, help='dimension of embeddings')
    subparser.add_argument('--n_pretrained', default=100, type=int, help='dimension of pretrained embeddings')
    subparser.add_argument('--bert', default='bert-base-chinese', help='which BERT model to use')
    subparser.add_argument('--struct_normalizer', default='token', choices=['token', 'sentence', 'target'], help='which struct normalizer to use')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx', help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx', help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllx', help='path to predicted result')
    subparser.add_argument('--output_toconll', action='store_true', help='whether to output in conllx format')
    parse(parser)


if __name__ == "__main__":
    main()
