# -*- coding: utf-8 -*-

import torch
from supar.utils import Config
from supar.utils.logging import init_logger, logger
from supar.utils.parallel import init_device


def parse(parser):
    parser.add_argument('--path', '-p', help='path to model file')
    parser.add_argument('--conf', '-c', default='', help='path to config file')
    parser.add_argument('--device', '-d', default='-1', help='ID of GPU to use')
    parser.add_argument('--seed', '-s', default=1, type=int, help='seed for generating random numbers')
    parser.add_argument('--threads', '-t', default=16, type=int, help='max num of threads')
    parser.add_argument('--batch-size', default=5000, type=int, help='batch size')
    parser.add_argument('--epochs', default=1000, type=int, help='number of epochs to train the model')
    parser.add_argument('--lr', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument('--lr_rate', default=20, type=float, help='learning rate multiplier for other layers (except encoder)')
    parser.add_argument('--nu', default='.9', type=float, help='adam params')
    parser.add_argument('--warmup', default=0.1, type=float, help='warmup ratio of all steps')
    parser.add_argument('--update_steps', default=1, type=int, help='gradient accumulation')
    parser.add_argument("--local_rank", type=int, default=-1, help='node rank for distributed training')
    args, unknown = parser.parse_known_args()
    args, unknown = parser.parse_known_args(unknown, args)
    args = Config.load(**vars(args), unknown=unknown)
    Parser = args.pop('Parser')

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    init_device(args.device, args.local_rank)
    init_logger(logger, f"{args.path}.{args.mode}.log")
    logger.info('\n' + str(args))

    if args.mode == 'train':
        parser = Parser.build(**args)
        parser.train(**args)
    elif args.mode == 'evaluate':
        parser = Parser.load(args.path)
        parser.evaluate(**args)
    elif args.mode == 'predict':
        parser = Parser.load(args.path)
        parser.predict(**args)
    elif args.mode == 'test':
        # if args.transform_path is not None:
        #     print('load_from_word_parser')
        #     parser = Parser.load_from_word_parser(args.path, args.transform_path)
        # else:
        #     parser = Parser.load(args.path)
        parser = Parser.load(args.path)
        parser.test(**args)
