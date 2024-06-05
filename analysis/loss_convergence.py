# -*- encodings: utf-8 -*-

import argparse
import re
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def extract_loss(log_file):
    with open(log_file, 'r') as f:
        text = f.read()
    "lr: 4.7253e-07 - loss: 0.1487arc loss: 0.12935549020767212, rel loss: 0.009955358691513538"
    # epochs = re.findall(r'lr: .*? - loss: .*?arc loss: (.*?), rel loss: (.*?)\n', text)
    results = re.findall(r'(s/it|it/s), lr: .*? - loss: (.*?)\n', text)
    print(len(results))
    values = []
    prev = float(results[0][1].strip())
    for i in range(1, len(results)):
        value = float(results[i][1].strip())
        if value == prev:
            continue
        prev = value
        values.append(value)
    return values


def draw(values: List[float], save_file):
    """
    Draw the loss convergence curve
    """
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(values)), values)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    # plt.legend(['explicit loss', 'latent loss'])
    plt.savefig(save_file)
    plt.show()


def draw_multi_loss(values, legend, save_file):
    """
    Draw the loss convergence curve
    """
    plt.figure(figsize=(8, 6))
    for i in range(len(values)):
        plt.plot(np.arange(len(values[i])), values[i])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(legend)
    plt.savefig(save_file)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Loss Convergence')
    parser.add_argument('--log', type=str, default='log.txt', help='log file')
    parser.add_argument('--save', type=str, default='loss_convergence.png', help='save file')
    args = parser.parse_args()

    values = extract_loss(args.log)
    draw(values, args.save)


if __name__ == '__main__':
    # main()
    loss_list = []
    for log_file in ['arr/log/ctb9/dep-malt.explicit-char-crf-dep.leftward.bert.epc=5.seed0.train',
                     'arr/log/ctb9/dep-malt.coarse2fine-char-crf-dep.label-loss=crf.bert.epc=5.seed0.train']:
        loss_list.append(extract_loss(log_file))
    draw_multi_loss(loss_list, ['explicit loss', 'latent loss'], 'loss_convergence.png')
