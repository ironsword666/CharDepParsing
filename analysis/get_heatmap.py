from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import re


plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.rc('font', family='SimSun', size=12)


def get_data(data_path: str, chars: List[str]):
    # ['<bos>', '金', '杯', '子', '的', '白', '开', '水']
    # transform the char-level sent to string format of list
    sent = str(chars)
    # escape the special characters in regex
    sent = sent.replace('[', r'\[')
    sent = sent.replace(']', r'\]')
    data_strs = []
    flag = False
    with open(data_path, 'r') as fr:
        for line in fr:
            if re.match(rf'{sent}', line):
                flag = True
                continue
            if not flag:
                continue
            # read the data until the next sentence
            if line.startswith('('):
                break
            else:
                data_strs.append(line.strip())

    datas = re.findall(r"tensor\(([\s\S]*?), device='cuda:0'\)", ''.join(data_strs))
    intra_probs = torch.tensor(eval(datas[0]))
    inter_probs = torch.tensor(eval(datas[1]))
    return (intra_probs + inter_probs).tolist()


def heatmap(data_value: List[List], chars: List[str], save_path):
    data = pd.DataFrame(data_value, index=chars, columns=chars)
    # sns.heatmap(data, vmax=1, vmin=0, annot=True, fmt='.2f', cmap='Blues')
    sns.heatmap(data, square=True, fmt='.2f', cmap='RdBu_r', center=0, cbar=False)

    # plt.savefig('analysis/heatmap-c2f-score.png')
    plt.savefig(save_path, bbox_inches='tight', transparent=True, pad_inches=0)
    # plt.show()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--chars', type=str, default='')

    args = parser.parse_args()

    chars = ['<bos>'] + list(args.chars)
    data = get_data(args.data_path, chars)
    chars[0] = '$'
    heatmap(data, chars, args.save_path)




