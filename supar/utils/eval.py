# -*- coding: utf-8 -*-

import subprocess


def get_conll05_results_with_official_script(gold_path, pred_path, script_path):
    # 'data/conll05-original-style/eval.sh'
    _SRL_CONLL_EVAL_SCRIPT = script_path
    child = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, gold_path, pred_path), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_info = child.communicate()[0]
    child2 = subprocess.Popen('sh {} {} {}'.format(
        _SRL_CONLL_EVAL_SCRIPT, pred_path, gold_path), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_info2 = child2.communicate()[0]
    # os.remove(tgt_temp_file)
    conll_recall = float(str(eval_info).strip().split("Overall")[1].strip().split('\\n')[0].split()[4])
    conll_precision = float(str(eval_info2).strip().split("Overall")[1].strip().split('\\n')[0].split()[4])
    conll_f1 = 2 * conll_recall * conll_precision / (conll_recall + conll_precision + 1e-12)

    return conll_precision/100, conll_recall/100, conll_f1/100
