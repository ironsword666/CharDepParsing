#! /bin/bash


seed=0
corpus_name="ctb7"
data_name="dep-malt"
data_dir="data/${corpus_name}/${data_name}"
encoder="bert"
epochs=5
label_loss="crf"
mode="predict"
method="${data_name}.coarse2fine-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
CUDA_VISIBLE_DEVICES=6 nohup python -m supar.cmds.coarse2fine_char_crf_dep predict \
        --device 0 \
        --path arr/exp/${corpus_name}/${method}/model \
        --proj \
        --partial \
        --data ${data_dir}/dev.conllx \
        --pred arr/results/${corpus_name}/${method}.pred \
        > arr/log/${corpus_name}/${method}.${mode} 2>&1 &