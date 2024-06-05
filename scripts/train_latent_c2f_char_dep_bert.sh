#! /bin/bash


for tuple in 0,0 1,2 2,4 3,6; do
    IFS=","; set -- $tuple; 
    seed=$1
    device=$2
    mode="train"
    corpus_name="ctb5"
    mkdir -p "arr/log/${corpus_name}"
    data_name="dep-malt"
    data_dir="data/${corpus_name}/${data_name}"
    mode="train"
    encoder="bert"
    epochs=10
    label_loss="crf"
    method="${data_name}.coarse2fine-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
    CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.coarse2fine_char_crf_dep train \
            --conf config/ctb.char_dep.${encoder}.ini  \
            --build \
            --device 0 \
            --seed ${seed} \
            --path arr/exp/${corpus_name}/${method}/model \
            --proj \
            --partial \
            --label_loss ${label_loss} \
            --c_span_mask \
            --i_span_mask \
            --encoder bert \
            --bert bert-base-chinese \
            --epochs ${epochs} \
            --batch-size 1000 \
            --train ${data_dir}/train.conllx \
            --dev ${data_dir}/dev.conllx \
            --test ${data_dir}/test.conllx \
            > arr/log/${corpus_name}/${method}.${mode} 2>&1 &
done