#! /bin/bash


for tuple in 0,3 1,4 2,5 3,6; do
    IFS=","; set -- $tuple; 
    seed=$1
    device=$2
    mode="train"
    corpus_name="ctb6"
    mkdir -p "arr/log/${corpus_name}"
    data_name="dep-malt"
    data_dir="data/${corpus_name}/${data_name}"
    encoder="bert"
    label_loss="crf"
    epochs=10
    mode="train"
    method="${data_name}.latent-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
    CUDA_VISIBLE_DEVICES=$device nohup python -m supar.cmds.latent_char_crf_dep train \
            --conf config/ctb.char_dep.${encoder}.ini  \
            --build \
            --device 0 \
            --seed ${seed} \
            --path arr/exp/${corpus_name}/${method}/model \
            --proj \
            --partial \
            --span_mask \
            --combine_mask \
            --label_loss ${label_loss} \
            --encoder bert \
            --bert bert-base-chinese \
            --epochs ${epochs} \
            --batch-size 1000 \
            --train ${data_dir}/train.conllx \
            --dev ${data_dir}/dev.conllx \
            --test ${data_dir}/test.conllx \
            > arr/log/${corpus_name}/${method}.${mode} 2>&1 &
done