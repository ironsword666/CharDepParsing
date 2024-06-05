#! /bin/bash



for tuple in 0,0 1,1 2,2 3,3; do
    IFS=","; set -- $tuple; 
    seed=$1
    device=$2
    corpus_name="ctb7"
    mkdir -p "arr/log/${corpus_name}"
    data_name="dep-malt"
    data_dir="data/${corpus_name}/${data_name}"
    mode="train"
    encoder="bert"
    epochs=10
    orientation="rightward" # leftward
    method="${data_name}.explicit-char-crf-dep.${orientation}.${encoder}.epc=${epochs}.seed${seed}"
    CUDA_VISIBLE_DEVICES=$device nohup python -m supar.cmds.explicit_char_crf_dep train \
            --conf config/ctb.char_dep.${encoder}.ini  \
            --build \
            --device 0 \
            --seed ${seed} \
            --path arr/exp/${corpus_name}/${method}/model \
            --tree \
            --proj \
            --orientation ${orientation} \
            --encoder bert \
            --bert bert-base-chinese \
            --epochs ${epochs} \
            --batch-size 1000 \
            --train ${data_dir}/train.conllx \
            --dev ${data_dir}/dev.conllx \
            --test ${data_dir}/test.conllx \
            > arr/log/${corpus_name}/${method}.${mode} 2>&1 &
done
