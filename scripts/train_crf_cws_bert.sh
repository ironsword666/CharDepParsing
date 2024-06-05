#! /bin/bash


for tuple in 0,3 1,4 2,5 3,6; do
    IFS=","; set -- $tuple; 
    seed=$1
    device=$2
    corpus_name="ctb7"
    mkdir -p "arr/log/${corpus_name}"
    data_name="segment"
    data_dir="data/${corpus_name}/${data_name}"
    encoder="bert"
    bert="bert-base-chinese"
    epochs=10
    mode="train"
    method="${data_name}.crf-cws.${encoder}.epc=${epochs}.seed${seed}"
    CUDA_VISIBLE_DEVICES=$device nohup python -u -m supar.cmds.crf_cws train \
            --conf config/ctb.cws.${encoder}.ini  \
            --build \
            --device 0 \
            --seed ${seed} \
            --path arr/exp/${corpus_name}/${method}/model \
            --encoder ${encoder} \
            --bert ${bert} \
            --epochs ${epochs} \
            --batch-size 1000 \
            --train ${data_dir}/train.seg \
            --dev ${data_dir}/dev.seg \
            --test ${data_dir}/test.seg \
            > arr/log/${corpus_name}/${method}.${mode} 2>&1 &
done