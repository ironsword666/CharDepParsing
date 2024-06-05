#! /bin/bash



for tuple in 0,3 1,4 2,5 3,6; do
    IFS=","; set -- $tuple; 
    seed=$1
    device=$2
    corpus_name="ctb6"
    mkdir -p "arr/log/${corpus_name}"
    data_name="dep-malt"
    data_dir="data/${corpus_name}/${data_name}"
    mode="train"
    encoder="bert"
    bert="bert-base-chinese"
    epochs=10
    method="${data_name}.word-crf-dep.${encoder}.epc=${epochs}.seed${seed}"
    CUDA_VISIBLE_DEVICES=$device nohup python -u -m supar.cmds.crf_dep train \
            --conf config/ptb.dep.${encoder}.ini  \
            --build \
            --device 0 \
            --seed ${seed} \
            --path arr/exp/${corpus_name}/${method}/model \
            --proj \
            --tree \
            --encoder ${encoder} \
            --bert ${bert} \
            --epochs ${epochs} \
            --batch-size 1000 \
            --update_steps 2 \
            --train ${data_dir}/train.conllx \
            --dev ${data_dir}/dev.conllx \
            --test ${data_dir}/test.conllx \
            > arr/log/${corpus_name}/${method}.${mode} 2>&1 &
done