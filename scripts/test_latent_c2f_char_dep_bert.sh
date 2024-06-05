#!/bin/bash
get_free_gpu () {
    gpu_id=$1
    # if free_time is larger than duration, then get the gpu
    duration=$2
    # indicate whether the gpu is free
    flag=0
    while true
    do
        gpu_mem=$(nvidia-smi -a -i ${gpu_id} | sed -En "s/^.*Used.*:\s(.*)\sMiB.*$/\1/p" | head -n1)
        # stat=$(nvidia-smi -a -i 1 | grep 'Used' | head -n1 | sed -En "s/^.*:\s(.*)\sMiB.*$/\1/p")
        # stat=$(nvidia-smi -a -i ${id} | grep 'Used' | head -n1)
        if [  ${gpu_mem} -le ${3:-10} ];then
            # gpu is free for a time
            if [ ${flag} -eq 1 ];then
                free_time=$(($(date +%s)-${start}))
                if [ ${free_time} -ge ${duration} ];then
                    echo 0
                    exit
                fi
            # gpu is just free
            else
                flag=1
                start=$(date +%s)
            fi
        else
            flag=0
        fi
        sleep 1
    done

}

seed=0
corpus_name="ctb7"
data_name="dep-sd"
data_dir="data/${corpus_name}/${data_name}"
label_loss="crf"
mode="test"
method="${data_name}.coarse2fine-char-crf-dep.label-loss=${label_loss}.save-all-epc.{encoder}.epc=10.seed${seed}"
CUDA_VISIBLE_DEVICES=0 nohup python -m supar.cmds.coarse2fine_char_crf_dep test \
        --conf config/ctb.char_dep.bert.ini  \
        --device 0 \
        --path arr/exp/${corpus_name}/${method}/model \
        --proj \
        --batch-size 1000 \
        --use_gold_tree \
        --data ${data_dir}/dev.conllx \
        --pred arr/results/${corpus_name}/${method}.pred \
        > arr/log/${corpus_name}/${method}.${mode} 2>&1 &