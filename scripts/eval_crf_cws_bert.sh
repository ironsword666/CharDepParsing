#! /bin/bash

function get_free_gpu() {
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
        if [  "${gpu_mem}" -le "${3:-10}" ];then
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
        sleep 3
    done

}


for tuple in 0,1 1,2 2,5 3,6; do
    IFS=","; set -- $tuple; 
    seed=$1
    device=$2
    corpus_name="ctb5"
    mkdir -p "arr/log/${corpus_name}"
    data_name="segment"
    data_dir="data/${corpus_name}/${data_name}"
    encoder="bert"
    mode="eval"
    method="${data_name}.crf-cws.${encoder}.seed${seed}"
    CUDA_VISIBLE_DEVICES=$device nohup python -u -m supar.cmds.crf_cws evaluate \
            --conf config/ctb.cws.${encoder}.ini  \
            --device "$(get_free_gpu "$device" 0)" \
            --seed ${seed} \
            --path arr/exp/${corpus_name}/${method}/model \
            --data ${data_dir}/test.seg \
            > arr/log/${corpus_name}/${method}.${mode} 2>&1 &
done