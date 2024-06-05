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


function joint_eval() {
    seed=$1
    device=$2
    corpus_name=$3
    mkdir -p "arr/log/${corpus_name}"
    data_name=$4
    data_dir="data/${corpus_name}/${data_name}"
    data_part=$5
    encoder="bert"
    epochs=10
    mode="evaluate"

    # leftward
    orientation="leftward"
    method="${data_name}.explicit-char-crf-dep.${orientation}.${encoder}.epc=${epochs}.seed${seed}"
    CUDA_VISIBLE_DEVICES=$device python -m supar.cmds.explicit_char_crf_dep $mode \
            --device $(get_free_gpu $device 0) \
            --path arr/exp/${corpus_name}/${method}/model \
            --tree \
            --proj \
            --orientation ${orientation} \
            --use_gold_seg \
            --batch-size 3000 \
            --data ${data_dir}/${data_part}.conllx \
            > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 

    # latent
    label_loss="crf"
    method="${data_name}.latent-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
    CUDA_VISIBLE_DEVICES=$device python -m supar.cmds.latent_char_crf_dep $mode \
            --device $(get_free_gpu $device 0) \
            --path arr/exp/${corpus_name}/${method}/model \
            --proj \
            --partial \
            --use_gold_seg \
            --batch-size 3000 \
            --data ${data_dir}/${data_part}.conllx \
            > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 

    # c2f
    label_loss="crf"
    # method="${data_name}.coarse2fine-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
    method="${data_name}.coarse2fine-char-crf-dep.label-loss=${label_loss}.coarse-word.${encoder}.epc=${epochs}.seed${seed}"
    CUDA_VISIBLE_DEVICES=${device} python -m supar.cmds.coarse2fine_char_crf_dep $mode \
            --device $(get_free_gpu $device 0) \
            --path arr/exp/${corpus_name}/${method}/model \
            --proj \
            --partial \
            --use_gold_seg \
            --batch-size 3000 \
            --data ${data_dir}/${data_part}.conllx \
            > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 
}

# done


for corpus_name in ctb5 ctb6 ctb7; do
    for data_name in dep-sd dep-malt; do
        for data_part in dev test; do
            for tuple in 0,3 1,4 2,5 3,6; do
                IFS=","; set -- $tuple; 
                # if the backgroup process is more than 4, then wait
                # exclude the grep process
                while [ $(ps -ef | grep "eval_joint" | grep -v "grep" | wc -l) -ge 6 ]
                do
                    sleep 1
                done
                joint_eval "$1" "$2" $corpus_name $data_name $data_part &
            done
        done
    done
done


# corpus_name="ctb5"
# data_name="dep-malt"
# for data_part in dev test; do
#     for tuple in 0,3 1,4 2,5 3,6; do
#         IFS=","; set -- $tuple; 
#         # if the backgroup process is more than 4, then wait
#         # exclude the grep process
#         while [ $(ps -ef | grep "eval_joint" | grep -v "grep" | wc -l) -ge 6 ]
#         do
#             sleep 1
#         done
#         joint_eval "$1" "$2" $corpus_name $data_name $data_part &
#     done
# done