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


function pipeline_predict() {
    seed=$1
    device=$2
    corpus_name=$3
    mkdir -p "arr/log/${corpus_name}"
    data_name="segment"
    data_dir="data/${corpus_name}/${data_name}"
    data_part=$4
    encoder="bert"
    mode="predict"
    method="${data_name}.crf-cws.${encoder}.seed${seed}"
    CUDA_VISIBLE_DEVICES=$device nohup python -u -m supar.cmds.crf_cws predict \
            --conf config/ctb.cws.${encoder}.ini  \
            --device "$(get_free_gpu "$device" 0)" \
            --seed ${seed} \
            --output_toconll \
            --path arr/exp/${corpus_name}/${method}/model \
            --data ${data_dir}/${data_part}.seg \
            --pred arr/results/${corpus_name}/${method}.${data_part}.conllx.pred \
            > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 

    cws_pred="arr/results/${corpus_name}/${method}.${data_part}.conllx.pred"

    data_name="dep-sd"
    data_dir="data/${corpus_name}/${data_name}"
    data_part=$4
    encoder="bert"
    mode="predict"
    method="${data_name}.word-crf-dep.${encoder}.seed${seed}"
    CUDA_VISIBLE_DEVICES=$device nohup python -u -m supar.cmds.crf_dep predict \
            --conf config/ptb.dep.${encoder}.ini  \
            --device "$(get_free_gpu "$device" 0)" \
            --seed ${seed} \
            --path arr/exp/${corpus_name}/${method}/model \
            --tree \
            --proj \
            --data ${cws_pred} \
            --pred arr/results/${corpus_name}/${method}.${data_part}.conllx.pred \
            > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 
}


function pipeline_eval() {
    seed=$1
    device=$2
    corpus_name=$3
    mkdir -p "arr/log/${corpus_name}"
    data_name="segment"
    data_dir="data/${corpus_name}/${data_name}"
    data_part=$5
    encoder="bert"
    epochs=10
    mode="predict"
    method="${data_name}.crf-cws.${encoder}.epc=${epochs}.seed${seed}"
    # method="${data_name}.crf-cws.${encoder}.seed${seed}"
    CUDA_VISIBLE_DEVICES=$device nohup python -u -m supar.cmds.crf_cws predict \
            --conf config/ctb.cws.${encoder}.ini  \
            --device "$(get_free_gpu "$device" 0)" \
            --seed ${seed} \
            --output_toconll \
            --path arr/exp/${corpus_name}/${method}/model \
            --data ${data_dir}/${data_part}.seg \
            --pred arr/results/${corpus_name}/${method}.${data_part}.conllx.pred \
            > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 

    cws_pred="arr/results/${corpus_name}/${method}.${data_part}.conllx.pred"

    data_name=$4
    # data_name="dep-malt"
    data_dir="data/${corpus_name}/${data_name}"
    data_part=$5
    encoder="bert"
    epochs=10
    mode="predict"
    # method="${data_name}.word-crf-dep.${encoder}.seed${seed}"
    method="${data_name}.word-crf-dep.${encoder}.epc=${epochs}.seed${seed}"
    CUDA_VISIBLE_DEVICES=$device nohup python -u -m supar.cmds.crf_dep predict \
            --conf config/ptb.dep.${encoder}.ini  \
            --device "$(get_free_gpu "$device" 0)" \
            --seed ${seed} \
            --path arr/exp/${corpus_name}/${method}/model \
            --tree \
            --proj \
            --data ${cws_pred} \
            --pred arr/results/${corpus_name}/${method}.${data_part}.conllx.pred \
            > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 

    mode="pipeline"
    dep_pred="arr/results/${corpus_name}/${method}.${data_part}.conllx.pred"
    python -m eval.eval_pipeline \
        --pred_file ${dep_pred} \
        --gold_file ${data_dir}/${data_part}.conllx \
        > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 
}

# for tuple in 0,1 1,2 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     corpus_name="ctb5"
#     data_part="test"
#     pipeline_eval "$1" "$2" $corpus_name $data_part &

# done

# for tuple in 0,1 1,2 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     corpus_name="ctb5"
#     data_part="dev"
#     pipeline_eval "$1" "$2" $corpus_name $data_part &

# done

# for tuple in 0,1 1,2 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     corpus_name="ctb5-big"
#     data_part="dev"
#     pipeline_eval "$1" "$2" $corpus_name $data_part &

# done

# for tuple in 0,1 1,2 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     corpus_name="ctb5-big"
#     data_part="test"
#     pipeline_eval "$1" "$2" $corpus_name $data_part &

# done

# for tuple in 0,1 1,2 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     corpus_name="ctb7"
#     data_part="dev"
#     pipeline_eval "$1" "$2" $corpus_name $data_part &

# done

# for tuple in 0,1 1,2 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     corpus_name="ctb7"
#     data_part="test"
#     pipeline_eval "$1" "$2" $corpus_name $data_part &

# done

# for tuple in 0,1 1,2 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     # ctb5, ctb6, ctb7, ctb9
#     corpus_name="ctb7"
#     # dev or test
#     data_part="dev"
#     pipeline_eval "$1" "$2" $corpus_name $data_part &

# done

# for tuple in 0,3 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     # ctb5, ctb6, ctb7, ctb9
#     corpus_name="ctb7"
#     # dev or test
#     data_part="test"
#     pipeline_eval "$1" "$2" $corpus_name $data_part &

# done
for corpus_name in ctb5 ctb6 ctb7; do
    for data_name in dep-sd dep-malt; do
        for data_part in dev test; do
            for tuple in 0,3 1,4 2,5 3,6; do
                IFS=","; set -- $tuple; 
                # if the backgroup process is more than 4, then wait
                # exclude the grep process
                while [ $(ps -ef | grep "eval_pipeline" | grep -v "grep" | wc -l) -ge 4 ]
                do
                    sleep 10
                done
                pipeline_eval "$1" "$2" $corpus_name $data_name $data_part &
            done
        done
    done
done