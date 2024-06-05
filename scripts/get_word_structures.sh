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
        if [  ${gpu_mem} -le ${3:-11} ];then
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

# for tuple in 0,1 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb7"
#     data_name="dep-sd"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     label_loss="crf"
#     epochs=10
#     mode="test"
#     method="${data_name}.coarse2fine-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.coarse2fine_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 0) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 &
# done

# for tuple in 0,1 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb7"
#     data_name="dep-malt"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     label_loss="crf"
#     epochs=10
#     mode="test"
#     method="${data_name}.coarse2fine-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.coarse2fine_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 20 9) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 &
# done

# for tuple in 0,1 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb7"
#     data_name="dep-sd"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     label_loss="crf"
#     epochs=10
#     mode="test"
#     method="${data_name}.latent-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.latent_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 40 9) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 &
# done

# for tuple in 0,1 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb7"
#     data_name="dep-malt"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     label_loss="crf"
#     epochs=10
#     mode="test"
#     method="${data_name}.latent-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.latent_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 60 9) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 &
# done

# # use gold seg

# for tuple in 0,1 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb7"
#     data_name="dep-sd"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     label_loss="crf"
#     epochs=10
#     mode="test"
#     method="${data_name}.coarse2fine-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.coarse2fine_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 80 9) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --use_gold_seg \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.gold-seg.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.gold-seg.${data_part} 2>&1 &
# done

# for tuple in 0,1 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb7"
#     data_name="dep-malt"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     label_loss="crf"
#     epochs=10
#     mode="test"
#     method="${data_name}.coarse2fine-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.coarse2fine_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 100 9) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --use_gold_seg \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.gold-seg.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.gold-seg.${data_part} 2>&1 &
# done

# for tuple in 0,1 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb7"
#     data_name="dep-sd"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     label_loss="crf"
#     epochs=10
#     mode="test"
#     method="${data_name}.latent-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.latent_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 0 9) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --use_gold_seg \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.gold-seg.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.gold-seg.${data_part} 2>&1 &
# done

# for tuple in 0,1 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb7"
#     data_name="dep-malt"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     label_loss="crf"
#     epochs=10
#     mode="test"
#     method="${data_name}.latent-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.latent_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 30 9) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --use_gold_seg \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.gold-seg.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.gold-seg.${data_part} 2>&1 &
# done

# for tuple in 0,1 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb5"
#     data_name="dep-sd"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     label_loss="crf"
#     epochs=10
#     mode="test"
#     method="${data_name}.coarse2fine-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.coarse2fine_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 0) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 &
# done

# for tuple in 0,1 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb6"
#     data_name="dep-sd"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     label_loss="crf"
#     epochs=10
#     mode="test"
#     method="${data_name}.coarse2fine-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.coarse2fine_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 0) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 &
# done

# for tuple in 0,1 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb6"
#     data_name="dep-malt"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     label_loss="crf"
#     epochs=10
#     mode="test"
#     method="${data_name}.coarse2fine-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.coarse2fine_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 20 9) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 &
# done

# for tuple in 0,1 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb6"
#     data_name="dep-sd"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     label_loss="crf"
#     epochs=10
#     mode="test"
#     method="${data_name}.latent-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.latent_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 40 9) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 &
# done

# for tuple in 0,1 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb6"
#     data_name="dep-malt"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     label_loss="crf"
#     epochs=10
#     mode="test"
#     method="${data_name}.latent-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.latent_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 60 9) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.${data_part} 2>&1 &
# done

# for tuple in 0,3 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb6"
#     data_name="dep-malt"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     epochs=10
#     mode="test"
#     # leftward
#     orientation="leftward"
#     method="${data_name}.explicit-char-crf-dep.${orientation}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.explicit_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 60 9) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --tree \
#             --proj \
#             --orientation ${orientation} \
#             --use_gold_seg \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.gold-seg.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.gold-seg.${data_part} 2>&1 &
# done

# for tuple in 0,3 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb6"
#     data_name="dep-malt"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     label_loss="crf"
#     epochs=10
#     mode="test"
#     method="${data_name}.coarse2fine-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.coarse2fine_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 30 9) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --use_gold_seg \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.gold-seg.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.gold-seg.${data_part} 2>&1 &
# done

# for tuple in 0,3 1,4 2,5 3,6; do
#     IFS=","; set -- $tuple; 
#     seed=$1
#     device=$2
#     corpus_name="ctb6"
#     data_name="dep-malt"
#     data_dir="data/${corpus_name}/${data_name}"
#     data_part="test"
#     encoder="bert"
#     label_loss="crf"
#     epochs=10
#     mode="test"
#     method="${data_name}.latent-char-crf-dep.label-loss=${label_loss}.${encoder}.epc=${epochs}.seed${seed}"
#     CUDA_VISIBLE_DEVICES=${device} nohup python -m supar.cmds.latent_char_crf_dep ${mode} \
#             --device $(get_free_gpu $device 0) \
#             --path arr/exp/${corpus_name}/${method}/model \
#             --use_gold_seg \
#             --batch-size 2000 \
#             --data ${data_dir}/${data_part}.conllx \
#             --pred arr/results/${corpus_name}/${method}.pred.gold-seg.${data_part} \
#             > arr/log/${corpus_name}/${method}.${mode}.gold-seg.${data_part} 2>&1 &
# done

