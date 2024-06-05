#! /bin/bash

encoder="bert"
results_archive="arr/joint_parsing.archive"
seeds="0 1 2 3"
# get results for the data_name in specified corpus_name of specified seed
for data_name in "dep-malt" "dep-sd"; do
    echo "The used dependency representation is ${data_name} !"
    for corpus_name in ctb5 ctb6 ctb7; do
        echo "The following results are for ${corpus_name} !"
        echo "Pipeline results:"
        for seed in $seeds; do
            # the dev and test will be automatically concatenated
            log_file="arr/log/${corpus_name}/${data_name}.word-crf-dep.${encoder}.epc=10.seed${seed}.pipeline"
            python -m eval.strip_log_for_pipe --log_file ${log_file}
        done
        echo "Leftward results:"
        for seed in $seeds; do
            # the dev and test will be automatically concatenated
            log_file="arr/log/${corpus_name}/${data_name}.explicit-char-crf-dep.leftward.${encoder}.epc=10.seed${seed}.train"
            python -m eval.strip_log_for_joint --log_file ${log_file}
        done
        # echo "Rightward results:"
        # for seed in $seeds; do
        #     log_file="arr/log/${corpus_name}/${data_name}.explicit-char-crf-dep.rightward.${encoder}.epc=10.seed${seed}.train"
        #     python -m eval.strip_log_for_joint --log_file ${log_file}
        # done
        echo "Latent results:"
        for seed in $seeds; do
            log_file="arr/log/${corpus_name}/${data_name}.latent-char-crf-dep.label-loss=crf.${encoder}.epc=10.seed${seed}.train"
            python -m eval.strip_log_for_joint --log_file ${log_file}
        done
        echo "Latent c2f results:"
        for seed in $seeds; do
            log_file="arr/log/${corpus_name}/${data_name}.coarse2fine-char-crf-dep.label-loss=crf.${encoder}.epc=10.seed${seed}.train"
            python -m eval.strip_log_for_joint --log_file ${log_file}
        done


    done
done