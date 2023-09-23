#!/bin/bash

loss='consensus_forth'
alpha=1
models_num=3
detach=1
learnable_q=1

epoch=30
alpha_list=(2 3)
gpu_list=(1 1 1)
prefix='forth_debug'
for i in ${!alpha_list[@]};do
    gpu=${gpu_list[i]}
    alpha=${alpha_list[i]}
    save=ckpt/consensus/${prefix}_loss${loss}_models${models_num}_aplha${alpha}_epoch${epoch}.pt
    experiment_name=consensus_${prefix}_loss${loss}_models${models_num}_aplha${alpha}_epochs${epochs}_gpu${gpu}
    folder_name=logs/consensus
    if [ ! -d ${folder_name} ]; then
        mkdir -p ${folder_name}
    fi
    log_filename=${folder_name}/${prefix}_loss${loss}_models${models_num}_aplha${alpha}_epoch${epoch}_gpu${gpu}.log
    nohup python -u consensus.py \
    --exp_name=${experiment_name} \
    --loss=${loss} \
    --models_num=${models_num} \
    --detach=${detach} \
    --learnable_q=${learnable_q} \
    --alpha=${alpha} \
    --gpu=${gpu} \
    --epochs=${epoch} \
    --save=${save} \
    > ${log_filename} 2>&1 &
done