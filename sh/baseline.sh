#!/bin/bash

loss='consensus_exclude'
alpha=1
models_num=1
detach=1
learnable_q=1

epoch=30
alpha_list=(1 1 1 1 1)
gpu_list=(0)
seed_list=(0)
prefix='baseline_4'
for i in ${!seed_list[@]};do
    seed=${seed_list[i]}
    gpu=${gpu_list[i]}
    alpha=${alpha_list[i]}
    save=ckpt/baseline/${prefix}_seed${seed}_epoch${epoch}.pt
    experiment_name=baseline_${prefix}_loss${loss}_models${models_num}_aplha${alpha}_seed${seed}_epochs${epochs}_gpu${gpu}
    folder_name=logs/baseline
    if [ ! -d ${folder_name} ]; then
        mkdir -p ${folder_name}
    fi
    log_filename=${folder_name}/${prefix}_seed${seed}_epoch${epoch}_gpu${gpu}.log
    nohup python -u main.py \
    --exp_name=${experiment_name} \
    --loss=${loss} \
    --models_num=${models_num} \
    --detach=${detach} \
    --learnable_q=${learnable_q} \
    --alpha=${alpha} \
    --gpu=${gpu} \
    --epochs=${epoch} \
    --save=${save} \
    --seed=${seed} \
    > ${log_filename} 2>&1 &
done