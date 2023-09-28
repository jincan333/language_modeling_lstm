#!/bin/bash

loss='consensus_forth'
models_num=2
detach=1
learnable_q=1

epochs=60
alpha_list=(0 0.5 0.8 0.9 1 1.1 1.2 1.5)
gpu_list=(0 0 1 1 2 2 3 3)
seeds=(0 0 0 0 0 0 0 0)
prefix='6.alpha'
for i in ${!alpha_list[@]};do
    gpu=${gpu_list[i]}
    alpha=${alpha_list[i]}
    seed=${seeds[i]}
    experiment_name=${prefix}_loss${loss}_models${models_num}_aplha${alpha}_epochs${epochs}_gpu${gpu}_seed${seed}
    save=ckpt/consensus/${experiment_name}.pt
    
    folder_name=logs/consensus
    if [ ! -d ${folder_name} ]; then
        mkdir -p ${folder_name}
    fi
    log_filename=${folder_name}/${experiment_name}.log
    nohup python -u consensus.py \
    --exp_name ${experiment_name} \
    --loss ${loss} \
    --models_num ${models_num} \
    --detach ${detach} \
    --learnable_q ${learnable_q} \
    --alpha ${alpha} \
    --gpu ${gpu} \
    --epochs ${epochs} \
    --save ${save} \
    --seed ${seed} \
    > ${log_filename} 2>&1 &
done