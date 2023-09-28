#!/bin/bash

loss='consensus_forth_symmetrized'
models_num=2
detach=1
learnable_q=1

epoch=60
alpha_list=(0.5 1)
gpu_list=(1 2)
seeds=(0 0 0 0)
T=2
prefix='alpha'
for i in ${!alpha_list[@]};do
    gpu=${gpu_list[i]}
    alpha=${alpha_list[i]}
    seed=${seeds[i]}
    save=ckpt/consensus/${prefix}_loss${loss}_models${models_num}_aplha${alpha}_epoch${epoch}_gpu${gpu}_seed${seed}.pt
    experiment_name=consensus_${prefix}_loss${loss}_models${models_num}_aplha${alpha}_epochs${epochs}_gpu${gpu}_seed${seed}_T${T}
    folder_name=logs/consensus
    if [ ! -d ${folder_name} ]; then
        mkdir -p ${folder_name}
    fi
    log_filename=${folder_name}/${prefix}_loss${loss}_models${models_num}_aplha${alpha}_epoch${epoch}_gpu${gpu}_seed${seed}_T${T}.log
    nohup python -u consensus.py \
    --exp_name ${experiment_name} \
    --loss ${loss} \
    --models_num ${models_num} \
    --detach ${detach} \
    --learnable_q ${learnable_q} \
    --alpha ${alpha} \
    --gpu ${gpu} \
    --epochs ${epoch} \
    --save ${save} \
    --seed ${seed} \
    --T ${T} \
    > ${log_filename} 2>&1 &
done