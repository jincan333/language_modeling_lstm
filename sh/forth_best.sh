#!/bin/bash

loss='consensus_forth'
models_num=2
detach=1
learnable_q=1
epochs=60
momentum=0.3
lr=30
clip=0.20
dropout=0.45
lr_gamma=0.25
alpha=1
gpu_list=(0 1 2 3)
seeds=(0)
prefix='1.multistep_1'
for i in ${!seeds[@]};do
    gpu=${gpu_list[i]}
    seed=${seeds[i]}
    experiment_name=${loss}_${prefix}_models${models_num}_aplha${alpha}_detach${detach}_epochs${epochs}_gpu${gpu}_seed${seed}
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
    --epochs ${epochs} \
    --momentum ${momentum} \
    --lr ${lr} \
    --clip ${clip} \
    --dropout ${dropout} \
    --lr_gamma ${lr_gamma} \
    --alpha ${alpha} \
    --gpu ${gpu} \
    --save ${save} \
    --seed ${seed} \
    > ${log_filename} 2>&1 &
done