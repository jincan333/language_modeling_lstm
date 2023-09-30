#!/bin/bash

loss='consensus_fifth'
models_num=2
detach=1
learnable_q=1
epochs=70
# epochs=60
lr_gammas=(0.3 0.4 0.5 0.6)
alpha_list=(1.1 1.1 1.1 1.1)
gpu_list=(0 1 2 3)
seeds=(0 0 0 0 0 0 0 0)
prefix='2.lr_gamma'
for i in ${!lr_gammas[@]};do
    gpu=${gpu_list[i]}
    alpha=${alpha_list[i]}
    seed=${seeds[i]}
    lr_gamma=${lr_gammas[i]}
    experiment_name=${loss}_${prefix}_models${models_num}_aplha${alpha}_detach${detach}_epochs${epochs}_gpu${gpu}_seed${seed}_lr_gamma${lr_gamma}
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
    --lr_gamma ${lr_gamma} \
    > ${log_filename} 2>&1 &
done