#!/bin/bash

loss='consensus_fifth'
models_num=2
detach=1
learnable_q=1

# epochs=60
epochs=70
alpha_list=(1 1 1 1)
student_lrs=(5 10 20 30)
gpu_list=(3 2 1 0)
seeds=(0 0 0 0 0 0 0 0)
prefix='1.test'
for i in ${!student_lrs[@]};do
    gpu=${gpu_list[i]}
    alpha=${alpha_list[i]}
    seed=${seeds[i]}
    student_lr=${student_lrs[i]}
    experiment_name=${loss}_${prefix}_models${models_num}_aplha${alpha}_detach${detach}_epochs${epochs}_gpu${gpu}_seed${seed}_student_lr${student_lr}
    save=ckpt/unequal_consensus/${experiment_name}.pt
    
    folder_name=logs/unequal_consensus
    if [ ! -d ${folder_name} ]; then
        mkdir -p ${folder_name}
    fi
    log_filename=${folder_name}/${experiment_name}.log
    nohup python -u unequal_consensus.py \
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
    --student_lr ${student_lr} \
    > ${log_filename} 2>&1 &
done