#!/bin/bash

loss='consensus_fifth'
models_num=2
detach=1
learnable_q=1
epochs=60
alpha_list=(0 0 0 0)
student_alpha_list=(1 1 1)
student_lrs=(30 30 30 30)
lr_gammas=(0.25 0.25 0.25 0.25)
student_ratio=(0 0 0 0)
seeds=(0 1 2)
gpu_list=(0 1 2)
prefix='4.baseline'
for i in ${!gpu_list[@]};do
    gpu=${gpu_list[i]}
    alpha=${alpha_list[i]}
    seed=${seeds[i]}
    lr_gamma=${lr_gammas[i]}
    student_lr=${student_lrs[i]}
    student_ratio=${student_ratio[i]}
    student_alpha=${alpha}
    experiment_name=${loss}_${prefix}_aplha${alpha}_detach${detach}_epochs${epochs}_gpu${gpu}_seed${seed}_student_lr${student_lr}_student_ratio${student_ratio}_student_alpha${student_alpha}
    save=ckpt/unequal_steps/${experiment_name}.pt
    
    log_folder_name=logs/unequal_steps
    if [ ! -d ${log_folder_name} ]; then
        mkdir -p ${log_folder_name}
    fi
    ckpt_folder_name=ckpt/unequal_steps
    if [ ! -d ${ckpt_folder_name} ]; then
        mkdir -p ${ckpt_folder_name}
    fi

    log_filename=${log_folder_name}/${experiment_name}.log
    nohup python -u unequal_steps.py \
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
    --student_lr ${student_lr} \
    --student_ratio ${student_ratio} \
    --student_alpha ${student_alpha} \
    > ${log_filename} 2>&1 &
done