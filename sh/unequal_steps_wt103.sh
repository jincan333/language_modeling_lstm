#!/bin/bash

models_num=2
detach=1
learnable_q=1
epochs=1
alpha_list=(1 1 1 1 1)
student_lrs=(30 30 30 30 30)
lr_gammas=(0.25 0.25 0.25 0.25)
student_ratio=(1 3 3 3)
seeds=(0 1 2)
gpu_list=(5)
prefix='0.test'
for i in ${!gpu_list[@]};do
    gpu=${gpu_list[i]}
    alpha=${alpha_list[i]}
    seed=${seeds[i]}
    lr_gamma=${lr_gammas[i]}
    student_lr=${student_lrs[i]}
    student_ratio=${student_ratio[i]}
    student_alpha=${alpha}
    experiment_name=${prefix}_aplha${alpha}_detach${detach}_epochs${epochs}_gpu${gpu}_seed${seed}_student_lr${student_lr}_student_ratio${student_ratio}_student_alpha${student_alpha}
    save=ckpt/unequal_steps_wt103/${experiment_name}.pt
    
    log_folder_name=logs/unequal_steps_wt103
    if [ ! -d ${log_folder_name} ]; then
        mkdir -p ${log_folder_name}
    fi
    ckpt_folder_name=ckpt/unequal_steps_wt103
    if [ ! -d ${ckpt_folder_name} ]; then
        mkdir -p ${ckpt_folder_name}
    fi

    log_filename=${log_folder_name}/${experiment_name}.log
    nohup python -u unequal_steps_wt103.py \
    --exp_name ${experiment_name} \
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