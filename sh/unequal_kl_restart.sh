#!/bin/bash

loss='consensus_fifth'
models_num=2
detach=1
learnable_q=1
epochs=60
alpha_list=(1 1 1 1)
student_lrs=(30 30 30 30)
lr_gammas=(0.25 0.25 0.25 0.25)
student_steps_list=(1 1 1 1)
student_epochs_list=(10 10 10 10)
distill_epochs_list=(5 10 15 20)
seeds=(0 0 0 0)
gpu_list=(3 2 1 0)
prefix='2.distill_epochs'
for i in ${!student_epochs_list[@]};do
    gpu=${gpu_list[i]}
    alpha=${alpha_list[i]}
    seed=${seeds[i]}
    lr_gamma=${lr_gammas[i]}
    student_lr=${student_lrs[i]}
    student_steps=${student_steps_list[i]}
    student_epochs=${student_epochs_list[i]}
    distill_epochs=${distill_epochs_list[i]}
    experiment_name=${loss}_${prefix}_aplha${alpha}_detach${detach}_epochs${epochs}_gpu${gpu}_seed${seed}_student_lr${student_lr}_student_steps${student_steps}_student_epochs${student_epochs}_distill_epochs${distill_epochs}
    save=ckpt/unequal_kl_restart/${experiment_name}.pt
    
    log_folder_name=logs/unequal_kl_restart
    if [ ! -d ${log_folder_name} ]; then
        mkdir -p ${log_folder_name}
    fi
    ckpt_folder_name=ckpt/unequal_kl_restart
    if [ ! -d ${ckpt_folder_name} ]; then
        mkdir -p ${ckpt_folder_name}
    fi

    log_filename=${log_folder_name}/${experiment_name}.log
    nohup python -u unequal_kl_restart.py \
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
    --student_steps ${student_steps} \
    --student_epochs ${student_epochs} \
    --distill_epochs ${distill_epochs} \
    > ${log_filename} 2>&1 &
done