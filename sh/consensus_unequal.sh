#!/bin/bash

loss='consensus_fifth'
models_num=2
detach=1
learnable_q=1
epochs=60
alpha_list=(1 1 1 1)
student_lrs=(30 30 30 30)
lr_gammas=(0.25 0.25 0.25 0.25)
gpu_list=(5 4 3 1)
student_steps_list=(1 2 3 4)
seeds=(0 0 0 0)
prefix='5.hiddens_debug_both_judge_depart'
for i in ${!gpu_list[@]};do
    gpu=${gpu_list[i]}
    alpha=${alpha_list[i]}
    seed=${seeds[i]}
    student_lr=${student_lrs[i]}
    student_steps=${student_steps_list[i]}
    lr_gamma=${lr_gammas[i]}
    experiment_name=${loss}_${prefix}_aplha${alpha}_detach${detach}_epochs${epochs}_gpu${gpu}_seed${seed}_student_lr${student_lr}_student_steps${student_steps}
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
    --student_steps ${student_steps} \
    --lr_gamma ${lr_gamma} \
    > ${log_filename} 2>&1 &
done