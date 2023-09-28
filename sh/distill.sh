#!/bin/bash

loss='consensus_forth'
models_num=2
detach=1
learnable_q=1

epochs=60
alpha_list=(1)
gpu_list=(3)
seeds=(0 0 0 0)
student_epochs=5
distill_epochs=20
prefix='1.debug'
for i in ${!alpha_list[@]};do
    gpu=${gpu_list[i]}
    alpha=${alpha_list[i]}
    seed=${seeds[i]}
    experiment_name=${prefix}_loss${loss}_models${models_num}_aplha${alpha}_epochs${epochs}_gpu${gpu}_seed${seed}_studentepochs${student_epochs}_distillepochs${distill_epochs}
    save=ckpt/distill/${experiment_name}.pt
    folder_name=logs/distill
    if [ ! -d ${folder_name} ]; then
        mkdir -p ${folder_name}
    fi
    log_filename=${folder_name}/${experiment_name}.log
    nohup python -u distill.py \
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
    --student_epochs ${student_epochs} \
    --distill_epochs ${distill_epochs} \
    > ${log_filename} 2>&1 &
done