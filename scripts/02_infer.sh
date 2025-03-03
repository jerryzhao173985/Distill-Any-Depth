#!/bin/bash

model_list=('1000' '1500' '2000' '2500' '3000' '3500' '4000' '4500' '4750')  # your model name

for model in "${model_list[@]}"; do
    python tools/testers/infer.py \
        --seed 1234 \
        --checkpoint "output/nyu_large_run/student_checkpoint_${model}.safetensors" \
        --processing_res 392 \
        --output_dir output/test1_1000_to_4750/${model} \
        --arch_name 'depthanything-base'
done
