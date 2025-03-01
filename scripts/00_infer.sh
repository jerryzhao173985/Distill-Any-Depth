#!/bin/bash

model_list=('xxx')  # your model name

for model in "${model_list[@]}"; do
    python tools/testers/infer.py \
        --seed 1234 \
        --checkpoint 'checkpoint/large/model.safetensors' \
        --processing_res 700 \
        --output_dir output/${model} \
        --arch_name 'depthanything-large'
done
