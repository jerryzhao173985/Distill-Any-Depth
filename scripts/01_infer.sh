#!/bin/bash

model_list=('5250' '5200' '5150' '5100' '5050' '5000')  # your model name

for model in "${model_list[@]}"; do
    python tools/testers/infer.py \
        --seed 1234 \
        --checkpoint "output/nyu_large_run/student_checkpoint_${model}.safetensors" \
        --processing_res 700 \
        --output_dir output/test1_5000/${model} \
        --arch_name 'depthanything-base'
done
