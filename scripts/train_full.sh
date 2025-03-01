#!/bin/bash

# Script for full training with optimal hyperparameters

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Define paths
DATASET_DIR="path/to/your/large/dataset"  # Replace with your actual dataset path
OUTPUT_DIR="output/full_training"
TEACHER_CHECKPOINT="checkpoint/large/model.safetensors"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run training with optimal hyperparameters
python tools/train_distillation.py \
  --dataset_dir ${DATASET_DIR} \
  --teacher_models depthanything-large \
  --teacher_checkpoints ${TEACHER_CHECKPOINT} \
  --student_arch depthanything-base \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 16 \
  --lr 1e-4 \
  --num_epochs 100 \
  --global_crop_size 560 \
  --local_crop_size 560 \
  --min_local_crop 384 \
  --normalization hybrid \
  --num_segments 4 \
  --lambda_lg 0.5 \
  --lambda_feat 1.0 \
  --lambda_grad 2.0 \
  --num_workers 8 \
  --seed 42 \
  --checkpoint_interval 5000 \
  --log_interval 100 \
  --val_split 0.1 \
  --use_scheduler \
  --scheduler_type cosine \
  --weight_decay 1e-5 \
  --gradient_clip 1.0 \
  --warmup_epochs 5 \
  --early_stopping 15

echo "Full training completed. Results saved to ${OUTPUT_DIR}" 