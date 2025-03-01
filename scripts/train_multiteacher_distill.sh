#!/bin/bash

# Configuration
DATASET_DIR="/path/to/dataset"  # Replace with actual dataset path (e.g., SA-1B subset)
OUTPUT_DIR="output/multi_teacher_distilled"
BATCH_SIZE=4
NUM_ITERATIONS=20000
LEARNING_RATE=5e-5
GLOBAL_CROP_SIZE=560
LOCAL_CROP_SIZE=560
MIN_LOCAL_CROP=384
NORMALIZATION="hybrid"
NUM_SEGMENTS=4
NUM_WORKERS=4
SEED=42
CHECKPOINT_INTERVAL=5000
LOG_INTERVAL=100

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run training with multiple teacher models
# This assumes you have checkpoints for both models
python tools/train_distillation.py \
    --dataset_dir ${DATASET_DIR} \
    --teacher_models depthanything-large depthanything-base \
    --teacher_checkpoints checkpoint/large/model.safetensors checkpoint/base/model.safetensors \
    --student_arch depthanything-base \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --num_iterations ${NUM_ITERATIONS} \
    --global_crop_size ${GLOBAL_CROP_SIZE} \
    --local_crop_size ${LOCAL_CROP_SIZE} \
    --min_local_crop ${MIN_LOCAL_CROP} \
    --normalization ${NORMALIZATION} \
    --num_segments ${NUM_SEGMENTS} \
    --lambda_lg 0.5 \
    --lambda_feat 1.0 \
    --lambda_grad 2.0 \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED} \
    --checkpoint_interval ${CHECKPOINT_INTERVAL} \
    --log_interval ${LOG_INTERVAL}

echo "Multi-teacher training completed. Results saved to ${OUTPUT_DIR}" 