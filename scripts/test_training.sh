#!/bin/bash

# Configuration for minimal testing
DATASET_DIR="data/train_test"  # Use our test dataset
OUTPUT_DIR="output/test_training"
BATCH_SIZE=2
NUM_ITERATIONS=10  # Minimal iterations for testing
LEARNING_RATE=5e-5
GLOBAL_CROP_SIZE=560
LOCAL_CROP_SIZE=560
MIN_LOCAL_CROP=384
NORMALIZATION="hybrid"
NUM_SEGMENTS=4
NUM_WORKERS=2
SEED=42
CHECKPOINT_INTERVAL=5
LOG_INTERVAL=1

# Clean previous test output (if any)
rm -rf ${OUTPUT_DIR}

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run training with minimal settings for testing
python tools/train_distillation.py \
    --dataset_dir ${DATASET_DIR} \
    --teacher_models depthanything-large \
    --teacher_checkpoints checkpoint/large/model.safetensors \
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
    --log_interval ${LOG_INTERVAL} \
    --debug

echo "Test training completed. Results saved to ${OUTPUT_DIR}" 