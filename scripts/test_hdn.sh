#!/bin/bash

# Script for testing HDN loss implementation

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Set MPS fallback for operations not implemented on MPS (like nanmedian)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Define paths
DATASET_DIR="data/train_test"
OUTPUT_DIR="output/test_hdn"
TEACHER_CHECKPOINT="checkpoint/large/model.safetensors"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Clean previous test output
rm -rf ${OUTPUT_DIR}/*

# Run training with HDN loss enabled
python tools/train_distillation.py \
  --dataset_dir ${DATASET_DIR} \
  --teacher_models depthanything-large \
  --teacher_checkpoints ${TEACHER_CHECKPOINT} \
  --student_arch depthanything-base \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 2 \
  --lr 5e-5 \
  --num_epochs 5 \
  --num_iterations 20 \
  --global_crop_size 560 \
  --local_crop_size 560 \
  --min_local_crop 384 \
  --normalization hybrid \
  --num_segments 4 \
  --lambda_lg 0.5 \
  --lambda_feat 1.0 \
  --lambda_grad 2.0 \
  --use_hdn_loss \
  --hdn_variant dr \
  --hdn_level 3 \
  --lambda_hdn 1.0 \
  --num_workers 2 \
  --checkpoint_interval 10 \
  --log_interval 1 \
  --device cpu \
  --debug

echo "HDN test completed. Results saved to ${OUTPUT_DIR}" 