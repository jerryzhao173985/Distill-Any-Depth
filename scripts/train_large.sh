#!/bin/bash
# Full training script for running on NYU Depth V2 dataset with optimized parameters

# Setup Python and Torch environment
export PYTHON_PATH=python
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Configuration for the NYU Depth V2 dataset
DATASET_DIR="data/nyu"
OUTPUT_DIR="output/nyu_depth_training"
TEACHER_CHECKPOINT="checkpoints/large/teacher_model_converted.safetensors"

# Training parameters
BATCH_SIZE=8
EPOCHS=30
IMAGE_SIZE=384
WORKERS=4

# Loss weights
LAMBDA_SC=0.5
LAMBDA_LG=0.5
LAMBDA_FEAT=1.0
LAMBDA_GRAD=0.2
LAMBDA_HDN=0.8

# Training optimization
LR=1e-4
WEIGHT_DECAY=1e-5
WARMUP_EPOCHS=2

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "=== Starting NYU Depth V2 training with optimized parameters ==="
echo "Dataset Directory: $DATASET_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Teacher Checkpoint: $TEACHER_CHECKPOINT"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Image Size: $IMAGE_SIZE"
echo "Learning Rate: $LR"
echo "Loss Weights: SC=$LAMBDA_SC, LG=$LAMBDA_LG, FEAT=$LAMBDA_FEAT, GRAD=$LAMBDA_GRAD, HDN=$LAMBDA_HDN"

# Execute the training command
$PYTHON_PATH tools/train_distillation.py \
    --dataset_dir "$DATASET_DIR" \
    --teacher_models depthanything-large \
    --teacher_checkpoints "$TEACHER_CHECKPOINT" \
    --student_arch depthanything-base \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --num_epochs $EPOCHS \
    --global_crop_size $IMAGE_SIZE \
    --local_crop_size $IMAGE_SIZE \
    --min_local_crop 256 \
    --lambda_sc $LAMBDA_SC \
    --lambda_lg $LAMBDA_LG \
    --lambda_feat $LAMBDA_FEAT \
    --lambda_grad $LAMBDA_GRAD \
    --use_hdn_loss \
    --hdn_variant dr \
    --hdn_level 3 \
    --lambda_hdn $LAMBDA_HDN \
    --num_workers $WORKERS \
    --weight_decay $WEIGHT_DECAY \
    --warmup_epochs $WARMUP_EPOCHS \
    --use_scheduler \
    --scheduler_type cosine \
    --checkpoint_interval 500 \
    --log_interval 50 \
    --visualize_interval 100 \
    --val_split 0.1 \
    --early_stopping 5 \
    --device mps \
    --use_nyu_dataset

echo "=== Training complete ==="
echo "Results saved to $OUTPUT_DIR"
echo "Visualizations available in $OUTPUT_DIR/visualizations" 