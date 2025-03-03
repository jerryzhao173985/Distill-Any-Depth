#!/bin/bash
# Test training script for NYU Depth V2 dataset with minimal epochs for testing

# Setup Python and Torch environment
export PYTHON_PATH=python
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Configuration for the NYU Depth V2 dataset
DATASET_DIR="data/nyu"
OUTPUT_DIR="output/nyu_large_run"
TEACHER_CHECKPOINT="checkpoints/depth_anything_v2_vitl.pth"

# Training parameters - full training run
BATCH_SIZE=16
EPOCHS=100
IMAGE_SIZE=392
WORKERS=1

# Loss weights
LAMBDA_SC=0.5
LAMBDA_LG=0.5
LAMBDA_FEAT=1.0
LAMBDA_GRAD=0.2
LAMBDA_HDN=0.8

# Training optimization
LR=1e-4
WEIGHT_DECAY=1e-5
WARMUP_EPOCHS=0

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "=== Starting NYU Depth V2 training run with large model ==="
echo "Dataset Directory: $DATASET_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Teacher Checkpoint: $TEACHER_CHECKPOINT"
echo "Teacher Model: depthanything-large"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Image Size: $IMAGE_SIZE"
echo "Learning Rate: $LR"
echo "Loss Weights: SC=$LAMBDA_SC, LG=$LAMBDA_LG, FEAT=$LAMBDA_FEAT, GRAD=$LAMBDA_GRAD, HDN=$LAMBDA_HDN"

# Execute the training command with full epochs
$PYTHON_PATH tools/train_distillation.py \
    --dataset_dir "$DATASET_DIR" \
    --teacher_models depthanything-large \
    --teacher_checkpoints "$TEACHER_CHECKPOINT" \
    --student_arch depthanything-base \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --num_epochs $EPOCHS \
    --num_iterations 0 \
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
    --checkpoint_interval 10 \
    --log_interval 5 \
    --visualize_interval 10 \
    --val_split 0.05 \
    --device mps \
    --use_nyu_dataset 
#    --debug

echo "=== Training complete ==="
echo "Results saved to $OUTPUT_DIR"
echo "Visualizations available in $OUTPUT_DIR/visualizations" 
