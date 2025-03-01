#!/bin/bash
# Full training script for running on larger datasets with optimized parameters

# Set environment variables
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export TORCH_USE_MPS_FALLBACK=1  # For Mac users with MPS

# Configuration - NYU Depth V2 is a widely available dataset for depth estimation
DATASET="nyu"  # NYU Depth V2 dataset
OUTPUT_DIR="output/full_training_nyu"
TEACHER_CHECKPOINT="checkpoints/teacher_model.safetensors"
BATCH_SIZE=8  # Reduced for compatibility with MacOS/MPS
EPOCHS=30     # Reduced for faster training but still effective
IMAGE_SIZE=384  # Higher resolution for full training
WORKERS=4      # Adjusted for better compatibility

# Loss weights from our hyperparameter tuning experiments
LAMBDA_SC=0.5    # Scale loss weight
LAMBDA_LG=0.5    # Local-global loss weight
LAMBDA_FEAT=1.0  # Feature alignment loss weight
LAMBDA_GRAD=0.2  # Increased gradient weight for better edge preservation
LAMBDA_HDN=0.8   # HDN loss weight

# Training parameters
LR=1e-4         # Learning rate
WEIGHT_DECAY=1e-5  # Weight decay for regularization
WARMUP_EPOCHS=2    # Learning rate warmup epochs

# Create output directory
mkdir -p $OUTPUT_DIR

# Log configuration
echo "=== Starting full training with optimized parameters ==="
echo "Dataset: $DATASET"
echo "Output Directory: $OUTPUT_DIR"
echo "Teacher Checkpoint: $TEACHER_CHECKPOINT"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Image Size: $IMAGE_SIZE"
echo "Learning Rate: $LR"
echo "Loss Weights: SC=$LAMBDA_SC, LG=$LAMBDA_LG, FEAT=$LAMBDA_FEAT, GRAD=$LAMBDA_GRAD, HDN=$LAMBDA_HDN"

# Execute training with optimized parameters
python tools/train_distillation.py \
  --dataset=$DATASET \
  --output-dir=$OUTPUT_DIR \
  --teacher=$TEACHER_CHECKPOINT \
  --batch-size=$BATCH_SIZE \
  --epochs=$EPOCHS \
  --image-size=$IMAGE_SIZE \
  --workers=$WORKERS \
  --lambda-sc=$LAMBDA_SC \
  --lambda-lg=$LAMBDA_LG \
  --lambda-feat=$LAMBDA_FEAT \
  --lambda-grad=$LAMBDA_GRAD \
  --lambda-hdn=$LAMBDA_HDN \
  --lr=$LR \
  --weight-decay=$WEIGHT_DECAY \
  --warmup-epochs=$WARMUP_EPOCHS \
  --save-checkpoint-freq=5 \
  --validate-freq=1 \
  --visualize-interval=200 \
  --save-val-results \
  --mixed-precision \
  --early-stopping=5

echo "=== Training complete ==="
echo "Results saved to $OUTPUT_DIR"
echo "Visualizations available in $OUTPUT_DIR/visualizations" 