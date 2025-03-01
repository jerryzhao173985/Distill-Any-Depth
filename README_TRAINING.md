# Training Distill-Any-Depth Models

This document provides instructions on how to train depth estimation models using the Cross-Context Distillation framework from the Distill-Any-Depth project.

## Overview

Distill-Any-Depth uses a distillation framework for zero-shot monocular depth estimation that leverages unlabeled RGB images. The key components are:

1. **Cross-Context Distillation**: This combines both local and global information through two mechanisms:
   - **Shared-Context Distillation**: Teacher and student models both process the same local crop
   - **Local-Global Distillation**: The teacher processes local crops, while the student produces global depth maps

2. **Multi-Teacher Distillation**: Randomly selects from multiple teacher models during training to reduce bias and improve generalization

3. **Depth Normalization Strategies**: Several methods to normalize depth maps:
   - **Global**: Normalize using global statistics (median and MAD)
   - **Hybrid**: Divide depth into segments, with each pixel normalized in its corresponding segment
   - **Local**: Focus only on the finest-scale grouping
   - **None**: Direct regression without normalization

## Dataset Preparation

The training process requires a large dataset of RGB images. The paper uses a subset of the SA-1B dataset (200K images). You can use any large-scale image dataset.

1. Download your preferred dataset (SA-1B, or another large image collection)
2. Organize images in a directory structure that can be parsed by the dataloader
3. Update the `DATASET_DIR` in the training scripts to point to your dataset

## Training Scripts

### Single Teacher Distillation

To train with a single teacher model:

```bash
./scripts/train_distill.sh
```

Before running, edit the script to set the correct paths and parameters:

- `DATASET_DIR`: Path to your training image dataset
- `OUTPUT_DIR`: Where to save checkpoints and logs
- Other hyperparameters: batch size, learning rate, etc.

### Multi-Teacher Distillation

To train with multiple teacher models:

```bash
./scripts/train_multiteacher_distill.sh
```

This uses multiple teacher models (e.g., DepthAnything-large and DepthAnything-base) and randomly selects one for each training iteration.

## Training Parameters

The main parameters you can configure:

- **Architecture**: 
  - `--student_arch`: Architecture for student model
  - `--teacher_models`: One or more teacher model architectures
  - `--teacher_checkpoints`: Corresponding checkpoint paths

- **Training Settings**:
  - `--batch_size`: Batch size for training (default: 4)
  - `--lr`: Learning rate (default: 5e-5)
  - `--num_iterations`: Number of training iterations (default: 20,000)
  - `--num_epochs`: Alternative to iterations, train for specific epochs

- **Distillation Settings**:
  - `--normalization`: Depth normalization strategy (global, hybrid, local, none)
  - `--num_segments`: Number of segments for hybrid/local normalization
  - `--lambda_lg`: Weight for local-global distillation loss (default: 0.5)
  - `--lambda_feat`: Weight for feature alignment loss (default: 1.0)
  - `--lambda_grad`: Weight for gradient preservation loss (default: 2.0)

- **Data Settings**:
  - `--global_crop_size`: Size of the global crop for local-global distillation
  - `--local_crop_size`: Size of the local crop for shared-context distillation
  - `--min_local_crop`: Minimum size of local crop sampling

## Monitoring Training

During training, the script logs progress and metrics to both the console and a log file in the output directory. The logs include:

- Loss values (total, shared-context, local-global, feature alignment, gradient)
- Training time
- Current step/epoch

The code automatically saves checkpoints at regular intervals (controlled by `--checkpoint_interval`).

## Advanced Configuration

### Custom Normalization

The framework supports four normalization strategies. You can modify `tools/train_distillation.py` to implement additional normalization methods.

### Adding New Teacher Models

To use a different teacher model:

1. Implement the model loading function in `load_teacher_model()`
2. Add the model name to the choices in the argument parser
3. Update the training script to use the new model

## Hardware Requirements

- The paper used an NVIDIA V100 GPU for training
- With batch size 4, training requires approximately 16GB VRAM
- Apple Silicon Macs can use MPS acceleration (built into the code)
- CPU-only training is possible but very slow

## Checkpoints and Inference

After training, the model checkpoints will be saved in the specified output directory:

- Intermediate checkpoints: `student_checkpoint_{step}.safetensors`
- Final model: `student_final.safetensors`

You can use the trained model for inference with the existing `infer.py` script:

```bash
python tools/testers/infer.py \
    --checkpoint 'path/to/student_final.safetensors' \
    --arch_name 'depthanything-base' \
    --output_dir 'output/inference'
``` 