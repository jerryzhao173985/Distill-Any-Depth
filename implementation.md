# Distill-Any-Depth Implementation Notes

This document tracks the implementation details for training Distill-Any-Depth models on the NYU Depth V2 dataset.

## Dataset Setup

### NYU Depth V2 Dataset

The NYU Depth V2 dataset has been organized in the following structure:

```
data/nyu/
├── nyu2_train/
│   ├── [xxxxx]_colors.png (RGB images)
│   └── [xxxxx]_depth.png (Depth images in uint8 format)
├── nyu2_test/
│   ├── [xxxxx]_colors.png (RGB images)
│   └── [xxxxx]_depth.png (Depth images in uint16 format)
├── nyu2_train.csv (Mapping of RGB to depth images for training)
└── nyu2_test.csv (Mapping of RGB to depth images for testing)
```

The CSV files contain paths to RGB and depth image pairs, with each line in the format:
```
data/nyu2_train/00001_colors.png,data/nyu2_train/00001_depth.png
```

### Important Dataset Notes:

1. **Depth Format Difference**: 
   - Training depth images are in uint8 format (0-255 values)
   - Testing depth images are in uint16 format (0-65535 values)
   - Our custom `NYUDataset` class handles these different formats by normalizing:
     - Train: `depth / 255.0`
     - Test: `depth / 65535.0`

2. **Data Augmentation**:
   - We apply standard image transformations (resize, normalize)
   - We use random crops for the local-global distillation training strategy

## Model Architecture

We're using the Depth-Anything architecture with two variants:
- Teacher model: `depthanything-large` 
- Student model: `depthanything-base`

### Teacher Checkpoint Handling

The teacher model checkpoint had a format mismatch with the expected keys:
- Original checkpoint uses `pretrained.*` key prefixes
- Training code expects `backbone.*` key prefixes

We created a conversion script (`tools/convert_checkpoint.py`) to transform the keys:
```python
# Convert pretrained.* keys to backbone.* keys
for key, value in checkpoint.items():
    if key.startswith('pretrained.'):
        # Convert pretrained.* to backbone.*
        new_key = 'backbone' + key[len('pretrained'):]
        new_checkpoint[new_key] = value
    else:
        # Keep other keys as they are
        new_checkpoint[key] = value
```

## Training Configuration

### Training Parameters:
- Batch size: 8
- Epochs: 30
- Image size: 384x384
- Learning rate: 1e-4
- Weight decay: 1e-5
- Warmup epochs: 2

### Loss Function Components:
- Shared-Context Distillation Loss (weight: 0.5)
- Local-Global Distillation Loss (weight: 0.5)
- Feature Alignment Loss (weight: 1.0)
- Gradient Preservation Loss (weight: 0.2)
- Hierarchical Depth Normalization Loss (weight: 0.8)
  - Using "dr" variant (depth ranges)
  - HDN level: 3

### Training Script

The main training command is executed through `scripts/train_large.sh`, which runs:
```bash
python tools/train_distillation.py \
    --dataset_dir data/nyu \
    --teacher_models depthanything-large \
    --teacher_checkpoints checkpoints/large/teacher_model_converted.safetensors \
    --student_arch depthanything-base \
    --output_dir output/nyu_depth_training \
    --batch_size 8 \
    --use_nyu_dataset \
    # ... additional parameters
```

## Custom Implementations

### NYU Dataset Loader
We implemented a custom `NYUDataset` class in `tools/data_loaders.py` that:
1. Reads image pairs from CSV files
2. Handles different depth image formats (uint8/uint16)
3. Applies appropriate transformations
4. Creates local crops for the distillation strategy

### Checkpoint Conversion
We created a conversion script at `tools/convert_checkpoint.py` to modify the teacher checkpoint's key format for compatibility.

## Visualization & Evaluation

Visualizations are generated during training:
- Saved to `output/nyu_depth_training/visualizations/`
- Generated every 100 steps

Model evaluation metrics:
- RMSE (Root Mean Square Error)
- REL (Relative Error)
- δ1, δ2, δ3 (Thresholded accuracy metrics)

Best model checkpoints are saved as:
- `student_best.safetensors`: Best model by validation metrics
- `student_final.safetensors`: Final model after training 