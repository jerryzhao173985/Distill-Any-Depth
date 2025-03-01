# Full Training Pipeline for Depth-Anything Distillation

This document provides instructions for running the full training pipeline to distill a smaller model from the original Depth-Anything model.

## Prerequisites

- PyTorch installed
- CUDA-capable GPU (recommended for full training)
- Dependencies from `requirements.txt`
- Teacher model checkpoint at `checkpoint/large/model.safetensors`

## Preparing a Large Dataset

For optimal training results, prepare a large and diverse dataset of images:

1. Create a directory for your dataset, e.g., `data/full_training`
2. Collect a diverse set of images (JPG, PNG) - aim for at least 10,000 images
3. Update the `DATASET_DIR` variable in `scripts/train_full.sh` to point to your dataset

Example directory structure:
```
data/
  full_training/
    image1.jpg
    image2.jpg
    subfolder1/
      image3.jpg
      image4.png
    ...
```

## Implementing Feature Alignment Loss

The feature alignment loss has been fully implemented to enable effective knowledge transfer from the teacher model to the student model. This loss encourages the student to learn similar feature representations as the teacher, leading to better depth prediction.

## Hyperparameter Configuration

The full training script (`scripts/train_full.sh`) includes optimized hyperparameters:

- Batch size: 16 (adjust based on your GPU memory)
- Learning rate: 1e-4 with cosine scheduler and 5-epoch warmup
- Training for 100 epochs with early stopping
- Validation split: 10% of the dataset
- Gradient clipping: 1.0
- Weight decay: 1e-5

Feel free to adjust these parameters based on your specific hardware and dataset.

## Running Full Training

1. Make sure the teacher model checkpoint is available at `checkpoint/large/model.safetensors`
2. Update the dataset path in `scripts/train_full.sh`
3. Run the training script:

```bash
./scripts/train_full.sh
```

The training will output:
- Regular checkpoints in `output/full_training/`
- Best model based on validation loss: `student_best.safetensors`
- Final model: `student_final.safetensors`
- Training logs and loss plots

## Monitoring Training

During training, you can monitor progress via:

1. Terminal output showing loss metrics
2. Log file at `output/full_training/training.log`
3. Loss plots generated every 5 epochs in `output/full_training/plots/`

## Validation

The training process includes validation on a held-out subset of your data. This helps:
- Prevent overfitting
- Track model generalization
- Trigger early stopping if performance plateaus
- Save the best model based on validation performance

## Expected Training Time

- On a high-end GPU (e.g., RTX 3090): ~2-3 days for 100 epochs with a large dataset
- On a mid-range GPU: 5-7 days
- CPU-only: Not recommended for full training

## Troubleshooting

- **Out of memory errors**: Reduce batch size or image dimensions
- **Slow training**: Increase number of workers, check disk I/O
- **Poor convergence**: Try different learning rates or longer warmup
- **Validation issues**: Increase validation split for better evaluation

## Next Steps After Training

Once training is complete:
1. Evaluate the model on test images
2. Compare with the teacher model for quality vs. speed tradeoff
3. Export to ONNX format for deployment if needed 