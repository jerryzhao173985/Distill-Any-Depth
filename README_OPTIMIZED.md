# Optimized Monocular Depth Distillation

This README describes the optimized features that have been implemented to enhance the monocular depth distillation training with Hierarchical Depth Normalization (HDN).

## Implemented Optimizations

### 1. Enhanced Feature Alignment
- The feature alignment loss function has been optimized for better handling of tensor dimension mismatches.
- It now uses more efficient tensor operations and handles dynamic feature shapes more robustly.
- The projection mechanism now targets the smaller dimension for both student and teacher features, reducing computational overhead.

### 2. Depth Visualization
- Training now includes automatic visualization of depth predictions at regular intervals.
- Visualizations include:
  - Predicted depth maps
  - Ground truth depth maps
  - Error maps highlighting the differences
- These visualizations are saved in the `output_dir/visualizations` directory.

### 3. Hyperparameter Tuning
- A new script `scripts/tune_loss_weights.py` has been added for automatic hyperparameter tuning.
- This script performs grid search over combinations of loss weights to find the optimal configuration.
- Results are organized and saved for easy comparison.

### 4. Full Training Support
- A dedicated script `scripts/train_large.sh` is now available for full-scale training.
- Optimized for larger datasets and longer training runs.
- Includes appropriate defaults for batch size, image resolution, and other parameters.

## Usage Instructions

### Hyperparameter Tuning

Run the hyperparameter tuning script to find the best loss weight configuration:

```bash
./scripts/tune_loss_weights.py \
  --dataset=nyu \
  --teacher=checkpoints/teacher_model.safetensors \
  --batch-size=8 \
  --epochs=5 \
  --output-dir=output/tuning
```

Optional arguments:
- `--sc-weights`, `--lg-weights`, `--feat-weights`, `--grad-weights`, `--hdn-weights`: List of weights to try for each loss component
- `--dry-run`: Print commands without executing them
- `--seed`: Random seed for reproducibility

### Full Training

After finding the optimal hyperparameters, run full training using:

```bash
./scripts/train_large.sh
```

You can modify the script to use your tuned hyperparameters and dataset configuration.

### Testing and Validation

The test script remains the same but now includes visualization:

```bash
./scripts/test_hdn.sh
```

## Visualizations

During training, the system automatically generates visualizations at regular intervals. The visualization frequency can be controlled using the `--visualize-interval` parameter.

Example visualizations include:
- Side-by-side comparisons of predicted and ground truth depth
- Error maps highlighting areas of inaccuracy
- Progression of depth quality throughout training

## Tips for Best Results

1. **Hyperparameter Tuning**: Start with a small-scale hyperparameter search to identify promising regions, then refine with a more focused search.

2. **Feature Alignment**: The optimized feature alignment works best when student and teacher models have compatible architectures. If architectures are very different, consider adjusting the projection mechanisms.

3. **Training Scale**: For best results on large datasets, consider:
   - Using larger batch sizes if memory allows
   - Increasing image resolution for fine detail
   - Training for more epochs (50+) with appropriate learning rate scheduling

4. **MPS Compatibility**: When using Apple Silicon, the `TORCH_USE_MPS_FALLBACK=1` environment variable helps avoid MPS-specific errors.

5. **Visualization Analysis**: Regularly check visualizations to identify potential issues:
   - Blurry predictions may indicate insufficient gradient preservation
   - Inconsistent depth scales suggest poor scale loss efficacy
   - Missing edge details might require higher gradient loss weight

## Troubleshooting

- **Memory Issues**: If encountering OOM errors, reduce batch size or image resolution
- **MPS Errors**: Use CPU (`--device=cpu`) for testing if MPS errors occur
- **Convergence Problems**: Try adjusting learning rates or loss weights

---

For further assistance, please refer to the original codebase documentation or open an issue. 