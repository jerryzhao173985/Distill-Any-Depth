#!/usr/bin/env python
# Evaluation script for Depth Anything distilled models

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

# Import the model and dataset classes
from data_loaders import NYUDataset
from depth_anything.dpt import DepthAnything, DepthAnythingV2

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Depth Anything distilled models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the distilled model checkpoint')
    parser.add_argument('--original_model_path', type=str, default='checkpoints/depth_anything_v2_vitb.pth', 
                        help='Path to the original model checkpoint')
    parser.add_argument('--dataset', type=str, choices=['nyu'], default='nyu', help='Dataset to evaluate on')
    parser.add_argument('--data_dir', type=str, default='data/nyu2_test', help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    return parser.parse_args()

def load_model(model_path, device, is_large=False):
    """Load the model from a checkpoint file."""
    if is_large:
        model = DepthAnything(
            encoder="vitl", 
            features=256, 
            out_channels=[256, 512, 1024, 1024], 
            use_bn=False, 
            use_clstoken=False
        ).to(device)
    else:
        model = DepthAnythingV2(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768]
        ).to(device)
    
    print(f"Loading model from {model_path}")
    if model_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        model_weights = load_file(model_path)
    else:
        model_weights = torch.load(model_path, map_location=device)
        if 'state_dict' in model_weights:
            model_weights = model_weights['state_dict']
    
    # Attempt to load the model weights
    try:
        model.load_state_dict(model_weights, strict=True)
        print("Model loaded successfully with strict=True")
    except Exception as e:
        print(f"Error loading model with strict=True: {e}")
        try:
            model.load_state_dict(model_weights, strict=False)
            print("Model loaded with strict=False - some weights may be missing or unexpected")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            raise
    
    model.eval()
    return model

def compute_depth_metrics(pred, gt, mask=None):
    """Compute depth estimation metrics."""
    if mask is None:
        mask = gt > 0
    
    # Apply mask
    pred = pred[mask]
    gt = gt[mask]
    
    if pred.shape[0] == 0:
        return {
            'abs_rel': np.nan,
            'abs_diff': np.nan,
            'sq_rel': np.nan,
            'rmse': np.nan,
            'rmse_log': np.nan,
            'a1': np.nan,
            'a2': np.nan,
            'a3': np.nan
        }
    
    # Calculate metrics
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    abs_diff = np.mean(np.abs(gt - pred))
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))
    
    return {
        'abs_rel': abs_rel,
        'abs_diff': abs_diff,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3
    }

def evaluate_model(model, dataloader, device):
    """Evaluate a model on a dataset."""
    metrics_sum = {
        'abs_rel': 0, 'abs_diff': 0, 'sq_rel': 0, 
        'rmse': 0, 'rmse_log': 0, 'a1': 0, 'a2': 0, 'a3': 0
    }
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            gt_depth = batch['depth'].cpu().numpy()
            
            # Forward pass
            pred_depth = model(images)
            pred_depth = pred_depth.cpu().numpy()
            
            # Compute metrics for each sample in the batch
            for i in range(pred_depth.shape[0]):
                mask = gt_depth[i] > 0
                if mask.sum() == 0:
                    continue
                
                # Scale prediction to match ground truth
                pred = pred_depth[i].squeeze()
                gt = gt_depth[i].squeeze()
                
                # Scale the prediction to match the ground truth
                pred = pred * (gt[mask].mean() / pred[mask].mean())
                
                # Compute metrics
                metrics = compute_depth_metrics(pred, gt, mask)
                for k in metrics:
                    if not np.isnan(metrics[k]):
                        metrics_sum[k] += metrics[k]
                sample_count += 1
    
    # Compute averages
    metrics_avg = {k: metrics_sum[k] / sample_count for k in metrics_sum}
    return metrics_avg

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    
    # Load models
    distilled_model = load_model(args.model_path, device, is_large=False)
    original_model = load_model(args.original_model_path, device, is_large=False)
    
    # Create dataset and dataloader
    if args.dataset == 'nyu':
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = NYUDataset(
            root_dir=args.data_dir,
            rgb_transform=transform,
            depth_transform=transforms.ToTensor(),
            is_test=True
        )
        
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
    
    # Evaluate models
    print("Evaluating distilled model...")
    distilled_metrics = evaluate_model(distilled_model, test_dataloader, device)
    
    print("Evaluating original model...")
    original_metrics = evaluate_model(original_model, test_dataloader, device)
    
    # Print results
    print("\nEvaluation Results:")
    print("\nDistilled Model Metrics:")
    for k, v in distilled_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nOriginal Model Metrics:")
    for k, v in original_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Compare models
    print("\nModel Comparison (Distilled vs Original):")
    for k in distilled_metrics:
        diff = distilled_metrics[k] - original_metrics[k]
        diff_percent = (diff / original_metrics[k]) * 100
        # For a1, a2, a3, higher is better, so invert the sign
        if k in ['a1', 'a2', 'a3']:
            diff_percent = -diff_percent
        print(f"  {k}: {diff_percent:+.2f}% ({distilled_metrics[k]:.4f} vs {original_metrics[k]:.4f})")
    
    # Save results to a file
    results_path = os.path.join(args.output_dir, f"{os.path.basename(args.model_path).split('.')[0]}_evaluation.txt")
    with open(results_path, 'w') as f:
        f.write("Distilled Model Metrics:\n")
        for k, v in distilled_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        
        f.write("\nOriginal Model Metrics:\n")
        for k, v in original_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        
        f.write("\nModel Comparison (Distilled vs Original):\n")
        for k in distilled_metrics:
            diff = distilled_metrics[k] - original_metrics[k]
            diff_percent = (diff / original_metrics[k]) * 100
            if k in ['a1', 'a2', 'a3']:
                diff_percent = -diff_percent
            f.write(f"  {k}: {diff_percent:+.2f}% ({distilled_metrics[k]:.4f} vs {original_metrics[k]:.4f})\n")
    
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main() 