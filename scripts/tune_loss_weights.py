#!/usr/bin/env python
"""
Script for hyperparameter tuning of loss weights in depth distillation.
This script performs grid search over different combinations of loss weights.
"""

import os
import sys
import json
import argparse
import itertools
import subprocess
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for depth distillation")
    
    # Base training script and arguments
    parser.add_argument('--train-script', type=str, default='tools/train_distillation.py',
                        help='Path to the training script')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to use for training')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for training')
    parser.add_argument('--teacher', type=str, required=True,
                        help='Path to teacher model checkpoint')
    parser.add_argument('--output-dir', type=str, default='output/hyperparameter_tuning',
                        help='Directory to save tuning results')
                        
    # Hyperparameter ranges to search
    parser.add_argument('--sc-weights', type=float, nargs='+', default=[0.1, 0.5, 1.0],
                        help='Scale loss weights to try')
    parser.add_argument('--lg-weights', type=float, nargs='+', default=[0.1, 0.5, 1.0],
                        help='Local-global loss weights to try')
    parser.add_argument('--feat-weights', type=float, nargs='+', default=[0.1, 0.5, 1.0],
                        help='Feature alignment loss weights to try')
    parser.add_argument('--grad-weights', type=float, nargs='+', default=[0.1, 0.5, 1.0],
                        help='Gradient loss weights to try')
    parser.add_argument('--hdn-weights', type=float, nargs='+', default=[0.1, 0.5, 1.0],
                        help='HDN loss weights to try')
    
    # Run control
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--run-sequential', action='store_true',
                        help='Run experiments sequentially instead of in parallel')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing them')
    
    return parser.parse_args()

def generate_experiment_configs(args):
    """Generate all experiment configurations based on parameter ranges"""
    # Create grid of hyperparameters
    param_grid = list(itertools.product(
        args.sc_weights,
        args.lg_weights,
        args.feat_weights,
        args.grad_weights,
        args.hdn_weights
    ))
    
    configs = []
    for i, (sc, lg, feat, grad, hdn) in enumerate(param_grid):
        # Create unique experiment name
        exp_name = f"exp_{i:03d}_sc{sc}_lg{lg}_feat{feat}_grad{grad}_hdn{hdn}"
        exp_dir = os.path.join(args.output_dir, exp_name)
        
        # Create configuration
        config = {
            'exp_name': exp_name,
            'output_dir': exp_dir,
            'lambda_sc': sc,
            'lambda_lg': lg,
            'lambda_feat': feat,
            'lambda_grad': grad,
            'lambda_hdn': hdn,
            'dataset': args.dataset,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'teacher': args.teacher,
            'seed': args.seed
        }
        configs.append(config)
    
    return configs

def build_command(config, train_script):
    """Build training command for a specific configuration"""
    cmd = [
        sys.executable,
        train_script,
        f"--dataset={config['dataset']}",
        f"--teacher={config['teacher']}",
        f"--output-dir={config['output_dir']}",
        f"--batch-size={config['batch_size']}",
        f"--epochs={config['epochs']}",
        f"--seed={config['seed']}",
        f"--lambda-sc={config['lambda_sc']}",
        f"--lambda-lg={config['lambda_lg']}",
        f"--lambda-feat={config['lambda_feat']}",
        f"--lambda-grad={config['lambda_grad']}",
        f"--lambda-hdn={config['lambda_hdn']}",
        "--visualize-interval=100",
        "--save-val-results"
    ]
    return cmd

def run_experiment(cmd, dry_run=False):
    """Run a single experiment with the specified command"""
    print(f"Running command: {' '.join(cmd)}")
    
    if dry_run:
        return 0
    
    # Create output directory if it doesn't exist
    output_dir = [arg.split('=')[1] for arg in cmd if arg.startswith('--output-dir=')][0]
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the command and capture output
    with open(os.path.join(output_dir, 'train.log'), 'w') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)
        
        # Wait for process to complete
        return_code = process.wait()
    
    return return_code

def collect_results(configs, args):
    """Collect and analyze results from all experiments"""
    results = []
    
    for config in configs:
        result_file = os.path.join(config['output_dir'], 'val_metrics.json')
        
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    metrics = json.load(f)
                
                # Add hyperparameters to metrics
                metrics.update({
                    'exp_name': config['exp_name'],
                    'lambda_sc': config['lambda_sc'],
                    'lambda_lg': config['lambda_lg'],
                    'lambda_feat': config['lambda_feat'],
                    'lambda_grad': config['lambda_grad'],
                    'lambda_hdn': config['lambda_hdn']
                })
                
                results.append(metrics)
            except Exception as e:
                print(f"Error reading results for {config['exp_name']}: {e}")
    
    # Sort results by validation loss
    if results:
        results.sort(key=lambda x: x.get('val_loss', float('inf')))
        
        # Write summary to file
        summary_file = os.path.join(args.output_dir, 'tuning_results.json')
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print best configuration
        print("\n===== Best Configuration =====")
        best = results[0]
        print(f"Experiment: {best['exp_name']}")
        print(f"Validation Loss: {best.get('val_loss', 'N/A')}")
        print(f"Loss Weights:")
        print(f"  Scale Loss: {best['lambda_sc']}")
        print(f"  Local-Global Loss: {best['lambda_lg']}")
        print(f"  Feature Alignment Loss: {best['lambda_feat']}")
        print(f"  Gradient Loss: {best['lambda_grad']}")
        print(f"  HDN Loss: {best['lambda_hdn']}")
    else:
        print("No results found.")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate experiment configurations
    configs = generate_experiment_configs(args)
    
    # Save configurations
    config_file = os.path.join(args.output_dir, 'experiment_configs.json')
    with open(config_file, 'w') as f:
        json.dump(configs, f, indent=2)
    
    print(f"Generated {len(configs)} experiment configurations")
    print(f"Configurations saved to {config_file}")
    
    if args.dry_run:
        print("Dry run mode - commands will be printed but not executed")
    
    # Run experiments
    for i, config in enumerate(configs):
        print(f"\n===== Running Experiment {i+1}/{len(configs)}: {config['exp_name']} =====")
        cmd = build_command(config, args.train_script)
        run_experiment(cmd, args.dry_run)
    
    # Collect and analyze results
    if not args.dry_run:
        collect_results(configs, args)

if __name__ == "__main__":
    main() 