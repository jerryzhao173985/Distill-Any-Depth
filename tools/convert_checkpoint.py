#!/usr/bin/env python3
import argparse
import os
import torch
from safetensors.torch import load_file, save_file

def convert_checkpoint(input_path, output_path):
    """Convert the checkpoint weights from 'pretrained' prefix to 'backbone' prefix"""
    print(f"Loading checkpoint from {input_path}")
    checkpoint = load_file(input_path)
    
    # Create a new state dict with updated keys
    new_checkpoint = {}
    
    # Process each key in the checkpoint
    for key, value in checkpoint.items():
        if key.startswith('pretrained.'):
            # Convert pretrained.* to backbone.*
            new_key = 'backbone' + key[len('pretrained'):]
            new_checkpoint[new_key] = value
        else:
            # Keep other keys as they are
            new_checkpoint[key] = value
    
    # Save the converted checkpoint
    print(f"Saving converted checkpoint to {output_path}")
    save_file(new_checkpoint, output_path)
    print("Conversion complete!")

def main():
    parser = argparse.ArgumentParser(description="Convert checkpoint format for DepthAnything model")
    parser.add_argument("--input", required=True, help="Path to the input checkpoint file (safetensors)")
    parser.add_argument("--output", required=True, help="Path to save the converted checkpoint (safetensors)")
    
    args = parser.parse_args()
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Convert the checkpoint
    convert_checkpoint(args.input, args.output)

if __name__ == "__main__":
    main() 