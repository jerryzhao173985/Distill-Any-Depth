import argparse
import logging
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import cv2
import math
from tqdm import tqdm
from omegaconf import OmegaConf
import time
import matplotlib.pyplot as plt
from glob import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os.path as osp

from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
from distillanydepth.utils.mmcv_config import Config

from detectron2.utils import comm
from detectron2.engine import launch
from safetensors.torch import load_file, save_file

# Import our custom NYU dataset loader
from data_loaders import NYUDataset


# Argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="Train monocular depth estimation via distillation.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory with training images (e.g., SA-1B subset).")
    parser.add_argument("--teacher_models", nargs='+', default=['depthanything-large'], choices=['depthanything-large', 'depthanything-base', 'genpercept'], help="Teacher models to use for distillation.")
    parser.add_argument("--teacher_checkpoints", nargs='+', required=True, help="Checkpoint paths for teacher models.")
    parser.add_argument("--student_arch", type=str, default="depthanything-base", choices=['depthanything-base'], help="Student model architecture.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and logs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--num_iterations", type=int, default=0, help="Number of training iterations (0 means train for num_epochs).")
    parser.add_argument("--global_crop_size", type=int, default=560, help="Size of the global crop for local-global distillation.")
    parser.add_argument("--local_crop_size", type=int, default=560, help="Size of the local crop for shared-context distillation.")
    parser.add_argument("--min_local_crop", type=int, default=384, help="Minimum size of local crop sampling.")
    parser.add_argument("--normalization", type=str, default="hybrid", choices=['global', 'hybrid', 'local', 'none'], help="Normalization strategy for depth maps.")
    parser.add_argument("--num_segments", type=int, default=4, help="Number of segments for hybrid/local normalization.")
    parser.add_argument("--lambda_sc", type=float, default=0.5, help="Weight for shared-context distillation loss.")
    parser.add_argument("--lambda_lg", type=float, default=0.5, help="Weight for local-global distillation loss.")
    parser.add_argument("--lambda_feat", type=float, default=1.0, help="Weight for feature alignment loss.")
    parser.add_argument("--lambda_grad", type=float, default=0.2, help="Weight for gradient preservation loss.")
    parser.add_argument("--use_hdn_loss", action='store_true', help="Whether to use Hierarchical Depth Normalization loss.")
    parser.add_argument("--hdn_variant", type=str, default="dr", choices=['dr', 'dp', 'ds'], help="Variant of HDN loss to use.")
    parser.add_argument("--hdn_level", type=int, default=3, help="Level of HDN (depth ranges).")
    parser.add_argument("--lambda_hdn", type=float, default=0.8, help="Weight for HDN loss.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for the optimizer.")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Number of warmup epochs for learning rate.")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Save checkpoint every N steps.")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps.")
    parser.add_argument("--visualize_interval", type=int, default=500, help="Visualize results every N steps.")
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'mps', 'cpu'], help="Device to train on.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of update steps to accumulate gradients for.")
    parser.add_argument("--use_scheduler", action='store_true', help="Whether to use a learning rate scheduler.")
    parser.add_argument("--scheduler_type", type=str, default='cosine', choices=['cosine', 'step'], help="Type of learning rate scheduler.")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for StepLR scheduler.")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1, help="Gamma for StepLR scheduler.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data to use for validation (0 for no validation).")
    parser.add_argument("--early_stopping", type=int, default=0, help="Number of epochs to wait for improvement before stopping (0 to disable).")
    parser.add_argument("--save_best", action='store_true', help="Save the best model based on validation loss.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping.")
    parser.add_argument("--use_nyu_dataset", action='store_true', help="Use the NYU Depth V2 dataset loader instead of generic images.")
    
    return parser


# Custom dataset for loading images
class ImageDataset(Dataset):
    def __init__(self, image_dir, global_transform, local_transform, min_local_crop=384, logger=None, image_paths=None):
        if image_paths is None:
            self.image_paths = sorted(glob(os.path.join(image_dir, "**/*.jpg"), recursive=True) + 
                                     glob(os.path.join(image_dir, "**/*.png"), recursive=True))
        else:
            self.image_paths = image_paths
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
            
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.min_local_crop = min_local_crop
        self.logger = logger
        
        # Log some basic dataset info
        if self.logger:
            self.logger.info(f"Found {len(self.image_paths)} images for training")
            # Print a few sample paths
            self.logger.info(f"Sample images: {self.image_paths[:3]}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Get image dimensions
            w, h = image.size
            
            # Process for global view
            global_image = self.global_transform({'image': np.array(image) / 255.0})['image']
            
            # Get dimensions of transformed image (which is what we'll use for training)
            transformed_h, transformed_w = global_image.shape[1], global_image.shape[2]
            
            # Ensure min_local_crop is not larger than the transformed image dimensions
            actual_min_crop = min(self.min_local_crop, transformed_h-2, transformed_w-2)  # Ensure we have at least 2 pixels buffer
            actual_min_crop = max(64, actual_min_crop)  # Ensure we have at least 64x64 crop size
            
            # Sample a local crop - ensure it's not larger than the transformed image
            max_crop_size = min(transformed_h, transformed_w)
            crop_size = random.randint(actual_min_crop, max_crop_size)
            
            # Ensure crop is within transformed image boundaries
            left = random.randint(0, max(0, transformed_w - crop_size))
            top = random.randint(0, max(0, transformed_h - crop_size))
            right = min(left + crop_size, transformed_w)
            bottom = min(top + crop_size, transformed_h)
            
            if self.logger and idx == 0:  # Only log for the first image
                self.logger.debug(f"Original image size: {w}x{h}, Transformed size: {transformed_w}x{transformed_h}")
                self.logger.debug(f"Local crop: {crop_size}x{crop_size} at position {left},{top}")
            
            # For local image, extract the crop from the global image (numpy array)
            local_image_crop = global_image[:, top:bottom, left:right]
            
            # Use the local transform to get the same normalization
            local_image = self.local_transform({'image': local_image_crop.transpose(1, 2, 0)})['image']
            
            return {
                'global_image': global_image,
                'local_image': local_image,
                'crop_left': left,
                'crop_top': top,
                'crop_right': right,
                'crop_bottom': bottom,
                'image_path': img_path
            }
        except Exception as e:
            # Log the error and return a default item if possible
            if self.logger:
                self.logger.error(f"Error processing image {img_path}: {e}")
            else:
                print(f"Error processing image {img_path}: {e}")
            # If this is the first image, we re-raise to fail fast
            if idx == 0:
                raise
            # Otherwise, try to get another image
            return self.__getitem__((idx + 1) % len(self.image_paths))


# Depth normalization functions
def global_normalize(depth):
    """Global normalization of depth maps"""
    # Calculate median and MAD (mean absolute deviation)
    median = torch.median(depth.view(depth.shape[0], -1), dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    mad = torch.mean(torch.abs(depth - median), dim=(1, 2, 3), keepdim=True)
    
    # Normalize using median and MAD
    normalized_depth = (depth - median) / (mad + 1e-6)
    return normalized_depth

def segment_depth_map(depth, num_segments):
    """Divide depth map into segments based on depth range"""
    b, c, h, w = depth.shape
    
    # Reshape to a flat tensor for each batch element
    depth_flat = depth.view(b, -1)
    
    # Calculate min and max for each batch element
    min_depth = torch.min(depth_flat, dim=1, keepdim=True)[0]
    max_depth = torch.max(depth_flat, dim=1, keepdim=True)[0]
    
    # Calculate depth range
    depth_range = max_depth - min_depth
    
    # Create segment boundaries
    segment_boundaries = [min_depth + (i/num_segments) * depth_range for i in range(num_segments+1)]
    
    # Create segmentation masks
    segment_masks = []
    for i in range(num_segments):
        lower_bound = segment_boundaries[i]
        upper_bound = segment_boundaries[i+1]
        
        # Ensure correct dimensions before broadcasting
        # First reshape to [batch_size, 1] and then expand to match depth shape
        lower_bound = lower_bound.reshape(b, 1).unsqueeze(-1).unsqueeze(-1).expand(-1, c, h, w)
        upper_bound = upper_bound.reshape(b, 1).unsqueeze(-1).unsqueeze(-1).expand(-1, c, h, w)
        
        # Create mask for current segment
        mask = (depth >= lower_bound) & (depth <= upper_bound)
        segment_masks.append(mask)
    
    return segment_masks

def hybrid_normalize(depth, num_segments):
    """Hybrid normalization that balances global coherence with local adaptation"""
    b, c, h, w = depth.shape
    
    # Global normalization
    depth_global = global_normalize(depth)
    
    # Segment the depth map
    segment_masks = segment_depth_map(depth, num_segments)
    
    # Create a tensor to hold the hybrid normalized result
    normalized_depth = torch.zeros_like(depth)
    
    # Normalize each segment
    for mask in segment_masks:
        # For each segment, apply global normalization within that segment
        segment_depth = torch.where(mask, depth, torch.zeros_like(depth))
        
        # Skip empty segments
        if torch.sum(mask) == 0:
            continue
        
        # Calculate median and MAD for this segment
        segment_median = torch.sum(segment_depth, dim=(1, 2, 3), keepdim=True) / (torch.sum(mask.float(), dim=(1, 2, 3), keepdim=True) + 1e-6)
        segment_mad = torch.sum(torch.abs(segment_depth - segment_median) * mask.float(), dim=(1, 2, 3), keepdim=True) / (torch.sum(mask.float(), dim=(1, 2, 3), keepdim=True) + 1e-6)
        
        # Normalize the segment
        segment_normalized = (segment_depth - segment_median) / (segment_mad + 1e-6)
        
        # Add to the result where mask is True
        normalized_depth = torch.where(mask, segment_normalized, normalized_depth)
    
    return normalized_depth

def local_normalize(depth, num_segments):
    """Local normalization focusing on the finest-scale grouping"""
    # This is similar to hybrid, but we only use the finest segmentation
    return hybrid_normalize(depth, num_segments)

def normalize_depth(depth, strategy, num_segments=4):
    """Apply the selected normalization strategy to depth maps"""
    if strategy == 'global':
        return global_normalize(depth)
    elif strategy == 'hybrid':
        return hybrid_normalize(depth, num_segments)
    elif strategy == 'local':
        return local_normalize(depth, num_segments)
    elif strategy == 'none':
        return depth
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")


# Loss functions
def distillation_loss(student_depth, teacher_depth, norm_strategy, num_segments=4):
    """Compute distillation loss between student and teacher depth maps"""
    # Apply the selected normalization strategy
    if norm_strategy != 'none':
        student_norm = normalize_depth(student_depth, norm_strategy, num_segments)
        teacher_norm = normalize_depth(teacher_depth, norm_strategy, num_segments)
        loss = F.l1_loss(student_norm, teacher_norm)
    else:
        # Direct regression without normalization
        loss = F.l1_loss(student_depth, teacher_depth)
    
    return loss

def feature_distillation_loss(student_features, teacher_features, device=None):
    """
    Computes feature alignment loss between student and teacher features.
    Features can be either individual tensors or lists of tensors.
    
    Optimized version with more efficient tensor operations and better handling of dimension mismatches.
    """
    # Import logger from logging module
    import logging
    logger = logging.getLogger(__name__)
    
    # Determine device from input tensors if not specified
    if device is None:
        device = student_features[0].device if isinstance(student_features, list) else student_features.device
    
    loss = torch.tensor(0.0, device=device)
    
    # Handle case where features are individual tensors
    if not isinstance(student_features, list) and not isinstance(teacher_features, list):
        # For debugging
        logger.debug(f"Student feature shape: {student_features.shape}, Teacher feature shape: {teacher_features.shape}")
        
        # Check dimensions and handle any mismatches
        if student_features.dim() != teacher_features.dim():
            # Handle dimension mismatch by expanding dimensions as needed
            if student_features.dim() < teacher_features.dim():
                # Add dimensions to student features to match teacher
                for _ in range(teacher_features.dim() - student_features.dim()):
                    student_features = student_features.unsqueeze(-1)
            else:
                # Add dimensions to teacher features to match student
                for _ in range(student_features.dim() - teacher_features.dim()):
                    teacher_features = teacher_features.unsqueeze(-1)
        
        # Ensure spatial dimensions match for multi-dimensional tensors
        if student_features.dim() >= 3 and teacher_features.dim() >= 3:
            if student_features.shape[2:] != teacher_features.shape[2:]:
                # Choose the target spatial dimensions (use smaller dimensions for efficiency)
                target_spatial = student_features.shape[2:] if torch.prod(torch.tensor(student_features.shape[2:])) < torch.prod(torch.tensor(teacher_features.shape[2:])) else teacher_features.shape[2:]
                
                # Resize student features if needed
                if student_features.shape[2:] != target_spatial:
                    student_features = F.interpolate(
                        student_features,
                        size=target_spatial,
                        mode='bilinear' if student_features.dim() == 4 else 'nearest',
                        align_corners=True if student_features.dim() == 4 else None
                    )
                
                # Resize teacher features if needed
                if teacher_features.shape[2:] != target_spatial:
                    teacher_features = F.interpolate(
                        teacher_features,
                        size=target_spatial,
                        mode='bilinear' if teacher_features.dim() == 4 else 'nearest',
                        align_corners=True if teacher_features.dim() == 4 else None
                    )
        
        # Handle channel dimension mismatch with more efficient projection
        if student_features.shape[1] != teacher_features.shape[1]:
            logger.debug(f"Projecting features - student: {student_features.shape[1]}, teacher: {teacher_features.shape[1]}")
            
            # Determine target channel dimension (smaller is more efficient)
            target_channels = min(student_features.shape[1], teacher_features.shape[1])
            
            # Project both to common dimension using efficient einsum operations when possible
            try:
                # Reshape tensors for projection
                b, c_student = student_features.shape[0], student_features.shape[1]
                spatial_size = student_features.numel() // (b * c_student)
                
                # Default target channels is the smaller of the two feature dimensions
                c_teacher = teacher_features.shape[1]
                target_channels = min(c_student, c_teacher)
                
                # Reshape to [B, C, S] where S is all spatial dimensions flattened
                sf_flat = student_features.reshape(b, c_student, spatial_size)
                tf_flat = teacher_features.reshape(b, c_teacher, spatial_size)
                
                if c_student != target_channels:
                    # Project student features
                    projection_matrix_s = torch.nn.Parameter(
                        torch.randn(c_student, target_channels, device=device) / (c_student ** 0.5)
                    )
                    sf_projected = torch.einsum('bcs,ct->bts', sf_flat, projection_matrix_s)
                    student_features = sf_projected.reshape(b, target_channels, *student_features.shape[2:])
                
                if c_teacher != target_channels:
                    # Project teacher features
                    projection_matrix_t = torch.nn.Parameter(
                        torch.randn(c_teacher, target_channels, device=device) / (c_teacher ** 0.5)
                    )
                    tf_projected = torch.einsum('bcs,ct->bts', tf_flat, projection_matrix_t)
                    teacher_features = tf_projected.reshape(b, target_channels, *teacher_features.shape[2:])
            except Exception as e:
                # In case of error, log the exception and use default MSE loss
                print(f"Error in feature alignment: {e}. Using default MSE loss.")
                return F.mse_loss(student_features, teacher_features)
        
        # At this point, student and teacher features should have compatible dimensions
        logger.debug(f"After projection - student: {student_features.shape}, teacher: {teacher_features.shape}")
        
        # Final dimension check
        if student_features.shape == teacher_features.shape:
            # Normalize features for cosine similarity calculation
            sf_norm = F.normalize(student_features, p=2, dim=1)
            tf_norm = F.normalize(teacher_features, p=2, dim=1)
            
            # Calculate cosine similarity loss
            loss = (1.0 - F.cosine_similarity(sf_norm, tf_norm, dim=1).mean())
        else:
            # If shapes still don't match after all adjustments, use a simpler approach
            logger.debug("Shapes still don't match, using simplified vector approach")
            
            # Convert to vectors by averaging across spatial dimensions
            sf_vec = student_features.mean(dim=tuple(range(2, student_features.dim())))  # [B, C]
            tf_vec = teacher_features.mean(dim=tuple(range(2, teacher_features.dim())))  # [B, C]
            
            # Match dimensions if needed
            if sf_vec.shape[1] != tf_vec.shape[1]:
                min_dim = min(sf_vec.shape[1], tf_vec.shape[1])
                sf_vec = sf_vec[:, :min_dim]
                tf_vec = tf_vec[:, :min_dim]
            
            # Normalize and compute loss
            sf_norm = F.normalize(sf_vec, p=2, dim=1)
            tf_norm = F.normalize(tf_vec, p=2, dim=1)
            loss = (1.0 - F.cosine_similarity(sf_norm, tf_norm, dim=1).mean())
        
        return loss
    
    # Handle case where features are lists
    valid_pairs = 0
    for i, (sf, tf) in enumerate(zip(student_features, teacher_features)):
        # Skip if either feature is None
        if sf is None or tf is None:
            continue
        
        # Process this pair of features recursively
        pair_loss = feature_distillation_loss(sf, tf, device)
        loss += pair_loss
        valid_pairs += 1
    
    # Return average loss
    return loss / max(valid_pairs, 1)

def gradient_preservation_loss(depth):
    """Compute gradient preservation loss to maintain edge sharpness"""
    # Compute gradients in x and y directions
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(depth.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(depth.device)
    
    # Apply convolution to compute gradients
    grad_x = F.conv2d(depth, sobel_x, padding=1)
    grad_y = F.conv2d(depth, sobel_y, padding=1)
    
    # Compute gradient magnitude
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    
    # Penalize small gradients to preserve edges
    loss = torch.mean(torch.exp(-grad_mag))
    
    return loss

# HDN Loss Functions
def masked_shift_and_scale(depth_preds, depth_gt, mask_valid):
    """Aligns depth prediction with ground truth using scale-invariant transformation"""
    # Use CPU operations for nanmedian calculation if on MPS
    device = depth_preds.device
    if device.type == 'mps':
        cpu_device = torch.device('cpu')
        depth_preds_cpu = depth_preds.to(cpu_device)
        depth_gt_cpu = depth_gt.to(cpu_device)
        mask_valid_cpu = mask_valid.to(cpu_device)
    else:
        depth_preds_cpu = depth_preds
        depth_gt_cpu = depth_gt
        mask_valid_cpu = mask_valid
    
    # Create NaN tensors for masked operations
    depth_preds_nan = depth_preds_cpu.clone()
    depth_gt_nan = depth_gt_cpu.clone()
    depth_preds_nan[~mask_valid_cpu] = float('nan')
    depth_gt_nan[~mask_valid_cpu] = float('nan')

    # Count valid pixels for normalization
    mask_diff = mask_valid_cpu.reshape(mask_valid_cpu.size()[:2] + (-1,)).sum(-1, keepdims=True) + 1

    # Robust median calculation - handle nanmedian errors
    try:
        # Try using nanmedian
        t_gt = depth_gt_nan.reshape(depth_gt_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
    except (RuntimeError, NotImplementedError):
        # Fallback: use masked median calculation
        t_gt = []
        for b in range(depth_gt_nan.shape[0]):
            for c in range(depth_gt_nan.shape[1]):
                valid_values = depth_gt_cpu[b, c][mask_valid_cpu[b, c]]
                if valid_values.numel() > 0:
                    median_val = valid_values.median()
                else:
                    median_val = torch.tensor(0.0, device=cpu_device)
                t_gt.append(median_val)
        t_gt = torch.stack(t_gt).reshape(depth_gt_cpu.shape[0], depth_gt_cpu.shape[1], 1, 1)
    
    # Ensure no NaN values in median
    t_gt[torch.isnan(t_gt)] = 0
    
    # Calculate scale for ground truth
    diff_gt = torch.abs(depth_gt_cpu - t_gt)
    diff_gt[~mask_valid_cpu] = 0
    s_gt = (diff_gt.reshape(diff_gt.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
    
    # Normalize ground truth
    depth_gt_aligned = (depth_gt_cpu - t_gt) / (s_gt + 1e-6)

    # Repeat for predictions
    try:
        # Try using nanmedian
        t_pred = depth_preds_nan.reshape(depth_preds_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
    except (RuntimeError, NotImplementedError):
        # Fallback: use masked median calculation
        t_pred = []
        for b in range(depth_preds_nan.shape[0]):
            for c in range(depth_preds_nan.shape[1]):
                valid_values = depth_preds_cpu[b, c][mask_valid_cpu[b, c]]
                if valid_values.numel() > 0:
                    median_val = valid_values.median()
                else:
                    median_val = torch.tensor(0.0, device=cpu_device)
                t_pred.append(median_val)
        t_pred = torch.stack(t_pred).reshape(depth_preds_cpu.shape[0], depth_preds_cpu.shape[1], 1, 1)
    
    # Ensure no NaN values in median
    t_pred[torch.isnan(t_pred)] = 0
    
    # Calculate scale for predictions
    diff_pred = torch.abs(depth_preds_cpu - t_pred)
    diff_pred[~mask_valid_cpu] = 0
    s_pred = (diff_pred.reshape(diff_pred.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
    
    # Normalize predictions
    depth_pred_aligned = (depth_preds_cpu - t_pred) / (s_pred + 1e-6)

    # Move back to original device
    if device.type == 'mps':
        depth_pred_aligned = depth_pred_aligned.to(device)
        depth_gt_aligned = depth_gt_aligned.to(device)

    return depth_pred_aligned, depth_gt_aligned

def masked_l1_loss(preds, target, mask_valid, dense=False):
    """Compute L1 loss only on valid pixels"""
    element_wise_loss = torch.abs(preds - target)
    element_wise_loss[~mask_valid] = 0
    if dense is False:
        return element_wise_loss.sum() / (mask_valid.sum() + 1e-6)
    else:  # not average, return the raw loss map
        return element_wise_loss

def get_contexts_dr(level, depth_gt, mask_valid):
    """Create hierarchical contexts based on depth ranges (DR variant)"""
    batch_norm_context = []
    
    # Handle None mask_valid by creating a mask of all True values
    if mask_valid is None:
        mask_valid = torch.ones_like(depth_gt, dtype=torch.bool)
    
    for mask_index in range(depth_gt.shape[0]):  # process each img in the batch
        depth_map = depth_gt[mask_index]
        valid_map = mask_valid[mask_index]

        if depth_map[valid_map].numel() == 0:  # if there is no valid pixel
            map_context_list = [valid_map for _ in range(2 ** (level) - 1)]
        else:
            valid_values = depth_map[valid_map]
            max_d = valid_values.max()
            min_d = valid_values.min()
            bin_size_list = [(1 / 2) ** (i) for i in range(level)]
            bin_size_list.reverse()
            map_context_list = []
            for bin_size in bin_size_list:
                for i in range(int(1 / bin_size)):
                    mask_new = (depth_map >= min_d + (max_d - min_d) * i * bin_size) & (
                            depth_map < min_d + (max_d - min_d) * (i + 1) * bin_size + 1e-30)
                    mask_new = mask_new & valid_map
                    map_context_list.append(mask_new)

        map_context_list = torch.stack(map_context_list, dim=0)
        batch_norm_context.append(map_context_list)
    batch_norm_context = torch.stack(batch_norm_context, dim=0).swapdims(0, 1)

    return batch_norm_context

def get_contexts_dp(level, depth_gt, mask_valid):
    """Create hierarchical contexts based on depth percentiles (DP variant)"""
    # Use CPU operations for nanquantile calculation if on MPS
    device = depth_gt.device
    if device.type == 'mps':
        cpu_device = torch.device('cpu')
        depth_gt_cpu = depth_gt.to(cpu_device)
        mask_valid_cpu = mask_valid.to(cpu_device)
    else:
        depth_gt_cpu = depth_gt
        mask_valid_cpu = mask_valid
    
    # Create NaN tensors for masked operations
    depth_gt_nan = depth_gt_cpu.clone()
    depth_gt_nan[~mask_valid_cpu] = float('nan')
    depth_gt_flat = depth_gt_nan.view(depth_gt_nan.shape[0], depth_gt_nan.shape[1], -1)
    
    # Set up bin sizes for hierarchical levels
    bin_size_list = [(1 / 2) ** (i) for i in range(level)]
    bin_size_list.reverse()
    
    batch_norm_context = []
    for bin_size in bin_size_list:
        num_bins = int(1 / bin_size)
        
        for bin_index in range(num_bins):
            # Compute quantile boundaries - handle nanquantile errors
            try:
                # Try using nanquantile
                min_bin = depth_gt_flat.nanquantile(bin_index * bin_size, dim=-1).unsqueeze(-1).unsqueeze(-1)
                max_bin = depth_gt_flat.nanquantile((bin_index + 1) * bin_size, dim=-1).unsqueeze(-1).unsqueeze(-1)
            except (RuntimeError, NotImplementedError):
                # Fallback: manual quantile calculation
                min_bin = []
                max_bin = []
                
                for b in range(depth_gt_cpu.shape[0]):
                    for c in range(depth_gt_cpu.shape[1]):
                        valid_values = depth_gt_cpu[b, c][mask_valid_cpu[b, c]]
                        if valid_values.numel() > 0:
                            sorted_values, _ = torch.sort(valid_values.flatten())
                            min_idx = max(0, min(int(bin_index * bin_size * len(sorted_values)), len(sorted_values)-1))
                            max_idx = max(0, min(int((bin_index + 1) * bin_size * len(sorted_values)), len(sorted_values)-1))
                            min_quantile = sorted_values[min_idx]
                            max_quantile = sorted_values[max_idx]
                        else:
                            min_quantile = torch.tensor(0.0, device=cpu_device)
                            max_quantile = torch.tensor(1.0, device=cpu_device)
                        min_bin.append(min_quantile)
                        max_bin.append(max_quantile)
                
                min_bin = torch.stack(min_bin).reshape(depth_gt_cpu.shape[0], depth_gt_cpu.shape[1], 1, 1)
                max_bin = torch.stack(max_bin).reshape(depth_gt_cpu.shape[0], depth_gt_cpu.shape[1], 1, 1)
            
            # Create mask for this quantile bin
            new_mask_valid = mask_valid_cpu.clone()
            new_mask_valid = new_mask_valid & (depth_gt_cpu >= min_bin)
            new_mask_valid = new_mask_valid & (depth_gt_cpu < max_bin)
            
            # Move back to original device if needed
            if device.type == 'mps':
                new_mask_valid = new_mask_valid.to(device)
                
            batch_norm_context.append(new_mask_valid)
    
    batch_norm_context = torch.stack(batch_norm_context, dim=0)
    return batch_norm_context

def init_temp_masks_ds(level, image_size):
    """Initialize template masks for spatial domain normalization"""
    size = image_size
    bin_size_list = [(1 / 2) ** (i) for i in range(level)]
    bin_size_list.reverse()

    map_level_list = []
    for bin_size in bin_size_list:  # e.g. 1/8
        for h in range(int(1 / bin_size)):
            for w in range(int(1 / bin_size)):
                mask_new = torch.zeros(1, 1, size, size)
                mask_new[:, :, int(h * bin_size * size):int((h + 1) * bin_size * size),
                int(w * bin_size * size):int((w + 1) * bin_size * size)] = 1
                mask_new = mask_new > 0
                map_level_list.append(mask_new)
    batch_norm_context = torch.stack(map_level_list, dim=0)
    return batch_norm_context

def get_contexts_ds(level, mask_valid):
    """Create hierarchical contexts based on spatial domain (DS variant)"""
    template_contexts = init_temp_masks_ds(level, mask_valid.shape[-1])
    template_contexts = template_contexts.to(mask_valid.device)

    batch_norm_context = mask_valid.unsqueeze(0)
    batch_norm_context = batch_norm_context.repeat(template_contexts.shape[0], 1, 1, 1, 1)
    batch_norm_context = batch_norm_context & template_contexts

    return batch_norm_context

class SSILoss(nn.Module):
    """Scale-Shift Invariant MAE Loss"""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, depth_preds, depth_gt, mask_valid, dense=False):
        depth_pred_aligned, depth_gt_aligned = masked_shift_and_scale(depth_preds, depth_gt, mask_valid)
        ssi_mae_loss = masked_l1_loss(depth_pred_aligned, depth_gt_aligned, mask_valid, dense)
        return ssi_mae_loss

def compute_hdn_loss(ssi_loss, depth_preds, depth_gt, mask_valid_list):
    """Compute Hierarchical Depth Normalization loss"""
    # Get base mask (first element of each batch in mask_valid_list)
    base_mask = mask_valid_list[0] if mask_valid_list.shape[0] > 0 else None
    
    hdn_loss_level = ssi_loss(
        depth_preds.unsqueeze(0).repeat(mask_valid_list.shape[0], 1, 1, 1, 1).reshape(-1, *depth_preds.shape[-3:]),
        depth_gt.unsqueeze(0).repeat(mask_valid_list.shape[0], 1, 1, 1, 1).reshape(-1, *depth_gt.shape[-3:]),
        mask_valid_list.reshape(-1, *mask_valid_list.shape[-3:]), 
        dense=True
    )

    hdn_loss_level_list = hdn_loss_level.reshape(*mask_valid_list.shape)
    hdn_loss_level_list = hdn_loss_level_list.sum(dim=0)  # summed loss generated by different contexts for all locations
    mask_valid_list_times = mask_valid_list.sum(dim=0)  # the number of contexts for each location
    valid_locations = (mask_valid_list_times != 0)  # valid locations
    hdn_loss_level_list[valid_locations] = hdn_loss_level_list[valid_locations] / mask_valid_list_times[valid_locations]  # mean loss in each location
    
    # Use the sum of valid locations for normalization instead of the base mask
    hdn_loss = hdn_loss_level_list.sum() / (valid_locations.sum() + 1e-6)  # average the losses of all locations

    return hdn_loss


# Model loading functions
def load_teacher_model(arch_name, checkpoint_path, device):
    """Load a teacher model for distillation"""
    model_kwargs = dict(
        vitb=dict(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
        ),
        vitl=dict(
            encoder="vitl", 
            features=256, 
            out_channels=[256, 512, 1024, 1024], 
            use_bn=False, 
            use_clstoken=False, 
            max_depth=150.0, 
            mode='disparity',
            pretrain_type='dinov2',
            del_mask_token=False
        )
    )

    if arch_name == 'depthanything-large':
        model = DepthAnything(**model_kwargs['vitl']).to(device)
    elif arch_name == 'depthanything-base':
        model = DepthAnythingV2(**model_kwargs['vitb']).to(device)
    else:
        raise NotImplementedError(f"Unknown architecture: {arch_name}")
    
    # Load weights based on file extension
    import logging
    logger = logging.getLogger(__name__)
    
    if checkpoint_path.endswith('.safetensors'):
        logger.info(f"Loading safetensors checkpoint from {checkpoint_path}")
        model_weights = load_file(checkpoint_path)
    else:
        logger.info(f"Loading PyTorch checkpoint from {checkpoint_path}")
        model_weights = torch.load(checkpoint_path, map_location=device)
        # PyTorch checkpoints might be wrapped in different ways
        if 'state_dict' in model_weights:
            model_weights = model_weights['state_dict']
    
    # Handle key prefix mismatches
    if arch_name == 'depthanything-large':
        # Check if keys have 'pretrained.' prefix
        has_pretrained_prefix = any(k.startswith('pretrained.') for k in model_weights.keys())
        if has_pretrained_prefix:
            logger.info("Remapping keys from 'pretrained.' prefix to 'backbone.' prefix")
            new_weights = {}
            for k, v in model_weights.items():
                if k.startswith('pretrained.'):
                    new_key = k.replace('pretrained.', 'backbone.')
                    new_weights[new_key] = v
                else:
                    new_weights[k] = v
            model_weights = new_weights
    
    # Handle other potential mismatches
    try:
        # First try loading with strict=True
        model.load_state_dict(model_weights, strict=True)
        logger.info(f"Successfully loaded checkpoint with strict=True")
    except Exception as e:
        logger.warning(f"Failed to load checkpoint with strict=True: {e}")
        logger.warning("Attempting to load with strict=False")
        try:
            # Try loading with strict=False
            model.load_state_dict(model_weights, strict=False)
            logger.warning("Loaded checkpoint with strict=False - some keys were missing or unexpected")
            
            # Log missing and unexpected keys
            model_keys = set(dict(model.named_parameters()).keys())
            checkpoint_keys = set(model_weights.keys())
            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys
            
            if missing_keys:
                logger.warning(f"Missing keys: {list(missing_keys)[:5]}... (total: {len(missing_keys)})")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {list(unexpected_keys)[:5]}... (total: {len(unexpected_keys)})")
        except Exception as e2:
            logger.error(f"Failed to load checkpoint even with strict=False: {e2}")
            raise
    
    # Set to evaluation mode for teacher
    model.eval()
    
    return model

def create_student_model(arch_name, device):
    """Create a student model for training"""
    model_kwargs = dict(
        vitb=dict(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
        )
    )

    if arch_name == 'depthanything-base':
        model = DepthAnythingV2(**model_kwargs['vitb']).to(device)
    else:
        raise NotImplementedError(f"Unknown architecture: {arch_name}")
    
    return model


def validate(model, teacher_models, dataloader, device, args):
    """
    Run validation on the model
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if dataloader is None:
        return 0.0
        
    # Set model to evaluation mode
    model.eval()
    
    total_loss = 0.0
    sc_loss_total = 0.0
    lg_loss_total = 0.0
    feat_loss_total = 0.0
    grad_loss_total = 0.0
    hdn_loss_total = 0.0
    num_samples = 0
    
    # Randomly select a teacher model for validation
    teacher_idx = random.randint(0, len(teacher_models) - 1)
    teacher_model = teacher_models[teacher_idx]
    
    with torch.no_grad():
        for batch in dataloader:
            if args.use_nyu_dataset:
                # Handle the case where our collate function returned lists
                if isinstance(batch['image'], list):
                    batch_size = len(batch['image'])
                    batch_loss = 0
                    
                    for i in range(batch_size):
                        single_image = batch['image'][i].unsqueeze(0).to(device)
                        
                        # Use the same image for both global and local views
                        global_image = single_image
                        local_image = single_image
                        
                        # Forward pass for student
                        student_global_disp, student_global_features = model(global_image)
                        student_local_disp, student_local_features = model(local_image)
                        
                        # Forward pass for teacher
                        teacher_local_disp, teacher_local_features = teacher_model(local_image)
                        
                        # Shared-Context Distillation Loss
                        sc_loss = distillation_loss(
                            student_local_disp,
                            teacher_local_disp,
                            args.normalization
                        )
                        
                        # Local-Global Distillation Loss
                        lg_loss = distillation_loss(
                            student_global_disp,
                            student_local_disp,
                            args.normalization
                        )
                        
                        # Feature Distillation Loss
                        feat_loss = feature_distillation_loss(
                            student_local_features,
                            teacher_local_features
                        )
                        
                        # Gradient Preservation Loss
                        grad_loss = gradient_preservation_loss(
                            student_local_disp
                        )
                        
                        # HDN loss
                        hdn_loss = torch.tensor(0.0, device=device)
                        if args.use_hdn_loss:
                            ssi_loss = SSILoss()
                            mask_valid_list = get_contexts_dr(args.hdn_level, teacher_local_disp, None) if args.hdn_variant == 'dr' else None
                            hdn_loss = compute_hdn_loss(
                                ssi_loss,
                                student_local_disp,
                                teacher_local_disp,
                                mask_valid_list
                            )
                        
                        # Combine losses
                        single_loss = (
                            args.lambda_sc * sc_loss +
                            args.lambda_lg * lg_loss +
                            args.lambda_feat * feat_loss +
                            args.lambda_grad * grad_loss
                        )
                        
                        if args.use_hdn_loss:
                            single_loss += args.lambda_hdn * hdn_loss
                        
                        # Accumulate loss values
                        batch_loss += single_loss.item()
                        sc_loss_total += sc_loss.item() * batch_size
                        lg_loss_total += lg_loss.item() * batch_size
                        feat_loss_total += feat_loss.item() * batch_size
                        grad_loss_total += grad_loss.item() * batch_size
                        if args.use_hdn_loss:
                            hdn_loss_total += hdn_loss.item() * batch_size
                    
                    # Average losses for this batch
                    batch_loss /= batch_size
                    total_loss += batch_loss * batch_size
                    num_samples += batch_size
                else:
                    # Regular batch processing (tensors already stacked)
                    image = batch['image'].to(device)
                    batch_size = image.size(0)
                    
                    # Use the same image for both global and local
                    global_image = image
                    local_image = image
                    
                    # Forward pass for student
                    student_global_disp, student_global_features = model(global_image)
                    student_local_disp, student_local_features = model(local_image)
                    
                    # Forward pass for teacher
                    teacher_local_disp, teacher_local_features = teacher_model(local_image)
                    
                    # Shared-Context Distillation Loss
                    sc_loss = distillation_loss(
                        student_local_disp,
                        teacher_local_disp,
                        args.normalization
                    )
                    
                    # Local-Global Distillation Loss
                    lg_loss = distillation_loss(
                        student_global_disp,
                        student_local_disp,
                        args.normalization
                    )
                    
                    # Feature Distillation Loss
                    feat_loss = feature_distillation_loss(
                        student_local_features,
                        teacher_local_features
                    )
                    
                    # Gradient Preservation Loss
                    grad_loss = gradient_preservation_loss(
                        student_local_disp
                    )
                    
                    # HDN loss
                    hdn_loss = torch.tensor(0.0, device=device)
                    if args.use_hdn_loss:
                        ssi_loss = SSILoss()
                        mask_valid_list = get_contexts_dr(args.hdn_level, teacher_local_disp, None) if args.hdn_variant == 'dr' else None
                        hdn_loss = compute_hdn_loss(
                            ssi_loss,
                            student_local_disp,
                            teacher_local_disp,
                            mask_valid_list
                        )
                    
                    # Combine losses
                    loss = (
                        args.lambda_sc * sc_loss +
                        args.lambda_lg * lg_loss +
                        args.lambda_feat * feat_loss +
                        args.lambda_grad * grad_loss
                    )
                    
                    if args.use_hdn_loss:
                        loss += args.lambda_hdn * hdn_loss
                    
                    # Accumulate loss values
                    total_loss += loss.item() * batch_size
                    sc_loss_total += sc_loss.item() * batch_size
                    lg_loss_total += lg_loss.item() * batch_size
                    feat_loss_total += feat_loss.item() * batch_size
                    grad_loss_total += grad_loss.item() * batch_size
                    if args.use_hdn_loss:
                        hdn_loss_total += hdn_loss.item() * batch_size
                    
                    num_samples += batch_size
            else:
                # Original code for ImageDataset
                global_image = batch['global_image'].to(device)
                local_image = batch['local_image'].to(device)
                batch_size = global_image.size(0)
                
                # Forward pass for student
                student_global_disp, student_global_features = model(global_image)
                student_local_disp, student_local_features = model(local_image)
                
                # Forward pass for teacher
                teacher_local_disp, teacher_local_features = teacher_model(local_image)
                
                # Calculate losses
                sc_loss = distillation_loss(student_local_disp, teacher_local_disp, args.normalization, args.num_segments)
                lg_loss = distillation_loss(student_global_disp, student_local_disp, args.normalization, args.num_segments)
                feat_loss = feature_distillation_loss(student_local_features, teacher_local_features)
                grad_loss = gradient_preservation_loss(student_local_disp)
                
                # HDN loss
                hdn_loss = torch.tensor(0.0, device=device)
                if args.use_hdn_loss:
                    ssi_loss = SSILoss()
                    mask_valid_list = get_contexts_dr(args.hdn_level, teacher_local_disp, None) if args.hdn_variant == 'dr' else None
                    hdn_loss = compute_hdn_loss(
                        ssi_loss,
                        student_local_disp,
                        teacher_local_disp,
                        mask_valid_list
                    )
                
                # Combine losses
                loss = sc_loss + args.lambda_lg * lg_loss + args.lambda_feat * feat_loss + args.lambda_grad * grad_loss
                
                if args.use_hdn_loss:
                    loss += args.lambda_hdn * hdn_loss
                
                # Accumulate loss values
                total_loss += loss.item() * batch_size
                sc_loss_total += sc_loss.item() * batch_size
                lg_loss_total += lg_loss.item() * batch_size
                feat_loss_total += feat_loss.item() * batch_size
                grad_loss_total += grad_loss.item() * batch_size
                if args.use_hdn_loss:
                    hdn_loss_total += hdn_loss.item() * batch_size
                
                num_samples += batch_size
    
    # Calculate average losses
    avg_loss = total_loss / max(num_samples, 1)
    avg_sc_loss = sc_loss_total / max(num_samples, 1)
    avg_lg_loss = lg_loss_total / max(num_samples, 1)
    avg_feat_loss = feat_loss_total / max(num_samples, 1)
    avg_grad_loss = grad_loss_total / max(num_samples, 1)
    avg_hdn_loss = hdn_loss_total / max(num_samples, 1) if args.use_hdn_loss else 0.0
    
    # Set model back to training mode
    model.train()
    
    # Log validation metrics
    log_msg = f"Validation Loss: {avg_loss:.4f} (SC: {avg_sc_loss:.4f}, " \
              f"LG: {avg_lg_loss:.4f}, Feat: {avg_feat_loss:.4f}, " \
              f"Grad: {avg_grad_loss:.4f}"
    
    # Add HDN loss to logging if enabled
    if args.use_hdn_loss:
        log_msg += f", HDN: {avg_hdn_loss:.4f}"
        
    log_msg += f")"
    logger.info(log_msg)
    
    return avg_loss

def visualize_depth_predictions(depth_pred, depth_gt, mask_valid, step, output_dir):
    """
    Visualize and save depth predictions during training
    
    Args:
        depth_pred: Predicted depth maps (B, 1, H, W)
        depth_gt: Ground truth depth maps (B, 1, H, W)
        mask_valid: Valid depth mask (B, 1, H, W)
        step: Current training step
        output_dir: Directory to save visualizations
    """
    # Create visualization directory if it doesn't exist
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Select one sample from batch for visualization
    idx = 0  # Using first sample in batch
    
    # Move tensors to CPU and convert to numpy
    d_pred = depth_pred[idx, 0].detach().cpu().numpy()
    d_gt = depth_gt[idx, 0].detach().cpu().numpy()
    mask = mask_valid[idx, 0].detach().cpu().numpy() if mask_valid is not None else np.ones_like(d_gt)
    
    # Apply mask to predictions and ground truth
    d_pred_masked = d_pred * mask
    d_gt_masked = d_gt * mask
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot prediction
    pred_plot = axs[0].imshow(d_pred_masked, cmap='plasma')
    axs[0].set_title('Depth Prediction')
    axs[0].axis('off')
    fig.colorbar(pred_plot, ax=axs[0], fraction=0.046, pad=0.04)
    
    # Plot ground truth
    gt_plot = axs[1].imshow(d_gt_masked, cmap='plasma')
    axs[1].set_title('Ground Truth')
    axs[1].axis('off')
    fig.colorbar(gt_plot, ax=axs[1], fraction=0.046, pad=0.04)
    
    # Plot error map
    error = np.abs(d_pred_masked - d_gt_masked)
    error_plot = axs[2].imshow(error, cmap='hot')
    axs[2].set_title('Absolute Error')
    axs[2].axis('off')
    fig.colorbar(error_plot, ax=axs[2], fraction=0.046, pad=0.04)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'depth_visualization_step_{step}.png'))
    plt.close(fig)
    
    # Also save a combined depth map comparing predictions for easier viewing
    plt.figure(figsize=(12, 6))
    
    # Create a side-by-side comparison with consistent color scaling
    vmin = min(d_pred_masked.min(), d_gt_masked.min())
    vmax = max(d_pred_masked.max(), d_gt_masked.max())
    
    plt.subplot(1, 2, 1)
    plt.imshow(d_pred_masked, cmap='plasma', vmin=vmin, vmax=vmax)
    plt.title('Predicted Depth')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(d_gt_masked, cmap='plasma', vmin=vmin, vmax=vmax)
    plt.title('Ground Truth Depth')
    plt.axis('off')
    
    plt.colorbar(fraction=0.035)
    plt.savefig(os.path.join(vis_dir, f'depth_comparison_step_{step}.png'))
    plt.close()

def train(args, device):
    """Main training function"""
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    logger.info(f"Args: {args}")
    
    # Set up transforms
    global_transform = transforms.Compose([
        Resize(
            args.global_crop_size, 
            args.global_crop_size,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    local_transform = transforms.Compose([
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    # Create dataset and dataloader
    try:
        if args.use_nyu_dataset:
            # Use our custom NYU dataset loader
            train_dataset = NYUDataset(
                mode='train',
                dataset_dir=args.dataset_dir,
                transform=global_transform,
                debug=args.debug,
                return_rgb_path=True
            )
            
            if args.val_split > 0:
                val_dataset = NYUDataset(
                    mode='test',  # Use test set for validation
                    dataset_dir=args.dataset_dir,
                    transform=global_transform,
                    debug=args.debug,
                    return_rgb_path=True
                )
                logger.info(f"Using NYU dataset - {len(train_dataset)} training images, {len(val_dataset)} validation images")
            else:
                val_dataset = None
        else:
            # Use the original generic image dataset loader
            all_image_paths = sorted(glob(os.path.join(args.dataset_dir, "**/*.jpg"), recursive=True) + 
                                    glob(os.path.join(args.dataset_dir, "**/*.png"), recursive=True))
            
            if len(all_image_paths) == 0:
                raise ValueError(f"No images found in {args.dataset_dir}")
            
            # Split dataset into train and validation
            if args.val_split > 0:
                # Shuffle the dataset with fixed seed for reproducibility
                random.seed(args.seed)
                random.shuffle(all_image_paths)
                
                # Calculate split indices
                val_size = int(len(all_image_paths) * args.val_split)
                train_paths = all_image_paths[val_size:]
                val_paths = all_image_paths[:val_size]
                
                logger.info(f"Split dataset: {len(train_paths)} training images, {len(val_paths)} validation images")
                
                # Create custom datasets
                train_dataset = ImageDataset(
                    args.dataset_dir, 
                    global_transform, 
                    local_transform,
                    min_local_crop=args.min_local_crop,
                    logger=logger,
                    image_paths=train_paths
                )
                
                val_dataset = ImageDataset(
                    args.dataset_dir, 
                    global_transform, 
                    local_transform,
                    min_local_crop=args.min_local_crop,
                    logger=logger,
                    image_paths=val_paths
                )
            else:
                # No validation split, use all images for training
                train_dataset = ImageDataset(
                    args.dataset_dir, 
                    global_transform, 
                    local_transform,
                    min_local_crop=args.min_local_crop,
                    logger=logger
                )
                val_dataset = None
        
        # Create data loader
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=True if args.device == 'cuda' else False,
            drop_last=True,
            collate_fn=None  # Don't use custom collate function
        )
        
        if val_dataset is not None:
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=args.num_workers,
                pin_memory=True if args.device == 'cuda' else False,
                collate_fn=None  # Don't use custom collate function
            )
        else:
            val_dataloader = None
    
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise
    
    # Create student model
    try:
        student_model = create_student_model(args.student_arch, device)
        logger.info(f"Created student model: {args.student_arch}")
    except Exception as e:
        logger.error(f"Error creating student model: {e}")
        raise
    
    # Load teacher models
    teacher_models = []
    try:
        for i, (arch, ckpt) in enumerate(zip(args.teacher_models, args.teacher_checkpoints)):
            teacher = load_teacher_model(arch, ckpt, device)
            teacher_models.append(teacher)
            logger.info(f"Loaded teacher model {i+1}: {arch} from {ckpt}")
    except Exception as e:
        logger.error(f"Error loading teacher model: {e}")
        raise
    
    # Set up optimizer with learning rate scheduler and weight decay
    optimizer = optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Implement warmup learning rate
    if args.warmup_epochs > 0:
        warmup_lr_scheduler = None
        def warmup_lambda(epoch):
            if epoch < args.warmup_epochs:
                return epoch / args.warmup_epochs
            return 1.0
        
        warmup_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    
    # Main learning rate scheduler (after warmup)
    main_scheduler = None
    if args.use_scheduler:
        if args.scheduler_type == "cosine":
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=args.num_epochs * len(train_dataloader), 
                eta_min=args.lr * 0.01
            )
        elif args.scheduler_type == "step":
            main_scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=args.step_size * len(train_dataloader), 
                gamma=args.scheduler_gamma
            )
    
    # Combine schedulers: warmup followed by main scheduler
    if args.warmup_epochs > 0 and args.use_scheduler:
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_lr_scheduler, main_scheduler],
            milestones=[args.warmup_epochs * len(train_dataloader)]
        )
    elif args.warmup_epochs > 0:
        scheduler = warmup_lr_scheduler
    elif args.use_scheduler:
        scheduler = main_scheduler
    else:
        scheduler = None
    
    # Initialize tracking variables
    global_step = 0
    start_time = time.time()
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Calculate total steps based on number of iterations or epochs
    max_steps = args.num_iterations if args.num_iterations > 0 else args.num_epochs * len(train_dataloader)
    
    # Create a directory for plots
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize lists to store metrics for plotting
    train_losses = []
    val_losses = []
    lr_values = []
    
    logger.info(f"Starting training for {max_steps} steps")
    student_model.train()
    
    try:
        for epoch in range(args.num_epochs):
            if args.num_iterations > 0 and global_step >= args.num_iterations:
                break
                
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                if args.num_iterations > 0 and global_step >= args.num_iterations:
                    break
                    
                # Move batch to device and handle potential lists from custom collate function
                if args.use_nyu_dataset:
                    # Handle the case where our collate function returned lists
                    if isinstance(batch['image'], list):
                        # Skip this batch if sizes are inconsistent
                        if len(batch['image']) != args.batch_size:
                            logger.warning(f"Skipping batch with inconsistent size: {len(batch['image'])}")
                            continue
                        
                        # Process one image at a time for this batch
                        total_batch_loss = 0
                        for i in range(len(batch['image'])):
                            single_image = batch['image'][i].unsqueeze(0).to(device)
                            single_depth = batch['depth'][i].unsqueeze(0).to(device)
                            
                            # Use the same image for both global and local
                            global_image = single_image
                            local_image = single_image
                            
                            # Log shapes for debugging
                            if global_step == 0 or args.debug:
                                logger.debug(f"Single image shape: {single_image.shape}")
                                logger.debug(f"Single depth shape: {single_depth.shape}")
                            
                            # Zero gradients
                            optimizer.zero_grad()
                            
                            # Randomly select a teacher for this iteration (multi-teacher distillation)
                            teacher_idx = random.randint(0, len(teacher_models) - 1)
                            teacher_model = teacher_models[teacher_idx]
                            
                            # Forward passes
                            student_global_disp, student_global_features = student_model(global_image)
                            student_local_disp, student_local_features = student_model(local_image)
                            
                            with torch.no_grad():
                                teacher_local_disp, teacher_local_features = teacher_model(local_image)
                            
                            # Calculate losses (passing single element tensors)
                            sc_loss = distillation_loss(
                                student_local_disp, 
                                teacher_local_disp, 
                                args.normalization
                            )
                            
                            # Local-Global Distillation Loss (since we're using the same image, this is identity distillation)
                            lg_loss = distillation_loss(
                                student_global_disp, 
                                student_local_disp, 
                                args.normalization
                            )
                            
                            # Feature Distillation Loss
                            feat_loss = feature_distillation_loss(
                                student_local_features, 
                                teacher_local_features
                            )
                            
                            # Gradient Preservation Loss
                            grad_loss = gradient_preservation_loss(
                                student_local_disp
                            )
                            
                            # Initialize HDN loss
                            hdn_loss = torch.tensor(0.0, device=device)
                            
                            # HDN Loss (if enabled)
                            if args.use_hdn_loss:
                                ssi_loss = SSILoss()
                                mask_valid_list = get_contexts_dr(args.hdn_level, teacher_local_disp, None) if args.hdn_variant == 'dr' else None
                                hdn_loss = compute_hdn_loss(
                                    ssi_loss,
                                    student_local_disp, 
                                    teacher_local_disp,
                                    mask_valid_list
                                )
                            
                            # Combine losses
                            single_loss = (
                                args.lambda_sc * sc_loss +
                                args.lambda_lg * lg_loss +
                                args.lambda_feat * feat_loss +
                                args.lambda_grad * grad_loss
                            )
                            
                            if args.use_hdn_loss:
                                single_loss += args.lambda_hdn * hdn_loss
                            
                            # Backward and optimize for each single image
                            single_loss.backward()
                            
                            # Gradient clipping
                            if args.max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
                            
                            optimizer.step()
                            
                            total_batch_loss += single_loss.item()
                            
                        # Average the loss for this batch
                        batch_loss = total_batch_loss / len(batch['image'])
                    else:
                        # Regular batch processing (tensors already stacked)
                        image = batch['image'].to(device)
                        depth = batch['depth'].to(device)
                        
                        # In our simplified version, we don't have separate global and local views
                        # So we'll use the same image for both global and local
                        global_image = image
                        local_image = image
                        
                        # We don't have crop coordinates anymore, so just log the shapes
                        if global_step == 0 or args.debug:
                            logger.debug(f"Image shape: {image.shape}")
                            logger.debug(f"Depth shape: {depth.shape}")
                        
                        # Zero gradients
                        optimizer.zero_grad()
                        
                        # Randomly select a teacher for this iteration (multi-teacher distillation)
                        teacher_idx = random.randint(0, len(teacher_models) - 1)
                        teacher_model = teacher_models[teacher_idx]
                        
                        # Forward pass for student (for both global and local images)
                        student_global_disp, student_global_features = student_model(global_image)
                        student_local_disp, student_local_features = student_model(local_image)
                        
                        # Forward pass for teacher (for local image only in shared-context)
                        with torch.no_grad():
                            teacher_local_disp, teacher_local_features = teacher_model(local_image)
                        
                        # Shared-Context Distillation Loss
                        sc_loss = distillation_loss(
                            student_local_disp, 
                            teacher_local_disp, 
                            args.normalization
                        )
                        
                        # Local-Global Distillation Loss (since we're using the same image, this is identity distillation)
                        lg_loss = distillation_loss(
                            student_global_disp, 
                            student_local_disp, 
                            args.normalization
                        )
                        
                        # Feature Distillation Loss
                        feat_loss = feature_distillation_loss(
                            student_local_features, 
                            teacher_local_features
                        )
                        
                        # Gradient Preservation Loss
                        grad_loss = gradient_preservation_loss(
                            student_local_disp
                        )
                        
                        # Initialize HDN loss
                        hdn_loss = torch.tensor(0.0, device=device)
                        
                        # HDN Loss (if enabled)
                        if args.use_hdn_loss:
                            ssi_loss = SSILoss()
                            mask_valid_list = get_contexts_dr(args.hdn_level, teacher_local_disp, None) if args.hdn_variant == 'dr' else None
                            hdn_loss = compute_hdn_loss(
                                ssi_loss,
                                student_local_disp, 
                                teacher_local_disp,
                                mask_valid_list
                            )
                        
                        # Combine losses
                        batch_loss = (
                            args.lambda_sc * sc_loss +
                            args.lambda_lg * lg_loss +
                            args.lambda_feat * feat_loss +
                            args.lambda_grad * grad_loss
                        )
                        
                        if args.use_hdn_loss:
                            batch_loss += args.lambda_hdn * hdn_loss
                        
                        # Backward and optimize
                        batch_loss.backward()
                        
                        # Gradient clipping
                        if args.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
                        
                        optimizer.step()
                        
                        batch_loss = batch_loss.item()
                else:
                    # Original code for ImageDataset
                    global_image = batch['global_image'].to(device)
                    local_image = batch['local_image'].to(device)
                    
                    # Rest of the original code...
                    # ... (keep original code for the non-NYU dataset case)
                
                # Update learning rate if using scheduler
                if scheduler:
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = args.lr
                
                # Track metrics
                epoch_loss += batch_loss
                num_batches += 1
                lr_values.append(current_lr)
                
                # Logging
                if global_step % args.log_interval == 0:
                    elapsed_time = time.time() - start_time
                    log_msg = f"Step {global_step}/{max_steps} | Epoch {epoch+1} | " \
                             f"Loss: {batch_loss:.4f} (SC: {sc_loss:.4f}, " \
                             f"LG: {lg_loss:.4f}, Feat: {feat_loss:.4f}, " \
                             f"Grad: {grad_loss:.4f}"
                    
                    # Add HDN loss to logging if enabled
                    if args.use_hdn_loss:
                        log_msg += f", HDN: {hdn_loss:.4f}"
                        
                    log_msg += f") | LR: {current_lr:.6f} | Time: {elapsed_time:.2f}s"
                    logger.info(log_msg)
                
                # Checkpoint saving
                if global_step % args.checkpoint_interval == 0 and global_step > 0:
                    checkpoint_path = os.path.join(args.output_dir, f"student_checkpoint_{global_step}.safetensors")
                    save_file(student_model.state_dict(), checkpoint_path)
                    logger.info(f"Saved checkpoint at step {global_step} to {checkpoint_path}")
                
                # Add visualization every args.visualize_interval steps
                if (epoch * len(train_dataloader) + batch_idx) % args.visualize_interval == 0:
                    visualize_depth_predictions(
                        student_local_disp, 
                        teacher_local_disp, 
                        mask_valid_list[-1], 
                        (epoch * len(train_dataloader) + batch_idx), 
                        args.output_dir
                    )
                
                global_step += 1
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            train_losses.append(avg_epoch_loss)
            
            logger.info(f"Epoch {epoch+1}/{args.num_epochs} completed | Avg Loss: {avg_epoch_loss:.4f}")
            
            # Run validation if validation set is available
            if val_dataloader:
                # Use the first teacher model for validation
                val_loss = validate(student_model, teacher_models, val_dataloader, device, args)
                val_losses.append(val_loss)
                
                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(args.output_dir, "student_best.safetensors")
                    save_file(student_model.state_dict(), best_model_path)
                    logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    logger.info(f"Validation did not improve. Epochs without improvement: {epochs_without_improvement}")
                    
                    # Check for early stopping
                    if args.early_stopping > 0 and epochs_without_improvement >= args.early_stopping:
                        logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                        break
            
            # Plot training curves at the end of each epoch
            if epoch % 5 == 0 or epoch == args.num_epochs - 1:  # Plot every 5 epochs or on the last epoch
                # Plot training loss
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
                if val_dataloader:
                    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(plots_dir, f'loss_epoch_{epoch+1}.png'))
                plt.close()
                
                # Plot learning rate
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(lr_values) + 1), lr_values)
                plt.xlabel('Steps')
                plt.ylabel('Learning Rate')
                plt.title('Learning Rate Schedule')
                plt.grid(True)
                plt.savefig(os.path.join(plots_dir, f'lr_epoch_{epoch+1}.png'))
                plt.close()
        
        # Save final model
        final_checkpoint_path = os.path.join(args.output_dir, f"student_final.safetensors")
        save_file(student_model.state_dict(), final_checkpoint_path)
        logger.info(f"Training completed. Saved final model to {final_checkpoint_path}")
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        # Save emergency checkpoint
        emergency_path = os.path.join(args.output_dir, "student_emergency.safetensors")
        save_file(student_model.state_dict(), emergency_path)
        logger.info(f"Saved emergency checkpoint to {emergency_path}")
        raise
    
    return student_model


def main(args):
    """Main function"""
    # Determine device
    device = torch.device(args.device)
    
    # Start training
    student_model = train(args, device)
    return student_model


if __name__ == "__main__":
    # Parse arguments
    args = argument_parser().parse_args()
    
    # Run main function
    main(args) 