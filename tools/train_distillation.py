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


# Argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="Train monocular depth estimation via distillation.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory with training images (e.g., SA-1B subset).")
    parser.add_argument("--teacher_models", nargs='+', default=['depthanything-large'], choices=['depthanything-large', 'depthanything-base', 'genpercept'], help="Teacher models to use for distillation.")
    parser.add_argument("--teacher_checkpoints", nargs='+', required=True, help="Checkpoint paths for teacher models.")
    parser.add_argument("--student_arch", type=str, default="depthanything-base", choices=['depthanything-base'], help="Student model architecture.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and logs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--num_iterations", type=int, default=20000, help="Number of training iterations.")
    parser.add_argument("--global_crop_size", type=int, default=560, help="Size of the global crop for local-global distillation.")
    parser.add_argument("--local_crop_size", type=int, default=560, help="Size of the local crop for shared-context distillation.")
    parser.add_argument("--min_local_crop", type=int, default=384, help="Minimum size of local crop sampling.")
    parser.add_argument("--normalization", type=str, default="hybrid", choices=['global', 'hybrid', 'local', 'none'], help="Normalization strategy for depth maps.")
    parser.add_argument("--num_segments", type=int, default=4, help="Number of segments for hybrid/local normalization.")
    parser.add_argument("--lambda_lg", type=float, default=0.5, help="Weight for local-global distillation loss.")
    parser.add_argument("--lambda_feat", type=float, default=1.0, help="Weight for feature alignment loss.")
    parser.add_argument("--lambda_grad", type=float, default=2.0, help="Weight for gradient preservation loss.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--checkpoint_interval", type=int, default=5000, help="Save checkpoint every N iterations.")
    parser.add_argument("--log_interval", type=int, default=100, help="Log training status every N iterations.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu", 
                        help="Device to use for training (cuda, mps, or cpu).")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with more verbose logging.")
    return parser


# Custom dataset for loading images
class ImageDataset(Dataset):
    def __init__(self, image_dir, global_transform, local_transform, min_local_crop=384, logger=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, "**/*.jpg"), recursive=True) + 
                                 glob(os.path.join(image_dir, "**/*.png"), recursive=True))
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

def feature_alignment_loss(student_features, teacher_features):
    """Compute feature alignment loss between student and teacher features"""
    # For testing purposes only: Skip feature alignment loss
    # This allows us to verify the rest of the training pipeline works
    return torch.tensor(0.0, device=student_features[0].device if isinstance(student_features, list) else student_features.device)
    
    # Feature alignment will be properly implemented in the full version
    # Below is a placeholder for the full implementation
    """
    loss = 0
    
    # Check if feature dimensions match
    if isinstance(student_features, torch.Tensor) and isinstance(teacher_features, torch.Tensor):
        # Handle case where we're comparing the final feature embeddings
        # Project student features to teacher dimension if needed
        if student_features.shape[1] != teacher_features.shape[1]:
            # Project student features to teacher feature dimension
            # This is a simplified version for testing
            student_features = F.interpolate(
                student_features.unsqueeze(2).unsqueeze(3), 
                size=(1, teacher_features.shape[1])
            ).squeeze(3).squeeze(2)
        
        # Normalize features
        sf_norm = F.normalize(student_features, p=2, dim=1)
        tf_norm = F.normalize(teacher_features, p=2, dim=1)
        
        # Compute cosine similarity loss
        loss += (1.0 - F.cosine_similarity(sf_norm, tf_norm, dim=1).mean())
        return loss
    
    # Handle case where we're comparing feature lists
    for i, (sf, tf) in enumerate(zip(student_features, teacher_features)):
        # Skip feature maps where dimensions are incompatible
        if sf.ndim != tf.ndim:
            continue
            
        # Downsample larger feature map if spatial sizes don't match
        if sf.shape[2:] != tf.shape[2:]:
            sf = F.interpolate(sf, size=tf.shape[2:], mode='bilinear', align_corners=True)
        
        # Skip if channel dimensions don't match
        if sf.shape[1] != tf.shape[1]:
            continue
        
        # Normalize features
        sf_norm = F.normalize(sf, p=2, dim=1)
        tf_norm = F.normalize(tf, p=2, dim=1)
        
        # Compute cosine similarity loss
        loss += (1.0 - F.cosine_similarity(sf_norm, tf_norm, dim=1).mean())
    
    num_valid_pairs = sum(1 for sf, tf in zip(student_features, teacher_features) 
                          if sf.ndim == tf.ndim and sf.shape[1] == tf.shape[1])
    
    return loss / max(num_valid_pairs, 1)
    """

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
    
    # Load weights from safetensors
    model_weights = load_file(checkpoint_path)
    model.load_state_dict(model_weights)
    
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


def extract_local_from_global(global_depth, crop_bbox, target_size):
    """Extract the local patch from a global depth map"""
    batch_size = global_depth.shape[0]
    crops = []
    h, w = global_depth.shape[2], global_depth.shape[3]
    
    # Process each item in the batch
    for i in range(batch_size):
        # Extract boundaries for this batch item
        left, top, right, bottom = crop_bbox[i]
        
        # Ensure crop is within image boundaries
        left = max(0, min(left, w-1))
        top = max(0, min(top, h-1))
        right = max(left+1, min(right, w))
        bottom = max(top+1, min(bottom, h))
        
        # Extract the crop based on bounding box
        crop = global_depth[i:i+1, :, top:bottom, left:right]
        
        # Log if crop dimensions were adjusted
        logger = logging.getLogger()
        if crop.shape[2] <= 0 or crop.shape[3] <= 0:
            logger.warning(f"Invalid crop dimensions: {crop.shape}. Using full image instead.")
            crop = global_depth[i:i+1]
        
        # Resize to target size if necessary
        if crop.shape[2:] != target_size:
            crop = F.interpolate(crop, size=target_size, mode='bilinear', align_corners=True)
        
        crops.append(crop)
    
    return torch.cat(crops, dim=0)


def train(args, device):
    """Main training function"""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "training.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    
    # Log arguments
    logger.info(f"Training arguments: {args}")
    logger.info(f"Using device: {device}")
    
    # Set up data transforms
    global_transform = transforms.Compose([
        Resize(args.global_crop_size, args.global_crop_size, 
               resize_target=False, keep_aspect_ratio=False, 
               ensure_multiple_of=14, resize_method='lower_bound', 
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])
    
    local_transform = transforms.Compose([
        Resize(args.local_crop_size, args.local_crop_size, 
               resize_target=False, keep_aspect_ratio=False, 
               ensure_multiple_of=14, resize_method='lower_bound', 
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])
    
    # Create dataset and dataloader
    try:
        dataset = ImageDataset(
            args.dataset_dir, 
            global_transform, 
            local_transform,
            min_local_crop=args.min_local_crop,
            logger=logger
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        logger.info(f"Created dataloader with {len(dataset)} images")
    except Exception as e:
        logger.error(f"Error creating dataset/dataloader: {e}")
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
    
    # Set up optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=args.lr)
    
    # Training loop
    global_step = 0
    start_time = time.time()
    
    # Calculate total steps based on number of iterations or epochs
    max_steps = args.num_iterations if args.num_iterations > 0 else args.num_epochs * len(dataloader)
    
    logger.info(f"Starting training for {max_steps} steps")
    student_model.train()
    
    try:
        for epoch in range(args.num_epochs):
            if args.num_iterations > 0 and global_step >= args.num_iterations:
                break
                
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
                if args.num_iterations > 0 and global_step >= args.num_iterations:
                    break
                    
                # Move batch to device
                global_image = batch['global_image'].to(device)
                local_image = batch['local_image'].to(device)
                
                # Extract crop coordinates and reconstruct the bbox
                crop_left = batch['crop_left'].tolist()
                crop_top = batch['crop_top'].tolist()
                crop_right = batch['crop_right'].tolist()
                crop_bottom = batch['crop_bottom'].tolist()
                
                # Construct crop_bbox as a list of tuples (one per batch item)
                crop_bbox = [(left, top, right, bottom) for left, top, right, bottom in 
                             zip(crop_left, crop_top, crop_right, crop_bottom)]
                
                # Log shapes
                if global_step == 0 or args.debug:
                    logger.debug(f"Global image shape: {global_image.shape}")
                    logger.debug(f"Local image shape: {local_image.shape}")
                    logger.debug(f"Crop bbox: {crop_bbox[0] if crop_bbox else None}")
                
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
                
                # Log shapes
                if global_step == 0 or args.debug:
                    logger.debug(f"Student global disp shape: {student_global_disp.shape}")
                    logger.debug(f"Student local disp shape: {student_local_disp.shape}")
                    logger.debug(f"Teacher local disp shape: {teacher_local_disp.shape}")
                
                # Shared-Context Distillation Loss
                sc_loss = distillation_loss(
                    student_local_disp, 
                    teacher_local_disp, 
                    args.normalization, 
                    args.num_segments
                )
                
                # Local-Global Distillation Loss
                # Extract the local patch from the global depth map
                student_global_local = extract_local_from_global(
                    student_global_disp, 
                    crop_bbox, 
                    (student_local_disp.shape[2], student_local_disp.shape[3])
                )
                
                lg_loss = distillation_loss(
                    student_global_local, 
                    teacher_local_disp, 
                    args.normalization, 
                    args.num_segments
                )
                
                # Feature Alignment Loss
                feat_loss = feature_alignment_loss(student_local_features, teacher_local_features)
                
                # Gradient Preservation Loss
                grad_loss = gradient_preservation_loss(student_global_disp)
                
                # Total Loss
                total_loss = sc_loss + args.lambda_lg * lg_loss + args.lambda_feat * feat_loss + args.lambda_grad * grad_loss
                
                # Backward pass and optimizer step
                total_loss.backward()
                optimizer.step()
                
                # Logging
                if global_step % args.log_interval == 0:
                    elapsed_time = time.time() - start_time
                    logger.info(f"Step {global_step}/{max_steps} | Epoch {epoch+1} | "
                              f"Loss: {total_loss.item():.4f} (SC: {sc_loss.item():.4f}, "
                              f"LG: {lg_loss.item():.4f}, Feat: {feat_loss.item():.4f}, "
                              f"Grad: {grad_loss.item():.4f}) | "
                              f"Time: {elapsed_time:.2f}s")
                
                # Checkpoint saving
                if global_step % args.checkpoint_interval == 0 and global_step > 0:
                    checkpoint_path = os.path.join(args.output_dir, f"student_checkpoint_{global_step}.safetensors")
                    save_file(student_model.state_dict(), checkpoint_path)
                    logger.info(f"Saved checkpoint at step {global_step} to {checkpoint_path}")
                
                global_step += 1
        
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