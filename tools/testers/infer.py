import argparse
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os.path as osp
import numpy as np
import torch
from PIL import Image
import cv2
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision.transforms import Compose
from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
from distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
from distillanydepth.utils.mmcv_config import Config
from detectron2.utils import comm
from detectron2.engine import launch
import torch.nn.functional as F
from glob import glob
# from huggingface_hub import hf_hub_download
from safetensors.torch import load_file  # 导入 safetensors 库

# Argument parser
def argument_parser():
    parser = argparse.ArgumentParser(description="Run single-image depth/surface normal estimation.")
    parser.add_argument("--arch_name", type=str, default="depthanything-large", choices=['depthanything-large', 'depthanything-base', 'midas'], help="Select a method for inference.")
    parser.add_argument("--mode", type=str, default="disparity", choices=['rel_depth', 'metric_depth', 'disparity'], help="Select a method for inference.")
    parser.add_argument("--checkpoint", type=str, default="prs-eth/marigold-v1-0", help="Checkpoint path or hub name.")
    parser.add_argument("--unet_ckpt_path", type=str, default=None, help="Checkpoint path for unet.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--denoise_steps", type=int, default=50, help="Diffusion denoising steps.")
    parser.add_argument("--ensemble_size", type=int, default=10, help="Number of predictions to be ensembled.")
    parser.add_argument("--half_precision", "--fp16", action="store_true", help="Run with half-precision (16-bit float).")
    parser.add_argument("--processing_res", type=int, default=0, help="Maximum resolution of processing.")
    parser.add_argument("--output_processing_res", action="store_true", help="Output depth at resized operating resolution.")
    parser.add_argument("--resample_method", type=str, default="bilinear", help="Resampling method used to resize images.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--use_cpu", action="store_true", help="Force using CPU even if CUDA or MPS is available.")
    parser.add_argument("--no_mps", action="store_true", help="Disable MPS even if available (will use CPU).")
    return parser

# Helper function to get appropriate device
def get_device(args):
    if torch.cuda.is_available() and not args.use_cpu:
        gpu_id = comm.get_rank() if torch.cuda.device_count() > 0 else 0
        device = torch.device(f"cuda:{gpu_id}")
        logging.info(f'Using CUDA device {gpu_id}')
        return device, "cuda"
    
    # Check for MPS availability (available on Mac with M-series chips)
    # Only available on PyTorch 1.12+ and macOS 12.3+
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and not args.use_cpu and not args.no_mps:
        device = torch.device("mps")
        logging.info('Using MPS (Metal Performance Shaders) for acceleration on Mac')
        return device, "mps"
    
    device = torch.device("cpu")
    logging.info('Using CPU for inference')
    return device, "cpu"

# Helper function for model loading
def load_model_by_name(arch_name, checkpoint_path, device):
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

    # Load model
    if arch_name == 'depthanything-large':
        model = DepthAnything(**model_kwargs['vitl']).to(device)
    elif arch_name == 'depthanything-base':
        model = DepthAnythingV2(**model_kwargs['vitb']).to(device)
    else:
        raise NotImplementedError(f"Unknown architecture: {arch_name}")
    
    # safetensors 
    model_weights = load_file(checkpoint_path)
    model.load_state_dict(model_weights)
    del model_weights
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return model

# Helper function for directory checks
def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# MPS-compatible inference function
def run_inference(model, input_tensor, device_type, arch_name):
    if device_type == 'cuda':
        with torch.autocast("cuda"):
            pred_disp, _ = model(input_tensor) if 'midas' not in arch_name else model(input_tensor)
    elif device_type == 'mps':
        # MPS doesn't support autocast yet, but works well with standard precision
        pred_disp, _ = model(input_tensor) if 'midas' not in arch_name else model(input_tensor)
    else:
        # For CPU, don't use autocast
        pred_disp, _ = model(input_tensor) if 'midas' not in arch_name else model(input_tensor)
    
    return pred_disp

# Image processing function
def process_images(validation_images, image_logs_folder, transform, model, device, device_type, arch_name):
    images = []
    for i, image_path in enumerate(validation_images):
        validation_image_np = cv2.imread(image_path, cv2.COLOR_BGR2RGB)[..., ::-1] / 255
        _, orig_H, orig_W = validation_image_np.shape
        validation_image = transform({'image': validation_image_np})['image']
        validation_image = torch.from_numpy(validation_image).unsqueeze(0).to(device)

        # Use the MPS-compatible inference function
        pred_disp = run_inference(model, validation_image, device_type, arch_name)
            
        pred_disp_np = pred_disp.cpu().detach().numpy()[0, :, :, :].transpose(1, 2, 0)
        pred_disp = (pred_disp_np - pred_disp_np.min()) / (pred_disp_np.max() - pred_disp_np.min())

        cmap = "Spectral_r" if args.mode != 'metric' else 'Spectral_r'
        depth_colored = colorize_depth_maps(pred_disp[None, ...], 0, 1, cmap=cmap).squeeze()
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        
        val_img_np = validation_image_np * 255
        h, w = val_img_np.shape[:2]
        depth_colored_hwc = cv2.resize(depth_colored_hwc, (w, h), cv2.INTER_LINEAR)

        image_out = Image.fromarray(np.concatenate([depth_colored_hwc], axis=1))
        images.append(image_out)
        image_out.save(osp.join(image_logs_folder, f'da_sota_{i}.jpg'))
        print(f'{i} OK')
        
        if device_type == 'cuda':
            torch.cuda.empty_cache()

    return images

def main(args, num_gpus):
    # Check for device availability and set device accordingly
    device, device_type = get_device(args)

    # Model preparation
    model = load_model_by_name(args.arch_name, args.checkpoint, device)
    model.eval()  # Set model to evaluation mode for inference

    # Image directory check
    check_directory(args.output_dir)
    image_logs_folder = osp.join(args.output_dir, 'image_logs')
    os.makedirs(image_logs_folder, exist_ok=True)

    # Load validation images using glob
    validation_images = glob('data/input/*')

    # Define image transformation
    resize_h, resize_w = args.processing_res, args.processing_res
    transform = Compose([
        Resize(resize_w, resize_h, resize_target=False, keep_aspect_ratio=False, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])

    # Use device_type for MPS-compatible processing
    images = process_images(validation_images, image_logs_folder, transform, model, device, device_type, args.arch_name)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = argument_parser().parse_args()
    
    # Handle device selection
    if torch.cuda.is_available() and not args.use_cpu:
        num_gpus = torch.cuda.device_count()
        launch(main, num_gpus, num_machines=1, machine_rank=0, dist_url='auto', args=(args, num_gpus))
    else:
        # For CPU or MPS (Apple Silicon), we don't need distributed launch
        num_gpus = 0
        main(args, num_gpus)
