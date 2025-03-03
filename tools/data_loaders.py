import os
import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import traceback
import torchvision.transforms as transforms

class NYUDataset(Dataset):
    """
    Dataset for loading NYU Depth V2 dataset
    Handles both training (uint8) and testing (uint16) depth images
    """
    def __init__(self, mode, dataset_dir='data/nyu', transform=None, debug=False, return_rgb_path=False):
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.debug = debug
        self.return_rgb_path = return_rgb_path
        self.workspace_root = os.path.abspath(os.getcwd())
        
        # Try multiple possible CSV file locations
        csv_filename = f'nyu2_{mode}.csv'
        csv_paths = [
            os.path.join(self.dataset_dir, csv_filename),  # Look in dataset_dir/nyu2_*.csv
            os.path.join('data', csv_filename),           # Look in data/nyu2_*.csv
            csv_filename                                  # Look in current directory
        ]
        
        csv_path = None
        for path in csv_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            raise FileNotFoundError(f"CSV file not found in any of these locations: {csv_paths}")
        
        if self.debug:
            print(f"Using CSV file: {csv_path}")
        
        self.pairs = pd.read_csv(csv_path, header=None).values
        if self.debug:
            print(f"Found {len(self.pairs)} image pairs for {mode}")
            print(f"Workspace root: {self.workspace_root}")
            for i in range(min(3, len(self.pairs))):
                rgb_path = self.pairs[i, 0]
                depth_path = self.pairs[i, 1]
                print(f"Sample pair {i}: RGB={rgb_path}, Depth={depth_path}")
                abs_rgb_path = os.path.join(self.workspace_root, rgb_path)
                abs_depth_path = os.path.join(self.workspace_root, depth_path)
                print(f"Absolute paths - RGB: {abs_rgb_path} (exists: {os.path.exists(abs_rgb_path)}), Depth: {abs_depth_path} (exists: {os.path.exists(abs_depth_path)})")
        
        # Define fixed output size for consistent tensors
        self.output_height = 480
        self.output_width = 640

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx, max_attempts=10):
        """
        Get a sample from the dataset.
        Includes retry logic to handle corrupted images.
        """
        for attempt in range(max_attempts):
            try:
                # Start with the current index, or try a random one if we're retrying
                if attempt == 0:
                    index = idx
                else:
                    index = np.random.randint(0, len(self.pairs))
                
                # Get file paths
                rgb_path = self.pairs[index, 0]
                depth_path = self.pairs[index, 1]
                
                # Convert to absolute paths
                abs_rgb_path = os.path.join(self.workspace_root, rgb_path)
                abs_depth_path = os.path.join(self.workspace_root, depth_path)
                
                if attempt == 0 and self.debug:
                    print(f"Loading idx {index}: RGB={abs_rgb_path}, Depth={abs_depth_path}")
                    print(f"Files exist: RGB={os.path.exists(abs_rgb_path)}, Depth={os.path.exists(abs_depth_path)}")
                
                # Load RGB image using OpenCV for compatibility with transforms
                rgb_img = cv2.imread(abs_rgb_path)
                if rgb_img is None:
                    raise FileNotFoundError(f"RGB image not found or corrupted: {abs_rgb_path}")
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                
                # Get target size from transform (assuming a Resize transform is in the chain)
                target_size = 392  # Default square size (updated to be divisible by patch size 14)
                if self.transform:
                    # Check if transform is a Compose object with transforms attribute
                    if hasattr(self.transform, 'transforms'):
                        transforms_list = self.transform.transforms
                    else:
                        # It's a single transform
                        transforms_list = [self.transform]
                    
                    # Look for Resize transform
                    for transform in transforms_list:
                        if hasattr(transform, '_Resize__width') and hasattr(transform, '_Resize__height'):
                            target_size = transform._Resize__width  # Assuming width = height for square crop
                            break
                
                # Resize RGB to square dimensions to match expected size
                rgb_img = cv2.resize(rgb_img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
                
                # Load depth image
                depth_img = cv2.imread(abs_depth_path, cv2.IMREAD_UNCHANGED)
                if depth_img is None:
                    raise FileNotFoundError(f"Depth image not found or corrupted: {abs_depth_path}")
                
                # Resize depth to match RGB size
                depth_img = cv2.resize(depth_img, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                
                # Normalize depth based on dtype
                if depth_img.dtype == np.uint16:
                    # For test set, normalize 16-bit depth values to 0-1 range
                    depth_img = depth_img.astype(np.float32) / 65535.0
                else:
                    # For train set, normalize 8-bit depth values to 0-1 range
                    depth_img = depth_img.astype(np.float32) / 255.0
                
                # Ensure depth has correct shape for transforms (H, W, 1)
                if len(depth_img.shape) == 2:
                    depth_img = depth_img[..., np.newaxis]
                
                # Apply transformations or directly convert to tensors with correct shape
                if self.transform:
                    # First convert numpy arrays to tensors
                    rgb_tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float()
                    depth_tensor = torch.from_numpy(depth_img.transpose(2, 0, 1)).float()
                    
                    # Apply transformations directly to the tensors if needed
                    # Note: Most transforms should be applied before this point
                    sample = {'image': rgb_tensor, 'depth': depth_tensor}
                    try:
                        sample = self.transform(sample)
                        rgb_tensor, depth_tensor = sample['image'], sample['depth']
                    except Exception as e:
                        if self.debug:
                            print(f"Warning: Failed to apply transforms: {str(e)}. Using pre-transformed tensors.")
                else:
                    # Convert numpy arrays to tensors with correct shape (CHW)
                    rgb_tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float()
                    depth_tensor = torch.from_numpy(depth_img.transpose(2, 0, 1)).float()
                
                # Additional check to ensure shapes are correct
                if rgb_tensor.shape[0] != 3:
                    if self.debug:
                        print(f"Warning: Unexpected RGB shape: {rgb_tensor.shape}, attempting to fix")
                    
                    # If the dimensions are swapped (e.g., [H, 3, W] instead of [3, H, W])
                    if len(rgb_tensor.shape) == 3 and rgb_tensor.shape[1] == 3:
                        # Swap dimensions to get [3, H, W]
                        rgb_tensor = rgb_tensor.permute(1, 0, 2)
                
                # Final check on tensor shapes
                if rgb_tensor.shape[1:] != depth_tensor.shape[1:]:
                    # Try to fix depth tensor shape if needed
                    if depth_tensor.shape[0] != 1:
                        depth_tensor = depth_tensor[:1]  # Keep only first channel
                    
                    if self.debug:
                        print(f"RGB shape: {rgb_tensor.shape}, Depth shape: {depth_tensor.shape}")
                    
                    if rgb_tensor.shape[1:] != depth_tensor.shape[1:]:
                        # Resize depth to match RGB if still mismatched
                        resize_transform = transforms.Resize(rgb_tensor.shape[1:])
                        depth_tensor = resize_transform(depth_tensor)
                
                # Return tensors
                if self.return_rgb_path:
                    return {'image': rgb_tensor, 'depth': depth_tensor, 'rgb_path': rgb_path}
                else:
                    return {'image': rgb_tensor, 'depth': depth_tensor}
                
            except Exception as e:
                if self.debug or attempt == 0:
                    print(f"Error loading sample {index} (attempt {attempt+1}/{max_attempts}): {str(e)}")
                    print(f"RGB path: {abs_rgb_path}, Depth path: {abs_depth_path}")
                    traceback.print_exc()
        
        # If we reached here, all attempts failed
        raise RuntimeError(f"Failed to load any valid samples after {max_attempts} attempts. Training cannot proceed.") 