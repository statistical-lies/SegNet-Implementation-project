import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# ==========================================
# 2. DATASET LOADER
# ==========================================
class CamVidDataset(Dataset):
    def __init__(self, txt_file, root_dir, width=480, height=360, max_images=None):
        """
        Args:
            txt_file (str): Path to train.txt or val.txt
            root_dir (str): Your local folder containing the 'CamVid' folder
        """
        self.root_dir = root_dir
        self.width = width
        self.height = height
        self.images = []
        self.masks = []

        # 1. Parse the text file
        # The text file contains lines like: /SegNet/CamVid/train/X.png /SegNet/CamVid/trainannot/X.png
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line: continue 
                
                # Split into image path and mask path
                parts = line.split(' ')
                if len(parts) != 2: continue
                
                img_path, mask_path = parts
                
                # 2. Fix the paths to match your local computer
                # The text file uses absolute paths from the author's computer.
                # We remove the "/SegNet/" prefix so it becomes relative: "CamVid/train/..."
                if img_path.startswith('/SegNet/'):
                    img_path = img_path.replace('/SegNet/', '')
                if mask_path.startswith('/SegNet/'):
                    mask_path = mask_path.replace('/SegNet/', '')
                
                # Combine with your local root directory (usually '.')
                self.images.append(os.path.join(self.root_dir, img_path))
                self.masks.append(os.path.join(self.root_dir, mask_path))
        
        if max_images:
            self.images = self.images[:max_images]
            self.masks = self.masks[:max_images]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # CamVid color map (Robust Version)
        self.class_colors = np.array([
            [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
            [0, 0, 192], [128, 128, 0], [192, 128, 128], [64, 64, 128],
            [64, 0, 128], [64, 64, 0], [0, 128, 192]
        ])

    def map_mask_to_class_id(self, mask):
        # Ensure mask is numpy array
        mask = np.array(mask)
        # Scenario 1: Mask is already integer (H, W)
        if len(mask.shape) == 2:
            tensor = torch.from_numpy(mask).long()
            # CRITICAL FIX: 
            # Map any value >= 11 (Void) to -100.
            # PyTorch CrossEntropyLoss ignores -100 by default.
            tensor[tensor >= 11] = -100 
            return tensor
            
        mask = mask.astype(np.int16)
        mask_class_id = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        
        # Map colors to IDs robustly
        for i, color in enumerate(self.class_colors):
            dist = np.abs(mask - color).sum(axis=2)
            mask_class_id[dist < 15] = i
            
        return torch.from_numpy(mask_class_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # Open image and mask
        # We check if the file exists to avoid errors if paths are slightly wrong
        if not os.path.exists(img_path):
             raise FileNotFoundError(f"Could not find image: {img_path}")
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        if mask.mode == 'P': mask = mask.convert('RGB')
        
        image = image.resize((self.width, self.height), Image.BILINEAR)
        mask = mask.resize((self.width, self.height), Image.NEAREST)
        
        if mask.mode != 'L': mask = mask.convert('RGB')
        
        return self.transform(image), self.map_mask_to_class_id(mask)