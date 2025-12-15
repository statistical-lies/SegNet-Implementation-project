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
    def __init__(self, txt_file, root_dir, width=480, height=360, train=False):
        self.root_dir = root_dir
        self.width = width
        self.height = height
        self.train = train
        self.images = []
        self.masks = []

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line: continue 
                parts = line.split(' ')
                if len(parts) != 2: continue
                img_path, mask_path = parts
                
                # Fix paths
                if img_path.startswith('/SegNet/'): img_path = img_path.replace('/SegNet/', '')
                if mask_path.startswith('/SegNet/'): mask_path = mask_path.replace('/SegNet/', '')
                
                self.images.append(os.path.join(self.root_dir, img_path))
                self.masks.append(os.path.join(self.root_dir, mask_path))

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # CamVid Color Map
        self.class_colors = np.array([
            [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
            [0, 0, 192], [128, 128, 0], [192, 128, 128], [64, 64, 128],
            [64, 0, 128], [64, 64, 0], [0, 128, 192]
        ])

    def map_mask_to_class_id(self, mask):
        mask = np.array(mask)
        if len(mask.shape) == 2:
            tensor = torch.from_numpy(mask).long()
            tensor[tensor >= 11] = -100 # Ignore Void
            return tensor
            
        mask = mask.astype(np.int16)
        mask_class_id = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        mask_class_id.fill(-100) # Default to Ignore
        
        for i, color in enumerate(self.class_colors):
            dist = np.abs(mask - color).sum(axis=2)
            mask_class_id[dist < 15] = i
            
        return torch.from_numpy(mask_class_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if mask.mode == 'P': mask = mask.convert('RGB')
        
        # 1. Resize First
        image = image.resize((self.width, self.height), Image.BILINEAR)
        mask = mask.resize((self.width, self.height), Image.NEAREST)

        # 2. Augmentation for the Training Only
        if self.train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        # 3. Finalize
        image = TF.to_tensor(image)
        image = self.normalize(image)
        if mask.mode != 'L': mask = mask.convert('RGB')
        
        return image, self.map_mask_to_class_id(mask)