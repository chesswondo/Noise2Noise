import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import Dataset
import torch
from typing import List, Tuple, Optional
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import pytorch_lightning as pl

class Noise2NoiseDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        patch_size: int = 128,
        noise_std: float = 0.15,
        is_validation: bool = False
    ):

        self.image_paths = image_paths
        self.noise_std = noise_std
        self.is_validation = is_validation
        self.patch_size = patch_size
        
        if is_validation:
            self.transform = A.Compose([
                A.CenterCrop(height=patch_size, width=patch_size),
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                # A.RandomCrop(height=patch_size, width=patch_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def add_noise(self, clean_tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(clean_tensor) * self.noise_std
        noisy_tensor = torch.clamp(clean_tensor + noise, 0.0, 1.0)
        return noisy_tensor
    
    def get_informative_crop(self, img, crop_size=256, threshold=20, max_attempts=10):
        """Getting informative crops from the image."""
        h, w, c = img.shape
        
        # If the image is too small
        if h < crop_size or w < crop_size:
            return A.RandomCrop(crop_size, crop_size)(image=img)['image']

        for _ in range(max_attempts):
            y = np.random.randint(0, h - crop_size + 1)
            x = np.random.randint(0, w - crop_size + 1)
            
            crop = img[y:y+crop_size, x:x+crop_size]
            
            # Calculate standard deviation
            if np.std(crop) > threshold:
                return crop
                
        return crop

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not self.is_validation:
            img = self.get_informative_crop(img, self.patch_size)

        augmented = self.transform(image=img)
        clean = augmented['image'] # Shape: [3, H, W]

        noisy_input = self.add_noise(clean)
        noisy_target = self.add_noise(clean)

        return noisy_input, noisy_target, clean


class DenoisingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        batch_size: int = 16,
        patch_size: int = 128,
        noise_std: float = 0.15,
        num_workers: int = 0
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.noise_std = noise_std
        self.num_workers = num_workers

    def _get_paths(self, directory: str) -> List[str]:
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        paths = []
        for ext in exts:
            paths.extend(list(Path(directory).rglob(ext)))
        return [str(p) for p in paths]

    def setup(self, stage: Optional[str] = None):
        train_paths = self._get_paths(self.train_dir)
        val_paths = self._get_paths(self.val_dir)

        print(f"Found {len(train_paths)} training images and {len(val_paths)} validation images.")

        if stage == 'fit' or stage is None:
            self.train_dataset = Noise2NoiseDataset(
                train_paths, 
                patch_size=self.patch_size, 
                noise_std=self.noise_std, 
                is_validation=False
            )
            self.val_dataset = Noise2NoiseDataset(
                val_paths, 
                patch_size=self.patch_size, 
                noise_std=self.noise_std, 
                is_validation=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )