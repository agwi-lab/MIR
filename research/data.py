import os

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Assuming these are defined elsewhere
from pathlib import Path
from typing import Union
import cv2


def load_exr(path: Path, as_torch: bool = False, channel_first: bool = False) -> Union[np.ndarray, torch.Tensor]:
    # not support alpha channel
    try:
        img: np.ndarray = cv2.cvtColor(cv2.imread(str(path), -1)[..., :3], cv2.COLOR_BGR2RGB)
    except TypeError:
        print(path)
        assert False
    if channel_first:
        img = img.transpose(2, 0, 1)
    if as_torch:
        img: torch.Tensor = torch.from_numpy(img)
    return img


class RDataset(Dataset):
    """
    Dataset that returns:
      - input image (from input_path, EXR)
      - RMSE (tensor scalar)
      - PSNR (tensor scalar, computed from CSV rmse and Imax)
    """
    def __init__(self, csv_path):
        super().__init__()
        self.df = pd.read_csv(csv_path)

        # Optional: keep only needed columns
        required_cols = ["rmse", "Imax", "input_path"]
        for c in required_cols:
            if c not in self.df.columns:
                raise ValueError(f"Missing column '{c}' in CSV")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        rmse_csv = float(row["rmse"])
        psnr_csv = float(row["psnr"])
        Imax = float(row["Imax"])
        input_path = row["input_path"]

        Imax = torch.tensor(Imax, dtype=torch.float32)
        N = torch.tensor(9, dtype=torch.float32)

        rmse = torch.tensor(rmse_csv, dtype=torch.float32)
        psnr = torch.tensor(psnr_csv, dtype=torch.float32)

        input_img = load_exr(input_path, as_torch=True) / Imax  # Normalize
        
        # Ensure input is in [C, H, W] format for CNN
        if input_img.dim() == 3:
            if input_img.shape[0] not in [1, 3, 4]:  # If channels are last
                input_img = input_img.permute(2, 0, 1)

        sample = {
            "input": input_img,
            "rmse": rmse,
            "psnr": psnr,
            'Imax': Imax,
            'N': N,
            "input_path": input_path,
        }
        return sample


def create_data_loaders(csv_path, batch_size=8, val_split=0.1, random_seed=42):
    """Create train, validation, and test data loaders"""
    # Load full dataset
    full_dataset = RDataset(csv_path)
    
    # Get dataset size
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Split indices
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    print(f"Dataset split: {train_size} train, {val_size} val")
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
