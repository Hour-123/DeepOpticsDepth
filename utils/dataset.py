"""
Dataset loading and preprocessing utilities for PyTorch.
Supports .npz files or separate .npy files (rgb, phase, depth).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PhaseCamDataset(Dataset):
    """
    PyTorch Dataset for PhaseCam3D data.
    Supports:
    1. .npz files containing 'rgb', 'dpphi', 'dp' arrays
    2. Separate .npy files: *_rgb.npy, *_phase.npy, *_depth.npy
    """
    def __init__(self, data_paths, image_size=139, num_depth_planes=21, augment=True):
        """
        Args:
            data_paths: List of .npz file paths, or dict with 'rgb', 'phase', 'depth' keys
            image_size: Size of images (default 139 based on converted data)
            num_depth_planes: Number of depth planes
            augment: Whether to apply data augmentation
        """
        self.image_size = image_size
        self.num_depth_planes = num_depth_planes
        self.augment = augment

        # Handle different input formats
        if isinstance(data_paths, dict):
            # Check if dict contains .npz files (new format with 'rgb', 'dpphi', 'dp' keys)
            if data_paths['rgb'][0].endswith('.npz'):
                # All .npz files contain the same arrays, use rgb paths
                self._load_npz_files(data_paths['rgb'])
            else:
                # Separate npy files (old format with 'rgb', 'phase', 'depth' keys)
                self._load_npy_files(data_paths)
        elif isinstance(data_paths, list):
            if data_paths[0].endswith('.npz'):
                self._load_npz_files(data_paths)
            else:
                raise ValueError("Unsupported file format. Use .npz files or provide dict with npy paths")
        else:
            raise ValueError("data_paths must be a list of .npz files or a dict with npy paths")

        self.num_samples = len(self.rgb)
        print(f"Dataset loaded: {self.num_samples} samples")

    def _load_npz_files(self, npz_paths):
        """Load data from .npz files."""
        rgb_list, dpphi_list, dp_list = [], [], []

        for path in npz_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")

            print(f"Loading {os.path.basename(path)}...")
            data = np.load(path)
            rgb_list.append(data['rgb'])
            dpphi_list.append(data['dpphi'])
            dp_list.append(data['dp'])

        self.rgb = np.concatenate(rgb_list, axis=0)
        self.dpphi = np.concatenate(dpphi_list, axis=0)
        self.dp = np.concatenate(dp_list, axis=0)

    def _load_npy_files(self, npy_dict):
        """Load data from separate .npy files."""
        rgb_paths = npy_dict['rgb']
        phase_paths = npy_dict['phase']
        depth_paths = npy_dict['depth']

        rgb_list, phase_list, depth_list = [], [], []

        for rgb_path, phase_path, depth_path in zip(rgb_paths, phase_paths, depth_paths):
            print(f"Loading {os.path.basename(rgb_path)}...")
            rgb = np.load(rgb_path)
            phase = np.load(phase_path)
            depth = np.load(depth_path)

            rgb_list.append(rgb)
            phase_list.append(phase)
            depth_list.append(depth)

        self.rgb = np.concatenate(rgb_list, axis=0)
        self.dpphi = np.concatenate(phase_list, axis=0)
        self.dp = np.concatenate(depth_list, axis=0)

        # Squeeze depth if needed (remove channel dimension if present)
        if self.dp.ndim == 4 and self.dp.shape[-1] == 1:
            self.dp = self.dp.squeeze(-1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. 读取数据
        rgb = self.rgb[idx].astype(np.float32)  # [H, W, 3]
        dpphi = self.dpphi[idx].astype(np.float32)  # [H, W, 21]
        dp = self.dp[idx].astype(np.float32)  # [H, W]

        # 2. 数据归一化
        # RGB: 原始范围 [0, 254]，执行标准归一化
        rgb = rgb / 255.0
        
        # DPPHI: 当前范围是 [0, 1]，已经是 Mask 形式，无需除以 255
        # 直接保持 [0, 1] 即可
        dpphi = dpphi 

        # DP: 当前范围 [52, 157]
        # 将其归一化到 [0, 1] 以匹配 Sigmoid 输出
        dp = (dp - 10.0) / 210.0
        dp = np.clip(dp, 0.0, 1.0)

        # 3. 后续处理 (Mask, NaN 处理, 维度转换)
        mask = np.isfinite(dp).astype(np.float32)
        rgb = np.nan_to_num(rgb, nan=0.0)
        dpphi = np.nan_to_num(dpphi, nan=0.0)
        dp = np.nan_to_num(dp, nan=0.0)

        rgb = np.transpose(rgb, (2, 0, 1))  # [3, H, W]
        dpphi = np.transpose(dpphi, (2, 0, 1))  # [21, H, W]

        if self.augment:
            rgb, dpphi, dp, mask = self._augment(rgb, dpphi, dp, mask)

        return (
            torch.from_numpy(rgb),
            torch.from_numpy(dpphi),
            torch.from_numpy(dp),
            torch.from_numpy(mask)
        )

    def _augment(self, rgb, dpphi, dp, mask):
        """Apply random flip augmentation."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            rgb = np.flip(rgb, axis=2).copy()
            dpphi = np.flip(dpphi, axis=2).copy()
            dp = np.flip(dp, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            rgb = np.flip(rgb, axis=1).copy()
            dpphi = np.flip(dpphi, axis=1).copy()
            dp = np.flip(dp, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        # Random brightness adjustment (only for RGB)
        brightness_factor = np.random.rand() * 0.3 + 0.8  # [0.8, 1.1]
        rgb = np.clip(rgb * brightness_factor, 0, 1)

        return rgb, dpphi, dp, mask


def get_dataloader(data_paths, batch_size, shuffle=True, augment=True,
                   num_workers=0, image_size=139, num_depth_planes=21):
    """
    Create a PyTorch DataLoader for PhaseCam3D data.

    Args:
        data_paths: List of .npz file paths, or dict with 'rgb', 'phase', 'depth' keys
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        augment: Whether to apply data augmentation
        num_workers: Number of data loading workers
        image_size: Size of images
        num_depth_planes: Number of depth planes

    Returns:
        DataLoader instance
    """
    dataset = PhaseCamDataset(
        data_paths,
        image_size=image_size,
        num_depth_planes=num_depth_planes,
        augment=augment
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True
    )
