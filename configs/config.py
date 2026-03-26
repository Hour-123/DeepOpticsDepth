"""
Configuration for PhaseCam3D training and testing.

All hyperparameters and paths are centralized here for easy modification.
"""
import os
import numpy as np
from datetime import datetime

# =============================================================================
# Paths
# =============================================================================
# Base directory (relative to project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_PATH = os.path.join(BASE_DIR, 'data', 'PhaseCam3D_preprocess')
ZERNIKE_PATH = os.path.join(BASE_DIR, 'zernike_basis.mat')

# Fisher Mask paths for RGB channels (3-channel independent optimization)
FISHER_MASK_DIR = os.path.join(BASE_DIR, 'FisherMask')
FISHER_MASK_PATH_R = os.path.join(FISHER_MASK_DIR, 'FisherMask_R.txt')
FISHER_MASK_PATH_G = os.path.join(FISHER_MASK_DIR, 'FisherMask_G.txt')
FISHER_MASK_PATH_B = os.path.join(FISHER_MASK_DIR, 'FisherMask_B.txt')

# Backward compatibility: single Fisher Mask path (for legacy code)
FISHER_MASK_PATH = FISHER_MASK_PATH_G  # Default to G channel

# Dataset paths (.npz files containing 'rgb', 'dpphi', 'dp' arrays)
# Each .npz file contains:
#   - rgb: [N, 278, 278, 3] - all-in-focus image (uint8)
#   - dpphi: [N, 278, 278, 21] - depth probability map for 21 defocus levels (uint8)
#   - dp: [N, 278, 278] - defocus coefficient ground truth (uint8)
NPZ_TRAIN_PATH = {
    'rgb': [
        os.path.join(DATA_PATH, 'train_A.npz'),
        # os.path.join(DATA_PATH, 'train_B.npz'),
        # os.path.join(DATA_PATH, 'train_C.npz'),
    ],
    'dpphi': [
        os.path.join(DATA_PATH, 'train_A.npz'),
        # os.path.join(DATA_PATH, 'train_B.npz'),
        # os.path.join(DATA_PATH, 'train_C.npz'),
    ],
    'dp': [
        os.path.join(DATA_PATH, 'train_A.npz'),
        # os.path.join(DATA_PATH, 'train_B.npz'),
        # os.path.join(DATA_PATH, 'train_C.npz'),
    ],
}
NPZ_VALID_PATH = {
    'rgb': [
        os.path.join(DATA_PATH, 'valid_A.npz'),
        # os.path.join(DATA_PATH, 'valid_B.npz'),
    ],
    'dpphi': [
        os.path.join(DATA_PATH, 'valid_A.npz'),
        # os.path.join(DATA_PATH, 'valid_B.npz'),
    ],
    'dp': [
        os.path.join(DATA_PATH, 'valid_A.npz'),
        # os.path.join(DATA_PATH, 'valid_B.npz'),
    ],
}
NPZ_TEST_PATH = {
    'rgb': [os.path.join(DATA_PATH, 'test.npz')],
    'dpphi': [os.path.join(DATA_PATH, 'test.npz')],
    'dp': [os.path.join(DATA_PATH, 'test.npz')],
}

# Timestamp for output directories
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Output paths with timestamp
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints', TIMESTAMP)
RESULTS_DIR = os.path.join(BASE_DIR, 'results', TIMESTAMP)
LOG_DIR = os.path.join(RESULTS_DIR, 'logs')

# =============================================================================
# Optical System Parameters
# =============================================================================
# PSF kernel sizes for RGB channels
# Reference size is 23x23 as mentioned in dataset description
PSF_SIZE_R = 31
PSF_SIZE_G = 27
PSF_SIZE_B = 23

# Wavelengths for RGB channels (in meters)
WAVELENGTHS = np.array([610, 530, 470]) * 1e-9

# Refractive index
REFRACTIVE_INDEX = 1.5

# =============================================================================
# Depth Parameters
# =============================================================================
# Depth plane defocus values
PHI_LIST = np.linspace(-10, 10, 21, dtype=np.float32)
NUM_DEPTH_PLANES = len(PHI_LIST)

# Image dimensions
# Input patch size is 278x278 to allow 23x23 PSF convolution to output 256x256
IMAGE_SIZE = 278  # Will be auto-detected from dataset if possible

# =============================================================================
# Training Parameters
# =============================================================================
# Learning rates
LR_OPTICAL = 1e-8   # Learning rate for phase mask optimization
LR_DIGITAL = 1e-4   # Learning rate for U-Net optimization

# Batch sizes
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_VALID = 4
BATCH_SIZE_TEST = 8

# Training schedule
MAX_ITERATIONS = 10
SAVE_INTERVAL = 1

# Data augmentation
SHUFFLE_BUFFER_SIZE = 5000

# =============================================================================
# Loss Parameters
# =============================================================================
# Noise standard deviation
NOISE_STD = 0.01

# Gradient loss weight
WEIGHT_GRADIENT = 1.0

# =============================================================================
# Hardware Configuration
# =============================================================================
# GPU settings
CUDA_VISIBLE_DEVICES = "0"

# DataLoader workers 
NUM_WORKERS = 0 # must be 0 for windows


def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("DeepOpticsDepth Configuration")
    print("=" * 60)
    print(f"Data path: {DATA_PATH}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"-" * 60)
    print(f"PSF sizes: R={PSF_SIZE_R}, G={PSF_SIZE_G}, B={PSF_SIZE_B}")
    print(f"Wavelengths: {WAVELENGTHS * 1e9} nm")
    print(f"Depth planes: {NUM_DEPTH_PLANES} ({PHI_LIST[0]} to {PHI_LIST[-1]})")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"-" * 60)
    print(f"Learning rates: optical={LR_OPTICAL}, digital={LR_DIGITAL}")
    print(f"Batch size (train): {BATCH_SIZE_TRAIN}")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
