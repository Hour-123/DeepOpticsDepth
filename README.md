# DeepOpticsDepth - PyTorch Implementation

This is a **PyTorch** implementation of **PhaseCam3D** for single-view depth estimation, developed as a Final Year Project at Tsinghua University (清华大学本科毕业设计).

The project is based on the PhaseCam3D paper from ICCP 2019, migrating the original TensorFlow 1.x implementation to modern PyTorch with full CUDA support.

## Project Structure

```
DeepOpticsDepth/
├── src/                          # Core source modules
│   ├── optics.py                 # Optical simulation (PSF, phase mask, blur) - PyTorch
│   └── unet.py                   # U-Net depth estimation network - PyTorch
│
├── utils/                        # Utility functions
│   └── dataset.py                # PyTorch Dataset for .npz data loading
│
├── configs/
│   └── config.py                 # Centralized configuration with timestamps
│
├── scripts/
│   ├── train.py                  # Training script with tqdm progress bars
│   ├── test.py                   # Testing script with visualization
│   └── convert_data.py           # TFRecord to NPZ converter (run once)
│
├── data/                         # Data directory (symlink to dataset location)
│   ├── PhaseCam3D_preprocess/    # Preprocessed dataset
│   │   ├── train_A.npz           # Training data (RGB, depth probability, depth)
│   │   ├── train_B.npz
│   │   ├── valid_A.npz
│   │   ├── valid_B.npz
│   │   └── test.npz
│   └── zernike_basis.mat         # Zernike polynomial basis (55 modes)
│
├── FisherMask/                   # Fisher mask height initialization (3-channel)
│   ├── FisherMask_R.txt          # Red channel height map
│   ├── FisherMask_G.txt          # Green channel height map
│   ├── FisherMask_B.txt          # Blue channel height map
│   └── FisherMask_phase.png      # Visualization
├── checkpoints/                  # Saved model checkpoints with timestamps
│   └── YYYYMMDD_HHMMSS/
│       ├── best_model.pth        # Best validation model
│       ├── HeightMap_R.txt       # Optimized phase mask height (Red channel)
│       ├── HeightMap_G.txt       # Optimized phase mask height (Green channel)
│       ├── HeightMap_B.txt       # Optimized phase mask height (Blue channel)
│       ├── HeightMap.txt         # Combined height maps (legacy format)
│       ├── zernike_coeffs.txt    # Zernike coefficients [55x3]
│       └── PSFs.npy              # Computed PSFs (3-channel independent)
│
├── results/                      # Training logs and outputs with timestamps
│   └── YYYYMMDD_HHMMSS/
│       ├── logs/                 # TensorBoard logs
│       ├── test_results/         # Test visualization outputs
│       ├── HeightMap.txt
│       ├── zernike_coeffs.txt
│       └── PSFs.npy
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support (recommended)
- NumPy, SciPy, Matplotlib, ImageIO, tqdm
- TensorBoard (for logging)

**No TensorFlow required for training/testing!**

### Installation

```bash
# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Windows-specific Notes

- Set `NUM_WORKERS = 0` in `configs/config.py` (required on Windows)
- Use Git Bash or WSL for symlink creation
- Ensure CUDA toolkit is properly installed for GPU support

## Data Setup

### Prerequisites

The project expects preprocessed data in NPZ format. Each `.npz` file contains three arrays:
- `rgb`: uint8 array of shape `[N, 278, 278, 3]` - all-in-focus RGB images
- `dpphi`: uint8 array of shape `[N, 278, 278, 21]` - depth probability maps for 21 depth planes
- `dp`: uint8 array of shape `[N, 278, 278]` - ground truth defocus coefficients

### Data Location

Configure the data path in `configs/config.py`:

```python
DATA_PATH = os.path.join(BASE_DIR, 'data', 'PhaseCam3D_preprocess')
```

Or create a symlink to your dataset location:

```bash
# Windows (as Administrator)
mklink /D data F:\datasets

# Linux/Mac
ln -s /path/to/datasets data
```

### Converting from Original TFRecord

If you have the original PhaseCam3D TFRecord files:

```bash
# Copy TFRecord files to a temporary location
cp /path/to/original/PhaseCam3D/Data/*.tfrecord temp_data/

# Install TensorFlow 1.x for conversion (use separate environment recommended)
pip install tensorflow-gpu==1.13.1

# Convert TFRecord to NPZ
python scripts/convert_data.py

# Move converted files to data directory
mv temp_data/*.npz data/PhaseCam3D_preprocess/
```

## Usage

### Training

Start training with default configuration:

```bash
python scripts/train.py
```

**Training Features:**
- `tqdm` progress bars with real-time loss display
- Automatic checkpoint saving (best model only)
- TensorBoard logging
- Outputs organized by timestamp: `results/YYYYMMDD_HHMMSS/`

**Monitor Training:**
```bash
tensorboard --logdir results --port 9001
```

**Training Details:**
- The model uses **Fisher Mask** for initialization (3-channel independent: `FisherMask_R.txt`, `FisherMask_G.txt`, `FisherMask_B.txt`)
- Joint optimization of optical parameters (Zernike coefficients [55x3] for RGB channels) and U-Net weights
- Each RGB channel has independent phase mask height map for wavelength-specific optimization
- Best model saved based on validation loss
- Automatic test runs after training completion

### Testing

Test with the best trained model:

```bash
# Test with latest checkpoint (from current config's RESULTS_DIR)
python scripts/test.py

# Test with specific checkpoint directory
python scripts/test.py --checkpoint_dir checkpoints/20250325_143052/ --num_batches 10
```

**Test Features:**
- Progress bar for batch processing
- Automatic visualization generation (depth prediction, ground truth, blur, sharp images)
- Loss curves plotted from TensorBoard logs
- Average test error reported

### Configuration

All hyperparameters are in `configs/config.py`. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `LR_OPTICAL` | Learning rate for phase mask (Zernike coeffs) | 1e-8 |
| `LR_DIGITAL` | Learning rate for U-Net weights | 1e-4 |
| `BATCH_SIZE_TRAIN` | Training batch size | 4 |
| `BATCH_SIZE_VALID` | Validation batch size | 4 |
| `BATCH_SIZE_TEST` | Test batch size | 16 |
| `MAX_ITERATIONS` | Total training iterations | 10000 |
| `SAVE_INTERVAL` | Validation/save interval (steps) | 50 |
| `WEIGHT_GRADIENT` | Weight for gradient loss | 1.0 |
| `PSF_SIZE_R/G/B` | PSF kernel sizes for RGB channels | 31/27/23 |
| `NUM_DEPTH_PLANES` | Number of depth planes | 21 |

## Technical Details

### End-to-End Optimization Framework

The system jointly optimizes:
1. **Optical Frontend**: Three independent phase masks (RGB channels) represented by Zernike polynomials (55 modes per channel)
2. **Digital Backend**: U-Net for depth estimation from optically coded images

```
Input RGB → [Phase Masks (R/G/B) + Optics] → Blurred Image → [U-Net] → Depth Map
                ↑                                                   ↓
         (Learnable Zernike                                   (Supervised
          Coefficients [55x3])                                Learning)
```

**3-Channel Independent Optimization**: Each RGB channel has its own phase mask height map, allowing independent optimization for wavelength-specific aberrations.

### Loss Function

Combined loss for end-to-end training:
- **L_RMS**: Root mean square error between predicted and ground truth depth
- **L_grad**: Gradient loss to preserve depth discontinuities

```
L_total = L_RMS + λ·L_grad  (λ = WEIGHT_GRADIENT)
```

### PSF Generation

Point Spread Functions are computed independently for each RGB channel from their respective phase mask height maps:

```
h_c(x,y) = Σ a_jc · Z_j(x,y)        # Height map for channel c (R/G/B)
φ^M_c = (2π/λ_c) · (n-1) · h_c      # Mask phase for channel c
φ^DF_c = W_m · (x²+y²) · (λ_g/λ_c)  # Defocus phase (scaled by wavelength)
PSF_c = |FFT{ A·exp[i(φ^M_c + φ^DF_c)] }|²
```

Each channel has independent Zernike coefficients `a_jc` [55x3], allowing wavelength-specific optimization.

### Data Normalization

| Data | Original Range | Normalized Range |
|------|---------------|------------------|
| RGB | [0, 254] | [0, 1] (divide by 255) |
| Depth Probability (dpphi) | [0, 1] | [0, 1] (unchanged) |
| Depth (dp) | [52, 157] | [0, 1] via `(dp - 10) / 210` |

## Output Structure with Timestamps

Every training run creates a new timestamped directory:

```
checkpoints/
├── 20250325_143052/          # Training run at 14:30:52
│   ├── best_model.pth        # Best validation checkpoint
│   ├── HeightMap_R.txt       # Optimized phase mask (Red channel, 23x23)
│   ├── HeightMap_G.txt       # Optimized phase mask (Green channel, 23x23)
│   ├── HeightMap_B.txt       # Optimized phase mask (Blue channel, 23x23)
│   ├── HeightMap.txt         # Combined height maps [69x23] (legacy format)
│   ├── zernike_coeffs.txt    # Zernike coefficients (55x3)
│   └── PSFs.npy              # Computed PSFs (21x23x23x3)
└── 20250325_154123/          # Another training run
    └── ...

results/
├── 20250325_143052/          # Corresponding results
│   ├── logs/                 # TensorBoard event files
│   ├── HeightMap_R.txt       # (copy of checkpoint)
│   ├── HeightMap_G.txt
│   ├── HeightMap_B.txt
│   ├── zernike_coeffs.txt
│   ├── PSFs.npy
│   └── test_results/         # Test visualizations
│       ├── loss_curves.png
│       ├── 0_phiHat.png      # Predicted depth
│       ├── 0_phiGT.png       # Ground truth depth
│       ├── 0_blur.png        # Blurred input
│       └── 0_sharp.png       # Sharp reference
└── 20250325_154123/
    └── ...
```

**Benefits:**
- Multiple experiments organized without conflicts
- Easy comparison between runs
- Automatic result preservation

## Key Improvements Over Original

1. **PyTorch Framework**: Modern PyTorch 2.0+ implementation with `torch.fft` for optical computation
2. **3-Channel Independent Optimization**: Each RGB channel has its own learnable phase mask (Zernike coefficients [55x3]) for wavelength-specific aberration correction
3. **Fisher Mask Initialization**: Proper initialization with channel-specific Fisher masks (R/G/B) for stable training convergence
4. **Progress Visualization**: `tqdm` progress bars with real-time loss metrics
5. **Timestamped Organization**: Automatic experiment tracking with datetime directories
6. **Flexible Data Loading**: Support for both NPZ bundles and separate NPY files
7. **Full CUDA Support**: Complete GPU acceleration for optical and neural components
8. **Data Normalization**: Proper handling of different data ranges during preprocessing

## References

### Original Paper

This implementation is based on the PhaseCam3D paper:

> Yicheng Wu, Vivek Boominathan, Huaijin Chen, Aswin Sankaranarayanan, and Ashok Veeraraghavan.
> "PhaseCam3D—Learning Phase Masks for Passive Single View Depth Estimation"
> *IEEE International Conference on Computational Photography (ICCP), 2019*

Original TensorFlow repository: https://github.com/YichengWu/PhaseCam3D

### Citation

```bibtex
@inproceedings{wu2019phasecam3d,
  title={PhaseCam3D—Learning Phase Masks for Passive Single View Depth Estimation},
  author={Wu, Yicheng and Boominathan, Vivek and Chen, Huaijin and Sankaranarayanan, Aswin and Veeraraghavan, Ashok},
  booktitle={2019 IEEE International Conference on Computational Photography (ICCP)},
  pages={1--12},
  year={2019},
  organization={IEEE}
}
```

## Project Info

- **Institution**: Tsinghua University (清华大学)
- **Type**: Undergraduate Final Year Project (本科毕业设计)
- **Year**: 2025
- **License**: Same as the original PhaseCam3D project

---

*For detailed technical documentation, see `PhaseCam3D 技术报告.md` (Chinese).*
