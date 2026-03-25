#!/usr/bin/env python
"""
Testing script for PhaseCam3D - PyTorch version.

This script evaluates a trained PhaseCam3D model on the test dataset
and saves visualization results.

Usage:
    python scripts/test.py [--checkpoint_dir PATH] [--output_dir PATH]

The script will:
    1. Load a trained model from checkpoint
    2. Run inference on the test dataset
    3. Save depth predictions, ground truth, and input images
    4. Compute and report average test error
"""
import os
import sys
import argparse
import numpy as np
import torch
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.image
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import imageio
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import config
from src import optics, unet
from utils import dataset

# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES


class PhaseCamModel(torch.nn.Module):
    """Complete PhaseCam3D model for inference."""

    def __init__(self, zernike_basis, aperture_index, num_modes, kernel_size,
                 height_map_init=None):
        super(PhaseCamModel, self).__init__()

        self.kernel_size = kernel_size
        self.crop = (kernel_size - 1) // 2

        # Generate defocus phase
        self.oof_phase = optics.generate_defocus_phase(
            config.PHI_LIST, kernel_size, config.WAVELENGTHS
        )

        # Zernike coefficients
        lambda_g = config.WAVELENGTHS[1]
        if height_map_init is not None:
            # Initialize from saved height map
            self.zernike_coeffs = torch.nn.Parameter(
                torch.zeros(num_modes, 1, dtype=torch.float32)
            )
        else:
            self.zernike_coeffs = torch.nn.Parameter(
                torch.zeros(num_modes, 1, dtype=torch.float32)
            )

        self.register_buffer('zernike_basis',
                            torch.from_numpy(zernike_basis).float())
        self.coeff_min = -lambda_g / 2
        self.coeff_max = lambda_g / 2

        # Store other parameters
        self.wavelengths = config.WAVELENGTHS
        self.aperture_index = aperture_index
        self.size_r = config.PSF_SIZE_R
        self.size_g = config.PSF_SIZE_G
        self.size_b = config.PSF_SIZE_B

        # Digital depth estimation network
        self.depth_network = unet.UNet()

        # Height map initialization
        self.height_map_init = height_map_init

    def get_height_map(self):
        """Compute height map from Zernike coefficients."""
        if self.height_map_init is not None and not self.training:
            # Use loaded height map for inference
            device = self.zernike_coeffs.device
            return torch.from_numpy(self.height_map_init).float().to(device)

        coeffs_clipped = torch.clamp(
            self.zernike_coeffs,
            self.coeff_min,
            self.coeff_max
        )
        g = torch.matmul(self.zernike_basis, coeffs_clipped).squeeze(-1)
        kernel_size = int(np.sqrt(g.shape[0]))
        height_map = torch.nn.functional.relu(g.view(kernel_size, kernel_size) +
                                               self.wavelengths[1])
        return height_map

    def get_psfs(self, device):
        """Generate PSFs from current height map."""
        height_map = self.get_height_map().to(device)
        psfs = optics.generate_psfs(
            height_map,
            self.oof_phase,
            self.wavelengths,
            self.aperture_index,
            self.size_r,
            self.size_g,
            self.size_b
        )
        return psfs

    def forward(self, rgb_image, depth_probability):
        """
        Forward pass.

        Args:
            rgb_image: [batch, 3, H, W]
            depth_probability: [batch, N_Phi, H, W]

        Returns:
            blurred_image, estimated_depth
        """
        device = rgb_image.device

        # Generate PSFs
        psfs = self.get_psfs(device)

        # Apply optical blur
        blur = optics.blur_image(rgb_image, depth_probability, psfs)

        # Add sensor noise (optional for testing)
        blur = optics.add_gaussian_noise(blur, config.NOISE_STD)

        # Digital depth estimation
        estimated_depth = self.depth_network(blur)

        return blur, estimated_depth


def load_model(checkpoint_dir, device):
    """Load trained model from checkpoint."""
    # Load Zernike basis
    zernike_data = sio.loadmat(config.ZERNIKE_PATH)
    zernike_basis = zernike_data['u2']
    aperture_index = zernike_data['idx'].astype(np.float32)

    num_modes = zernike_basis.shape[1]
    kernel_size = config.PSF_SIZE_B

    # Load height map - try checkpoint dir first, then find in parent dirs
    height_map_path = os.path.join(checkpoint_dir, 'HeightMap.txt')
    if not os.path.exists(height_map_path):
        # Try parent of checkpoint dir
        height_map_path = os.path.join(os.path.dirname(checkpoint_dir), 'HeightMap.txt')
    if not os.path.exists(height_map_path):
        # Try results dir from config
        height_map_path = os.path.join(config.RESULTS_DIR, 'HeightMap.txt')

    if not os.path.exists(height_map_path):
        raise FileNotFoundError(
            f"Height map not found. Please train the model first or provide a valid checkpoint.\n"
            f"Searched in: {checkpoint_dir} and parent directories"
        )

    height_map_init = np.loadtxt(height_map_path)
    print(f"Loaded height map from: {height_map_path}")

    # Create model
    model = PhaseCamModel(zernike_basis, aperture_index, num_modes,
                         kernel_size, height_map_init)
    model = model.to(device)

    # Load checkpoint - prefer best_model.pth
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_checkpoint_path):
        checkpoint_path = best_checkpoint_path
    else:
        # Fallback: load any .pth file
        checkpoint_files = [f for f in os.listdir(checkpoint_dir)
                           if f.endswith('.pth')]
        if not checkpoint_files:
            raise ValueError(f"No checkpoint found in {checkpoint_dir}")
        checkpoint_files.sort()
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Training step: {checkpoint.get('step', 'unknown')}")
    if 'best_valid_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['best_valid_loss']:.6f}")

    return model


def save_results(depth_pred, depth_gt, blur, rgb, crop, output_dir,
                batch_idx, batch_size):
    """Save visualization results.

    Args:
        depth_pred: Predicted depth [batch, 1, H, W]
        depth_gt: Ground truth depth [batch, 1, H, W]
        blur: Blurred image [batch, 3, H, W]
        rgb: Input RGB image [batch, 3, H, W]
        crop: Crop size
        output_dir: Output directory
        batch_idx: Batch index
        batch_size: Number of images to save (full batch)
    """
    # Convert to numpy
    depth_pred = depth_pred.cpu().numpy()
    depth_gt = depth_gt.cpu().numpy()
    blur = blur.cpu().numpy()
    rgb = rgb.cpu().numpy()

    # Crop RGB to valid region
    rgb_cropped = rgb[:, :, crop:-crop, crop:-crop]

    for j in range(batch_size):
        # Save depth prediction
        matplotlib.image.imsave(
            os.path.join(output_dir, f'{j}_phiHat.png'),
            depth_pred[j, 0, :, :],
            vmin=0.0, vmax=1.0,
            cmap='jet'
        )

        # Save ground truth
        matplotlib.image.imsave(
            os.path.join(output_dir, f'{j}_phiGT.png'),
            depth_gt[j, 0, :, :],
            vmin=0.0, vmax=1.0,
            cmap='jet'
        )

        # Save blurred image
        blur_img = np.transpose(blur[j], (1, 2, 0))  # CHW -> HWC
        imageio.imwrite(
            os.path.join(output_dir, f'{j}_blur.png'),
            np.uint8(np.clip(blur_img, 0, 1) * 255)
        )

        # Save sharp image
        sharp_img = np.transpose(rgb_cropped[j], (1, 2, 0))  # CHW -> HWC
        imageio.imwrite(
            os.path.join(output_dir, f'{j}_sharp.png'),
            np.uint8(np.clip(sharp_img, 0, 1) * 255)
        )


def load_tensorboard_scalars(log_dir):
    """Load scalar data from TensorBoard logs.

    Args:
        log_dir: Directory containing events.out.tfevents files

    Returns:
        dict: {tag: [(step, value), ...]}
    """
    scalars = {}

    # Find events file
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not event_files:
        print(f"Warning: No TensorBoard events found in {log_dir}")
        return scalars

    events_path = os.path.join(log_dir, event_files[0])
    print(f"Loading TensorBoard logs from: {events_path}")

    # Load events
    ea = event_accumulator.EventAccumulator(events_path)
    ea.Reload()

    # Get all scalar tags
    tags = ea.Tags()['scalars']
    print(f"Found {len(tags)} scalar tags: {tags}")

    for tag in tags:
        events = ea.Scalars(tag)
        scalars[tag] = [(e.step, e.value) for e in events]

    return scalars


def plot_loss_curves(scalars, output_dir):
    """Plot training and validation loss curves.

    Args:
        scalars: Dict of scalar data from tensorboard
        output_dir: Directory to save plots
    """
    # Common loss tags to plot
    loss_groups = {
        'Total Loss': {
            'train': 'cost/train',
            'valid': 'cost/valid',
        },
        'RMS Loss': {
            'train': 'cost/rms_train',
            'valid': 'cost/rms_valid',
        },
        'Gradient Loss': {
            'train': 'cost/grad_train',
            'valid': 'cost/grad_valid',
        }
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (title, tags) in enumerate(loss_groups.items()):
        ax = axes[idx]

        # Plot train
        train_tag = tags.get('train')
        if train_tag and train_tag in scalars:
            steps, values = zip(*scalars[train_tag])
            ax.plot(steps, values, 'b-', label='Train', linewidth=1.5)

        # Plot valid
        valid_tag = tags.get('valid')
        if valid_tag and valid_tag in scalars:
            steps, values = zip(*scalars[valid_tag])
            ax.plot(steps, values, 'r-', label='Valid', linewidth=1.5)

        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to: {plot_path}")


def test():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test PhaseCam3D model')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=config.CHECKPOINT_DIR,
        help='Directory containing trained model checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(config.RESULTS_DIR, 'test_all'),
        help='Directory to save test results'
    )
    parser.add_argument(
        '--num_batches',
        type=int,
        default=1,
        help='Number of test batches to evaluate (default: 1 batch = 40 images)'
    )
    args = parser.parse_args()

    # Create output directory with timestamp if not specified
    if args.output_dir == os.path.join(config.RESULTS_DIR, 'test_all'):
        args.output_dir = os.path.join(config.RESULTS_DIR, f'test_{config.TIMESTAMP}')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of batches: {args.num_batches} (batch_size={config.BATCH_SIZE_TEST}, total images: {args.num_batches * config.BATCH_SIZE_TEST})")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint_dir, device)
    model.eval()

    # Load and plot training loss curves from TensorBoard logs
    print("\nLoading training logs...")
    # Find results directory (parent of checkpoint or use RESULTS_DIR)
    if 'checkpoints' in args.checkpoint_dir:
        results_dir = args.checkpoint_dir.replace('checkpoints', 'results')
    else:
        results_dir = config.RESULTS_DIR

    log_dir = os.path.join(results_dir, 'logs')
    if os.path.exists(log_dir):
        scalars = load_tensorboard_scalars(log_dir)
        if scalars:
            plot_loss_curves(scalars, args.output_dir)
    else:
        print(f"Warning: Log directory not found: {log_dir}")

    # Load test data from NPZ files
    print("Loading test dataset...")
    test_loader = dataset.get_dataloader(
        config.NPZ_TEST_PATH,
        batch_size=config.BATCH_SIZE_TEST,
        shuffle=False,
        augment=False,
        num_workers=0,
        image_size=config.IMAGE_SIZE,
        num_depth_planes=config.NUM_DEPTH_PLANES
    )

    # Run tests with tqdm
    test_losses = []
    print("Running inference on test set...")

    crop = model.crop

    # Determine number of batches to run
    total_batches = min(args.num_batches, len(test_loader))

    with torch.no_grad():
        pbar = tqdm(total=total_batches, desc="Testing", unit="batch")

        for i, (rgb_test, dpphi_test, phi_test, mask_test) in enumerate(test_loader):
            if i >= args.num_batches:
                break

            # Move to device
            rgb_test = rgb_test.to(device)
            dpphi_test = dpphi_test.to(device)
            phi_test = phi_test.to(device)

            # Crop ground truth
            phi_gt_test = phi_test[:, crop:-crop, crop:-crop].unsqueeze(1)

            # Forward pass
            blur_test, phi_hat_test = model(rgb_test, dpphi_test)

            # Compute error
            cost = 20 * torch.sqrt(torch.mean((phi_gt_test - phi_hat_test) ** 2))
            test_losses.append(cost.item())

            # Update progress bar
            pbar.set_postfix({'loss': f'{cost.item():.4f}'})
            pbar.update(1)

            # Save results (save all images in batch)
            save_results(
                phi_hat_test, phi_gt_test, blur_test, rgb_test, crop,
                args.output_dir, i, config.BATCH_SIZE_TEST
            )

        pbar.close()

    # Compute and save average loss
    avg_loss = np.mean(test_losses)
    total_images = len(test_losses) * config.BATCH_SIZE_TEST
    print(f"\nAverage Test Loss = {avg_loss:.6f}")
    print(f"Total batches: {len(test_losses)}, Total images: {total_images}")

    loss_file = os.path.join(
        args.output_dir,
        f'test_loss_{avg_loss:.6f}.txt'
    )
    np.savetxt(loss_file, test_losses)
    print(f"Test losses saved to: {loss_file}")


if __name__ == "__main__":
    test()
