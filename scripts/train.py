#!/usr/bin/env python
"""
Training script for PhaseCam3D - PyTorch version.

This script trains the end-to-end optical system and depth estimation network.
Usage:
    python scripts/train.py

The script will:
    1. Load the dataset from .npz files
    2. Build the optical system with learnable phase mask
    3. Train both optical and digital parameters jointly
    4. Save checkpoints and logs to the results directory
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import config
from src import optics, unet
from utils import dataset

# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES


def cost_rms(ground_truth, prediction, mask=None):
    """Root mean square error, scaled to (-10, 10) range.

    Args:
        ground_truth: [batch, 1, H, W]
        prediction: [batch, 1, H, W]
        mask: [batch, H, W] or None - 1 for valid pixels, 0 for NaN/invalid
    """
    diff_sq = (ground_truth - prediction) ** 2
    if mask is not None:
        # Add channel dimension to mask
        mask = mask.unsqueeze(1)
        # Weighted mean over valid pixels only
        loss = torch.sum(diff_sq * mask) / (torch.sum(mask) + 1e-8)
    else:
        loss = torch.mean(diff_sq)
    return 20 * torch.sqrt(loss)


def cost_gradient(ground_truth, prediction, mask=None):
    """Gradient loss to preserve edges in depth estimation.

    Args:
        ground_truth: [batch, 1, H, W]
        prediction: [batch, 1, H, W]
        mask: [batch, H, W] or None
    """
    # Compute image gradients using finite differences
    gt_y = ground_truth[:, :, 1:, :] - ground_truth[:, :, :-1, :]
    gt_x = ground_truth[:, :, :, 1:] - ground_truth[:, :, :, :-1]

    pred_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    pred_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]

    # Pad to original size
    gt_y = torch.nn.functional.pad(gt_y, (0, 0, 0, 1))
    gt_x = torch.nn.functional.pad(gt_x, (0, 1, 0, 0))
    pred_y = torch.nn.functional.pad(pred_y, (0, 0, 0, 1))
    pred_x = torch.nn.functional.pad(pred_x, (0, 1, 0, 0))

    cost_x = cost_rms(gt_x, pred_x, mask)
    cost_y = cost_rms(gt_y, pred_y, mask)

    return cost_x + cost_y


class PhaseCamModel(nn.Module):
    """Complete PhaseCam3D model."""

    def __init__(self, zernike_basis, aperture_index, num_modes, kernel_size):
        super(PhaseCamModel, self).__init__()

        self.kernel_size = kernel_size
        self.crop = (kernel_size - 1) // 2

        # Generate defocus phase
        self.oof_phase = optics.generate_defocus_phase(
            config.PHI_LIST, kernel_size, config.WAVELENGTHS
        )

        # Learnable Zernike coefficients
        lambda_g = config.WAVELENGTHS[1]
        self.zernike_coeffs = nn.Parameter(
            torch.zeros(num_modes, 3, dtype=torch.float32)
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

    # def get_height_map(self):
    #     """Compute height map from Zernike coefficients."""
    #     coeffs_clipped = torch.clamp(
    #         self.zernike_coeffs,
    #         self.coeff_min,
    #         self.coeff_max
    #     )
    #     g = torch.matmul(self.zernike_basis, coeffs_clipped).squeeze(-1)
    #     kernel_size = int(np.sqrt(g.shape[0]))
    #     height_map = nn.functional.relu(g.view(kernel_size, kernel_size) +
    #                                    self.wavelengths[1])
    #     return height_map

    def get_height_maps(self):
        """Compute height maps for each channel from Zernike coefficients."""
        coeffs_clipped = torch.clamp(
            self.zernike_coeffs,
            self.coeff_min,
            self.coeff_max
        )
        g = torch.matmul(self.zernike_basis, coeffs_clipped).t()
        kernel_size = int(np.sqrt(g.shape[1]))
        wavelengths_tensor = torch.from_numpy(self.wavelengths).float().to(g.device)
        height_maps = nn.functional.relu(g.view(3, kernel_size, kernel_size) + wavelengths_tensor.view(3, 1, 1))
        return height_maps

    def get_psfs(self, device):
        """Generate PSFs from current height maps."""
        height_maps = self.get_height_maps().to(device)
        psfs = optics.generate_psfs_multi(
            height_maps,
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

        # Add sensor noise during training
        if self.training:
            blur = optics.add_gaussian_noise(blur, config.NOISE_STD)

        # Digital depth estimation
        estimated_depth = self.depth_network(blur)

        return blur, estimated_depth


def train():
    """Main training function."""
    # Create output directories with timestamp
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Save configuration to results directory
    config_path = os.path.join(config.RESULTS_DIR, 'config.txt')
    with open(config_path, 'w') as f:
        f.write(f"Timestamp: {config.TIMESTAMP}\n")
        f.write(f"Checkpoint dir: {config.CHECKPOINT_DIR}\n")
        f.write(f"Results dir: {config.RESULTS_DIR}\n")
        f.write(f"Log dir: {config.LOG_DIR}\n")

    # Print configuration
    config.print_config()
    print(f"\nOutput directory: {config.RESULTS_DIR}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load Zernike basis
    print("Loading Zernike basis...")
    zernike_data = sio.loadmat(config.ZERNIKE_PATH)
    zernike_basis = zernike_data['u2']  # [N_B*N_B, N_modes]
    aperture_index = zernike_data['idx'].astype(np.float32)

    num_modes = zernike_basis.shape[1]
    kernel_size = config.PSF_SIZE_B

    # Create model
    print("Building model...")
    model = PhaseCamModel(zernike_basis, aperture_index, num_modes, kernel_size)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Fisher Mask 初始化 (三通道独立)
    if os.path.exists(config.FISHER_MASK_PATH_R) and \
       os.path.exists(config.FISHER_MASK_PATH_G) and \
       os.path.exists(config.FISHER_MASK_PATH_B):
        print("Initializing with Fisher Mask (3-channel independent)...")

        # 加载三个通道的 Fisher Mask
        fisher_height_r = np.loadtxt(config.FISHER_MASK_PATH_R)
        fisher_height_g = np.loadtxt(config.FISHER_MASK_PATH_G)
        fisher_height_b = np.loadtxt(config.FISHER_MASK_PATH_B)

        # 分别拟合每个通道的 Zernike 系数
        coeffs_r = optics.fit_zernike_to_height_map(fisher_height_r, zernike_basis)
        coeffs_g = optics.fit_zernike_to_height_map(fisher_height_g, zernike_basis)
        coeffs_b = optics.fit_zernike_to_height_map(fisher_height_b, zernike_basis)

        # 合并三个通道的系数 [num_modes, 3]
        initial_coeffs = torch.cat([coeffs_r, coeffs_g, coeffs_b], dim=1)

        # 将拟合得到的系数写入模型的参数中
        with torch.no_grad():
            model.zernike_coeffs.copy_(initial_coeffs.to(device))
        print("Successfully initialized with Fisher Mask (R/G/B channels)!")
    else:
        print("Fisher Mask files not found (expected FisherMask_R.txt, FisherMask_G.txt, FisherMask_B.txt), starting from zeros.")

    # Separate parameters for optical and digital components
    optical_params = [model.zernike_coeffs]
    digital_params = list(model.depth_network.parameters())

    # Optimizers
    opt_optical = optim.Adam(optical_params, lr=config.LR_OPTICAL)
    opt_digital = optim.Adam(digital_params, lr=config.LR_DIGITAL)

    # Learning rate schedulers (optional)
    scheduler_optical = optim.lr_scheduler.StepLR(
        opt_optical, step_size=10000, gamma=0.5
    )
    scheduler_digital = optim.lr_scheduler.StepLR(
        opt_digital, step_size=10000, gamma=0.5
    )

    # TensorBoard writer
    writer = SummaryWriter(config.LOG_DIR)

    # Load data from NPZ files
    print("\nLoading datasets...")
    train_loader = dataset.get_dataloader(
        config.NPZ_TRAIN_PATH,
        batch_size=config.BATCH_SIZE_TRAIN,
        shuffle=True,
        augment=True,
        num_workers=config.NUM_WORKERS,
        image_size=config.IMAGE_SIZE,
        num_depth_planes=config.NUM_DEPTH_PLANES
    )

    valid_loader = dataset.get_dataloader(
        config.NPZ_VALID_PATH,
        batch_size=config.BATCH_SIZE_VALID,
        shuffle=False,
        augment=False,
        num_workers=config.NUM_WORKERS,
        image_size=config.IMAGE_SIZE,
        num_depth_planes=config.NUM_DEPTH_PLANES
    )

    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    # Create iterators for validation
    valid_iter = iter(valid_loader)

    # Training loop with tqdm
    global_step = 0
    total_iterations = config.MAX_ITERATIONS
    pbar = tqdm(total=total_iterations, desc="Training", unit="iter")

    # Track best model
    best_valid_loss = float('inf')
    best_step = -1

    # Compute epochs needed
    num_epochs = (total_iterations + len(train_loader) - 1) // len(train_loader)

    for epoch in range(num_epochs):
        model.train()

        # Iterate through batches without inner progress bar
        for batch_idx, (rgb_batch, dpphi_batch, phi_batch, mask_batch) in enumerate(train_loader):
            if global_step >= total_iterations:
                break

            # Move to device
            rgb_batch = rgb_batch.to(device)
            dpphi_batch = dpphi_batch.to(device)
            phi_batch = phi_batch.to(device)
            mask_batch = mask_batch.to(device)

            # Crop ground truth and mask to valid region
            crop = model.crop
            phi_gt = phi_batch[:, crop:-crop, crop:-crop].unsqueeze(1)
            mask_cropped = mask_batch[:, crop:-crop, crop:-crop]

            # Forward pass
            blur_train, phi_hat_train = model(rgb_batch, dpphi_batch)

            # Compute losses with mask (ignore NaN pixels)
            cost_rms_train = cost_rms(phi_gt, phi_hat_train, mask_cropped)
            cost_grad_train = cost_gradient(phi_gt, phi_hat_train, mask_cropped)
            loss_train = cost_rms_train + config.WEIGHT_GRADIENT * cost_grad_train

            # Backward pass
            opt_optical.zero_grad()
            opt_digital.zero_grad()
            loss_train.backward()
            opt_optical.step()
            opt_digital.step()

            # Update learning rates
            scheduler_optical.step()
            scheduler_digital.step()

            # Logging and validation
            if global_step % config.SAVE_INTERVAL == 0:
                model.eval()

                with torch.no_grad():
                    # Validation
                    try:
                        rgb_valid, dpphi_valid, phi_valid, mask_valid = next(valid_iter)
                    except StopIteration:
                        valid_iter = iter(valid_loader)
                        rgb_valid, dpphi_valid, phi_valid, mask_valid = next(valid_iter)

                    rgb_valid = rgb_valid.to(device)
                    dpphi_valid = dpphi_valid.to(device)
                    phi_valid = phi_valid.to(device)
                    mask_valid = mask_valid.to(device)

                    phi_gt_valid = phi_valid[:, crop:-crop, crop:-crop].unsqueeze(1)
                    mask_valid_cropped = mask_valid[:, crop:-crop, crop:-crop]
                    blur_valid, phi_hat_valid = model(rgb_valid, dpphi_valid)

                    cost_rms_valid = cost_rms(phi_gt_valid, phi_hat_valid, mask_valid_cropped)
                    cost_grad_valid = cost_gradient(phi_gt_valid, phi_hat_valid, mask_valid_cropped)
                    loss_valid = cost_rms_valid + config.WEIGHT_GRADIENT * cost_grad_valid

                    # Get height maps and PSFs (3 channels)
                    height_maps = model.get_height_maps().cpu().numpy()  # [3, N_B, N_B]
                    psfs = model.get_psfs(device).cpu().numpy()
                    coeffs = model.zernike_coeffs.detach().cpu().numpy()

                # Update main progress bar
                pbar.set_postfix({
                    'train_loss': f'{loss_train.item():.4f}',
                    'valid_loss': f'{loss_valid.item():.4f}',
                    'best_loss': f'{best_valid_loss:.4f}',
                    'best_step': best_step
                })

                # TensorBoard logging
                writer.add_scalar('cost/train', loss_train.item(), global_step)
                writer.add_scalar('cost/valid', loss_valid.item(), global_step)
                writer.add_scalar('cost/rms_train', cost_rms_train.item(), global_step)
                writer.add_scalar('cost/rms_valid', cost_rms_valid.item(), global_step)
                writer.add_scalar('cost/grad_train', cost_grad_train.item(), global_step)
                writer.add_scalar('cost/grad_valid', cost_grad_valid.item(), global_step)
                writer.add_histogram('zernike_coeffs', coeffs, global_step)
                # writer.add_image('HeightMap', height_map[np.newaxis, ...],
                #                 global_step, dataformats='CHW')
                # writer.add_image('valid/sharp',
                #                 rgb_valid[0, :, crop:-crop, crop:-crop].cpu(),
                #                 global_step)
                # writer.add_image('valid/blur', blur_valid[0].cpu(), global_step)
                # writer.add_image('valid/depth_pred', phi_hat_valid[0].cpu(),
                #                 global_step)
                # writer.add_image('valid/depth_gt', phi_gt_valid[0].cpu(),
                #                 global_step)

                # Save checkpoint only if it's the best model
                is_best = loss_valid.item() < best_valid_loss
                if is_best:
                    best_valid_loss = loss_valid.item()
                    best_step = global_step

                    # Save best model checkpoint (overwrite previous best)
                    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'opt_optical_state_dict': opt_optical.state_dict(),
                        'opt_digital_state_dict': opt_digital.state_dict(),
                        'loss_train': loss_train.item(),
                        'loss_valid': loss_valid.item(),
                        'best_valid_loss': best_valid_loss,
                    }, checkpoint_path)

                    # Save height maps (3 channels) and PSFs
                    # height_maps shape: [3, N_B, N_B] - save as separate files for clarity
                    np.savetxt(
                        os.path.join(config.CHECKPOINT_DIR, 'HeightMap_R.txt'),
                        height_maps[0]
                    )
                    np.savetxt(
                        os.path.join(config.CHECKPOINT_DIR, 'HeightMap_G.txt'),
                        height_maps[1]
                    )
                    np.savetxt(
                        os.path.join(config.CHECKPOINT_DIR, 'HeightMap_B.txt'),
                        height_maps[2]
                    )
                    # # Also save combined version for compatibility
                    # np.savetxt(
                    #     os.path.join(config.CHECKPOINT_DIR, 'HeightMap.txt'),
                    #     height_maps.reshape(-1, height_maps.shape[-1])
                    # )
                    # np.savetxt(
                    #     os.path.join(config.CHECKPOINT_DIR, 'zernike_coeffs.txt'),
                    #     coeffs
                    # )
                    # np.save(
                    #     os.path.join(config.CHECKPOINT_DIR, 'PSFs.npy'),
                    #     psfs
                    # )

                    # Also save to results dir for easy access
                    np.savetxt(
                        os.path.join(config.RESULTS_DIR, 'HeightMap_R.txt'),
                        height_maps[0]
                    )
                    np.savetxt(
                        os.path.join(config.RESULTS_DIR, 'HeightMap_G.txt'),
                        height_maps[1]
                    )
                    np.savetxt(
                        os.path.join(config.RESULTS_DIR, 'HeightMap_B.txt'),
                        height_maps[2]
                    )
                    np.savetxt(
                        os.path.join(config.RESULTS_DIR, 'HeightMap.txt'),
                        height_maps.reshape(-1, height_maps.shape[-1])
                    )
                    np.savetxt(
                        os.path.join(config.RESULTS_DIR, 'zernike_coeffs.txt'),
                        coeffs
                    )
                    np.save(
                        os.path.join(config.RESULTS_DIR, 'PSFs.npy'),
                        psfs
                    )

                model.train()

            global_step += 1
            pbar.update(1)

    pbar.close()
    writer.close()
    print(f"\nTraining completed!")
    print(f"Best model at step {best_step} with validation loss: {best_valid_loss:.6f}")
    print(f"Results saved to: {config.RESULTS_DIR}")

    # 自动运行测试 (只在有最优模型时)
    if best_step >= 0:
        print("\n" + "=" * 60)
        print("Starting automatic testing with best model...")
        print("=" * 60)

        import subprocess
        test_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test.py')
        subprocess.run([
            sys.executable, test_script,
            '--checkpoint_dir', config.CHECKPOINT_DIR,
            '--output_dir', os.path.join(config.RESULTS_DIR, 'test_results')
        ], check=True)
    else:
        print("\nNo best model found (possibly no validation was performed).")


if __name__ == "__main__":
    train()
