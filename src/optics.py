"""
Optical system simulation for PhaseCam3D - PyTorch version.

This module implements the optical components:
- Phase mask generation using Zernike polynomials
- PSF (Point Spread Function) computation for RGB channels
- Image blurring through depth-dependent PSFs
- End-to-end optical system simulation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_defocus_phase(phi_list, kernel_size, wavelengths):
    """
    Generate out-of-focus phase for different depth planes.

    Args:
        phi_list: List of defocus values (e.g., np.linspace(-10, 10, 21))
        kernel_size: Size of the PSF kernel (N_B)
        wavelengths: Array of wavelengths for R, G, B channels [3]

    Returns:
        OOFphase: Array of shape [N_Phi, N_B, N_B, 3]
    """
    N = kernel_size
    x0 = np.linspace(-1.1, 1.1, N)
    xx, yy = np.meshgrid(x0, x0)

    num_phi = len(phi_list)
    num_wavelengths = len(wavelengths)
    OOFphase = np.empty([num_phi, N, N, num_wavelengths], dtype=np.float32)

    for j in range(num_phi):
        phi = phi_list[j]
        for k in range(num_wavelengths):
            # Scale phase by wavelength ratio
            OOFphase[j, :, :, k] = phi * (xx ** 2 + yy ** 2) * wavelengths[1] / wavelengths[k]

    return OOFphase


def fft2dshift(input_tensor):
    """
    Shift the FFT output to center the zero frequency component.
    Equivalent to MATLAB's fftshift for 2D tensors.

    Args:
        input_tensor: Input tensor of shape [batch, height, width]

    Returns:
        Shifted tensor of same shape
    """
    dim = input_tensor.shape[-1]
    if dim % 2 == 0:
        print('Warning: Please make the size of kernel odd')

    # Use torch.fft.fftshift for PyTorch >= 1.9
    return torch.fft.fftshift(input_tensor, dim=(-2, -1))


def generate_psfs(height_map, oof_phase, wavelengths, aperture_index,
                  size_r, size_g, size_b):
    """
    Generate RGB Point Spread Functions from height map.

    Args:
        height_map: Phase mask height map [N_B, N_B] (tensor)
        oof_phase: Out-of-focus phase [N_Phi, N_B, N_B, 3] (numpy array)
        wavelengths: Array of wavelengths [3] in meters
        aperture_index: Binary aperture mask [N_B, N_B]
        size_r: PSF size for red channel
        size_g: PSF size for green channel
        size_b: PSF size for blue channel (reference size)

    Returns:
        PSFs: Point spread functions [N_Phi, N_B, N_B, 3]
    """
    n = 1.5  # Refractive index
    kernel_size = size_b
    num_phi = oof_phase.shape[0]

    device = height_map.device

    # Convert numpy arrays to tensors
    oof_phase = torch.from_numpy(oof_phase).to(device)
    aperture_index = torch.from_numpy(aperture_index).to(device)

    psfs_list = []

    # Red channel
    oof_phase_r = oof_phase[:, :, :, 0]  # [N_Phi, N_B, N_B]
    phase_r = 2 * np.pi / wavelengths[0] * (n - 1) * height_map.unsqueeze(0) + oof_phase_r
    pupil_r = aperture_index.unsqueeze(0) * torch.exp(1j * phase_r)
    # Pad to size_r
    pad_r = (size_r - kernel_size) // 2
    pupil_r = F.pad(pupil_r, (pad_r, pad_r, pad_r, pad_r), mode='constant', value=0)
    norm_r = size_r * size_r * torch.sum(aperture_index ** 2)
    fft_r = torch.fft.fft2(pupil_r)
    fft_r_shifted = torch.fft.fftshift(fft_r, dim=(-2, -1))
    psf_r = torch.abs(fft_r_shifted) ** 2 / norm_r
    # Crop back to kernel_size
    if pad_r > 0:
        psf_r = psf_r[:, pad_r:-pad_r, pad_r:-pad_r]

    # Green channel
    oof_phase_g = oof_phase[:, :, :, 1]
    phase_g = 2 * np.pi / wavelengths[1] * (n - 1) * height_map.unsqueeze(0) + oof_phase_g
    pupil_g = aperture_index.unsqueeze(0) * torch.exp(1j * phase_g)
    pad_g = (size_g - kernel_size) // 2
    pupil_g = F.pad(pupil_g, (pad_g, pad_g, pad_g, pad_g), mode='constant', value=0)
    norm_g = size_g * size_g * torch.sum(aperture_index ** 2)
    fft_g = torch.fft.fft2(pupil_g)
    fft_g_shifted = torch.fft.fftshift(fft_g, dim=(-2, -1))
    psf_g = torch.abs(fft_g_shifted) ** 2 / norm_g
    if pad_g > 0:
        psf_g = psf_g[:, pad_g:-pad_g, pad_g:-pad_g]

    # Blue channel (reference size)
    oof_phase_b = oof_phase[:, :, :, 2]
    phase_b = 2 * np.pi / wavelengths[2] * (n - 1) * height_map.unsqueeze(0) + oof_phase_b
    pupil_b = aperture_index.unsqueeze(0) * torch.exp(1j * phase_b)
    norm_b = kernel_size * kernel_size * torch.sum(aperture_index ** 2)
    fft_b = torch.fft.fft2(pupil_b)
    fft_b_shifted = torch.fft.fftshift(fft_b, dim=(-2, -1))
    psf_b = torch.abs(fft_b_shifted) ** 2 / norm_b

    # Stack RGB channels
    psfs = torch.stack([psf_r, psf_g, psf_b], dim=-1)  # [N_Phi, N_B, N_B, 3]

    return psfs


def blur_image(rgb_image, depth_probability, psfs):
    """
    Apply depth-dependent blur to an RGB image using PSFs.

    Args:
        rgb_image: Sharp RGB image [batch, 3, H, W]
        depth_probability: Depth probability map [batch, N_Phi, H, W]
        psfs: Point spread functions [N_Phi, N_B, N_B, 3]

    Returns:
        Blurred image [batch, 3, H', W'] where H' = H - N_B + 1
    """
    kernel_size = psfs.shape[1]
    crop = (kernel_size - 1) // 2
    num_depth = psfs.shape[0]
    batch_size = rgb_image.shape[0]

    # Prepare output
    H, W = rgb_image.shape[2], rgb_image.shape[3]
    H_out = H - 2 * crop
    W_out = W - 2 * crop

    blurred_channels = []

    for c in range(3):  # For each RGB channel
        sharp_c = rgb_image[:, c:c+1, :, :]  # [batch, 1, H, W]
        psfs_c = psfs[:, :, :, c]  # [N_Phi, N_B, N_B]

        # Reshape PSFs for group convolution
        # [N_Phi, 1, N_B, N_B]
        psfs_c = psfs_c.unsqueeze(1)

        # Apply convolution for each depth plane
        # We need to convolve the image with each PSF
        blur_all = []
        for d in range(num_depth):
            psf_d = psfs_c[d:d+1, ...]  # [1, 1, N_B, N_B]
            # Convolve entire batch with this PSF
            blur_d = F.conv2d(sharp_c, psf_d, padding='valid')  # [batch, 1, H', W']
            blur_all.append(blur_d)

        blur_all = torch.cat(blur_all, dim=1)  # [batch, N_Phi, H', W']

        # Weight by depth probability (cropped to valid region)
        dp_cropped = depth_probability[:, :, crop:-crop, crop:-crop]  # [batch, N_Phi, H', W']

        # Weighted sum over depth planes
        blur_c = torch.sum(blur_all * dp_cropped, dim=1, keepdim=True)  # [batch, 1, H', W']
        blurred_channels.append(blur_c)

    blur = torch.cat(blurred_channels, dim=1)  # [batch, 3, H', W']

    return blur


def add_gaussian_noise(images, std=0.01):
    """Add Gaussian noise to images."""
    noise = torch.randn_like(images) * std
    return torch.clamp(images + noise, min=0.0)


class OpticalSystem(nn.Module):
    """
    End-to-end optical system: blur -> noise -> depth estimation.

    This module combines the optical components with the digital
    depth estimation network.
    """

    def __init__(self, zernike_basis, num_modes, kernel_size, wavelengths,
                 aperture_index, size_r, size_g, size_b, depth_network,
                 add_noise=True, noise_std=0.01):
        """
        Args:
            zernike_basis: Zernike basis matrix [N_B*N_B, N_modes]
            num_modes: Number of Zernike modes to use
            kernel_size: Size of the height map [N_B, N_B]
            wavelengths: Array of wavelengths [3]
            aperture_index: Binary aperture mask [N_B, N_B]
            size_r, size_g, size_b: PSF sizes for RGB channels
            depth_network: Digital depth estimation network (nn.Module)
            add_noise: Whether to add Gaussian noise
            noise_std: Standard deviation of noise
        """
        super(OpticalSystem, self).__init__()

        self.kernel_size = kernel_size
        self.wavelengths = wavelengths
        self.aperture_index = aperture_index
        self.size_r = size_r
        self.size_g = size_g
        self.size_b = size_b
        self.add_noise = add_noise
        self.noise_std = noise_std

        # Convert zernike_basis to tensor
        self.register_buffer('zernike_basis', torch.from_numpy(zernike_basis).float())

        # Generate defocus phase (fixed, not learnable)
        from configs import config
        oof_phase = generate_defocus_phase(config.PHI_LIST, kernel_size, wavelengths)
        self.register_buffer('oof_phase', torch.from_numpy(oof_phase).float())

        # Learnable Zernike coefficients
        # Default constraint: clip to [-lambda/2, lambda/2]
        lambda_g = wavelengths[1]
        self.zernike_coeffs = nn.Parameter(
            torch.zeros(num_modes, 1, dtype=torch.float32)
        )
        self.coeff_min = -lambda_g / 2
        self.coeff_max = lambda_g / 2

        # Digital depth estimation network
        self.depth_network = depth_network

    def get_height_map(self):
        """Compute height map from Zernike coefficients."""
        # Clip coefficients
        coeffs_clipped = torch.clamp(
            self.zernike_coeffs,
            self.coeff_min,
            self.coeff_max
        )

        # Linear combination of Zernike modes
        g = torch.matmul(self.zernike_basis, coeffs_clipped).squeeze(-1)

        # Height map should be all positive (add lambda_g and apply ReLU)
        height_map = F.relu(g + self.wavelengths[1])

        kernel_size = int(np.sqrt(g.shape[0]))
        height_map = height_map.view(kernel_size, kernel_size)

        return height_map

    def get_psfs(self):
        """Generate PSFs from current height map."""
        height_map = self.get_height_map()
        psfs = generate_psfs(
            height_map,
            self.oof_phase.numpy(),  # Convert to numpy for compatibility
            self.wavelengths,
            self.aperture_index,
            self.size_r,
            self.size_g,
            self.size_b
        )
        return psfs

    def forward(self, rgb_image, depth_probability):
        """
        Forward pass through the optical system.

        Args:
            rgb_image: Input RGB image [batch, 3, H, W]
            depth_probability: Depth probability map [batch, N_Phi, H, W]

        Returns:
            blurred_image, estimated_depth
        """
        # Generate PSFs
        psfs = self.get_psfs()

        # Apply optical blur
        blur = blur_image(rgb_image, depth_probability, psfs)

        # Add sensor noise
        if self.add_noise and self.training:
            blur_noisy = add_gaussian_noise(blur, self.noise_std)
        else:
            blur_noisy = blur

        # Digital depth estimation
        estimated_depth = self.depth_network(blur_noisy)

        return blur_noisy, estimated_depth


def create_height_map_from_zernike(zernike_basis, num_modes, kernel_size,
                                   wavelengths, constraint=None):
    """
    Create a height map from Zernike polynomial coefficients.

    Args:
        zernike_basis: Zernike basis matrix [N_B*N_B, N_modes]
        num_modes: Number of Zernike modes to use
        kernel_size: Size of the height map [N_B, N_B]
        wavelengths: Array of wavelengths [3]
        constraint: Optional constraint (min, max) tuple for coefficients

    Returns:
        height_map tensor, zernike_coeffs parameter
    """
    # Default constraint: clip to [-lambda/2, lambda/2]
    if constraint is None:
        constraint = (-wavelengths[1] / 2, wavelengths[1] / 2)

    zernike_coeffs = nn.Parameter(
        torch.zeros(num_modes, 1, dtype=torch.float32)
    )

    # Linear combination of Zernike modes
    g = torch.matmul(
        torch.from_numpy(zernike_basis).float(),
        zernike_coeffs
    ).squeeze(-1)

    # Height map should be all positive
    height_map = F.relu(g.view(kernel_size, kernel_size) + wavelengths[1])

    return height_map, zernike_coeffs

def fit_zernike_to_height_map(height_map_target, zernike_basis):
    """
    使用最小二乘法将高度图映射为 Zernike 系数
    h = B * a  =>  a = (B^T B)^-1 B^T h
    """
    # 将 23x23 的高度图展平为 529 向量
    h = height_map_target.flatten() - 530e-9 # 减去偏置背景
    # 使用伪逆计算系数
    coeffs = np.linalg.lstsq(zernike_basis, h, rcond=None)[0]
    return torch.from_numpy(coeffs).float().unsqueeze(-1)
