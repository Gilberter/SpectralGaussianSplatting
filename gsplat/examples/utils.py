import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps

def spectral_angle_mapper(x, y, eps=1e-8):
    """
    Spectral Angle Mapper (SAM) metric.
    Args:
        x, y: [..., C] — works for any leading dims (B,H,W,C) or (N,C) etc.
    Returns:
        angle in radians [...] — mean over all pixels as scalar if reduce=True
    """
    # F.normalize is safer + faster than manual norm division
    # it handles zero vectors gracefully
    x_norm = F.normalize(x, p=2, dim=-1)  # [..., C]
    y_norm = F.normalize(y, p=2, dim=-1)  # [..., C]

    # clamp is still needed for float precision at ±1
    cos = (x_norm * y_norm).sum(-1).clamp(-1.0 + eps, 1.0 - eps)

    return torch.acos(cos)  # [...] radians

import torch
import numpy as np

def compute_per_band_metrics(pred_image, gt_image, mask=None):
    """
    Compute PSNR and RMSE per band for hyperspectral images
    
    Args:
        gt_image: Ground truth image [1, B, H, W] 
        pred_image: Predicted image [1, B, H, W]
        mask: Optional [1, H, W] or [H, W] with valid pixels (True = valid)
    
    Returns:
        psnr_per_band: Tensor of PSNR values per band [B]
        rmse_per_band: Tensor of RMSE values per band [B]
        mse_per_band: Tensor of MSE values per band [B]
    """
    
    # Ensure images are the same shape
    assert gt_image.shape == pred_image.shape, f"Shape mismatch: {gt_image.shape} vs {pred_image.shape}"
    
    # Extract dimensions: [1, B, H, W]
    batch, B, H, W = gt_image.shape
    assert batch == 1, f"Expected batch size 1, got {batch}"
    
    # Handle mask
    if mask is not None:
        # Mask could be [1, H, W]
        
        # Reshape mask for broadcasting: [H, W] -> [1, 1, H, W]
        mask = mask.reshape(1, 1, H, W)
        
        # Apply mask
        gt_masked = gt_image * mask
        pred_masked = pred_image * mask
        n_valid_pixels = mask.sum().float()  # Total valid pixels
    else:
        gt_masked = gt_image
        pred_masked = pred_image
        n_valid_pixels = H * W
    
    # Compute per-band MSE - sum over H and W dimensions
    # Shape: [1, B, H, W] -> sum over dims 2,3 (H,W) -> [1, B]
    mse_per_band = ((gt_masked - pred_masked) ** 2).sum(dim=(2, 3)) / n_valid_pixels
    mse_per_band = mse_per_band.squeeze(0)  # Remove batch dimension -> [B]
    
    # Compute per-band RMSE
    rmse_per_band = torch.sqrt(mse_per_band + 1e-8)  # Add epsilon for numerical stability
    
    # Compute per-band PSNR
    # Find max value per band - max over H and W dimensions
    max_val_per_band = gt_masked.max(dim=2)[0].max(dim=2)[0]  # [1, B]
    max_val_per_band = max_val_per_band.squeeze(0)  # [B]
    
    # Handle case where max is 0 (avoid division by zero)
    max_val_per_band = torch.clamp(max_val_per_band, min=1e-8)
    
    # PSNR = 20 * log10(max_val / sqrt(MSE))
    psnr_per_band = 20 * torch.log10(max_val_per_band / torch.sqrt(mse_per_band + 1e-8))
    
    return psnr_per_band, rmse_per_band, mse_per_band

def sam_metric(x, y, reduce=True):
    """
    Returns mean SAM in degrees (standard reporting convention).
    Args:
        x, y: [B, H, W, C] or [N, C]
    """
    angle = spectral_angle_mapper(x, y)  # [B, H, W] or [N]
    angle_deg = torch.rad2deg(angle)
    return angle_deg.mean() if reduce else angle_deg

class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_dims = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_dims, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_dims, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img


import os
import numpy as np
import cv2
import argparse

# XYZ from chromaticity helper
def xyz_from_xy(x,y):
    return np.array((x,y,1-x-y))

ILUMINANT = {
    'D65': xyz_from_xy(0.3127, 0.3291),
    'E':  xyz_from_xy(1/3, 1/3),
}

COLOR_SPACE = {
    'sRGB': (xyz_from_xy(0.64, 0.33),
             xyz_from_xy(0.30, 0.60),
             xyz_from_xy(0.15, 0.06),
             ILUMINANT['D65']),

    'AdobeRGB': (xyz_from_xy(0.64, 0.33),
                 xyz_from_xy(0.21, 0.71),
                 xyz_from_xy(0.15, 0.06),
                 ILUMINANT['D65']),

    'AppleRGB': (xyz_from_xy(0.625, 0.34),
                 xyz_from_xy(0.28, 0.595),
                 xyz_from_xy(0.155, 0.07),
                 ILUMINANT['D65']),

    'UHDTV': (xyz_from_xy(0.708, 0.292),
              xyz_from_xy(0.170, 0.797),
              xyz_from_xy(0.131, 0.046),
              ILUMINANT['D65']),

    'CIERGB': (xyz_from_xy(0.7347, 0.2653),
               xyz_from_xy(0.2738, 0.7174),
               xyz_from_xy(0.1666, 0.0089),
               ILUMINANT['E']),
}


# CIE Analytical CMF (Color Matching Function)

def piecewise_gaussian(x,mu,tau1,tau2):
    result = np.zeros_like(x) 
    left_mask = x < mu
    right_mask = x >= mu
    result[left_mask] = np.exp((-0.5*tau1**2)*((x[left_mask]-mu)**2))
    result[right_mask] = np.exp((-0.5*tau2**2)*((x[right_mask]-mu)**2))
    return result


def x_bar_1931(wavelength):
    return (1.056 * piecewise_gaussian(wavelength, 599.8, 0.0264, 0.0323) +
            0.362 * piecewise_gaussian(wavelength, 442.0, 0.0624, 0.0374) -
            0.065 * piecewise_gaussian(wavelength, 501.1, 0.0490, 0.0382))

def y_bar_1931(wavelength):
    return (0.821 * piecewise_gaussian(wavelength, 568.8, 0.0213, 0.0247) +
            0.286 * piecewise_gaussian(wavelength, 530.9, 0.0613, 0.0322))

def z_bar_1931(wavelength):
    return (1.217 * piecewise_gaussian(wavelength, 437.0, 0.0845, 0.0278) +
            0.681 * piecewise_gaussian(wavelength, 459.0, 0.0385, 0.0725)) 

def compute_cmf_1931(wavelength):
    x_bar = x_bar_1931(wavelength) # shape (N,)
    y_bar = y_bar_1931(wavelength)
    z_bar = z_bar_1931(wavelength)
    return np.stack([x_bar, y_bar, z_bar], axis=1) # shape (N,3) where N is the number of wavelengths (bands)



class ColourSystem:

    def __init__(self,start=450,end=650,bands=21, cspace='sRGB'):

        wavelength = np.linspace(start,end,bands)

        self.cmf = compute_cmf_1931(wavelength)

        self.red, self.green, self.blue, self.white = COLOR_SPACE[cspace]

        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T
        self.MI = np.linalg.inv(self.M)

        # White scaling array
        self.wscale = self.MI.dot(self.white)

        # xyz -> rgb transformation matrix
        self.A = self.MI / self.wscale[:, np.newaxis]

        # Cache the transform matrix
        self._transform = self._compute_transform()

    def _compute_transform(self):
            """spectrum (N, bands) -> rgb (N, 3), no cross-band normalization."""
            XYZ = self.cmf           # (bands, 3)
            RGB = XYZ @ self.A.T     # (bands, 3)
            # Normalize so white spectrum maps to (1,1,1)
            white_rgb = np.ones((1, XYZ.shape[0])) @ RGB  # sum over bands
            RGB = RGB / white_rgb    # scale each channel
            return RGB               # (bands, 3)

    def spec_to_rgb(self, spec):
        """Convert a spectrum to an rgb value."""
        return spec @ self._transform  # (N, 3)
    


# Module-level cache
_colour_system_cache = {}

def spectrum_to_rgb(spectrum, start=450, end=650, bands=21, apply_gamma=True):
    import torch

    assert spectrum.shape[-1] == bands, \
        f"Expected {bands} bands, got {spectrum.shape[-1]}"

    is_torch = isinstance(spectrum, torch.Tensor)
    device = spectrum.device if is_torch else None
    dtype = spectrum.dtype if is_torch else None

    spec_np = spectrum.detach().cpu().numpy() if is_torch else np.asarray(spectrum, dtype=np.float64)

    original_shape = spec_np.shape
    spec_2d = spec_np.reshape(-1, bands)

    # Use cached ColourSystem
    key = (start, end, bands)
    if key not in _colour_system_cache:
        _colour_system_cache[key] = ColourSystem(start=start, end=end, bands=bands)
    cs = _colour_system_cache[key]

    rgb_2d = cs.spec_to_rgb(spec_2d)
    rgb_2d = np.clip(rgb_2d, 0, 1)

    if apply_gamma:
        rgb_2d = np.where(
            rgb_2d < 0.0031308,
            12.92 * rgb_2d,
            1.055 * (rgb_2d ** (1.0 / 2.4)) - 0.055
        )

    rgb = rgb_2d.reshape(original_shape[:-1] + (3,))

    if is_torch:
        rgb = torch.from_numpy(rgb).to(device=device, dtype=dtype)

    return rgb