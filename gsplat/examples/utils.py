import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps

def _apply_colormap(x: np.ndarray, cmap: str = "magma") -> np.ndarray:
    """
    Apply a matplotlib-style colormap to a [H,W] float array in [0,1].
    Returns [H, W, 3] uint8. Works without a display (SSH safe).
    Uses only numpy — no plt.show() or GUI calls.
    """
    import matplotlib.cm as cm
    mapper = cm.get_cmap(cmap)
    rgba = mapper(x)                          # [H, W, 4] float64 in [0,1]
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb

    
def spectral_angle_mapper(x, y, eps=1e-8):
    """
    Spectral Angle Mapper (SAM) metric.
    Args:
        x, y works for any leading dims (B,H,W,BANDS)
    Returns:
        angle in radians [...] — mean over all pixels as scalar if reduce=True
    """

    # F.normalize is safer + faster than manual norm division
    # it handles zero vectors gracefully
    x_norm = F.normalize(x, p=2, dim=-1)  # [..., C]
    y_norm = F.normalize(y, p=2, dim=-1)  # [..., C]

    # clamp is still needed for float precision at ±1
    cos = (x_norm * y_norm).sum(-1).clamp(-1.0 + eps, 1.0 - eps)
    sam_map = torch.acos(cos) #radians
    return sam_map.mean()

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


def spectral_kl_loss(
    X_gt: torch.Tensor,     # (H, W, B)
    X_pred: torch.Tensor,   # (H, W, B)
    normalization: str = "l1",
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Spectral KL divergence loss.

    Computes:
        KL(D_gt || D_pred)

    after applying a spectral normalization strategy.

    Args:
        X_gt:
            Ground-truth hyperspectral image (H, W, B)

        X_pred:
            Predicted hyperspectral image (H, W, B)

        normalization:
            Spectral normalization method:
                - "softmax"
                - "l1"
                - "l2"

        eps:
            Numerical stability

        reduction:
            "mean" | "sum"

    Returns:
        Scalar spectral KL loss
    """

    # ------------------------------------------------------------------
    # Spectral normalization
    # ------------------------------------------------------------------

    if normalization == "softmax":

        # Probability distribution over wavelengths
        D_gt = F.softmax(X_gt, dim=-1)
        D_pred = F.softmax(X_pred, dim=-1)

    elif normalization == "l1":

        # Sum-to-one normalization
        D_gt = X_gt / (X_gt.sum(dim=-1, keepdim=True) + eps)
        D_pred = X_pred / (X_pred.sum(dim=-1, keepdim=True) + eps)

    elif normalization == "l2":

        # Unit spectral vector normalization
        D_gt = F.normalize(X_gt, p=2, dim=-1, eps=eps)
        D_pred = F.normalize(X_pred, p=2, dim=-1, eps=eps)

        # KL requires positive values
        D_gt = D_gt.abs()
        D_pred = D_pred.abs()

        # Renormalize to probability simplex
        D_gt = D_gt / (D_gt.sum(dim=-1, keepdim=True) + eps)
        D_pred = D_pred / (D_pred.sum(dim=-1, keepdim=True) + eps)

    else:
        raise ValueError(
            f"Unknown normalization '{normalization}'. "
            f"Choose from ['softmax', 'l1', 'l2']."
        )

    # ------------------------------------------------------------------
    # KL divergence
    # ------------------------------------------------------------------

    log_D_pred = torch.log(D_pred + eps)

    kl = F.kl_div(
        log_D_pred,
        D_gt,
        reduction="none",
        log_target=False,
    )  # (H, W, B)

    # Sum over spectral dimension
    kl = kl.sum(dim=-1)  # (H, W)

    # ------------------------------------------------------------------
    # Reduction
    # ------------------------------------------------------------------

    if reduction == "mean":
        return kl.mean()

    elif reduction == "sum":
        return kl.sum()

    else:
        raise ValueError(f"Invalid reduction: {reduction}")



def spectral_sam_loss(
    X_gt: torch.Tensor,     # (H, W, B)
    X_pred: torch.Tensor,   # (H, W, B)
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Spectral Angle Mapper (SAM) loss.

    Measures spectral shape similarity independently of magnitude.

    SAM(x, y) = arccos( <x,y> / (||x|| ||y||) )

    Args:
        X_gt:
            Ground-truth hyperspectral image (H, W, B)

        X_pred:
            Predicted hyperspectral image (H, W, B)

        eps:
            Numerical stability

        reduction:
            "mean" | "sum"

    Returns:
        Scalar SAM loss in radians
    """

    # Dot product
    dot = (X_gt * X_pred).sum(dim=-1)  # (H, W)

    # Norms
    norm_gt = torch.norm(X_gt, dim=-1).clamp(min=eps)
    norm_pred = torch.norm(X_pred, dim=-1).clamp(min=eps)

    # Cosine similarity
    cos_theta = dot / (norm_gt * norm_pred)

    # Numerical stability
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)

    # Spectral angle
    sam = torch.acos(cos_theta)  # (H, W)

    if reduction == "mean":
        return sam.mean()

    elif reduction == "sum":
        return sam.sum()

    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def spectral_loss(
    X_gt: torch.Tensor,    # (H, W, B) raw logits / raw spectral values GT
    X_pred: torch.Tensor,  # (H, W, B) raw logits / raw spectral values predicted
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    L_spectral = α * Σ_{h,w} KL(D_gt || D_pred) + β * Σ_{h,w} (1 - cos(D_gt, D_pred))

    D_λ(h,w) = softmax(X(h,w))  along the spectral (B) dimension.

    Args:
        X_gt   : (H, W, B) raw spectral values for ground truth
        X_pred : (H, W, B) raw spectral values for prediction
        alpha  : weight for KL divergence term
        beta   : weight for cosine similarity term
        eps    : numerical stability for cosine norm

    Returns:
        scalar loss tensor with gradients attached to X_pred
    """
    # ── 1. Normalize to probability distributions along spectral dim ──────────
    D_gt   = F.softmax(X_gt,   dim=-1)   # (H, W, B)  — no grad needed
    D_pred = F.softmax(X_pred, dim=-1)   # (H, W, B)  — gradients flow here

    # ── 2. KL divergence: KL(D_gt || D_pred) = Σ_b D_gt * log(D_gt / D_pred) ─
    # F.kl_div expects LOG-PROBABILITIES for the prediction
    log_D_pred = torch.log(D_pred + eps)              # (H, W, B)
    # reduction='none' → (H, W, B), then sum over spectral dim → (H, W)
    kl_per_pixel = F.kl_div(
        log_D_pred,
        D_gt,
        reduction="none",
        log_target=False,
    ).sum(dim=-1)                                      # (H, W)
    kl_loss = kl_per_pixel.sum()                       # scalar Σ_{h,w}

    # ── 3. Cosine similarity: (1 - cos(D_gt, D_pred)) per pixel ──────────────
    # Both vectors are already positive (softmax), but we keep the general form.
    dot      = (D_gt * D_pred).sum(dim=-1)             # (H, W)
    norm_gt  = D_gt.norm(dim=-1).clamp(min=eps)        # (H, W)
    norm_pred = D_pred.norm(dim=-1).clamp(min=eps)     # (H, W)
    cosine_sim      = dot / (norm_gt * norm_pred)      # (H, W)  ∈ [-1, 1]
    cosine_per_pixel = 1.0 - cosine_sim                # (H, W)
    cosine_loss = cosine_per_pixel.sum()               # scalar Σ_{h,w}

    # ── 4. Combined loss ──────────────────────────────────────────────────────
    return alpha * kl_loss + beta * cosine_loss

    
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


### Apperance Module V2


### Wavelength Dependet Apperance Module




class WavelengthEncoder(nn.Module):
    """
    Learns wavelength-specific offsets (δSH_λ) to add to base SH coefficients.

    For each wavelength λ (normalized to [0,1]):
      1. Positional embedding: γ(λ) = [sin(2^k π λ), cos(2^k π λ)] for k=0..L-1
      2. MLP maps γ(λ) → δSH_λ  of shape (sh_coeffs,)
      3. SH+_λ = SH_λ + δSH_λ   (broadcast over all N Gaussians)

    Args:
        sh_degree   : degree of spherical harmonics (3 → 16 coeffs, 4 → 25, etc.)
        n_freq_bands: L in the paper, number of sinusoidal frequency bands
        hidden_dim  : width of the MLP hidden layers
    """

    def __init__(self, sh_degree: int = 3, n_freq_bands: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.sh_degree = sh_degree
        self.n_freq_bands = n_freq_bands

        # (degree+1)^2 SH coefficients per wavelength
        self.n_sh_coeffs = (sh_degree + 1) ** 2  # 16 for degree 3

        # Positional embedding output dim: sin + cos per band
        pe_dim = 2 * n_freq_bands  # 16 for L=8

        # MLP: γ(λ) → δSH_λ of shape (n_sh_coeffs,)
        self.mlp = nn.Sequential(
            nn.Linear(pe_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_sh_coeffs),
        )

        # Initialize last layer to near-zero so offsets start small
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Register frequency bands as a buffer (not a learned param)
        freqs = 2.0 ** torch.arange(n_freq_bands)  # [L]
        self.register_buffer("freqs", freqs)

    def positional_embedding(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wavelengths: (B,) normalized wavelengths in [0, 1]
        Returns:
            γ(λ): (B, 2*L) sinusoidal embeddings
        """
        # (B, 1) * (L,) → (B, L)
        angles = 2.0 * math.pi * wavelengths.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, 2L)

    def forward(
        self,
        sh0: torch.Tensor,          # (N, 1, 3)  — DC SH band
        shN: torch.Tensor,          # (N, K-1, 3) — higher SH bands
        wavelengths: torch.Tensor,  # (B,) wavelength values, normalized [0,1]
    ) -> torch.Tensor:
        """
        Returns SH+_λ for every Gaussian × wavelength combination.

        Output shape: (N, B, K, 1)
            N = number of Gaussians
            B = number of wavelength bands
            K = number of SH coefficients = (sh_degree+1)^2
            last dim = 1  (monochromatic; hyperspectral replaces RGB)
        """
        N = sh0.shape[0]
        B = wavelengths.shape[0]
        K = self.n_sh_coeffs

        # --- 1. Build base SH: (N, K, 3) → collapse RGB to scalar per band ---
        # In hyperspectral 3DGS each wavelength IS a channel, so we use
        # a single scalar SH coefficient set.  We average RGB as a simple
        # bridge; in a full hyperspectral implementation sh would be (N, K, B).
        sh_base = torch.cat([sh0, shN], dim=1)  # (N, K, B)
        sh_scalar = sh_base.mean(dim=-1)         # (N, K)  — one value per coeff

        # --- 2. Positional embedding for each wavelength ---
        gamma = self.positional_embedding(wavelengths)  # (B, 2L)

        # --- 3. MLP → per-wavelength offset ---
        delta_sh = self.mlp(gamma)                      # (B, K)

        # --- 4. Add offset (broadcast N and B) ---
        # sh_scalar : (N, K) → (N, 1, K)
        # delta_sh  : (B, K) → (1, B, K)
        sh_plus = sh_scalar.unsqueeze(1) + delta_sh.unsqueeze(0)  # (N, B, K)

        return sh_plus  # (N, B, K)


# ─── Integration helper ──────────────────────────────────────────────────────

def get_wavelength_modulated_colors(
    splats: dict,
    wavelength_encoder: WavelengthEncoder,
    wavelengths: torch.Tensor,   # (B,) normalized wavelengths
) -> torch.Tensor:
    """
    Drop-in replacement for the standard:
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)  # (N, K, 3)

    Returns (N, B, K) — one SH set per Gaussian per wavelength band.
    """
    sh0 = splats["sh0"]   # (N, 1, 3)
    shN = splats["shN"]   # (N, K-1, 3)
    return wavelength_encoder(sh0, shN, wavelengths)  # (N, B, K)



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


