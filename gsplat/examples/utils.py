import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps
import torch.nn as nn
import math

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

def spectral_kl_loss(
    X_gt: torch.Tensor,     # (H, W, B)
    X_pred: torch.Tensor,   # (H, W, B)
    normalization: str = "l1",
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    

    if X_gt.shape[0] == 1:
        X_gt = X_gt.squeeze(0)
        X_pred = X_pred.squeeze(0)


    # 🛑 Crucial: Force non-negativity for physical spectrum scaling
    if normalization in ["l1", "l2"]:
        X_gt = torch.clamp(X_gt, min=0.0)
        X_pred = torch.clamp(X_pred, min=0.0)

    # ------------------------------------------------------------------
    # Spectral normalization
    # ------------------------------------------------------------------
    if normalization == "softmax":
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

        # Renormalize to probability simplex
        D_gt = D_gt / (D_gt.sum(dim=-1, keepdim=True) + eps)
        D_pred = D_pred / (D_pred.sum(dim=-1, keepdim=True) + eps)
    else:
        raise ValueError(f"Unknown normalization '{normalization}'.")

    # ------------------------------------------------------------------
    # Safe KL divergence calculation
    # ------------------------------------------------------------------
    # Add eps to both to guarantee target > 0 and pred > 0
    log_D_pred = torch.log(D_pred.clamp(min=eps))

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
    ignore_dark_pixels:bool = True, # mask out near-zero pixels where SAM is undefined
    dark_threshold: float = 1e-4,   # pixels with gt norm below this are ignored
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

    if X_gt.shape[0] == 1:
        X_gt   = X_gt.squeeze(0)
        X_pred = X_pred.squeeze(0)


    # Dot product
    dot = (X_gt * X_pred).sum(dim=-1)  # (H, W)

    # Norms
    norm_gt = torch.norm(X_gt, dim=-1).clamp(min=eps)
    norm_pred = torch.norm(X_pred, dim=-1).clamp(min=eps)

    # Cosine similarity
    cos_theta = dot / (norm_gt * norm_pred).clamp(min=1e-8)

    # Numerical stability
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)

    # Spectral angle
    sam = torch.acos(cos_theta)  # (H, W)


    if ignore_dark_pixels:
        valid = norm_gt > dark_threshold   # (...,)  True where gt is non-dark
        if reduction == "mean":
            # Avoid division by zero if all pixels are dark (edge case)
            n_valid = valid.sum().clamp(min=1)
            return (sam * valid).sum() / n_valid
        elif reduction == "sum":
            return (sam * valid).sum()
        elif reduction == "none":
            return sam * valid
        else:
            raise ValueError(f"Invalid reduction '{reduction}'.")

    if reduction == "mean":
        return sam.mean()
    elif reduction == "sum":
        return sam.sum()
    elif reduction == "none":
        return sam
    else:
        raise ValueError(f"Invalid reduction '{reduction}'.")

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
      2. MLP maps γ(λ) → δSH_λ of shape (K,)  — one offset per SH coeff
      3. SH+_λ = SH_λ + δSH_λ   broadcast over N Gaussians

    Works for both:
      - RGB SH:  sh0 [N, 1, 3],   shN [N, K-1, 3]   → output [N, K, 3]
      - HSI SH:  sh0 [N, 1, B],   shN [N, K-1, B]   → output [N, K, B]

    The MLP learns a scalar offset per SH coefficient per wavelength.
    That scalar is then broadcast across all N Gaussians.

    Args:
        sh_degree    : degree of spherical harmonics (3 → 16 coeffs, 4 → 25, etc.)
        n_freq_bands : L in the positional encoding, number of sinusoidal frequency bands
        hidden_dim   : width of the MLP hidden layers
    """

    def __init__(
        self,
        sh_degree: int = 3,
        n_freq_bands: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.n_freq_bands = n_freq_bands

        # (degree+1)^2 SH coefficients
        self.n_sh_coeffs = (sh_degree + 1) ** 2   # e.g. 16 for degree 3

        pe_dim = 2 * n_freq_bands                  # sin + cos per band

        # MLP: γ(λ) → δSH  of shape (n_sh_coeffs,)
        # One scalar offset per SH coefficient, shared across all Gaussians
        # and broadcast across the channel (RGB or spectral band) dimension.
        self.mlp = nn.Sequential(
            nn.Linear(pe_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_sh_coeffs),
        )

        # Start near zero so the model trains from the base SH coefficients
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Frequency bands — not learned
        freqs = 2.0 ** torch.arange(n_freq_bands)  # [L]
        self.register_buffer("freqs", freqs)

    def positional_embedding(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal positional encoding for scalar wavelength values.

        Args:
            wavelengths: (C,) normalized wavelengths in [0, 1]
                         C = 3 for RGB, C = num_spectral_bands for HSI

        Returns:
            γ(λ): (C, 2*L)
        """
        # (C, 1) * (L,) → (C, L)
        angles = 2.0 * math.pi * wavelengths.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (C, 2L)

    def forward(
        self,
        sh0: torch.Tensor,           # (N, 1, C)   DC SH band
        shN: torch.Tensor,           # (N, K-1, C) higher SH bands
        wavelengths: torch.Tensor,   # (C,) normalized wavelengths in [0, 1]
    ) -> torch.Tensor:
        """
        Compute per-channel SH offsets and add them to the base SH coefficients.

        C is 3 for RGB, or num_spectral_bands for HSI.

        The MLP sees each wavelength independently and outputs K scalar offsets.
        Those offsets are broadcast over N Gaussians, then applied per channel.

        Args:
            sh0         : (N, 1, C)
            shN         : (N, K-1, C)
            wavelengths : (C,) one normalized wavelength per channel

        Returns:
            (N, K, C)  — same shape as torch.cat([sh0, shN], dim=1)
        """
        # sh_base: (N, K, C)
        sh_base = torch.cat([sh0, shN], dim=1)
        N, K, C = sh_base.shape

        assert K == self.n_sh_coeffs, (
            f"Expected {self.n_sh_coeffs} SH coeffs (sh_degree={self.sh_degree}), "
            f"got {K}. Check that sh_degree matches the splat tensors."
        )

        assert wavelengths.shape[0] == C, (
            f"wavelengths has {wavelengths.shape[0]} entries but sh tensors have "
            f"{C} channels. Pass one wavelength per channel."
        )

        # --- 1. Positional embedding: one encoding per channel ---
        gamma = self.positional_embedding(wavelengths)  # (C, 2L)

        # --- 2. MLP: one (K,) offset vector per channel ---
        delta_sh = self.mlp(gamma)                      # (C, K)

        # --- 3. Reshape for broadcasting over N Gaussians ---
        # delta_sh: (C, K) → (1, K, C)
        delta_sh = delta_sh.permute(1, 0).unsqueeze(0)  # (1, K, C)

        # sh_base: (N, K, C) + (1, K, C) → (N, K, C)
        sh_plus = sh_base + delta_sh

        return sh_plus  # (N, K, C)
        

 
# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
 
class NaiveHSIUnmixer(nn.Module):
    """
    Learnable per-pixel abundances + global endmembers.
 
    Parameters
    ----------
    height, width : int
        Spatial dimensions of the HSI image.
    num_bands : int
        Number of spectral bands.
    num_endmembers : int
        Number of pure material spectra to learn (hyperparameter).
    init_mode : str
        How to initialize endmembers: 'random' or 'kmeans'.
        'kmeans' picks random pixels as initial endmembers (VCA-lite).
    """
 
    def __init__(
        self,
        height: int,
        width: int,
        num_bands: int,
        num_endmembers: int = 5,
        init_mode: str = "random",
        gt_pixels: torch.Tensor = None,  # [H*W, B] for kmeans init
    ):
        super().__init__()
        self.H = height
        self.W = width
        self.B = num_bands
        self.M = num_endmembers
 
        # ---- Abundances: [H, W, M] (raw logits, softmax applied in forward) ----
        # Initialized near uniform — no prior on which material dominates
        self.abundances = nn.Parameter(
            torch.zeros(height, width, num_endmembers)
        )
 
        # ---- Endmembers: [M, B] (raw values, ReLU applied to enforce non-negativity) ----
        if init_mode == "kmeans" and gt_pixels is not None:
            # Pick M random pixels as starting endmembers (Vertex Component Analysis lite)
            idx = torch.randperm(gt_pixels.shape[0])[:num_endmembers]
            init_em = gt_pixels[idx].float()  # [M, B]
            self.endmembers = nn.Parameter(init_em)
        else:
            # Random init scaled to plausible reflectance range [0, 1]
            self.endmembers = nn.Parameter(
                torch.rand(num_endmembers, num_bands)
            )
 
    def forward(self) -> torch.Tensor:
        """
        Returns reconstructed HSI: [H, W, B]
 
        Steps:
          1. softmax over endmember dim → abundances sum to 1 per pixel (simplex constraint)
          2. relu on endmembers       → non-negative spectra
          3. einsum(abundances, endmembers) → [H, W, B]
        """
        # [H, W, M] — each pixel's mixture weights sum to 1
        abund = F.softmax(self.abundances, dim=-1)
 
        # [M, B] — non-negative pure spectra
        em = F.relu(self.endmembers)
 
        # [H, W, B] = [H, W, M] @ [M, B]
        hsi_pred = torch.einsum("hwm,mb->hwb", abund, em)
 
        return hsi_pred
 
    @torch.no_grad()
    def get_abundances(self) -> torch.Tensor:
        """Returns the softmax-normalized abundance maps [H, W, M]."""
        return F.softmax(self.abundances, dim=-1)
 
    @torch.no_grad()
    def get_endmembers(self) -> torch.Tensor:
        """Returns the relu-clipped endmember spectra [M, B]."""
        return F.relu(self.endmembers)
 


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


