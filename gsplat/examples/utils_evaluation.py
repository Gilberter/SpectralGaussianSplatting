import os
import numpy as np
import cv2
import argparse
import torch

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

class EndmemberTracker:

    def __init__(self):
        self.steps = []
        self.history = []

    @torch.no_grad()
    def update(self, endmembers, step):

        E = torch.sigmoid(
            endmembers.detach()
        ).float().cpu()

        self.steps.append(step)

        # store CPU tensor only
        self.history.append(E)

    def save(self, path, fps=2):
        """
        Saves:
            - compressed history (.npz)
            - animated gif (.gif)

        history shape: [T, M, B]
        """

        from pathlib import Path
        import numpy as np
        import matplotlib.pyplot as plt
        import imageio.v2 as imageio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------
        # SAVE RAW DATA
        # ---------------------------------------------------------

        history = torch.stack(self.history)  # [T, M, B]

        np.savez_compressed(
            str(path.with_suffix(".npz")),
            history=history.numpy(),
            steps=np.array(self.steps),
        )

        # ---------------------------------------------------------
        # CREATE GIF
        # ---------------------------------------------------------

        history_np = history.numpy()

        frames = []

        T, M, B = history_np.shape

        x = np.arange(B)

        for t in range(T):

            fig, ax = plt.subplots(figsize=(8, 5))

            for m in range(M):
                ax.plot(
                    x,
                    history_np[t, m],
                    linewidth=2,
                    label=f"EM {m}"
                )

            ax.set_title(f"Step {self.steps[t]}")
            ax.set_xlabel("Band")
            ax.set_ylabel("Reflectance")

            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)

            if M <= 15:
                ax.legend()

            fig.tight_layout()

            # -----------------------------------------------------
            # Convert matplotlib figure -> numpy image
            # -----------------------------------------------------

            fig.canvas.draw()

            frame = np.frombuffer(
                fig.canvas.buffer_rgba(),
                dtype=np.uint8
            )

            frame = frame.reshape(
                fig.canvas.get_width_height()[::-1] + (4,)
            )

            # remove alpha channel
            frame = frame[..., :3]

            frames.append(frame)

            plt.close(fig)

        # ---------------------------------------------------------
        # SAVE GIF
        # ---------------------------------------------------------

        gif_path = path.with_suffix(".gif")

        imageio.mimsave(
            gif_path,
            frames,
            fps=fps,
            loop=0,
        )

        print(f"Saved endmember history to: {path.with_suffix('.npz')}")
        print(f"Saved GIF to: {gif_path}")


class EndmemberHealthDiagnostics:
    """Evaluate endmember quality and detect dead/collapsed endmembers."""
    
    @staticmethod
    def compute_endmember_stats(endmembers: torch.Tensor, abundances: torch.Tensor) -> dict:
        """
        Args:
            endmembers: [M, num_bands] sigmoid(logits)
            abundances: [C, H, W, M] logits (pre-softmax)
        
        Returns:
            dict with diagnostic metrics
        """
        M, B = endmembers.shape
        a_pos = torch.softmax(abundances, dim=-1)  # [C, H, W, M]
        
        # 1. Mean usage per endmember
        mean_usage = a_pos.mean(dim=[0, 1, 2])  # [M]
        max_usage = a_pos.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]  # [M]
        
        # 2. Detect dead endmembers (< 0.5% average usage)
        dead_threshold = 0.005
        dead_mask = mean_usage < dead_threshold
        n_dead = dead_mask.sum().item()
        
        # 3. Endmember diversity: pairwise cosine similarity
        E_norm = torch.nn.functional.normalize(endmembers, dim=-1)
        sim_matrix = E_norm @ E_norm.T  # [M, M]
        
        # Off-diagonal similarities (should be low for diverse endmembers)
        off_diag_idx = ~torch.eye(M, dtype=torch.bool, device=sim_matrix.device)
        mean_similarity = sim_matrix[off_diag_idx].mean()
        max_similarity = sim_matrix[off_diag_idx].max()
        
        # 4. Endmember spread (how different are they in logit space)
        E_logits = torch.logit(endmembers.clamp(0.01, 0.99))
        E_spread = torch.std(E_logits, dim=0).mean()
        
        # 5. Spectral smoothness (penalizes noisy endmembers)
        spectral_smoothness = torch.mean((endmembers[:, 1:] - endmembers[:, :-1]) ** 2)
        
        return {
            "mean_usage": mean_usage.cpu().numpy(),           # [M] usage rates
            "max_usage": max_usage.cpu().numpy(),             # [M] peak usage
            "n_dead": n_dead,                                 # Count
            "dead_mask": dead_mask.cpu().numpy(),             # [M] bool
            "mean_similarity": mean_similarity.item(),        # Scalar: should be < 0.6
            "max_similarity": max_similarity.item(),          # Scalar: should be < 0.8
            "endmember_spread": E_spread.item(),              # Scalar: higher = more diverse
            "spectral_smoothness": spectral_smoothness.item(),# Scalar: lower = smoother
            "range": f"[{endmembers.min().item()},{endmembers.max().item()}]"
        }

    def print_diagnostics(self, stats: dict):
        """Pretty print endmember health."""
        print("\n" + "="*60)
        print("ENDMEMBER HEALTH DIAGNOSTICS")
        print("="*60)
        
        print(f"\n❌ DEAD ENDMEMBERS: {stats['n_dead']}")
        print(f"   Mean usage: {stats['mean_usage']}")
        print(f"   Dead mask: {stats['dead_mask']}")
        print(f"   Range Endmember Logits: {stats['range']}")
        
        print(f"\n📊 DIVERSITY METRICS:")
        print(f"   Mean similarity: {stats['mean_similarity']:.3f} (target: < 0.6)")
        print(f"   Max similarity:  {stats['max_similarity']:.3f} (target: < 0.8)")
        print(f"   Spread (logit):  {stats['endmember_spread']:.3f} (higher = better)")
        
        print(f"\n🔧 QUALITY METRICS:")
        print(f"   Spectral smoothness: {stats['spectral_smoothness']:.4f} (lower = smoother)")
        
        # Diagnosis
        print("\n🔍 DIAGNOSIS:")
        if stats['n_dead'] > 0:
            dead_idx = torch.where(stats['dead_mask'])[0]
            print(f"   ⚠️  PROBLEM: {stats['n_dead']} dead endmembers (idx: {dead_idx.tolist()})")
            print(f"       → Apply resurrection regularization")
            print(f"       → Check initialization")
        
        if stats['mean_similarity'] > 0.7:
            print(f"   ⚠️  PROBLEM: Endmembers too similar")
            print(f"       → Apply diversity regularization")
            print(f"       → Increase corpus entropy weight")
        
        print()


class AppearanceMaterialSeparationDiagnostics:
    """Check if endmembers hijack appearance (psi) or stay material-only."""
    
    @staticmethod
    def compute_material_purity(
        endmembers: torch.Tensor,
        psi_rendered: torch.Tensor,
        abundances: torch.Tensor,
        render_alphas: torch.Tensor
    ) -> dict:
        """
        Check if endmembers stay material-only or get contaminated by appearance.
        
        Args:
            endmembers: [M, B] — material reflectances
            psi_rendered: [C, H, W, M] — rendered view-dependent modulation
            abundances: [C, H, W, M] — rendered abundances (post-softmax)
        
        Returns:
            Metrics showing material purity
        """
        M = endmembers.shape[0]
        
        # 1. PSI STATISTICS (should be near 1.0 for material-only interpretation)
        psi_mean = psi_rendered.mean()
        psi_std = psi_rendered.std()
        psi_per_endmember = psi_rendered.mean(dim=[0, 1, 2])  # [M]
        psi_deviation = torch.abs(psi_rendered - 1.0).mean()
        
        # 2. EXTREME PSI VALUES (contamination indicator)
        n_psi_low = (psi_rendered < 0.5).sum() / psi_rendered.numel()  # Fraction < 0.5
        n_psi_high = (psi_rendered > 2.0).sum() / psi_rendered.numel() # Fraction > 2.0
        
        # 3. PSI CORRELATION WITH ENDMEMBER (hijacking test)
        # If psi correlates with abundance of specific endmembers,
        # those endmembers are "hijacking" appearance
        psi_global_mean = psi_rendered.mean(dim=-1, keepdim=True)  # [C, H, W, 1]
        correlations = []
        for m in range(M):
            # Correlation between abundance[m] and psi[m]
            a_m = abundances[..., m]  # [C, H, W]
            psi_m = psi_rendered[..., m]  # [C, H, W]
            
            # Flatten and compute correlation
            a_flat = a_m.reshape(-1)
            psi_flat = psi_m.reshape(-1)
            
            corr = torch.corrcoef(torch.stack([a_flat, psi_flat]))[0, 1]
            correlations.append(corr.item() if not torch.isnan(corr) else 0.0)
        
        correlations = np.array(correlations)
        max_correlation = np.max(np.abs(correlations))
        mean_correlation = np.mean(np.abs(correlations))
        
        # 4. OPAQUE vs. SPECULAR REGIONS
        alpha = render_alphas  # [C, H, W, 1]
        opaque_mask = alpha > 0.8
        
        if opaque_mask.sum() > 0:
            psi_opaque = psi_rendered[opaque_mask.squeeze(-1)]
            psi_opaque_mean = psi_opaque.mean().item()
            psi_opaque_std = psi_opaque.std().item()
        else:
            psi_opaque_mean = float('nan')
            psi_opaque_std = float('nan')
        
        return {
            "psi_global_mean": psi_mean.item(),              # Should be ~1.0
            "psi_global_std": psi_std.item(),                # Should be small (~0.1-0.3)
            "psi_per_endmember": psi_per_endmember.detach().cpu().numpy(),  # [M]
            "psi_deviation_from_1": psi_deviation.item(),    # Should be small
            "fraction_psi_low": n_psi_low.item(),            # Should be ~0
            "fraction_psi_high": n_psi_high.item(),          # Should be ~0
            "psi_em_correlations": correlations,             # [M] should be ~0
            "max_psi_em_correlation": max_correlation,       # Should be < 0.3
            "mean_psi_em_correlation": mean_correlation,     # Should be < 0.1
            "psi_opaque_mean": psi_opaque_mean,              # Should be ~1.0
            "psi_opaque_std": psi_opaque_std,                # Should be small
        }
    
    def print_diagnostics(self, stats: dict):
        """Pretty print material purity diagnostics."""
        print("\n" + "="*60)
        print("APPEARANCE-MATERIAL SEPARATION DIAGNOSTICS")
        print("="*60)
        
        print(f"\n📊 PSI (APPEARANCE) STATISTICS:")
        print(f"   Global mean: {stats['psi_global_mean']:.3f} (target: 1.0)")
        print(f"   Global std:  {stats['psi_global_std']:.3f} (target: < 0.2)")
        print(f"   Deviation from 1.0: {stats['psi_deviation_from_1']:.3f}")
        
        print(f"\n⚠️  PSI OUTLIERS:")
        print(f"   Fraction < 0.5: {stats['fraction_psi_low']:.4f} (target: ~0)")
        print(f"   Fraction > 2.0: {stats['fraction_psi_high']:.4f} (target: ~0)")
        
        print(f"\n🔗 ENDMEMBER-PSI HIJACKING TEST:")
        print(f"   Per-endmember correlations: {stats['psi_em_correlations']}")
        print(f"   Max correlation: {stats['max_psi_em_correlation']:.3f} (target: < 0.2)")
        print(f"   Mean correlation: {stats['mean_psi_em_correlation']:.3f} (target: < 0.1)")
        
        print(f"\n🏗️  OPAQUE SURFACES (MATERIAL-ONLY TEST):")
        print(f"   Opaque psi mean: {stats['psi_opaque_mean']:.3f} (target: ~1.0)")
        print(f"   Opaque psi std:  {stats['psi_opaque_std']:.3f} (target: small)")
        
        print("\n🔍 DIAGNOSIS:")
        if stats['psi_global_mean'] > 1.3 or stats['psi_global_mean'] < 0.7:
            print(f"   ⚠️  PROBLEM: Psi drifting away from 1.0")
            print(f"       → Endmembers hijacking appearance")
            print(f"       → Add psi regularization toward 1.0")
        
        if stats['max_psi_em_correlation'] > 0.3:
            print(f"   ⚠️  PROBLEM: Strong psi-endmember correlation")
            print(f"       → Specific endmembers hijacking appearance")
            print(f"       → Check endmember initialization and regularization")
        
        if not np.isnan(stats['psi_opaque_mean']) and abs(stats['psi_opaque_mean'] - 1.0) > 0.2:
            print(f"   ⚠️  PROBLEM: Opaque surfaces not material-only")
            print(f"       → Violates physical assumptions")
            print(f"       → Increase material-only constraint weight")
        
        print()

class AbundancePatternDiagnostics:
    """Check if abundance maps are physically plausible."""
    
    @staticmethod
    def compute_abundance_stats(abundances_raw: torch.Tensor) -> dict:
        """
        Args:
            abundances_raw: [C, H, W, M] logits (pre-softmax)
        
        Returns:
            Diagnostic metrics
        """
        # Normalize to probabilities
        a_pos = torch.softmax(abundances_raw, dim=-1)  # [C, H, W, M]
        
        # 1. SPARSITY: How many active materials per pixel (should be 1-2)
        # Compute entropy per pixel
        pixel_entropy = -(a_pos * torch.log(a_pos + 1e-8)).sum(dim=-1)  # [C, H, W]
        entropy_mean = pixel_entropy.mean()
        entropy_std = pixel_entropy.std()
        
        # Fraction of pixels with < 1.5 effective materials
        # (entropy < log(4) ≈ 1.39 means dominated by 1-2 materials)
        sparse_pixels = (pixel_entropy < np.log(4)).sum() / pixel_entropy.numel()
        
        # 2. SPATIAL COHERENCE: Similar abundances in neighboring pixels
        # Compute gradient magnitude per endmember
        a_diff_h = torch.abs(a_pos[:, 1:, :, :] - a_pos[:, :-1, :, :])  # Height gradient
        a_diff_w = torch.abs(a_pos[:, :, 1:, :] - a_pos[:, :, :-1, :])  # Width gradient
        
        spatial_smoothness = (a_diff_h.mean() + a_diff_w.mean()) / 2.0
        
        # 3. CORRELATION STRUCTURE: Do adjacent endmembers correlate?
        # (Good materials are spectrally distinct, so correlations should be low)
        a_flat = a_pos.reshape(-1, a_pos.shape[-1])  # [N_pixels, M]
        a_corr = torch.corrcoef(a_flat.T)  # [M, M]
        
        off_diag_idx = ~torch.eye(a_pos.shape[-1], dtype=torch.bool)
        mean_corr = torch.abs(a_corr[off_diag_idx]).mean()
        
        # 4. MATERIAL MIXING: Pixels where 3+ materials are significant (> 0.2)
        mixing_level = (a_pos > 0.2).sum(dim=-1)  # Count > 0.2
        heavy_mixing = (mixing_level > 3).sum() / a_pos.shape[0] / a_pos.shape[1] / a_pos.shape[2]
        
        return {
            "pixel_entropy_mean": entropy_mean.item(),       # Target: 0.3-0.8 (sparse)
            "pixel_entropy_std": entropy_std.item(),
            "fraction_sparse_pixels": sparse_pixels.item(),  # Target: > 0.8
            "spatial_smoothness": spatial_smoothness.item(), # Lower = more coherent
            "abundance_correlation": mean_corr.item(),       # Target: < 0.3
            "heavy_mixing_fraction": heavy_mixing.item(),    # Target: < 0.1
            "max_abundance": a_pos.max().item(),
            "min_abundance": a_pos.min().item(),
            "max_abundance_logits": abundances_raw.max().item(),
            "min_abundance_logits": abundances_raw.min().item(),
        }
    
    def print_diagnostics(self, stats: dict):
        """Pretty print abundance diagnostics."""
        print("\n" + "="*60)
        print("ABUNDANCE PATTERN DIAGNOSTICS")
        print("="*60)
        
        print(f"\n🌲 SPARSITY (materials per pixel):")
        print(f"   Entropy mean: {stats['pixel_entropy_mean']:.3f} (target: 0.3-0.8)")
        print(f"   Entropy std:  {stats['pixel_entropy_std']:.3f}")
        print(f"   Sparse pixels (≤2 materials): {stats['fraction_sparse_pixels']:.1%} (target: > 80%)")
        
        print(f"\n🔗 SPATIAL COHERENCE:")
        print(f"   Smoothness: {stats['spatial_smoothness']:.3f} (lower = more coherent)")
        print(f"   Abundance correlation: {stats['abundance_correlation']:.3f} (target: < 0.3)")
        
        print(f"\n🎨 MATERIAL MIXING:")
        print(f"   Heavy mixing (3+ active): {stats['heavy_mixing_fraction']:.1%} (target: < 10%)")
        print(f"   Abundance range: [{stats['max_abundance_logits']:.3f}, {stats['max_abundance_logits']:.3f}]")
        print(f"   Abundance range logits: [{stats['min_abundance']:.3f}, {stats['max_abundance']:.3f}]")

        print("\n🔍 DIAGNOSIS:")
        if stats['fraction_sparse_pixels'] < 0.5:
            print(f"   ⚠️  PROBLEM: Pixels too mixed (>2 materials)")
            print(f"       → Increase pixel sparsity loss weight")
        
        if stats['spatial_smoothness'] > 0.3:
            print(f"   ⚠️  PROBLEM: Abundant maps noisy (not coherent)")
            print(f"       → Add total variation regularization")
        
        print()


class SpectralReconstructionDiagnostics:
    """Check if spectrum is correctly reconstructed from materials."""
    
    @staticmethod
    def compute_reconstruction_errors(
        spectrum_gt: torch.Tensor,
        abundances: torch.Tensor,
        endmembers: torch.Tensor,
        psi_rendered: torch.Tensor,
        device
    ) -> dict:
        """
        Physically: spectrum = sum_m(a_m * E_m * psi_m) + noise
        
        Check if reconstruction explains the spectrum.
        """
        # Normalize abundances
        a_pos = torch.softmax(abundances, dim=-1).to(device)  # [C, H, W, M]
        E_sig = torch.sigmoid(endmembers).to(device)  # [M, B]
        
        # Reconstruct spectrum
        # a_pos: [C, H, W, M], E_sig: [M, B], psi: [C, H, W, M]
        # Result should be [C, H, W, B]
        
        # Reshape for matrix multiply
        C, H, W, M = a_pos.shape
        B = E_sig.shape[-1]
        
        a_flat = a_pos.reshape(-1, M)  # [N, M]
        psi_flat = psi_rendered.reshape(-1, M)  # [N, M]
        
        # Material reconstruction (a * E)
        recon_material = a_flat @ E_sig  # [N, B]
        
        # With appearance (a * E * psi)
        a_weighted = a_flat * psi_flat  # [N, M] element-wise
        recon_appearance = a_weighted @ E_sig  # [N, B]
        
        spec_flat = spectrum_gt.reshape(-1, B)  # [N, B]
        
        # Errors
        mae_material = torch.mean(torch.abs(recon_material - spec_flat))
        mae_appearance = torch.mean(torch.abs(recon_appearance - spec_flat))
        mse_material = torch.mean((recon_material - spec_flat) ** 2)
        mse_appearance = torch.mean((recon_appearance - spec_flat) ** 2)
        
        mse_material_clamp = torch.clamp(mse_material, min=1e-10)
        mse_appearance_clamp = torch.clamp(mse_appearance, min=1e-10)
        psnr_material = 10.0 * torch.log10(1.0 / mse_material_clamp)
        psnr_appearance = 10.0 * torch.log10(1.0 / mse_appearance_clamp)

        # RMSE per band (find problematic bands)
        rmse_per_band_material = torch.sqrt(torch.mean((recon_material - spec_flat) ** 2, dim=0))
        rmse_per_band_appearance = torch.sqrt(torch.mean((recon_appearance - spec_flat) ** 2, dim=0))
        
        # Spectral angle mapper (SAM)
        def compute_sam(x, y):
            x_norm = torch.nn.functional.normalize(x + 1e-8, dim=-1)
            y_norm = torch.nn.functional.normalize(y + 1e-8, dim=-1)
            cos_angle = torch.sum(x_norm * y_norm, dim=-1)
            cos_angle = torch.clamp(cos_angle, -1, 1)
            return torch.acos(cos_angle).mean() * 180 / np.pi
        
        sam_material = compute_sam(recon_material, spec_flat)
        sam_appearance = compute_sam(recon_appearance, spec_flat)
        
        return {
            "mae_material_only": mae_material.item(),        # Should be low
            "mae_with_appearance": mae_appearance.item(),    # Should be lower
            "improvement_mae": (mae_material - mae_appearance).item(),
            "mse_material_only": mse_material.item(),
            "mse_with_appearance": mse_appearance.item(),
            "psnr_material_only": psnr_material.item(),      # New metric
            "psnr_with_appearance": psnr_appearance.item(),
            "sam_material_only": sam_material.item(),        # degrees
            "sam_with_appearance": sam_appearance.item(),
            "rmse_per_band_material": rmse_per_band_material.cpu().numpy(),
            "rmse_per_band_appearance": rmse_per_band_appearance.cpu().numpy(),
            "range": f"[{E_sig.min().item()},{E_sig.max().item()}]"

        }
    
    def print_diagnostics(self, stats: dict):
        """Pretty print reconstruction diagnostics."""
        print("\n" + "="*60)
        print("SPECTRAL RECONSTRUCTION DIAGNOSTICS")
        print("="*60)
        
        print(f"   Range Endmember Sigmoid: {stats['range']}")

        print(f"\n📉 ERROR METRICS:")
        print(f"   MAE (material only): {stats['mae_material_only']:.4f}")
        print(f"   MAE (with appearance): {stats['mae_with_appearance']:.4f}")
        print(f"   Improvement: {stats['improvement_mae']:.4f}")
        
        print(f"\n   RMSE (material only): {np.sqrt(stats['mse_material_only']):.4f}")
        print(f"   RMSE (with appearance): {np.sqrt(stats['mse_with_appearance']):.4f}")
        
        print(f"\n   SAM (material only): {stats['sam_material_only']:.2f}°")
        print(f"   SAM (with appearance): {stats['sam_with_appearance']:.2f}°")
        
        print(f"\n📊 RMSE PER BAND:")
        rmse_material = stats['rmse_per_band_material']
        rmse_appear = stats['rmse_per_band_appearance']
        print(f"   Material only: {rmse_material}")
        print(f"   With appearance: {rmse_appear}")
        

        print(f"\n📺 PSNR METRICS (Higher is Better):")
        print(f"   PSNR (material only): {stats['psnr_material_only']:.2f} dB")
        print(f"   PSNR (with appearance): {stats['psnr_with_appearance']:.2f} dB")
        print(f"   Gain: {stats['psnr_with_appearance'] - stats['psnr_material_only']:.2f} dB")
        


        # Find problematic bands
        worst_bands_material = np.argsort(rmse_material)[-3:][::-1]
        worst_bands_appear = np.argsort(rmse_appear)[-3:][::-1]
        print(f"   Worst bands (material): {worst_bands_material}")
        print(f"   Worst bands (appearance): {worst_bands_appear}")
        
        print("\n🔍 DIAGNOSIS:")
        if stats['improvement_mae'] < 0.001:
            print(f"   ⚠️  PROBLEM: Appearance doesn't improve reconstruction")
            print(f"       → Psi might be converging to 1.0 (good!)")
            print(f"       → Or appearance is irrelevant for this scene")
        
        if stats['mae_material_only'] > 0.1:
            print(f"   ⚠️  PROBLEM: Poor material reconstruction")
            print(f"       → Check endmember quality and abundance accuracy")
            print(f"       → May need more materials (increase num_endmembers)")
        
        print()

def validation_step_with_diagnostics(self, step, cfg, renders, render_alphas, info, gt_spectrum, device):
    """Run validation with diagnostic metrics."""
    print("\n" + "="*60)
    print(f"Step {step}")
    print("="*60)
    # 1. Endmember Health
    em_diag = EndmemberHealthDiagnostics()
    em_stats = em_diag.compute_endmember_stats(
        self.endmembers,
        info["abundances"]
    )
    em_diag.print_diagnostics(em_stats)
    
    # 3. Appearance-Material Separation
    sep_diag = AppearanceMaterialSeparationDiagnostics()
    sep_stats = sep_diag.compute_material_purity(
        torch.sigmoid(self.endmembers),
        info["render_psi"],
        info["abundances"],
        render_alphas
    )
    sep_diag.print_diagnostics(sep_stats)
    
    # 4. Abundance Patterns
    abund_diag = AbundancePatternDiagnostics()
    abund_stats = abund_diag.compute_abundance_stats(
        info["abundances"]
    )
    abund_diag.print_diagnostics(abund_stats)
    
    # 5. Spectral Reconstruction
    recon_diag = SpectralReconstructionDiagnostics()
    recon_stats = recon_diag.compute_reconstruction_errors(
        gt_spectrum,
        info["abundances"],
        self.endmembers,
        info["render_psi"],
        device
    )
    recon_diag.print_diagnostics(recon_stats)


import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def save_unmixing_diagnostics(
    save_dir,
    abundances,      # [H,W,M]
    psi=None,        # [H,W,M]
    gt=None,         # [H,W,B]
    pred=None,       # [H,W,B]
    endmembers=None, # [M,B]
    step=0,
):
    """
    Save diagnostic visualizations for ELMM-SH.

    Parameters
    ----------
    abundances : [H,W,M]
    psi        : [H,W,M]
    gt         : [H,W,B]
    pred       : [H,W,B]
    endmembers : [M,B]
    """

    os.makedirs(save_dir, exist_ok=True)

    if torch.is_tensor(abundances):
        abundances = abundances.detach().cpu().numpy()
        abundances = abundances[0]

    if psi is not None and torch.is_tensor(psi):
        psi = psi.detach().cpu().numpy()
        psi = psi[0]

    if gt is not None and torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()
        gt = np.transpose(gt[0], (1, 2, 0))

    if pred is not None and torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
        pred = np.transpose(pred[0], (1, 2, 0))

    if endmembers is not None and torch.is_tensor(endmembers):
        endmembers = endmembers.detach().cpu().numpy()

    H, W, M = abundances.shape

    # ==========================================================
    # 1. Abundance maps
    # ==========================================================
    cols = min(4, M)
    rows = (M + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

    if rows == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    for k in range(M):
        im = axes[k].imshow(abundances[..., k], cmap="viridis")
        axes[k].set_title(f"Abundance {k}")
        axes[k].axis("off")
        plt.colorbar(im, ax=axes[k])

    for ax in axes[M:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"step_{step:06d}_abundances.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    # ==========================================================
    # 2. Psi maps
    # ==========================================================
    if psi is not None:

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

        if rows == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for k in range(M):
            im = axes[k].imshow(psi[..., k], cmap="coolwarm")
            axes[k].set_title(
                f"Psi {k}\nμ={psi[...,k].mean():.3f}"
            )
            axes[k].axis("off")
            plt.colorbar(im, ax=axes[k])

        for ax in axes[M:]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"step_{step:06d}_psi.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

    # ==========================================================
    # 3. Effective abundances a * psi
    # ==========================================================
    if psi is not None:

        apsi = abundances * psi

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

        if rows == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for k in range(M):
            im = axes[k].imshow(apsi[..., k], cmap="viridis")
            axes[k].set_title(f"a·psi {k}")
            axes[k].axis("off")
            plt.colorbar(im, ax=axes[k])

        for ax in axes[M:]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"step_{step:06d}_apsi.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

    # ==========================================================
    # 4. Dominant endmember
    # ==========================================================
    dominant = np.argmax(abundances, axis=-1)

    plt.figure(figsize=(8, 8))
    plt.imshow(dominant, cmap="tab20")
    plt.title("Dominant Endmember")
    plt.colorbar()
    plt.axis("off")

    plt.savefig(
        os.path.join(save_dir, f"step_{step:06d}_dominant_em.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    # ==========================================================
    # 5. Entropy map
    # ==========================================================
    entropy = -np.sum(
        abundances * np.log(abundances + 1e-8),
        axis=-1
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(entropy, cmap="magma")
    plt.title(
        f"Abundance Entropy\nmean={entropy.mean():.3f}"
    )
    plt.colorbar()
    plt.axis("off")

    plt.savefig(
        os.path.join(save_dir, f"step_{step:06d}_entropy.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    # ==========================================================
    # 6. Reconstruction error map
    # ==========================================================
    if gt is not None and pred is not None:

        rmse = np.sqrt(
            np.mean((gt - pred) ** 2, axis=-1)
        )

        plt.figure(figsize=(8, 8))
        plt.imshow(rmse, cmap="inferno")
        plt.title(
            f"Pixel RMSE\nmean={rmse.mean():.4f}"
        )
        plt.colorbar()
        plt.axis("off")

        plt.savefig(
            os.path.join(save_dir, f"step_{step:06d}_rmse.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

    # ==========================================================
    # 7. Endmember spectra
    # ==========================================================
    if endmembers is not None:

        plt.figure(figsize=(10, 5))

        for k in range(endmembers.shape[0]):
            plt.plot(
                endmembers[k],
                label=f"EM {k}"
            )

        plt.xlabel("Band")
        plt.ylabel("Reflectance")
        plt.title("Endmember Spectra")
        plt.legend()

        plt.savefig(
            os.path.join(save_dir, f"step_{step:06d}_endmembers.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()