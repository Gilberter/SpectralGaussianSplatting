import torch
import numpy as np
import imageio
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
COLMAP_DIR  = "/disk/SYNTHETIC_NESPOF_DATA/hotdog_colmap/sparse/1"
HSI_DIR     = "/disk/SYNTHETIC_NESPOF_DATA/hotdog_npy"
RGB_DIR     = "/disk/SYNTHETIC_NESPOF_DATA/hotdog_rgb"
OUTPUT_DIR  = "/disk/SYNTHETIC_NESPOF_DATA/tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from datasets.colmap import HSIParser, HyperspectralDataset

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — COLMAP SPARSE MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 1: COLMAP sparse model")
print("="*60)

from read_write_model import read_cameras_binary, read_images_binary

cameras = read_cameras_binary(f"{COLMAP_DIR}/cameras.bin")
images  = read_images_binary(f"{COLMAP_DIR}/images.bin")

print(f"Num cameras : {len(cameras)}")
print(f"Num images  : {len(images)}")

# Print first camera intrinsics
cam = list(cameras.values())[0]
print(f"Camera model: {cam.model}  W={cam.width}  H={cam.height}  params={cam.params}")

# Print a few image entries
for img_id, img in list(images.items())[:3]:
    print(f"  image_id={img_id}  camera_id={img.camera_id}  name={img.name}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HSI PARSER
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 2: HSIParser — Hyperspectral (.npy)")
print("="*60)

parser_hsi = HSIParser(
    data_dir   = HSI_DIR,
    colmap_dir = COLMAP_DIR,
    factor     = 1,
    normalize  = True,
    test_every = 8,
)

print(f"Num images parsed      : {len(parser_hsi.image_names)}")
print(f"Scene scale            : {parser_hsi.scene_scale:.4f}")
print(f"Points shape           : {parser_hsi.points.shape}")
print(f"Points range X         : [{parser_hsi.points[:,0].min():.3f}, {parser_hsi.points[:,0].max():.3f}]")
print(f"Ks_dict keys (first 3) : {list(parser_hsi.Ks_dict.keys())[:3]}")
print(f"imsize_dict (first)    : {list(parser_hsi.imsize_dict.values())[0]}")
print(f"camtoworlds shape      : {parser_hsi.camtoworlds.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HSI DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 3: HyperspectralDataset — train split")
print("="*60)

dataset_hsi_train = HyperspectralDataset(parser=parser_hsi, split="train")
dataset_hsi_val   = HyperspectralDataset(parser=parser_hsi, split="val")

print(f"Train samples : {len(dataset_hsi_train)}")
print(f"Val   samples : {len(dataset_hsi_val)}")

# Inspect one sample
sample_hsi = dataset_hsi_train.__getitem__(0)
print(f"\nSample keys   : {list(sample_hsi.keys())}")

img_hsi  = sample_hsi["image"]    # [H, W, B]
mask_hsi = sample_hsi["mask"]     # [H, W] or [H, W, 1]
K_hsi    = sample_hsi["K"]
c2w_hsi  = sample_hsi["camtoworld"]

print(f"image shape   : {img_hsi.shape}   dtype={img_hsi.dtype}")
print(f"mask  shape   : {mask_hsi.shape}  dtype={mask_hsi.dtype}")
print(f"K     shape   : {K_hsi.shape}")
print(f"c2w   shape   : {c2w_hsi.shape}")
print(f"image min/max : {img_hsi.min():.4f} / {img_hsi.max():.4f}")
print(f"mask  foreground pixels : {mask_hsi.sum().item()}")
print(f"mask  background pixels : {(~mask_hsi.bool()).sum().item() if mask_hsi.dtype==torch.bool else 'N/A'}")
print(f"Spectral vector pixel (0,0)     : {img_hsi[0,0,:]}")
print(f"Spectral vector pixel (mid,mid) : {img_hsi[img_hsi.shape[0]//2, img_hsi.shape[1]//2, :]}")

# Band statistics
band_mean = img_hsi.mean(dim=(0,1))   # [B]
band_std  = img_hsi.std(dim=(0,1))    # [B]
print(f"\nPer-band mean (first 5): {band_mean[:5].tolist()}")
print(f"Per-band std  (first 5): {band_std[:5].tolist()}")

# Spatial std (variance across bands per pixel) — foreground detection
spatial_std = img_hsi.std(dim=-1)    # [H, W]
fg_mask_auto = spatial_std > 1e-4
print(f"\nAuto foreground mask (std>1e-4): {fg_mask_auto.sum().item()} px foreground")

# ── Save HSI visualizations ───────────────────────────────────────────────────
def apply_colormap_np(x: np.ndarray, cmap: str = "magma") -> np.ndarray:
    """[H,W] float [0,1] -> [H,W,3] uint8"""
    import matplotlib.cm as cm
    rgba = cm.get_cmap(cmap)(x)
    return (rgba[:, :, :3] * 255).astype(np.uint8)

# Save mid-band with magma colormap
mid_b = img_hsi.shape[-1] // 2
band_img = img_hsi[:, :, mid_b].numpy()
band_img_norm = (band_img - band_img.min()) / (band_img.max() - band_img.min() + 1e-8)
imageio.imwrite(
    f"{OUTPUT_DIR}/hsi_train0_band{mid_b}_magma.png",
    apply_colormap_np(band_img_norm, "magma"),
)
print(f"\nSaved: hsi_train0_band{mid_b}_magma.png")

# Save all 3 bands (short/mid/long) side by side
B = img_hsi.shape[-1]
bands_to_show = [0, B//2, B-1]
band_strip = []
for b in bands_to_show:
    bimg = img_hsi[:, :, b].numpy()
    bimg = (bimg - bimg.min()) / (bimg.max() - bimg.min() + 1e-8)
    band_strip.append(apply_colormap_np(bimg, "magma"))
imageio.imwrite(
    f"{OUTPUT_DIR}/hsi_train0_bands_short_mid_long.png",
    np.concatenate(band_strip, axis=1),
)
print(f"Saved: hsi_train0_bands_short_mid_long.png")

# Save mask
mask_vis = (mask_hsi.numpy().astype(np.float32))
if mask_vis.ndim == 3:
    mask_vis = mask_vis.squeeze(-1)
imageio.imwrite(
    f"{OUTPUT_DIR}/hsi_train0_mask.png",
    (mask_vis * 255).astype(np.uint8),
)
print(f"Saved: hsi_train0_mask.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RGB PARSER
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 4: HSIParser — RGB mode (rgb_dir=True)")
print("="*60)

parser_rgb = HSIParser(
    data_dir   = RGB_DIR,
    colmap_dir = COLMAP_DIR,
    factor     = 1,
    normalize  = True,
    test_every = 8,
    rgb_dir    = True,
)

print(f"Num images parsed : {len(parser_rgb.image_names)}")
print(f"Scene scale       : {parser_rgb.scene_scale:.4f}")
print(f"Points shape      : {parser_rgb.points.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — RGB DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 5: HyperspectralDataset — RGB train split")
print("="*60)

dataset_rgb_train = HyperspectralDataset(parser=parser_rgb, split="train")
dataset_rgb_val   = HyperspectralDataset(parser=parser_rgb, split="val")

print(f"Train samples : {len(dataset_rgb_train)}")
print(f"Val   samples : {len(dataset_rgb_val)}")

sample_rgb = dataset_rgb_train.__getitem__(0)
img_rgb    = sample_rgb["image"]    # [H, W, 3]
mask_rgb   = sample_rgb["mask"]

print(f"\nSample keys  : {list(sample_rgb.keys())}")
print(f"image shape  : {img_rgb.shape}   dtype={img_rgb.dtype}")
print(f"mask  shape  : {mask_rgb.shape}")
print(f"image min/max: {img_rgb.min():.4f} / {img_rgb.max():.4f}")

# Verify it is actually 3 channels
assert img_rgb.shape[-1] == 3, f"Expected 3 channels, got {img_rgb.shape[-1]}"
print("PASS: RGB image has 3 channels")

# Save RGB image
rgb_np = img_rgb.numpy()
rgb_np = np.clip(rgb_np, 0, 1)
imageio.imwrite(
    f"{OUTPUT_DIR}/rgb_train0_image.png",
    (rgb_np * 255).astype(np.uint8),
)
print(f"Saved: rgb_train0_image.png")

mask_rgb_vis = mask_rgb.numpy().astype(np.float32)
if mask_rgb_vis.ndim == 3:
    mask_rgb_vis = mask_rgb_vis.squeeze(-1)
imageio.imwrite(
    f"{OUTPUT_DIR}/rgb_train0_mask.png",
    (mask_rgb_vis * 255).astype(np.uint8),
)
print(f"Saved: rgb_train0_mask.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DATALOADER TEST (HSI + RGB)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 6: DataLoader batch test")
print("="*60)

for mode, dataset in [("HSI", dataset_hsi_train), ("RGB", dataset_rgb_train)]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  = 4,
        shuffle     = True,
        num_workers = 2,
        pin_memory  = True,
    )
    batch = next(iter(loader))

    img_b  = batch["image"]    # [4, H, W, C]
    mask_b = batch["mask"]     # [4, H, W] or [4, H, W, 1]
    K_b    = batch["K"]
    c2w_b  = batch["camtoworld"]

    print(f"\n[{mode}] batch image  : {img_b.shape}  min={img_b.min():.4f}  max={img_b.max():.4f}")
    print(f"[{mode}] batch mask   : {mask_b.shape}")
    print(f"[{mode}] batch K      : {K_b.shape}")
    print(f"[{mode}] batch c2w    : {c2w_b.shape}")

    # Verify shapes are consistent across batch
    assert img_b.shape[0] == 4,  "Batch size mismatch"
    assert K_b.shape   == (4, 3, 3), f"K shape wrong: {K_b.shape}"
    assert c2w_b.shape == (4, 4, 4), f"c2w shape wrong: {c2w_b.shape}"
    print(f"[{mode}] PASS: all batch shapes consistent")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CAMERA / GEOMETRY SANITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 7: Camera and geometry sanity checks")
print("="*60)

for mode, parser in [("HSI", parser_hsi), ("RGB", parser_rgb)]:
    c2w = parser.camtoworlds  # [N, 3, 4] or [N, 4, 4]
    print(f"\n[{mode}] camtoworlds shape : {c2w.shape}")
    print(f"[{mode}] translation range  : [{c2w[:, :3, 3].min():.3f}, {c2w[:, :3, 3].max():.3f}]")
    # Check rotations are valid (det ~ 1)
    R = c2w[:, :3, :3]
    dets = np.linalg.det(R)
    print(f"[{mode}] rotation det range : [{dets.min():.4f}, {dets.max():.4f}]  (should be ~1.0)")
    assert np.allclose(dets, 1.0, atol=1e-3), f"[{mode}] BAD rotation matrices! det={dets}"
    print(f"[{mode}] PASS: rotation matrices valid")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ALL TESTS PASSED")
print(f"Output images saved to: {OUTPUT_DIR}")
print("="*60)
print(f"  hsi_train0_band{mid_b}_magma.png       — single band with magma colormap")
print(f"  hsi_train0_bands_short_mid_long.png   — 3 bands side by side")
print(f"  hsi_train0_mask.png                   — foreground mask")
print(f"  rgb_train0_image.png                  — RGB image")
print(f"  rgb_train0_mask.png                   — RGB mask")
print("="*60)