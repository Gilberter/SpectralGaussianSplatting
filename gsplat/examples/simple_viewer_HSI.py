import argparse
import math
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser
from pathlib import Path
from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization

from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer import GsplatViewer, GsplatRenderTabState

from utils import spectrum_to_rgb



def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    if args.ckpt is None:
        (
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
        ) = load_test_data(device=device, scene_grid=args.scene_grid)

        assert world_size <= 2
        means = means[world_rank::world_size].contiguous()
        means.requires_grad = True
        quats = quats[world_rank::world_size].contiguous()
        quats.requires_grad = True
        scales = scales[world_rank::world_size].contiguous()
        scales.requires_grad = True
        opacities = opacities[world_rank::world_size].contiguous()
        opacities.requires_grad = True
        colors = colors[world_rank::world_size].contiguous()
        colors.requires_grad = True

        viewmats = viewmats[world_rank::world_size][:1].contiguous()
        Ks = Ks[world_rank::world_size][:1].contiguous()

        sh_degree = None
        C = len(viewmats)
        N = len(means)
        print("rank", world_rank, "Number of Gaussians:", N, "Number of Cameras:", C)

        # batched render
        for _ in tqdm.trange(1):
            render_colors, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, S, 3]
                viewmats,  # [C, 4, 4]
                Ks,  # [C, 3, 3]
                width,
                height,
                render_mode="RGB+D",
                packed=False,
                distributed=world_size > 1,
            )
        C = render_colors.shape[0]
        assert render_colors.shape == (C, height, width, 4)
        assert render_alphas.shape == (C, height, width, 1)
        render_colors.sum().backward()

        render_rgbs = render_colors[..., 0:3]
        render_depths = render_colors[..., 3:4]
        render_depths = render_depths / render_depths.max()

        # dump batch images
        os.makedirs(args.output_dir, exist_ok=True)
        canvas = (
            torch.cat(
                [
                    render_rgbs.reshape(C * height, width, 3),
                    render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
                    render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
                ],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        imageio.imsave(
            f"{args.output_dir}/render_rank{world_rank}.png",
            (canvas * 255).astype(np.uint8),
        )
    else:
        means, quats, scales, opacities, spectrum = [], [], [], [], []
        for ckpt_path in args.ckpt:
            ckpt = torch.load(ckpt_path, map_location=device)["splats"]
            means.append(ckpt["means"])
            quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
            scales.append(torch.exp(ckpt["scales"]))
            opacities.append(torch.sigmoid(ckpt["opacities"]))
            spectrum.append(torch.sigmoid(ckpt["spectrum"]))  # [N, B]
    means     = torch.cat(means, dim=0)
    quats     = torch.cat(quats, dim=0)
    scales    = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    spectrum  = torch.cat(spectrum, dim=0)   # [N, B]
    num_bands = spectrum.shape[-1]
    sh_degree = None
    print(f"Number of Gaussians: {len(means)}, Spectral bands: {num_bands}")
    viewer_ref = [None]
    # viewer_render_fn — replace the render + output block:
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)

        # Update band slider max at runtime
        render_tab_state.hs_num_bands = num_bands
        render_tab_state.hs_num_bands = num_bands
        if viewer_ref[0] is not None:
            handles = viewer_ref[0]._rendering_tab_handles
            if "hs_band_slider" in handles:
                handles["hs_band_slider"].max      = num_bands - 1
                handles["hs_band_slider"].disabled = (render_tab_state.hs_display_mode != "single_band")
                handles["hs_false_r_slider"].max      = num_bands - 1
                handles["hs_false_r_slider"].disabled = (render_tab_state.hs_display_mode != "false_color")
                handles["hs_false_g_slider"].max      = num_bands - 1
                handles["hs_false_g_slider"].disabled = (render_tab_state.hs_display_mode != "false_color")
                handles["hs_false_b_slider"].max      = num_bands - 1
                handles["hs_false_b_slider"].disabled = (render_tab_state.hs_display_mode != "false_color")
                # Update wavelength label
                wl = (render_tab_state.hs_wavelength_start +
                    (render_tab_state.hs_wavelength_end - render_tab_state.hs_wavelength_start) *
                    render_tab_state.hs_band_index / max(num_bands - 1, 1))
                if "hs_band_label" in handles:
                    handles["hs_band_label"].value = f"{wl:.1f} nm  (band {render_tab_state.hs_band_index}/{num_bands-1})"

        width  = render_tab_state.render_width  if render_tab_state.preview_render else render_tab_state.viewer_width
        height = render_tab_state.render_height if render_tab_state.preview_render else render_tab_state.viewer_height

        c2w     = torch.from_numpy(camera_state.c2w).float().to(device)
        K       = torch.from_numpy(camera_state.get_K((width, height))).float().to(device)
        viewmat = c2w.inverse()

        render_colors, render_alphas, info = rasterization(
            means,
            quats,
            scales,
            opacities,
            spectrum,           # [N, B] — no SH, direct spectral colors
            viewmat[None],
            K[None],
            width,
            height,
            sh_degree=None,     # disable SH, spectrum is already in color space
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            render_mode="RGB",  # returns [1, H, W, B]
            packed=False,
        )
        render_tab_state.total_gs_count    = len(means)
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        spectral = render_colors[0].clamp(0, 1)  # [H, W, B]

        mode = render_tab_state.hs_display_mode

        if mode == "rgb_converted":
            renders = spectrum_to_rgb(
                spectral,
                start=render_tab_state.hs_wavelength_start,
                end=render_tab_state.hs_wavelength_end,
                bands=num_bands,
                apply_gamma=True,
            ).clamp(0, 1).cpu().numpy()  # [H, W, 3]

        elif mode == "single_band":
            b = render_tab_state.hs_band_index
            band = spectral[..., b:b+1]           # [H, W, 1]
            renders = apply_float_colormap(
                band, render_tab_state.colormap
            ).cpu().numpy()                        # [H, W, 3]

        elif mode == "false_color":
            r = render_tab_state.hs_false_r_band
            g = render_tab_state.hs_false_g_band
            b = render_tab_state.hs_false_b_band
            false_color = torch.stack([
                spectral[..., r],
                spectral[..., g],
                spectral[..., b],
            ], dim=-1)                             # [H, W, 3]
            renders = false_color.cpu().numpy()

        return renders

    server = viser.ViserServer(port=args.port, verbose=False)
    viewer_ref[0] = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --ckpt results/garden/ckpts/ckpt_6999_rank0.pt \
        --output_dir results/garden/ \
        --port 8082
    
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --output_dir results/garden/ \
        --port 8082
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--with_ut", action="store_true", help="use uncentered transform"
    )
    parser.add_argument("--with_eval3d", action="store_true", help="use eval 3D")
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
