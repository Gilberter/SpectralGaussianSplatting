import viser
from pathlib import Path
from typing import Literal
from typing import Tuple, Callable
from nerfview import Viewer, RenderTabState


class GsplatRenderTabState(RenderTabState):
    # non-controlable parameters
    total_gs_count: int = 0
    rendered_gs_count: int = 0

    # controlable parameters
    max_sh_degree: int = 5
    near_plane: float = 1e-2
    far_plane: float = 1e2
    radius_clip: float = 0.0
    eps2d: float = 0.3
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal[
        "rgb", "depth(accumulated)", "depth(expected)", "alpha"
    ] = "rgb"
    normalize_nearfar: bool = False
    inverse: bool = False
    colormap: Literal[
        "turbo", "viridis", "magma", "inferno", "cividis", "gray"
    ] = "turbo"
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Hyperspectral parameters

    hs_display_mode: Literal["rgb_converted", "single_band", "false_color"] = "single_band"
    hs_band_index: int = 0
    hs_num_bands: int = 21
    hs_band_wavelength_label: str = "Band 0"  # display only
    hs_wavelength_start: float = 450.0
    hs_wavelength_end: float = 650.0
    hs_false_r_band: int = 20
    hs_false_g_band: int = 10
    hs_false_b_band: int = 0


class GsplatViewer(Viewer):
    """
    Viewer for gsplat.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mode: Literal["rendering", "training"] = "rendering",
    ):
        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("gsplat viewer")

    def _init_rendering_tab(self):
        self.render_tab_state = GsplatRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        server = self.server
        with self._rendering_folder:
            with server.gui.add_folder("Gsplat"):
                total_gs_count_number = server.gui.add_number(
                    "Total",
                    initial_value=self.render_tab_state.total_gs_count,
                    disabled=True,
                    hint="Total number of splats in the scene.",
                )
                rendered_gs_count_number = server.gui.add_number(
                    "Rendered",
                    initial_value=self.render_tab_state.rendered_gs_count,
                    disabled=True,
                    hint="Number of splats rendered.",
                )

                max_sh_degree_number = server.gui.add_number(
                    "Max SH",
                    initial_value=self.render_tab_state.max_sh_degree,
                    min=0,
                    max=5,
                    step=1,
                    hint="Maximum SH degree used",
                )

                @max_sh_degree_number.on_update
                def _(_) -> None:
                    self.render_tab_state.max_sh_degree = int(
                        max_sh_degree_number.value
                    )
                    self.rerender(_)

                near_far_plane_vec2 = server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
                    step=1e-3,
                    hint="Near and far plane for rendering.",
                )

                @near_far_plane_vec2.on_update
                def _(_) -> None:
                    self.render_tab_state.near_plane = near_far_plane_vec2.value[0]
                    self.render_tab_state.far_plane = near_far_plane_vec2.value[1]
                    self.rerender(_)

                radius_clip_slider = server.gui.add_number(
                    "Radius Clip",
                    initial_value=self.render_tab_state.radius_clip,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    hint="2D radius clip for rendering.",
                )

                @radius_clip_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radius_clip = radius_clip_slider.value
                    self.rerender(_)

                eps2d_slider = server.gui.add_number(
                    "2D Epsilon",
                    initial_value=self.render_tab_state.eps2d,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    hint="Epsilon added to the egienvalues of projected 2D covariance matrices.",
                )

                @eps2d_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.eps2d = eps2d_slider.value
                    self.rerender(_)

                backgrounds_slider = server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color for rendering.",
                )

                @backgrounds_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.backgrounds = backgrounds_slider.value
                    self.rerender(_)

                render_mode_dropdown = server.gui.add_dropdown(
                    "Render Mode",
                    ("rgb", "depth(accumulated)", "depth(expected)", "alpha"),
                    initial_value=self.render_tab_state.render_mode,
                    hint="Render mode to use.",
                )

                @render_mode_dropdown.on_update
                def _(_) -> None:
                    if "depth" in render_mode_dropdown.value:
                        normalize_nearfar_checkbox.disabled = False
                        inverse_checkbox.disabled = False
                    else:
                        normalize_nearfar_checkbox.disabled = True
                        inverse_checkbox.disabled = True
                    self.render_tab_state.render_mode = render_mode_dropdown.value
                    self.rerender(_)

                normalize_nearfar_checkbox = server.gui.add_checkbox(
                    "Normalize Near/Far",
                    initial_value=self.render_tab_state.normalize_nearfar,
                    disabled=True,
                    hint="Normalize depth with near/far plane.",
                )

                @normalize_nearfar_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.normalize_nearfar = (
                        normalize_nearfar_checkbox.value
                    )
                    self.rerender(_)

                inverse_checkbox = server.gui.add_checkbox(
                    "Inverse",
                    initial_value=self.render_tab_state.inverse,
                    disabled=True,
                    hint="Inverse the depth.",
                )

                @inverse_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.inverse = inverse_checkbox.value
                    self.rerender(_)

                colormap_dropdown = server.gui.add_dropdown(
                    "Colormap",
                    ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                    initial_value=self.render_tab_state.colormap,
                    hint="Colormap used for rendering depth/alpha.",
                )

                @colormap_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.colormap = colormap_dropdown.value
                    self.rerender(_)

                rasterize_mode_dropdown = server.gui.add_dropdown(
                    "Anti-Aliasing",
                    ("classic", "antialiased"),
                    initial_value=self.render_tab_state.rasterize_mode,
                    hint="Whether to use classic or antialiased rasterization.",
                )

                @rasterize_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.rasterize_mode = rasterize_mode_dropdown.value
                    self.rerender(_)

                camera_model_dropdown = server.gui.add_dropdown(
                    "Camera",
                    ("pinhole", "ortho", "fisheye"),
                    initial_value=self.render_tab_state.camera_model,
                    hint="Camera model used for rendering.",
                )

                @camera_model_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.camera_model = camera_model_dropdown.value
                    self.rerender(_)

            with server.gui.add_folder("Hyperspectral"):

                hs_display_mode_dropdown = server.gui.add_dropdown(
                    "HS Display",
                    ("rgb_converted", "single_band", "false_color"),
                    initial_value=self.render_tab_state.hs_display_mode,
                    hint="How to display hyperspectral data.",
                )

                @hs_display_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.hs_display_mode = hs_display_mode_dropdown.value
                    # Show/hide relevant controls
                    hs_band_slider.disabled = hs_display_mode_dropdown.value != "single_band"
                    hs_false_r_slider.disabled = hs_display_mode_dropdown.value != "false_color"
                    hs_false_g_slider.disabled = hs_display_mode_dropdown.value != "false_color"
                    hs_false_b_slider.disabled = hs_display_mode_dropdown.value != "false_color"
                    self.rerender(_)

                hs_band_slider = server.gui.add_slider(
                    "Band Index",
                    min=0,
                    max=self.render_tab_state.hs_num_bands - 1,
                    step=1,
                    initial_value=self.render_tab_state.hs_band_index,
                    disabled=True,
                    hint="Which spectral band to display.",
                )

                # Add a text field to show the current wavelength
                hs_band_label = server.gui.add_text(
                    "Wavelength",
                    initial_value=self.render_tab_state.hs_band_wavelength_label,
                    disabled=True,
                )

                @hs_band_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.hs_band_index = int(hs_band_slider.value)
                    # Update wavelength label based on band index
                    start = self.render_tab_state.hs_wavelength_start
                    end   = self.render_tab_state.hs_wavelength_end
                    bands = self.render_tab_state.hs_num_bands
                    wl = start + (end - start) * hs_band_slider.value / max(bands - 1, 1)
                    hs_band_label.value = f"{wl:.1f} nm"
                    self.rerender(_)

                hs_false_r_slider = server.gui.add_slider(
                    "False R Band",
                    min=0, max=self.render_tab_state.hs_num_bands - 1,
                    step=1,
                    initial_value=self.render_tab_state.hs_false_r_band,
                    disabled=True,
                    hint="Band mapped to Red in false color.",
                )
                @hs_false_r_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.hs_false_r_band = int(hs_false_r_slider.value)
                    self.rerender(_)

                hs_false_g_slider = server.gui.add_slider(
                    "False G Band",
                    min=0, max=self.render_tab_state.hs_num_bands - 1,
                    step=1,
                    initial_value=self.render_tab_state.hs_false_g_band,
                    disabled=True,
                )
                @hs_false_g_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.hs_false_g_band = int(hs_false_g_slider.value)
                    self.rerender(_)

                hs_false_b_slider = server.gui.add_slider(
                    "False B Band",
                    min=0, max=self.render_tab_state.hs_num_bands - 1,
                    step=1,
                    initial_value=self.render_tab_state.hs_false_b_band,
                    disabled=True,
                )
                @hs_false_b_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.hs_false_b_band = int(hs_false_b_slider.value)
                    self.rerender(_)

        self._rendering_tab_handles.update(
            {
                "total_gs_count_number": total_gs_count_number,
                "rendered_gs_count_number": rendered_gs_count_number,
                "near_far_plane_vec2": near_far_plane_vec2,
                "radius_clip_slider": radius_clip_slider,
                "eps2d_slider": eps2d_slider,
                "backgrounds_slider": backgrounds_slider,
                "render_mode_dropdown": render_mode_dropdown,
                "normalize_nearfar_checkbox": normalize_nearfar_checkbox,
                "inverse_checkbox": inverse_checkbox,
                "colormap_dropdown": colormap_dropdown,
                "rasterize_mode_dropdown": rasterize_mode_dropdown,
                "camera_model_dropdown": camera_model_dropdown,
                "hs_display_mode_dropdown": hs_display_mode_dropdown,
                "hs_false_r_slider": hs_false_r_slider,
                "hs_false_g_slider": hs_false_g_slider,
                "hs_false_b_slider": hs_false_b_slider,
                
            }
        )
        # Add to handles
        self._rendering_tab_handles["hs_band_slider"] = hs_band_slider
        self._rendering_tab_handles["hs_band_label"]  = hs_band_label
        super()._populate_rendering_tab()

    def _after_render(self):
        self._rendering_tab_handles["total_gs_count_number"].value = self.render_tab_state.total_gs_count
        self._rendering_tab_handles["rendered_gs_count_number"].value = self.render_tab_state.rendered_gs_count

        # Sync band slider max with actual num_bands from the model
        slider = self._rendering_tab_handles.get("hs_band_slider")
        if slider is not None:
            slider.max = max(self.render_tab_state.hs_num_bands - 1, 1)
            # Clamp current value in case it's out of range
            if slider.value > slider.max:
                slider.value = slider.max
                self.render_tab_state.hs_band_index = int(slider.value)
