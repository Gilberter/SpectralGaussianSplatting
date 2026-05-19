import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
#from .spec_to_rgb import spectrum_to_rgb

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
#from datasets.colmap import Dataset, Parser, HSIParser, HyperspectralDataset
from datasets.colmap import Dataset, Parser, HSIParser, HyperspectralDataset

from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, RootMeanSquaredErrorUsingSlidingWindow, SpectralAngleMapper
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed, sam_metric, compute_per_band_metrics, spectral_angle_mapper, _apply_colormap, WavelengthEncoder, get_wavelength_modulated_colors, spectral_kl_loss
from utils_color import spectrum_to_rgb
from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap

from gsplat.cuda._torch_impl import _spherical_harmonics_hs


@dataclass
class Config:

    # HSI CONFIGURATION
    use_hyperspectral: bool = True
    num_spectral_bands: int = 21 # nespof dataset
    hyperspectral_data_dir: str = "/" # directory with the .npy
    rgb_data_dir: str = "/"
    colmap_dir: str = "/"

    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"        

    # Path to the Mip-NeRF 360 dataset
    #data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "/"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Rendering Mode

    rendering_mode: Literal["rgb","spectral", "rgb_sh", "spectral_sh"] = "rgb"


    # Port for the viewer server
    #port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: Optional[int] = None # default nespof dataset
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 2000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20
    # SH FALSE DESACTIVATE TRUE ACTIVATE
    sh_hyperspectral: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    ## Enable apperance optimization using positional embeddings for wavelength dependent apperance
    wave_opt: bool = False
    wave_embed_dim: int = 64
    wave_opt_lr: float = 1e-3
    wave_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Enable KL divergence loss for spectral distribution matching
    kl_loss: bool = False
    # Weight for KL loss
    kl_lambda: float = 1e-4
    # Temperature for softmax in KL loss (controls sharpness)
    kl_temperature: float = 1

    sam_loss: bool = False

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    # just test True
    just_test: bool = False


    feature_dim: int = 32

    use_wandb: bool = True
    wandb_project: str = "gsplat"
    wandb_entity: str = "higilberter-universidad-industrial-de-santander"
    wandb_run_name: Optional[str] = None
    wandb_key = os.getenv('WANDB_API_KEY')
    wandb_steps: int = 1000
    wandb_path_challenge:str = ""

    # max refine steps
    max_refine_steps: int = 25000

    noise_lr:float = 5e4 # default 5e5

    min_opacity:float = 0.01 # Default 0.005

    max_gaussians:int= 1_000_000

    strategy_depth: Literal["None","progressive","cosine_warmup","exponential"] = "progressive"
    depth_loss_to_compute: List[Literal["SSIL", "MSS"]] = field(default_factory=lambda: ["SSIL"])

    ground_depth_loss: bool = False
    ground_depth_lambda: float = 2.3
    ground_seg_dir: str = ""
    ground_depth_start_step: int = 1000

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    #parser: Parser,
    parser: HSIParser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: Optional[int] = None,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    # Hyperspectral conf
    num_spectral_bands: int = 21,
    use_hyperspectral:bool = True,
    sh_hyperspectral:bool = False    

) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:


    # POINT CLOUD INITIALIZATION
    if init_type == "sfm": # from structure from motion COLMAP

        points = torch.from_numpy(parser.points).float()
        # rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()

    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        # rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    # Splits Gaussian across multiple GPUs
    points = points[world_rank::world_size]
    # rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4] random rotations
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,] opacity in logit space (-Inf,Inf)

    # Learnable Parameters
    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    #use_hyperspectral is obvious
    
    if use_hyperspectral:
        if sh_hyperspectral:
            #print("HYPERSPECTRAL + SH")
            bands = torch.rand((N, (sh_degree + 1) ** 2, num_spectral_bands))
            bands[:, 0, :] = rgb_to_sh(torch.full((N, num_spectral_bands), 0.5))
            params.append(("sh0", torch.nn.Parameter(bands[:, :1, :]), sh0_lr))
            params.append(("shN", torch.nn.Parameter(bands[:, 1:, :]), shN_lr))
        else:
            #print("HYPERSPECTRAL (no SH)")
            spectrum = torch.rand((N, num_spectral_bands))
            params.append(("spectrum", torch.nn.Parameter(spectrum), sh0_lr))

    else:  # RGB
        if sh_degree is not None and sh_degree >= 0:
            print("RGB + SH")
            colors = torch.rand((N, (sh_degree + 1) ** 2, 3))
            colors[:, 0, :] = rgb_to_sh(torch.full((N, 3), 0.5))
            params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
            params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
        else:
            print("RGB (no SH)")
            colors = torch.full((N, 3), 0.5)
            params.append(("sh0", torch.nn.Parameter(rgb_to_sh(colors)), sh0_lr))
    
    # parameter dictionary
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1


    BS = batch_size * world_size # batch size scaling
    # batch_size per gpu
    # world_size number of gpus
    optimizer_class = None # optimizer
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [
                {
                    "params": splats[name], 
                    "lr": lr * math.sqrt(BS), # learning rate with BS batch size
                    "name": name
                }
            ],
            eps=1e-15 / math.sqrt(BS), # extremely small epsilon
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True, # CUDA fused Adam Kernel
        )
        for name, _, lr in params
    }
    print(f"Splats {splats}")
    # splat and optimizers
    return splats, optimizers
    # splats holds all learnable tensors
    # splats = ParameterDict({
    # "means":      Parameter([N, 3]),
    # "scales":     Parameter([N, 3]),
    # "quats":      Parameter([N, 4]),
    # "opacities":  Parameter([N]),
    # if sh_hyperspectral
    # "spectrum":   Parameter([N, num_bands]) # Hyperspectral
    # else
    # "sh0":  Parameter([N,1,num_bands]),
    # "shN": Parameter([N,K-1,num_bands]),
    # })
    # optimizer each optimizer owns exactly one tensor has its own hyperparameters


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, 
        local_rank: int, # gpu index
        world_rank, # global process ID
        world_size: int,  # total gpus
        cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank) 

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")
        # loss curves
        # PNSR / SSIM
        # number of splats

        if self.cfg.use_hyperspectral:


            self.parser = HSIParser(
                data_dir=cfg.hyperspectral_data_dir,
                colmap_dir = cfg.colmap_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
            )
        else:
            print("TRUE RGB")
            self.parser = HSIParser(
                data_dir=cfg.hyperspectral_data_dir,
                colmap_dir = cfg.colmap_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
                rgb_dir = True
            )


        self.trainset = HyperspectralDataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        print("Dataset length:", len(self.trainset))
        self.valset = HyperspectralDataset(self.parser, split="val")

        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale

        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            sh_hyperspectral=cfg.sh_hyperspectral, # False 
            use_hyperspectral=cfg.use_hyperspectral # True
        )
        #print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        # Default Densification Strategy gradient-based densification
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")
#

        # Camera pose optimization
        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        # Pose Noise Injection

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        # Appereance Optimization Module
        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None

            # per image embeddings
            # neural color correction
            # view dependent appearance

            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)


            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)


        self.wave_optimizer = []
        if cfg.wave_opt:

            self.wave_module = WavelengthEncoder(
                sh_degree = cfg.sh_degree, 
                n_freq_bands = cfg.num_spectral_bands, 
                hidden_dim = cfg.wave_embed_dim
            ).to(self.device)
            
            self.wave_optimizer = [
                # Positional-frequency spectral modulation MLP
                torch.optim.Adam(
                    self.wave_module.mlp.parameters(),
                    lr=cfg.wave_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.wave_opt_reg,
                ),
            ]


        # BILATERAL GRID (Photometric Correction    )

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.sam = SpectralAngleMapper(reduction ='elementwise_mean').to(self.device)
        self.rmse = RootMeanSquaredErrorUsingSlidingWindow().to(self.device)

        #print("Losses")

        # Learned Perceptual Image Patch Similatiry
        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        # if not self.cfg.disable_viewer:
        #     self.server = viser.ViserServer(port=cfg.port, verbose=False)
        #     self.viewer = GsplatViewer(
        #         server=self.server,
        #         render_fn=self._viewer_render_fn,
        #         output_dir=Path(cfg.result_dir),
        #         mode="training",
        #     )

    #print("Rasterizer")
    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:

        

        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3] scales are stored in log-space and rendered in exp space
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,] opacity is stored in logit space and redered in sigmoid space

        # Hyperspectral

        if self.cfg.use_hyperspectral:
            # Hyperspectral
            if self.cfg.sh_hyperspectral:
                colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)
                sh_degree_for_render = kwargs.pop("sh_degree", self.cfg.sh_degree)

            else:
                colors = torch.sigmoid(self.splats["spectrum"])  # [.., N, num_bands]
                sh_degree_for_render = kwargs.pop("sh_degree", self.cfg.sh_degree)
                sh_degree_for_render = None  # Disable SH processing
        else:
            # RGB
            if self.cfg.sh_degree is not None:
                colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)
                sh_degree_for_render = kwargs.pop("sh_degree", self.cfg.sh_degree)
            else:
                colors = torch.cat([self.splats["sh0"]]) # (..,N,3)
                sh_degree_for_render = kwargs.pop("sh_degree", self.cfg.sh_degree)
                sh_degree_for_render = None  # Disable SH processing


                
        

        #print("sh_degree_for_render",sh_degree_for_render)
        #print("colors shape", colors.shape)
        #  Appearance Optimization Module
        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)

        if self.cfg.wave_opt:
            # have in mind that this is not generazible, we can do it generazible
            # hardcoded
            wavelengths = torch.linspace(
                450.0,
                650.0,
                cfg.num_spectral_bands,
                device=self.splats["sh0"].device,
            )

            # Normalize to [0,1]
            wavelengths = (wavelengths - 450.0) / (650.0 - 450.0)
            delta_sh = self.wave_module(sh0=self.splats['sh0'],shN=self.splats['shN'],wavelengths=wavelengths)
            #print(f"Shape Colors {colors.shape} and Shape Delth Sh {delta_sh.shape}")
            colors = colors + delta_sh
            print(f"Antes Sigmoid Color Range in Wave Opt Min:{colors.min()} Max: {colors.max()}")
            colors = torch.sigmoid(colors)
            print(f"Despues sigmoid Color Range in Wave Opt Min:{colors.min()} Max: {colors.max()}")

        # Rasterization Mode
        # Classic faster,harder edges
        # Antialiased smoother gradients better quality
        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        # Camera model, affects the projection, Jacobians and gradient flow
        if camera_model is None:
            camera_model = self.cfg.camera_model


        # Rasterization
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4] Convert world to camera space
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad, # only visible gaussian receive gradients
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            sh_hyperspectral = cfg.sh_hyperspectral,
            use_hyperspectral = cfg.use_hyperspectral,
            sh_degree = sh_degree_for_render,
            rendering_mode = self.cfg.rendering_mode,
            **kwargs,
        )
        # foreground only supervision
        # ignoring invalid pixels

        if masks is not None:
            render_colors[~masks] = 0
        
        # #print("function rasterize splats")
        # #print("render", render_colors.shape)
        # #print("rebnder_alphas", render_alphas.shape)
        return render_colors, render_alphas, info

    def train(self):

        # PHASE 0 GLOBAL SET UP AND REPRUDICIBILITY

        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)


        # PHASE 1 Traning horizon and LR Schedules

        max_steps = cfg.max_steps #step based not epoch based
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
            # LR decays smoothly
            # Final LR approx 1% of the initial
            # Geometry stabilized over time
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )
        
        # Phase 2 DATALOADER
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        # trainloader
        # {
        #     image: [B, H, W, B]
        #     K: [B, 3, 3]
        #     camtoworld: [B, 4, 4]
        #     image_id: [B]
        #     mask: optional
        #     points, depths: optional
        # }
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        # PHASE 3 MAIN TRAINING LOOP
        for step in pbar:
            # if not cfg.disable_viewer:
            #     while self.viewer.state == "paused":
            #         time.sleep(0.01)
            #     self.viewer.lock.acquire()
            #     tic = time.time()

            try:
                data = next(trainloader_iter) # load batch
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            # pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            pixels = data["image"].to(device) # [1, H, W, BANDS]

            num_train_rays_per_step = ( # only for performance reporting
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]

            # if cfg.depth_loss:
            #     points = data["points"].to(device)  # [1, M, 2]
            #     depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            # if cfg.pose_noise:
            #     camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            # if cfg.pose_opt:
            #     camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            # start with low frequency color
            # gradually add high frequency SH
            if cfg.sh_hyperspectral or cfg.sh_degree is not None:
                

                sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            else:
                sh_degree_to_use = None

            if step % cfg.sh_degree_interval and sh_degree_to_use <= cfg.sh_degree:
                print(f"Now Using SH Degree {sh_degree_to_use}/{cfg.sh_degree}")
            # forward
            # the rendering
            self.sh_degree_to_use = sh_degree_to_use
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use, # adding degrees each sh_degree_inverval
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                #render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                render_mode="RGB",
                masks=masks,
            )
            # ##print("renders", renders.shape)
            # #print("alphas", alphas.shape)
            # #print("info batch",info['batch_ids'])
            # #print("info ids", info['camera_ids'])
            # #print("gaussian ids", info['gaussian_ids'])
            # #print("info dic", info.keys())
            # #print("radii", info['radii'].shape)
            # RGB AND DEPTH
            # if renders.shape[-1] == 4:
            #     colors, depths = renders[..., 0:3], renders[..., 3:4]
            # else:
            #     colors, depths = renders, None
            colors, depths = renders, None
            
            # PHOTOMETRIC POST PROCESSING

            if cfg.use_bilateral_grid:

                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]

            # RANDOM BACKGROUND REGULARIZATION

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            # STRATEGY HOOK PRE BACKWARD
            # the densification strategy needsper gaussian image plane gradients to decide
            # the step_pre_backward make sure those gradients are retained
            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            #print("after pre_backward")
            
            
            # loss - Photometric Loss

            # colors and pixels [1, H, W, BANDS]
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )


            # Photometric loss (L1 + SSIM)
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda 

            # add the loss from the paper Diffusion Denoised HSI
            
            # KL Divergence Loss - for spectral distribution matching
            klloss = None
            if cfg.kl_loss:
                
                klloss = spectral_kl_loss(colors,pixels)

                if torch.isnan(klloss).any():
                    print("Warning: klloss is NaN!")
                        # Optional: raise ValueError("klloss is NaN")

                    assert klloss.item() is not None
            
            if cfg.kl_loss:
                loss += klloss * 1e-3
            
            samloss = None
            if cfg.sam_loss:
                samloss = spectral_angle_mapper(colors,pixels) 

            if cfg.sam_loss:
                loss += samloss * 1e-3

            # OPTIONAL DEPTH LOSS


            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            
            # regulatization optional grid artifacts
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0: #prevents too many opaque splats
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            
            if cfg.scale_reg > 0.0: # prevents exploding Gaussians
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            loss.backward() # backward pass

            desc = f"loss={loss.item():.3f}| ssim= {ssimloss} l1loss = {l1loss} "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.kl_loss and klloss is not None:
                desc += f"kl loss={klloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            if cfg.sam_loss:
                desc += f"sam loss={samloss}"
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )
            #print(f"world_rank {world_rank} and tb_every {cfg.tb_every}")
            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                assert world_rank == 0, print(f"tb_every {cfg.tb_every}, step {step} , step%tb_every {step % cfg.tb_every}")
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.kl_loss and klloss is not None:
                    self.writer.add_scalar("train/klloss", klloss.item(), step)
                if cfg.sam_loss and samloss is not None:
                    self.writer.add_scalar("train/samloss", samloss.item(), step) 
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)

                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                assert world_rank == 0, print(f"tb_every {cfg.tb_every}, step {step} , step%tb_every {step % cfg.tb_every}")
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": (time.time() - global_tic),
                    "num_GS": len(self.splats["means"]),
                    "Global Time": global_tic ,
                    "sh_degree": self.sh_degree_to_use
                }
                #print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                #print(" PLY")

                # PLY HYPERSPECTRAL

                if cfg.use_hyperspectral:
                    # For hyperspectral mode, skip PLY export as it's designed for RGB+SH
                    # Consider saving spectrum features separately if needed
                    #print(f"Skipping PLY export for hyperspectral mode at step {step}")
                    #print(spectrum.shape)
                    if cfg.sh_hyperspectral:
                        print("PLY SH Hyperspectral")
                    else:
                        #print("PLY NO SH HYPERSPECTRAL")
                        means = self.splats["means"]
                        scales = self.splats["scales"]
                        quats = self.splats["quats"]
                        opacities = self.splats["opacities"]
                        sh0 = self.splats["spectrum"].unsqueeze(1)  # [N, B] -> [N, 1, B]
                        export_splats(
                            means=means,
                            scales=scales,
                            quats=quats,
                            opacities=opacities,
                            sh0=sh0,
                            format="ply_hs",
                            save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                        )
                else:
                    # RGB mode with SH coefficients
                    # if self.cfg.app_opt:
                    #     # eval at origin to bake the appearance into the colors
                    #     rgb = self.app_module(
                    #         features=self.splats["features"],
                    #         embed_ids=None,
                    #         dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                    #         sh_degree=cfg.sh_degree_to_use,
                    #     )
                    #     rgb = rgb + self.splats["colors"]
                    #     rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    #     sh0 = rgb_to_sh(rgb)
                    #     shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                    # else:
                    #     sh0 = self.splats["sh0"]
                    #     shN = self.splats["shN"]

                    if self.cfg.sh_degree is not None:
                        sh0 = self.splats["sh0"]
                        shN = self.splats["shN"]
                    else: 
                        sh0 = self.splats["sh0"].unsqueeze(1)
                        shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                    means = self.splats["means"]
                    scales = self.splats["scales"]
                    quats = self.splats["quats"]
                    opacities = self.splats["opacities"]
                    export_splats(
                        means=means,
                        scales=scales,
                        quats=quats,
                        opacities=opacities,
                        sh0=sh0,
                        shN=shN,
                        format="ply",
                        save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                    )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.wave_optimizer:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            # Adaptive part of Gaussian Splatting
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                #print(f"step {step} evaluation")
                self.eval(step)
                #self.render_traj(step)
                

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                #print("compression disable")
                self.run_compression(step=step)

            # if not cfg.disable_viewer:
            #     self.viewer.lock.release()
            #     num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
            #     num_train_rays_per_sec = (
            #         num_train_rays_per_step * num_train_steps_per_sec
            #     )
            #     # Update the viewer state.
            #     self.viewer.render_tab_state.num_train_rays_per_sec = (
            #         num_train_rays_per_sec
            #     )
            #     # Update the scene.
            #     self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        #print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
        )
        ellipse_time = 0
        # Separate metric buckets

        metric_psnr = []
        metric_ssim = []
        metric_lpips = []

        os.makedirs(f"{cfg.result_dir}/rgb", exist_ok=True)
        os.makedirs(f"{cfg.result_dir}/hsi", exist_ok=True)
        for i, data in enumerate(valloader):
            #print("data shape",data.keys())
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            spectrum_gt = data["image"].to(device) 
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = spectrum_gt.shape[1:3]

            assert spectrum_gt.ndim == 4, (
                f"spectrum_gt must be [B,H,W,C], got {spectrum_gt.shape}"
            )

            assert spectrum_gt.shape[0] == 1, (
                f"Batch size must be 1, got {spectrum_gt.shape[0]}"
            )

            assert torch.isfinite(spectrum_gt).all(), (
                "spectrum_gt contains NaN or Inf"
            )

            assert spectrum_gt.shape[-1] == cfg.num_spectral_bands, (
                f"Expected {cfg.num_spectral_bands} spectral bands, "
                f"got {spectrum_gt.shape[-1]}"
            )

            tic = time.time()
            spectrum_pre, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, Bands] 
    
            ellipse_time += max(time.time() - tic, 1e-10)

           # mask = masks.detach().cpu().permute(1,2,0)

            if world_rank == 0:
                # FOR HSI USING THE FUNCTION SPECTRUM_TO_RGB

                assert spectrum_pre.ndim == 4, (
                    f"spectrum_pre must be [B,H,W,C], got {spectrum_pre.shape}"
                )

                assert spectrum_pre.shape == spectrum_gt.shape, (
                    f"Prediction shape mismatch: "
                    f"{spectrum_pre.shape} vs {spectrum_gt.shape}"
                )

                assert torch.isfinite(spectrum_pre).all(), (
                    "spectrum_pre contains NaN or Inf"
                )
                
                print(f"ANTES spectrum_pre range: [{spectrum_pre.min():.4f}, {spectrum_pre.max():.4f}]")
                print(f"ANTES rgb_pre range: [{rgb_pre.min():.4f}, {rgb_pre.max():.4f}]")


                if spectrum_gt.max() > 1:
                    spectrum_gt = spectrum_gt / spectrum_gt.max().clamp(min=1e-8)
                    spectrum_gt = torch.clamp(spectrum_gt,0.0,1.0)
                    #print(f"Pixels only positive values min {pixels.min()} max {pixels.max()}  Colors min {colors.min()} Colors max {colors.max()}")

                if spectrum_pre.max() > 1:
                    spectrum_pre = spectrum_pre / spectrum_pre.max().clamp(min=1e-8)
                    spectrum_pre = torch.clamp(spectrum_pre,0.0,1.0)

                # change the RGB and Hyperspectral setup
                spectrum_pre = torch.clamp(spectrum_pre,0,1).contiguous().permute(0,3,1,2)
                spectrum_gt = torch.clamp(spectrum_gt,0,1).contiguous().permute(0,3,1,2)

                print(f"DESPUES spectrum_pre range: [{spectrum_pre.min():.4f}, {spectrum_pre.max():.4f}]")
                print(f"DESPUES rgb_pre range: [{rgb_pre.min():.4f}, {rgb_pre.max():.4f}]")
                
                rgb_pre = spectrum_to_rgb(spectrum_pre, start=450, end=650, bands=cfg.num_spectral_bands).permute(0,3,1,2)
                rgb_gt = spectrum_to_rgb(spectrum_gt, start=450, end=650, bands=cfg.num_spectral_bands).permute(0,3,1,2)



                assert rgb_pre.ndim == 4
                assert rgb_gt.ndim == 4

                assert rgb_pre.shape[1] == 3, (
                    f"RGB prediction must have 3 channels, got {rgb_pre.shape}"
                )

                assert rgb_gt.shape[1] == 3, (
                    f"RGB GT must have 3 channels, got {rgb_gt.shape}"
                )

                assert torch.isfinite(rgb_pre).all(), (
                    "rgb_pre contains NaN or Inf"
                )

                assert torch.isfinite(rgb_gt).all(), (
                    "rgb_gt contains NaN or Inf"
                )
                

                spectrum_pre_cpu = spectrum_pre.detach().cpu()
                spectrum_gt_cpu = spectrum_gt.detach().cpu()
                rgb_pre_cpu = rgb_pre.detach().cpu()
                rgb_gt_cpu = rgb_gt.detach().cpu()
                

                # METRICS INSIDE THE GPU FOR THE TENSOR IN PYTORCH CURRENT GRAPH

                metric_psnr.append((i, self.psnr(spectrum_gt, spectrum_pre).item()))
                metric_ssim.append((i, self.ssim(spectrum_gt, spectrum_pre).item()))
                metric_lpips.append((i, self.lpips(rgb_gt, rgb_pre).item()))

                ## RGB SAVING NUMPY

                rgb_pred_np = (
                    rgb_pre_cpu[0]
                    .permute(1, 2, 0)
                    .numpy()
                )

                rgb_gt_np = (
                    rgb_gt_cpu[0]
                    .permute(1, 2, 0)
                    .numpy()
                )

                rgb_pred_np = (rgb_pred_np * 255).astype(np.uint8)
                rgb_gt_np = (rgb_gt_np * 255).astype(np.uint8)




                rgb_canvas = np.concatenate(
                    [
                        rgb_gt_np,
                        rgb_pred_np
                    ],
                    axis=1,
                )



                imageio.imwrite(
                    os.path.join(
                        f"{cfg.result_dir}/rgb",
                        f"{stage}_step{step:04d}_{i:04d}.png"
                    ),
                    rgb_canvas,
                )


                # Hyperspectral  SAVING NUMPY

                B = spectrum_pre.shape[1]
                
                band_indices = {
                    "band_short": 0,
                    "band_mid":   B // 2,
                    "band_long":  B - 1,
                }

                band_rows = []

                for band_name, b_idx in band_indices.items():
                    
                    # GT BAND
                    gt_band = (
                        spectrum_gt_cpu[0, b_idx]
                        .numpy()
                    )
                    # PRED BAND
                    pred_band = (
                        spectrum_pre_cpu[0, b_idx]
                        .numpy()
                    )
                   
                    gt_cm = _apply_colormap(
                        gt_band,
                        cmap="magma",
                    )

                    pred_cm = _apply_colormap(
                        pred_band,
                        cmap="magma",
                    )

                    row = np.concatenate(
                        [
                            gt_cm,
                            pred_cm
                        ],
                        axis=1,
                    )

                    band_rows.append(row)

                hsi_canvas = np.concatenate(
                    band_rows,
                    axis=0,
                )

                imageio.imwrite(
                    os.path.join(
                        f"{cfg.result_dir}/hsi",
                        f"{stage}_step{step:04d}_{i:04d}.png"
                    ),
                    hsi_canvas,
                )

            del spectrum_pre
            del spectrum_gt
            del rgb_pre
            del rgb_gt

        psnr_values = [item[1] for item in metric_psnr]
        ssim_values = [item[1] for item in metric_ssim]
        lpips_values = [item[1] for item in metric_lpips]

        metrics_dict = {
            "psnr_mean": float(np.mean(psnr_values)),
            "psnr_std": float(np.std(psnr_values)),

            "ssim_mean": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),

            "lpips_mean": float(np.mean(lpips_values)),
            "lpips_std": float(np.std(lpips_values)),

            "num_samples": len(metric_psnr),

            "per_image": {
                "psnr": metric_psnr,
                "ssim": metric_ssim,
                "lpips": metric_lpips,
            },

            "ellipse_time_seconds": float(ellipse_time),
        }

        json_path = os.path.join(
            self.render_dir,
            f"{stage}_step{step:04d}_metrics.json"
        )

        with open(json_path, "w") as f:
            json.dump(
                metrics_dict,
                f,
                indent=4,
            )
          

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        #print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device
        #print(f"CAMTOWORLS {self.parser.camtoworlds}")
        
        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if camtoworlds_all.shape[0] < 2:
            return
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        
        writer_hs = imageio.get_writer(f"{video_dir}/traj_{step}_hyperspectral.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=self.sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, BANDS + 1 (Depth)]
            spectral = torch.clamp(renders[..., :-1], 0.0, 1.0)  # [1, H, W, Bands + 1(Depth)]
            depths = renders[..., -1:]  # [1, H, W, 1]
            # Normalize depth for visualization
            depths_vis = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
            
            if spectral.shape[-1] > 3:
                rgb_from_hs = spectrum_to_rgb(
                    spectral.squeeze(0),  # [H, W, B]
                )  # [H, W, 3]
                rgb_from_hs = torch.clamp(rgb_from_hs, 0.0, 1.0).unsqueeze(0)  # [1, H, W, 3]
            else:
                #RGB
                rgb_from_hs = torch.clamp(spectral, 0.0, 1.0)

            
            canvas_list = [rgb_from_hs, depths_vis.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)

            

            mid_band = spectral.shape[-1] // 2
            band_vis = spectral[..., mid_band : mid_band + 1]  # [1, H, W, 1]
            band_vis = torch.clamp(band_vis, 0.0, 1.0).repeat(1, 1, 1, 3)  # [1, H, W, 3]
            hs_canvas = torch.cat([band_vis, depths_vis.repeat(1, 1, 1, 3)], dim=2)
            hs_canvas = hs_canvas.squeeze(0).cpu().numpy()
            hs_canvas = (hs_canvas * 255).astype(np.uint8)
            writer_hs.append_data(hs_canvas)

        writer.close()
        writer_hs.close()
        #print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        #print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):


    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

     # Disable viewer for hyperspectral mode (it expects RGB output)
    if cfg.use_hyperspectral and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print(f"HYPERSPECTRAL DATA SET {cfg.hyperspectral_data_dir}")
            print(f"HYPERSPECTRAL FOLDER RESULT {cfg.result_dir}")
            print("Viewer is disabled in hyperspectral mode (requires RGB visualization).")

    if cfg.use_hyperspectral:
        print("HYPERSPECTRAL TRUE")
    if cfg.sh_hyperspectral:
        print("SH HYPERSPECTRAL TRUE")
        print(f"SH DEGREE {cfg.sh_degree} ")
    if cfg.kl_loss:
        print("KL LOSS TRUE")
    
    if cfg.just_test:
        return 0

    runner = Runner(local_rank, world_rank, world_size, cfg)

    

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    # if not cfg.disable_viewer:
    #     runner.viewer.complete()
    #     #print("Viewer running... Ctrl+C to exit.")
    #     time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
