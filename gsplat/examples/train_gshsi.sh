#!/bin/bash
#SBATCH --job-name=gsplat_hyperspectral
#SBATCH --output=./logs/colmap_rendering_%j.out
#SBATCH --account=gs_hyperspectral
#SBATCH --error=./logs/error_rendering_%j.log
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --mem=15G
#SBATCH --time 6:00:00

echo "Running job: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate gsplat_hs

export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export CUDA_LAUNCH_BLOCKING=1

# export WANDB_MODE=offline
# export WANDB_DIR="./wandb_logs"

export WANDB_API_KEY=$(cat ~/.wandb_api_key 2>/dev/null || echo "")
export WANDB_MODE=online          # real-time streaming
export WANDB_DIR="$RESULT_DIR"    # keep wandb files in temp dir


# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
}

print_info() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

if [ $# -lt 2 ]; then
    print_error "Not enough arguments"
    echo "Usage: SCENE DENSIFICATION [FLAGS]"
    echo "  SCENE: Scene name (e.g., ajar or hotdog)"
    echo "  DENSIFICATION: 'classic' or 'mcmc'"
    echo ""
    echo "Flags:"
    echo "  --absgrad               Enable absolute gradients"
    echo "  --antialiased           Enable antialiased rasterization"
    echo "  --app-opt               Enable appearance optimization"
    echo "  --wave-opt              Enable Wavelength dependent apperance optimization"
    echo "  --post-processing {bilateral_grid,ppisp}      Enable bilateral grid post-processing"
    echo "  --data-factor N         Downsample factor (default: 1)"
    echo "  --max-steps N           Max steps (default: 40000)"
    echo "  --colmap-dir PATH       Path to COLMAP sparse folder"
    echo "  --data-dir PATH         Path to dataset"
    echo "  --packed                Enable Packed Mode"
    echo "  --kl_loss               Enable KL Loss "

    echo "  --depth_loss            Enable Packed Mode"
    echo "  --depth_ground            Enable Packed Mode"
    echo "  --depth_model {da3metric-large, da3mono-large}            Enable Packed Mode"
    echo "  --strategy_depth        Strategy "
    echo "  --difix                  Enable Difix3D distillation (uses simple_trainer_difix_wandb.py)"

    exit 1
fi

# -------------------------
# Arguments
# -------------------------


SCENE=$1
DENSIFICATION=$2
RENDER=$3
shift 3

# Validate densification type
if [[ ! "$DENSIFICATION" =~ ^(classic|mcmc)$ ]]; then
    print_error "DENSIFICATION must be 'classic' or 'mcmc', got: $DENSIFICATION"
    exit 1
fi

if [[ ! "$RENDER" =~ ^(spectral_sh|spectral|rgb_sh|rgb)$ ]]; then
    print_error "The rendering must be 'spectral_sh' or 'spectral' or 'rgb_sh' or 'rgb', got: $RENDER"
    exit 1
fi

# Feature flags
ABSGRAD=false
ANTIALIASED=false
APP_OPT=false
BILATERAL=false
PPISP=false
PACKED=false
DEPTH_LOSS=false
DEPTH_GROUND=false
DEPTH_MODEL="DA3MONO-LARGE"
STRATEGY_DEPTH="progressive"
SSIM_LAMBDA=0.2
DIFIX=false
CKPT_PATH=""
MAX_GAUSSIANS=1000000
TEST_EVERY=8

KL_LOSS=false
SAM_LOSS=false
WAVE_OPT=false
SH_HSI=false

NUM_SAVES=2

NOISE_LR=500000
## wandb

USE_WANDB=false # use wandb
WANDB_STEPS_EVAL=1000
WANDB_RUN_NAME=""

# Hyperparameters
DATA_FACTOR=1
MAX_STEPS=40000
MAX_REFINE_STEPS=25000
OPACITY_REG=0.01
SCALE_REG=0.01
APP_EMBED_DIM=16
FEATURE_DIM=32
GROW_GRAD2D=0.0008
# PARAMETERS FOR ABUDANCES AND ENDMEMBERS
AE_OPT=false
NUM_ENDMEMBERS=5
AE_HEIGHT=512
AE_WIDTH=512


# Bilateral grid parameters
BILATERAL_SHAPE_X=16
BILATERAL_SHAPE_Y=16
BILATERAL_SHAPE_W=8

# Paths (set defaults, can be overridden)

# PATHS FOR GSPLAT HYPERSPECTRAL
#
# /disk/SYNTHETIC_NESPOF_DATA/${SCENE}_extracted
# 
# 

RGB_DIR="/disk/SYNTHETIC_NESPOF_DATA/${SCENE}_rgb"
NPY_DIR="/disk/SYNTHETIC_NESPOF_DATA/${SCENE}_npy"

COLMAP_DIR="/disk/SYNTHETIC_NESPOF_DATA/${SCENE}_colmap/sparse/0"


# DATA_DIR="/disk/SN-NVS-2026-raw/${SCENE}"
#CHALLENGE_DIR="/disk/SN-NVS-2026-raw/${SCENE}-challenge" new Output
OUTPUT_DIR="/disk/all_results_HSI/${SCENE}-results"

# GROUND_DIR=/disk/SN-NVS-2026-raw/${SCENE}/mask/masks
# DEPTH_DIR=${DATA_DIR}/dae3/${DEPTH_MODEL}/depth_maps.npz



# Parse flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --absgrad)
            ABSGRAD=true
            shift
            ;;
        --antialiased)
            ANTIALIASED=true
            shift
            ;;
        --app-opt)
            APP_OPT=true
            shift
            ;;
        --depth-loss)
            DEPTH_LOSS=true
            shift
            ;;
        --depth-ground)
            DEPTH_GROUND=true
            shift
            ;;

        --ckpt)
            CKPT_PATH=$2
            shift 2
            ;;

        --num-saves)
            NUM_SAVES=$2
            shift 2
            ;;
            
        --kl-loss)
            KL_LOSS=true
            shift
            ;;

        --sam-loss)
            SAM_LOSS=true
            shift
            ;;

        --wave-opt)
            WAVE_OPT=true
            shift
            ;;

        --sh-hsi)
            SH_HSI=true
            shift
            ;;

        --post-processing)
            case $2 in
                bilateral*) # Matches bilateral or bilateral_grid
                    BILATERAL=true ;;
                ppisp)
                    PPISP=true ;;
                *)
                    echo "Warning: Unknown post-processing method '$2'" ;;
            esac
            shift 2
            ;;
        --data-factor)
            DATA_FACTOR=$2
            shift 2
            ;;
        --max-steps)
            MAX_STEPS=$2
            shift 2
            ;;
        --max-refine)
            MAX_REFINE_STEPS=$2
            shift 2
            ;;
        --opacity-reg)
            OPACITY_REG=$2
            shift 2
            ;;
        --scale-reg)
            SCALE_REG=$2
            shift 2
            ;;

        --test)
            TEST_EVERY=$2
            shift 2
            ;;
        --data-dir)
            DATA_DIR=$2
            shift 2
            ;;
        --wandb-steps)
            WANDB_STEPS_EVAL=$2
            shift 2
            ;;
        --colmap-dir)
            COLMAP_DIR=$2
            shift 2
            ;;
        --feature-dim)
            FEATURE_DIM=$2
            shift 2
            ;;
        --app-embed-dim)
            APP_EMBED_DIM=$2
            shift 2
            ;;

        --ae-opt)
            AE_OPT=true
            shift
            ;;

        --num-endmembers)
            NUM_ENDMEMBERS=$2
            shift 2
            ;;

        --ssim-lambda)
            SSIM_LAMBDA=$2
            shift 2
            ;;

        --strategy-depth)
            STRATEGY_DEPTH=$2
            shift 2
            ;;
        --max-gass)
            MAX_GAUSSIANS=$2
            shift 2
            ;;
        --grow-grad2d)
            GROW_GRAD2D=$2
            shift 2
            ;;

        --bilateral-shape)
            BILATERAL_SHAPE_X=$2
            BILATERAL_SHAPE_Y=$3
            BILATERAL_SHAPE_W=$4
            shift 4
            ;;
        *)
            print_warning "Unknown flag: $1"
            shift
            ;;
    esac
done

FEATURES=()

## ADD FEATURES TO ADD TO THE NAME OF THE FOLDER
if [ "$ABSGRAD" = true ]; then
    FEATURES+=("absgrad")
fi

if [ "$ANTIALIASED" = true ]; then
    FEATURES+=("antialiased")
fi

if [ "$WAVE_OPT" = true ]; then
    FEATURES+=("waveopt")
fi

if [ "$SAM_LOSS" = true ]; then
    FEATURES+=("samloss")
fi

if [ "$KL_LOSS" = true ]; then
    FEATURES+=("klloss")
fi

if [ "$APP_OPT" = true ]; then
    FEATURES+=("appopt")
fi

if [ "$PPISP" = true ]; then
    FEATURES+=("ppisp")
fi

if [ "$BILATERAL" = true ]; then
    FEATURES+=("bilateral")
fi

if [ "$AE_OPT" = true ]; then
    FEATURES+=("aeopt")
fi



if [ "$DEPTH_LOSS" = true ]; then
    SCENE_OUTPUT="results-depth-prior"
else
    SCENE_OUTPUT="${SCENE}-${RENDER}"
fi

# If no features, use baseline
if [ ${#FEATURES[@]} -eq 0 ]; then
    FEATURES=("baseline")
fi

FEATURE_NAME=$(IFS=_; echo "${FEATURES[*]}")

# Generate configuration suffix based on non-default parameters
CONFIG_SUFFIX=""
if [ "$MAX_STEPS" != "40000" ]; then
    CONFIG_SUFFIX="${CONFIG_SUFFIX}_s${MAX_STEPS}"
fi


OUTPUT_BASE_DIR="${OUTPUT_DIR}/${SCENE_OUTPUT}/${DENSIFICATION}/${FEATURE_NAME}${CONFIG_SUFFIX}"

# Handle multiple runs (run1, run2, etc.)
if [ -d "$OUTPUT_BASE_DIR" ]; then
    i=1
    while [ -d "${OUTPUT_BASE_DIR}_run${i}" ]; do
        ((i++))
    done
    OUTPUT_DIR="${OUTPUT_BASE_DIR}_run${i}"
    RUN_NUM=$i
else
    OUTPUT_DIR="$OUTPUT_BASE_DIR"
    RUN_NUM=0
fi


# Temporary training directory – lives in /tmp and is cleaned up after eval
RESULT_DIR="/tmp/gsplat_train_${SCENE}_${DENSIFICATION}_${FEATURE_NAME}${CONFIG_SUFFIX}_run${RUN_NUM}"

WANDB_RUN_NAME="${SCENE}_${DENSIFICATION}_${FEATURE_NAME}_run${RUN_NUM}"

WANDB_PATH_CHALLENGE="${OUTPUT_DIR}/sparse/0"


#mkdir -p "$RESULT_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p ./logs

LOG_FILE="$OUTPUT_DIR/experiment_log.txt"
CONFIG_FILE="$OUTPUT_DIR/config.txt"


print_header "COMPUTING SAVE INTERVALS FOR $NUM_SAVES STEPS"

SAVE_STEPS=""

for ((i=1; i<=NUM_SAVES; i++)); do

    PERCENT=$(( i * 100 / NUM_SAVES ))

    STEP=$(( MAX_STEPS * PERCENT / 100 - 1 ))

    [ $STEP -lt 0 ] && STEP=0

    SAVE_STEPS="${SAVE_STEPS}${STEP}"

    # Add comma except last element
    if [ $i -lt $NUM_SAVES ]; then
        SAVE_STEPS="${SAVE_STEPS} "
    fi

    echo "   $PERCENT% → Step $STEP"

done

SAVE_STEPS="${SAVE_STEPS}"

print_info "Final Save Steps string: $SAVE_STEPS"

{
    print_header "EXPERIMENT CONFIGURATION"

    echo ""
    echo "📋 EXPERIMENT IDENTITY"
    printf "%-30s %s\n" "Scene:" "$SCENE"
    printf "%-30s %s\n" "Densification:" "$DENSIFICATION"
    printf "%-30s %s\n" "Feature Set:" "$FEATURE_NAME"
    printf "%-30s %s\n" "Run Number:" "$RUN_NUM"
    printf "%-30s %s\n" "WANDB STEPS EVALUATION:" "$WANDB_STEPS_EVAL"


    echo ""
    echo "🔧 TRAINING PARAMETERS"
    printf "%-30s %s\n" "Max Steps:" "$MAX_STEPS"
    printf "%-30s %s\n" "Max Steps:" "$MAX_REFINE_STEPS"

    printf "%-30s %s\n" "Data Factor:" "$DATA_FACTOR"
    printf "%-30s %s\n" "Batch Size:" "1"
    printf "%-30s %s\n" "SH Degree:" "3"

    echo ""
    echo "🎯 FEATURES ENABLED"
    [ "$ABSGRAD" = true ]     && printf "%-30s %s\n" "Absgrad:"       "✓ ON" || printf "%-30s %s\n" "Absgrad:"       "✗ OFF"
    [ "$ANTIALIASED" = true ] && printf "%-30s %s\n" "Antialiased:"   "✓ ON" || printf "%-30s %s\n" "Antialiased:"   "✗ OFF"
    [ "$APP_OPT" = true ]     && printf "%-30s %s\n" "App Opt:"       "✓ ON" || printf "%-30s %s\n" "App Opt:"       "✗ OFF"
    [ "$BILATERAL" = true ]   && printf "%-30s %s\n" "Bilateral Grid:" "✓ ON" || printf "%-30s %s\n" "Bilateral Grid:" "✗ OFF"
    [ "$PPISP" = true ]       && printf "%-30s %s\n" "PPISP:"         "✓ ON" || printf "%-30s %s\n" "PPISP:"         "✗ OFF"

    [ "$KL_LOSS" = true ]       && printf "%-30s %s\n" "KL_LOSS:"         "✓ ON" || printf "%-30s %s\n" "KL_LOSS:"         "✗ OFF"
    [ "$SAM_LOSS" = true ]       && printf "%-30s %s\n" "SAM_LOSS:"         "✓ ON" || printf "%-30s %s\n" "SAM_LOSS:"         "✗ OFF"
    [ "$WAVE_OPT" = true ]       && printf "%-30s %s\n" "WAVE_OPT:"         "✓ ON" || printf "%-30s %s\n" "WAVE_OPT:"         "✗ OFF"
    [ "$NUM_SAVES" = true ]       && printf "%-30s %s\n" "NUM_SAVES:"         "✓ ON" || printf "%-30s %s\n" "NUM_SAVES:"         "✗ OFF"
    
    echo ""
    echo "📊 HYPERPARAMETERS"
    printf "%-30s %s\n" "Depth Strategy:" "$STRATEGY_DEPTH"
    printf "%-30s %s\n" "Opacity Reg:" "$OPACITY_REG"
    printf "%-30s %s\n" "Scale Reg:" "$SCALE_REG"
    printf "%-30s %s\n" "Feature Dim:" "$FEATURE_DIM"
    printf "%-30s %s\n" "App Embed Dim:" "$APP_EMBED_DIM"
    printf "%-30s %s\n" "SSIM_LAMBDA:" "$SSIM_LAMBDA"

    [ "$ABSGRAD" = true ]   && printf "%-30s %s\n" "Grow Grad2D:" "$GROW_GRAD2D"
    [ "$BILATERAL" = true ] && printf "%-30s %s\n" "Bilateral Grid Shape:" "($BILATERAL_SHAPE_X, $BILATERAL_SHAPE_Y, $BILATERAL_SHAPE_W)"
    [ "$PPISP" = true ]     && printf "%-30s %s\n" "PPISP:" "✓ ON" || printf "%-30s %s\n" "PPISP:" "✗ OFF"


    echo ""
    echo "📂 PATHS"
    printf "%-30s %s\n" "RGB_DIR Directory:" "$RGB_DIR"
    printf "%-30s %s\n" "RGB_DIR Directory:" "$NPY_DIR"
    printf "%-30s %s\n" "COLMAP Directory:" "$COLMAP_DIR"
    printf "%-30s %s\n" "Temp Train Directory:" "$RESULT_DIR"
    printf "%-30s %s\n" "Output Directory:" "$OUTPUT_DIR"

    echo ""
    echo "⏱️ TIMING"
    printf "%-30s %s\n" "Start Time:" "$(date '+%Y-%m-%d %H:%M:%S')"

} | tee "$LOG_FILE"

cp "$LOG_FILE" "$CONFIG_FILE"

print_info "Configuration logged to: $LOG_FILE"


FLAGS=""
FLAGS="$FLAGS --disable_viewer"
FLAGS="$FLAGS --data_factor $DATA_FACTOR"
FLAGS="$FLAGS --opacity_reg $OPACITY_REG"
FLAGS="$FLAGS --scale_reg $SCALE_REG"
FLAGS="$FLAGS --feature_dim $FEATURE_DIM"
FLAGS="$FLAGS --max_steps $MAX_STEPS"
#FLAGS="$FLAGS --wandb_run_name $WANDB_RUN_NAME"
#FLAGS="$FLAGS --wandb_steps $WANDB_STEPS_EVAL"
FLAGS="$FLAGS --ssim_lambda $SSIM_LAMBDA"
#FLAGS="$FLAGS --max_refine_steps $MAX_REFINE_STEPS"
#FLAGS="$FLAGS --wandb_path_challenge $WANDB_PATH_CHALLENGE"
#FLAGS="$FLAGS --max_gaussians $MAX_GAUSSIANS"
FLAGS="$FLAGS --test_every $TEST_EVERY"



FLAGS="$FLAGS --hyperspectral_data_dir $NPY_DIR"
FLAGS="$FLAGS --rgb_data_dir $RGB_DIR"

FLAGS="$FLAGS --rendering_mode $RENDER"


# Add conditional flags
if [ "$ABSGRAD" = true ]; then
 #   FLAGS="$FLAGS --absgrad"
    # --strategy.absgrad and --strategy.grow_grad2d only exist on DefaultStrategy,
    # not MCMCStrategy. Only pass them for the 'classic' (default) densification.
    if [ "$DENSIFICATION" = "classic" ]; then
        FLAGS="$FLAGS --strategy.absgrad --strategy.grow_grad2d $GROW_GRAD2D"
    fi
fi

if [ "$ANTIALIASED" = true ]; then
    FLAGS="$FLAGS --antialiased"
fi

if [ "$APP_OPT" = true ]; then
    FLAGS="$FLAGS --app_opt --app_embed_dim $APP_EMBED_DIM"
fi

if [ "$BILATERAL" = true ]; then
    FLAGS="$FLAGS --post_processing bilateral_grid --bilateral_grid_fused"
    FLAGS="$FLAGS --bilateral_grid_shape $BILATERAL_SHAPE_X $BILATERAL_SHAPE_Y $BILATERAL_SHAPE_W"
fi

if [ "$DEPTH_LOSS" = true ]; then
    FLAGS="$FLAGS --depth_loss --strategy_depth $STRATEGY_DEPTH"
fi

if [ "$PPISP" = true ]; then
    FLAGS="$FLAGS --post_processing ppisp"
fi

if [ -n "$CKPT_PATH" ]; then
    FLAGS="$FLAGS --ckpt $CKPT_PATH"
fi


if [ "$KL_LOSS" = true ]; then
    FLAGS="$FLAGS --kl_loss"
fi

if [ "$SAM_LOSS" = true ]; then
    FLAGS="$FLAGS --sam_loss"
fi

if [ "$WAVE_OPT" = true ]; then
    FLAGS="$FLAGS --wave_opt"
fi

if [ "$RENDER" = "spectral_sh" ]; then
    FLAGS="$FLAGS --sh_hyperspectral"
fi

if [ "$AE_OPT" = true ]; then
    FLAGS="$FLAGS --ae_opt"
fi
# [ "$GROUND_LOSS"    -eq 1 ] && FLAGS="$FLAGS --ground_depth_loss --ground_seg_dir $GROUND_DIR "

# -------------------------
# Paths
# -------------------------

if [ "$DENSIFICATION" = "mcmc" ]; then
    DENSIFICATION_CMD="mcmc"
else
    DENSIFICATION_CMD="default"
fi

# ============================================================================
# TRAINING
# ============================================================================

print_header "STARTING TRAINING"
print_info "Scene: $SCENE"
print_info "Densification: $DENSIFICATION"
print_info "Features: $FEATURE_NAME"
print_info "Temp output: $RESULT_DIR"

START_TIME=$(date +%s)

# -------------------------
# Training with MEMORY OPTIMIZATIONS
# -------------------------


TRAINER_SCRIPT="/home/hensemberk/dev/gsplat_hyperspectral/gsplat/examples/simple_trainer_HSI.py"


srun python "$TRAINER_SCRIPT" "$DENSIFICATION_CMD" \
    --max_steps $MAX_STEPS \
    --hyperspectral_data_dir $NPY_DIR \
    --rgb_data_dir $RGB_DIR \
    --save_steps $SAVE_STEPS \
    --eval_steps $SAVE_STEPS \
    --colmap_dir $COLMAP_DIR \
    --result_dir $OUTPUT_DIR \
    --data_factor $DATA_FACTOR \
    --no-normalize_world_space \
    \
    --sh_degree 3 \
    \
    --batch_size 1 \
    \
    $FLAGS \
    2>&1 | tee -a "$LOG_FILE"

# --ssim_lambda 0.2 \
#

TRAIN_EXIT=${PIPESTATUS[0]}   # ← must be immediately after, PIPESTATUS is consumed on next command
echo "Training exit code: $TRAIN_EXIT"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))


# # ============================================================================
# # POST-TRAINING: EVAL + COPY CKPT + CLEANUP TEMP DIR
# # ============================================================================

# print_header "TRAINING COMPLETED"
# print_info "Duration: $((DURATION / 60)) minutes ($DURATION seconds)"

# REPO_ROOT="/home/hensemberk/dev/gsplat_hyperspectral/gsplat"
# CKPT_SRC_DIR="$RESULT_DIR/ckpts"  
# CKPT_DEST_DIR="$OUTPUT_DIR/ckpts"

# mkdir -p "$CKPT_DEST_DIR"

# print_info "Source checkpoint directory: $CKPT_SRC_DIR"
# print_info "Destination checkpoint directory: $CKPT_DEST_DIR"

# if compgen -G "$CKPT_SRC_DIR/ckpt_*.pt" > /dev/null 2>&1; then
#     print_info "Found training checkpoints. Copying all..."
    
#     CKPT_COUNT=0
#     for ckpt_file in "$CKPT_SRC_DIR"/ckpt_*.pt; do
#         cp "$ckpt_file" "$CKPT_DEST_DIR/" && {
#             CKPT_COUNT=$((CKPT_COUNT + 1))
#             print_info "  [$(basename "$ckpt_file")] ✓"
#         } || {
#             print_error "Failed to copy $(basename "$ckpt_file")"
#         }
#     done
    
#     print_info "Total checkpoints copied: $CKPT_COUNT"
    
#     # DYNAMIC FIX: Find the absolute latest checkpoint by step number from the destination folder
#     # Sorts numerically by name so ckpt_1499_rank0.pt comes before ckpt_9999_rank0.pt properly
#     LATEST_CKPT=$(ls -v "$CKPT_DEST_DIR"/ckpt_*.pt 2>/dev/null | tail -1)
    
# else
#     print_warning "No checkpoints found in $CKPT_SRC_DIR"
#     LATEST_CKPT=""
# fi

# if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT" ]; then
#     print_info "Latest checkpoint resolved for evaluation: $(basename "$LATEST_CKPT")"

#     # ── 1. Run evaluation (outputs go to OUTPUT_DIR inside CHALLENGE_DIR) ──
#     print_header "RUNNING EVALUATION"

#     srun python "$REPO_ROOT/examples/eval_gsplat_hsi.py" \
#         --ckpt "$LATEST_CKPT" \
#         --data_dir "$CHALLENGE_DIR" \
#         --result_folder "$OUTPUT_DIR" \
#         --specific \
#         2>&1 | tee -a "$LOG_FILE"

#     # ── 2. Summary ──
#     print_info "Evaluation complete. All checkpoints safely preserved in: $CKPT_DEST_DIR"

#     # # ── 3. Delete the temporary training directory ──
#     rm -rf "$RESULT_DIR"
#     print_info "Temp training directory removed: $RESULT_DIR"

# else
#     print_warning "No valid checkpoint found for evaluation."
#     print_warning "Skipping eval and temp-dir cleanup."
# fi


# ============================================================================
# GENERATE SUMMARY  (saved to OUTPUT_DIR alongside logs + ckpt)
# ============================================================================

SUMMARY_FILE="$OUTPUT_DIR/SUMMARY.md"

cat > "$SUMMARY_FILE" << EOF
# Experiment Summary

## Configuration
- **Scene**: $SCENE
- **Densification**: $DENSIFICATION
- **Features**: $FEATURE_NAME
- **Run**: ${RUN_NUM}

## Parameters
- **Max Steps**: $MAX_STEPS
- **Data Factor**: $DATA_FACTOR
- **Opacity Reg**: $OPACITY_REG
- **Scale Reg**: $SCALE_REG

## Features Enabled
- Absgrad: $ABSGRAD
- Antialiased: $ANTIALIASED
- App Opt: $APP_OPT
- Bilateral Grid: $BILATERAL
- PPISP: $PPISP
- DEPTH: $DEPTH_LOSS

- KL_LOSS Grid: $KL_LOSS
- SAM_LOSS: $SAM_LOSS
- WAVE_OPT: $WAVE_OPT
- NUM_SAVES: $NUM_SAVES

## Results
- **Start Time**: $(date -d @$START_TIME '+%Y-%m-%d %H:%M:%S')
- **End Time**: $(date '+%Y-%m-%d %H:%M:%S')
- **Duration**: $((DURATION / 60)) minutes
- **Output Dir**: $OUTPUT_DIR
- **Checkpoint**: $CKPT_DEST_DIR/$(basename $CKPT)

## Paths
- Data: $DATA_DIR
- COLMAP: $COLMAP_DIR
- Challenge: $CHALLENGE_DIR
- Depth Dir: $DEPTH_DIR

EOF

print_info "Summary saved to: $SUMMARY_FILE"

echo ""
print_header "✅ EXPERIMENT FINISHED SUCCESSFULLY"
# # ============================================================================
# # Post-processing & Wandb Sync
# # ============================================================================

# # WANDB_DIR_IN_TMP="$RESULT_DIR/wandb"

# # if [ -f "$CKPT" ] && [ "$TRAIN_EXIT" -eq 0 ]; then

# #     # ── 1. Sync wandb BEFORE deleting /tmp ──
# #     if [ -d "$WANDB_DIR_IN_TMP" ]; then
# #         echo "🔄 Syncing wandb offline runs to cloud..."
# #         export WANDB_MODE=online

# #         SYNC_FAILED=0

# #         # offline runs are named offline-run-YYYYMMDD_HHMMSS-<id>
# #         while IFS= read -r run_dir; do
# #             echo "  → Syncing: $(basename $run_dir)"
# #             wandb sync "$run_dir" \
# #                 && echo "  ✓ Synced: $(basename $run_dir)" \
# #                 || { echo "  ⚠ Failed: $(basename $run_dir)"; SYNC_FAILED=1; }
# #         done < <(find "$WANDB_DIR_IN_TMP" -maxdepth 1 -type d -name "offline-run-*")

# #         # Check if any runs were actually found
# #         RUN_COUNT=$(find "$WANDB_DIR_IN_TMP" -maxdepth 1 -type d -name "offline-run-*" | wc -l)
# #         if [ "$RUN_COUNT" -eq 0 ]; then
# #             echo "⚠ No offline-run-* directories found in $WANDB_DIR_IN_TMP"
# #             echo "⚠ Contents of wandb dir:"
# #             ls -la "$WANDB_DIR_IN_TMP"
# #         elif [ $SYNC_FAILED -eq 0 ]; then
# #             echo "✓ All $RUN_COUNT wandb run(s) synced successfully"
# #         else
# #             echo "⚠ Some wandb runs failed to sync (non-fatal)"
# #         fi
# #     else
# #         echo "⚠ No wandb directory found at $WANDB_DIR_IN_TMP — skipping sync"
# #     fi

# #     # ── 2. Clean up temp directory AFTER sync ──
# #     echo "🧹 Removing temp training directory: $RESULT_DIR"
# #     rm -rf "$RESULT_DIR" \
# #         && echo "✓ Temp directory removed" \
# #         || echo "⚠ Failed to remove temp dir: $RESULT_DIR"

# # else
# #     echo "⚠ Training failed (exit=$TRAIN_EXIT) or checkpoint not found at: $CKPT"
# #     echo "⚠ Keeping temp dir for inspection: $RESULT_DIR"
# # fi


# if [ -f "$CKPT" ] && [ "$TRAIN_EXIT" -eq 0 ]; then
#     CKPT_DEST_DIR="$OUTPUT_DIR/ckpts"
#     mkdir -p "$CKPT_DEST_DIR"
#     cp "$CKPT" "$CKPT_DEST_DIR/"
#     print_info "Checkpoint copied to: $CKPT_DEST_DIR/$(basename $CKPT)"

#     echo "🧹 Removing temp training directory: $RESULT_DIR"
#     rm -rf "$RESULT_DIR" \
#         && echo "✓ Temp directory removed" \
#         || echo "⚠ Failed to remove temp dir: $RESULT_DIR"
# else
#     echo "⚠ Training failed or checkpoint not found"
#     echo "⚠ Keeping temp dir: $RESULT_DIR"
# fi