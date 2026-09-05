import os
import json
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from scipy.stats import spearmanr
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler

# ============================================================
# OPTIONAL LPIPS
# ============================================================
try:
    import lpips

    LPIPS_AVAILABLE = True
    print("LPIPS module available")
except ImportError:
    lpips = None
    LPIPS_AVAILABLE = False
    print("lpips not installed. LPIPS will be skipped.")


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # -----------------------------
    # Paths
    # -----------------------------
    DATA_ROOT = (
        "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    )

    VAE_CHECKPOINT = "results/vae_mixed/autoencoder_best.pth"

    # IMPORTANT: use diffusion_best2
    DIFFUSION_CHECKPOINT = "diffusion/diffusion_best2.pth"

    LATENT_STATS_PATH = "latent_pairs/latent_stats.pt"
    VAL_LATENTS_PATH = "latent_pairs/val_pairs_scaled.pt"

    SAVE_DIR = Path("evaluation_results_diffusion_best_claude2")
    VIS_DIR = SAVE_DIR / "visual_comparisons"
    PT_DIR = SAVE_DIR / "generated_samples"

    # -----------------------------
    # Model / latent configuration
    # -----------------------------
    LATENT_CHANNELS = 8

    # -----------------------------
    # Evaluation
    # -----------------------------
    SLICES_TO_EVAL = range(70, 100, 5)

    VALIDATION_SPLIT = 0.10
    NUM_TEST_CASES = 5

    # -----------------------------
    # Reproducibility
    # -----------------------------
    SEED = 42

    # -----------------------------
    # Hyperparameter search
    # -----------------------------
    RUN_HYPERPARAMETER_SEARCH = True

    # Number of validation latent pairs used for tuning.
    # Increase to 20-50 for a more stable thesis-level search.
    NUM_TUNING_SAMPLES = 20

    # Fixed candidates from your previous search. Still used when
    # USE_WIDE_SEARCH=False (below), and as the ORIGIN of the fine
    # search grid otherwise -- kept exactly as-is so nothing about
    # the original narrow search changes unless you opt in.
    HYPERPARAMETER_COMBINATIONS = [
        (2.6, 200, 1.0, "g2.6_s200_e1.0"),
        (2.6, 200, 0.8, "g2.6_s200_e0.8"),
        (2.8, 200, 0.8, "g2.8_s200_e0.8"),
        (3.0, 160, 0.8, "g3.0_s160_e0.8"),
        (2.6, 180, 0.8, "g2.6_s180_e0.8"),
        (2.6, 180, 1.0, "g2.6_s180_e1.0"),
        (2.6, 200, 0.9, "g2.6_s200_e0.9"),
        (2.8, 180, 1.0, "g2.8_s180_e1.0"),
        (2.4, 200, 1.0, "g2.4_s200_e1.0"),
    ]

    # -----------------------------
    # ADDED: two-stage coarse-to-fine wide search
    # -----------------------------
    # Your original grid only covers guidance 2.4-3.0 and eta 0.8-1.0
    # -- a narrow slice of the space, never tried low guidance or a
    # fully deterministic sampler (eta=0). USE_WIDE_SEARCH runs a
    # cheap coarse sweep over a much wider range first (fewer steps,
    # to keep it affordable), picks the most promising region(s), then
    # runs a focused fine grid there at full step count and more
    # tuning samples. Set False to fall back to the original fixed
    # HYPERPARAMETER_COMBINATIONS list untouched.
    USE_WIDE_SEARCH = False

    # Coarse stage: cheap (fewer steps), wide range, on
    # NUM_TUNING_SAMPLES pairs.
    COARSE_GUIDANCE_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    COARSE_ETA_VALUES = [0.0, 0.5, 1.0]
    COARSE_STEPS = 100  # cheaper than the full 150-200 used elsewhere

    # Fine stage: how many top coarse (guidance, eta) regions to
    # refine, how far to search around each, and at what resolution.
    FINE_TOP_K_REGIONS = 2
    FINE_GUIDANCE_RADIUS = 0.4   # search chosen_guidance +/- this, in steps below
    FINE_GUIDANCE_STEP = 0.2
    FINE_ETA_VALUES = [0.0, 0.2, 0.5, 0.8, 1.0]
    FINE_STEPS_VALUES = [150, 200]  # try both around the coarse winner

    # Fine stage uses more tuning samples than the coarse stage for a
    # more trustworthy final ranking (coarse stage always uses
    # NUM_TUNING_SAMPLES to stay cheap).
    NUM_TUNING_SAMPLES_FINE = 40

    # -----------------------------
    # ADDED: tuning/eval leakage check
    # -----------------------------
    # Checks whether the (case, slice) identities used to TUNE
    # best_params overlap with the (case, slice) identities used in
    # the FINAL evaluation. If val_pairs_scaled.pt was built from the
    # same cases/slices run_evaluation_final() evaluates on, tuning on
    # it and then reporting metrics on it is leakage -- the reported
    # numbers would be optimistic. Requires PairedLatentDataset items
    # to carry a "case"/"slice" (or similar) identifier; if they
    # don't, this prints a warning and skips rather than guessing.
    RUN_LEAKAGE_CHECK = True

    # Fallback if hyperparameter search is disabled.
    DEFAULT_GUIDANCE = 2.6
    DEFAULT_STEPS = 200
    DEFAULT_ETA = 1.0

    # Classifier-free guidance:
    # Keep this enabled because your previous search found it useful.
    USE_CFG = True

    # Evaluation images are normalized to [0, 1].
    METRIC_DATA_RANGE = 1.0

    # Skip slices that are effectively empty.
    EMPTY_SLICE_THRESHOLD = 0.01

    # -----------------------------
    # ADDED: Diagnostics
    # -----------------------------
    # Diagnostic 1: check whether EMPTY_SLICE_THRESHOLD on the
    # [-1,1]-normalized tensor actually catches background-heavy
    # slices. Purely informational -- does NOT change the main
    # evaluation loop's behavior.
    RUN_EMPTY_SLICE_DIAGNOSTIC = True
    EMPTY_SLICE_DIAGNOSTIC_NUM_CASES = 3
    EMPTY_SLICE_DIAGNOSTIC_SLICE_STEP = 5
    EMPTY_SLICE_DIAGNOSTIC_BG_FRAC_THRESHOLD = 0.95  # "truly empty" cutoff

    # Diagnostic 2: during the DDIM hyperparameter search, also decode
    # each candidate's generated latents and compute image-space
    # PSNR/SSIM, then report whether the latent-MSE ranking agrees
    # with the image-space ranking.
    EVALUATE_SEARCH_IN_IMAGE_SPACE = True

    # ADDED: which criterion actually picks best_params for the final
    # evaluation run. "latent_mse" reproduces the original behavior.
    # "psnr" / "ssim" select using the image-space diagnostic instead
    # (forces EVALUATE_SEARCH_IN_IMAGE_SPACE on if not already set).
    SELECT_BEST_BY = "latent_mse"  # one of: "latent_mse", "psnr", "ssim"

    # -----------------------------
    # ADDED: cached search results (skip the ~30+ min DDIM search)
    # -----------------------------
    # If True, main() skips search_best_ddim_parameters() entirely
    # (and doesn't even load the validation latent dataset) and picks
    # best_params straight from CSVs already saved by a previous run.
    # Falls back to KNOWN_BEST_CONFIGS below if those CSVs aren't
    # found on disk -- so this still works on a fresh machine/folder.
    USE_CACHED_SEARCH_RESULTS = False

    # These match the filenames search_best_ddim_parameters() and
    # report_search_objective_agreement() already write to SAVE_DIR.
    CACHED_SEARCH_CSV = SAVE_DIR / "ddim_hyperparameter_search.csv"
    CACHED_IMAGE_SPACE_CSV = SAVE_DIR / "diagnostic_search_objective_agreement.csv"

    # Hardcoded fallback, taken directly from your last full search
    # run (20 tuning samples, 9 candidates, seed=42):
    #   Best by latent MSE : g2.6_s180_e1.0
    #   Best by PSNR       : g2.8_s180_e1.0
    #   Best by SSIM       : g2.4_s200_e1.0
    # Re-run the full search if you change NUM_TUNING_SAMPLES,
    # HYPERPARAMETER_COMBINATIONS, the checkpoint, or the seed --
    # these numbers are only valid for that exact configuration.
    KNOWN_BEST_CONFIGS = {
        "latent_mse": {
            "guidance": 2.6, "steps": 180, "eta": 1.0,
            "name": "g2.6_s180_e1.0",
        },
        "psnr": {
            "guidance": 2.8, "steps": 180, "eta": 1.0,
            "name": "g2.8_s180_e1.0",
        },
        "ssim": {
            "guidance": 2.4, "steps": 200, "eta": 1.0,
            "name": "g2.4_s200_e1.0",
        },
    }


cfg = Config()


# ============================================================
# REPRODUCIBILITY
# ============================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic settings for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


seed_everything(cfg.SEED)


# ============================================================
# DEVICE / DIRECTORIES
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache() if torch.cuda.is_available() else None

cfg.SAVE_DIR.mkdir(parents=True, exist_ok=True)
cfg.VIS_DIR.mkdir(parents=True, exist_ok=True)
cfg.PT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DIFFUSION BEST2 - FINAL EVALUATION")
print("=" * 70)
print(f"Device: {device}")
print(f"Diffusion checkpoint: {cfg.DIFFUSION_CHECKPOINT}")
print(f"Scheduler: scaled_linear_beta")
print(f"Seed: {cfg.SEED}")
print("=" * 70)


# ============================================================
# LPIPS
# ============================================================
lpips_model = None

if LPIPS_AVAILABLE:
    try:
        lpips_model = lpips.LPIPS(net="alex").to(device)
        lpips_model.eval()
        print("LPIPS model initialized.")
    except Exception as e:
        print(f"Failed to initialize LPIPS: {e}")
        LPIPS_AVAILABLE = False
        lpips_model = None


# ============================================================
# MODEL LOADING
# ============================================================
def load_vae(device, ckpt_path):
    print(f"\nLoading VAE from: {ckpt_path}")

    vae = AutoencoderKL(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 128, 256),
        latent_channels=8,
        num_res_blocks=2,
        attention_levels=(False, False, True),
        norm_num_groups=32,
    ).to(device)

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    if "model" not in ckpt:
        raise KeyError(
            f"'model' key not found in VAE checkpoint: {ckpt_path}"
        )

    vae.load_state_dict(ckpt["model"])
    vae.eval()

    print("VAE loaded successfully.")
    return vae


def load_diffusion(device, ckpt_path):
    print(f"Loading diffusion model from: {ckpt_path}")

    unet = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=16,
        out_channels=8,
        num_channels=(64, 128, 256),
        attention_levels=(False, True, True),
        num_head_channels=(32, 64, 128),
        num_res_blocks=3,
    ).to(device)

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    # Prefer EMA if available, otherwise fall back to model/state_dict.
    if "ema_state_dict" in ckpt:
        source_state = ckpt["ema_state_dict"]
        print("Using EMA diffusion weights.")
    elif "model" in ckpt:
        source_state = ckpt["model"]
        print("Using diffusion 'model' weights.")
    elif "state_dict" in ckpt:
        source_state = ckpt["state_dict"]
        print("Using diffusion 'state_dict' weights.")
    else:
        raise KeyError(
            "Checkpoint does not contain 'ema_state_dict', 'model', "
            "or 'state_dict'."
        )

    current_state = unet.state_dict()

    matched = 0
    for name, value in source_state.items():
        if name in current_state:
            current_state[name] = value
            matched += 1

    if matched == 0:
        raise RuntimeError(
            "No diffusion parameters matched the UNet architecture."
        )

    unet.load_state_dict(current_state)
    unet.eval()

    print(f"Loaded {matched} parameter tensors.")
    print("Diffusion model loaded successfully.")

    return unet


vae = load_vae(device, cfg.VAE_CHECKPOINT)
unet = load_diffusion(device, cfg.DIFFUSION_CHECKPOINT)


# ============================================================
# LATENT STATISTICS
# ============================================================
print(f"\nLoading latent statistics from: {cfg.LATENT_STATS_PATH}")

latent_stats = torch.load(
    cfg.LATENT_STATS_PATH,
    map_location=device,
    weights_only=False,
)

LATENT_MEAN = (
    latent_stats["mean"]
    .to(device)
    .float()
    .view(1, cfg.LATENT_CHANNELS, 1, 1)
)

LATENT_STD = (
    latent_stats["std"]
    .to(device)
    .float()
    .view(1, cfg.LATENT_CHANNELS, 1, 1)
)

print("Latent statistics loaded.")
print(f"Latent mean shape: {tuple(LATENT_MEAN.shape)}")
print(f"Latent std shape:  {tuple(LATENT_STD.shape)}")


# ============================================================
# DDIM SCHEDULER
# ============================================================
# IMPORTANT:
# Your selected evaluation schedule is scaled_linear_beta.
# Keep these values consistent with the schedule that produced
# your best diffusion generation results.
# ============================================================
ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    schedule="scaled_linear_beta",
    beta_start=0.00085,
    beta_end=0.012,
    clip_sample=False,
    set_alpha_to_one=False,
)

ddim_scheduler.alphas_cumprod = (
    ddim_scheduler.alphas_cumprod.to(device)
)

print("\nDDIM scheduler initialized:")
print("  schedule = scaled_linear_beta")
print("  train timesteps = 1000")
print("  beta_start = 0.00085")
print("  beta_end   = 0.012")


# ============================================================
# DATA / IMAGE PREPROCESSING
# ============================================================
def normalize_image_percentile(img):
    """Percentile normalization to [-1, 1]."""
    low = np.percentile(img, 1)
    high = np.percentile(img, 99)

    img = np.clip(img, low, high)
    img = (img - low) / (high - low + 1e-8)

    return img * 2.0 - 1.0


def load_brats_slice(case_path, case_name, modality, slice_idx):
    """Load a BraTS slice using the same preprocessing used for evaluation."""
    file_path = case_path / f"{case_name}_{modality}.nii"

    vol = nib.load(file_path).get_fdata()

    # Same spatial crop used in the original evaluation code.
    crop = vol[48:208, 48:208, slice_idx]

    crop = normalize_image_percentile(crop)

    return (
        torch.from_numpy(crop)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )


# ============================================================
# ADDED: DIAGNOSTIC 1 -- EMPTY SLICE FILTER CHECK
# ============================================================
def diagnose_empty_slice_filter():
    """
    Checks whether cfg.EMPTY_SLICE_THRESHOLD, applied to
    abs(tensor).mean() on the [-1,1]-normalized tensor (exactly what
    the main evaluation loop does), actually distinguishes
    background-heavy slices from slices with real anatomy.

    This does NOT modify cfg.EMPTY_SLICE_THRESHOLD or the main loop.
    It only prints a report so you can decide whether the filter
    needs to change.
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: EMPTY-SLICE FILTER CHECK")
    print("=" * 70)

    all_cases = sorted(Path(cfg.DATA_ROOT).glob("BraTS20_Training_*"))

    if len(all_cases) == 0:
        print(f"No cases found under {cfg.DATA_ROOT}. Skipping diagnostic.")
        return

    sample_cases = all_cases[: cfg.EMPTY_SLICE_DIAGNOSTIC_NUM_CASES]
    step = cfg.EMPTY_SLICE_DIAGNOSTIC_SLICE_STEP
    bg_cutoff = cfg.EMPTY_SLICE_DIAGNOSTIC_BG_FRAC_THRESHOLD
    threshold = cfg.EMPTY_SLICE_THRESHOLD

    rows = []

    for case_path in sample_cases:
        case_name = case_path.name
        t1ce_file = case_path / f"{case_name}_t1ce.nii"

        if not t1ce_file.exists():
            continue

        vol = nib.load(t1ce_file).get_fdata()
        num_slices = vol.shape[2]

        for s in range(0, num_slices, step):
            raw_crop = vol[48:208, 48:208, s]

            # Fraction of pixels that are (near-)exactly raw background.
            bg_frac = float(np.mean(np.abs(raw_crop) < 1e-6))

            # What the current code actually checks.
            norm_crop = normalize_image_percentile(raw_crop)
            current_metric = float(np.abs(norm_crop).mean())

            would_skip = current_metric < threshold

            rows.append(
                {
                    "case": case_name,
                    "slice": s,
                    "bg_frac": bg_frac,
                    "abs_mean_normalized": current_metric,
                    "would_skip": would_skip,
                }
            )

    if not rows:
        print("No slices could be read. Skipping diagnostic.")
        return

    df = pd.DataFrame(rows)

    df.to_csv(
        cfg.SAVE_DIR / "diagnostic_empty_slice_filter.csv",
        index=False,
    )

    truly_empty = df[df["bg_frac"] > bg_cutoff]
    caught = truly_empty[truly_empty["would_skip"]]

    non_empty = df[df["bg_frac"] <= bg_cutoff]
    false_positive = non_empty[non_empty["would_skip"]]

    print(f"Cases sampled           : {len(sample_cases)}")
    print(f"Slices sampled          : {len(df)}")
    print(f"Current threshold       : {threshold}")
    print(f"'Truly empty' cutoff    : bg_frac > {bg_cutoff}")
    print("-" * 70)
    print(f"Truly empty slices      : {len(truly_empty)}")
    print(f"  ...caught by filter   : {len(caught)}  "
          f"({(len(caught) / max(1, len(truly_empty))) * 100:.1f}%)")
    print(f"Non-empty slices        : {len(non_empty)}")
    print(f"  ...false-positive skip: {len(false_positive)}  "
          f"({(len(false_positive) / max(1, len(non_empty))) * 100:.1f}%)")
    print("-" * 70)
    print(
        f"abs_mean_normalized range across ALL slices: "
        f"[{df['abs_mean_normalized'].min():.4f}, "
        f"{df['abs_mean_normalized'].max():.4f}]"
    )
    print(
        f"abs_mean_normalized range across TRULY EMPTY slices: "
        f"[{truly_empty['abs_mean_normalized'].min():.4f}, "
        f"{truly_empty['abs_mean_normalized'].max():.4f}]"
        if len(truly_empty) > 0
        else "abs_mean_normalized range across TRULY EMPTY slices: n/a"
    )

    if len(truly_empty) > 0 and len(caught) == 0:
        print(
            "\n*** The current filter caught ZERO truly empty slices. "
            "It is very likely non-functional on this data. ***"
        )
    elif len(truly_empty) > 0 and len(caught) < len(truly_empty):
        print(
            f"\n*** The current filter caught only "
            f"{len(caught)}/{len(truly_empty)} truly empty slices. "
            f"Consider a background-fraction-based filter instead. ***"
        )
    else:
        print("\nFilter appears to behave as intended on this sample.")

    print(
        f"\nFull table saved to: "
        f"{cfg.SAVE_DIR / 'diagnostic_empty_slice_filter.csv'}"
    )
    print("=" * 70)


# ============================================================
# DETERMINISTIC NOISE
# ============================================================
def make_fixed_noise_like(reference, seed):
    """
    Generate reproducible Gaussian noise directly on the target device.
    The same seed gives the same starting latent for a candidate/sample.
    """
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(int(seed))

    return torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def stable_seed(*values):
    """
    Convert arbitrary values into a deterministic integer seed.
    """
    text = "|".join(str(v) for v in values)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Keep safely inside torch/manual_seed's useful integer range.
    return int(digest[:8], 16)


# ============================================================
# MODEL FORWARD
# ============================================================
def predict_noise(
    latent,
    condition,
    timestep,
    guidance_scale,
):
    """
    Conditional + optional classifier-free guidance.
    """
    timestep_tensor = torch.full(
        (latent.shape[0],),
        int(timestep),
        device=latent.device,
        dtype=torch.long,
    )

    if not cfg.USE_CFG:
        return unet(
            torch.cat([latent, condition], dim=1),
            timestep_tensor,
        )

    noise_cond = unet(
        torch.cat([latent, condition], dim=1),
        timestep_tensor,
    )

    noise_uncond = unet(
        torch.cat([latent, torch.zeros_like(condition)], dim=1),
        timestep_tensor,
    )

    return (
        noise_uncond
        + guidance_scale * (noise_cond - noise_uncond)
    )


# ============================================================
# DDIM SAMPLING
# ============================================================
@torch.no_grad()
def ddim_sample_with_params(
    condition,
    unet,
    scheduler,
    num_inference_steps,
    guidance_scale,
    eta,
    start_noise=None,
    verbose=False,
):
    """
    Standard DDIM reverse sampling.

    Important:
    - If start_noise is supplied, the sampling run is reproducible.
    - The same start_noise can be reused to fairly compare different
      guidance/steps/eta settings.
    """
    scheduler.set_timesteps(num_inference_steps)

    if start_noise is None:
        latent = torch.randn_like(condition)
    else:
        latent = start_noise.clone()

    iterator = (
        tqdm(
            scheduler.timesteps,
            desc=f"DDIM ({num_inference_steps} steps)",
            leave=False,
        )
        if verbose
        else scheduler.timesteps
    )

    for t in iterator:
        noise_pred = predict_noise(
            latent=latent,
            condition=condition,
            timestep=int(t),
            guidance_scale=guidance_scale,
        )

        latent, _ = scheduler.step(
            model_output=noise_pred,
            timestep=int(t),
            sample=latent,
            eta=float(eta),
        )

    return latent


# ============================================================
# METRICS
# ============================================================
def compute_metrics(real, synth, compute_lpips=True):
    """
    All image metrics are computed on [0, 1] images.

    PSNR and SSIM therefore use data_range=1.0, rather than a
    per-image dynamic range.
    """
    real_t = real.detach().float()
    synth_t = synth.detach().float()

    real_np = real_t.squeeze().cpu().numpy()
    synth_np = synth_t.squeeze().cpu().numpy()

    if real_np.shape != synth_np.shape:
        from skimage.transform import resize

        synth_np = resize(
            synth_np,
            real_np.shape,
            preserve_range=True,
            anti_aliasing=True,
        )

    mse = mean_squared_error(real_np, synth_np)
    mae = np.mean(np.abs(real_np - synth_np))

    try:
        p = psnr(
            real_np,
            synth_np,
            data_range=cfg.METRIC_DATA_RANGE,
        )
    except Exception:
        p = 0.0

    try:
        min_dim = min(real_np.shape)

        if min_dim >= 7:
            win_size = 7
        else:
            win_size = min_dim

        if win_size % 2 == 0:
            win_size -= 1

        win_size = max(3, win_size)

        s = ssim(
            real_np,
            synth_np,
            data_range=cfg.METRIC_DATA_RANGE,
            win_size=win_size,
        )
    except Exception:
        s = 0.0

    lp = 0.0

    if compute_lpips and LPIPS_AVAILABLE and lpips_model is not None:
        try:
            # real/synth are [0,1]
            real_lpips = (
                real_t.clamp(0, 1)
                .repeat(1, 3, 1, 1)
                * 2
                - 1
            ).to(device)

            synth_lpips = (
                synth_t.clamp(0, 1)
                .repeat(1, 3, 1, 1)
                * 2
                - 1
            ).to(device)

            lp = lpips_model(
                real_lpips,
                synth_lpips,
            ).item()

        except Exception as e:
            print(f"LPIPS failed: {e}")
            lp = 0.0

    return mse, mae, p, s, lp


# ============================================================
# HYPERPARAMETER SEARCH
# ============================================================
def select_tuning_samples(val_dataset, num_samples):
    n = len(val_dataset)

    if n == 0:
        raise RuntimeError("Validation latent dataset is empty.")

    count = min(num_samples, n)

    # Deterministic subset selection.
    rng = random.Random(cfg.SEED)

    indices = list(range(n))
    rng.shuffle(indices)

    return indices[:count]


# ============================================================
# ADDED: DIAGNOSTIC 2 -- SEARCH OBJECTIVE AGREEMENT CHECK
# ============================================================
@torch.no_grad()
def report_search_objective_agreement(cached_samples, search_results):
    """
    For each candidate already evaluated by search_best_ddim_parameters,
    decode the generated latents back to image space and compute
    PSNR/SSIM, then compare the ranking induced by latent MSE (what
    the search currently optimizes) against the ranking induced by
    PSNR/SSIM (what you actually report as final metrics).

    This does NOT change best_params -- it only prints/saves a report
    so you can see whether the two objectives agree.
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: SEARCH OBJECTIVE AGREEMENT (latent MSE vs image space)")
    print("=" * 70)

    rows = []

    for candidate in tqdm(search_results, desc="Decoding candidates for image-space check"):
        guidance = candidate["guidance"]
        steps = candidate["steps"]
        eta = candidate["eta"]
        name = candidate["name"]

        psnrs = []
        ssims = []

        for item in cached_samples:
            generated_scaled = ddim_sample_with_params(
                condition=item["condition"],
                unet=unet,
                scheduler=ddim_scheduler,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                eta=float(eta),
                start_noise=item["noise"],
                verbose=False,
            )

            # item["target"] is the SCALED latent (same space the
            # search already compares against). Unscale + decode both
            # sides for a fair image-space comparison.
            generated_unscaled = generated_scaled * LATENT_STD + LATENT_MEAN
            target_unscaled = item["target"] * LATENT_STD + LATENT_MEAN

            synth_img = vae.decode(generated_unscaled)
            real_img = vae.decode(target_unscaled)

            synth_01 = ((synth_img.clamp(-1, 1) + 1) / 2.0).clamp(0, 1)
            real_01 = ((real_img.clamp(-1, 1) + 1) / 2.0).clamp(0, 1)

            _, _, p, s, _ = compute_metrics(real_01, synth_01, compute_lpips=False)
            psnrs.append(p)
            ssims.append(s)

        rows.append(
            {
                "name": name,
                "guidance": guidance,
                "steps": steps,
                "eta": eta,
                "mean_latent_mse": candidate["mean_latent_mse"],
                "mean_psnr": float(np.mean(psnrs)),
                "mean_ssim": float(np.mean(ssims)),
            }
        )

    df = pd.DataFrame(rows)

    df["latent_rank"] = df["mean_latent_mse"].rank(method="min", ascending=True)
    df["psnr_rank"] = df["mean_psnr"].rank(method="min", ascending=False)
    df["ssim_rank"] = df["mean_ssim"].rank(method="min", ascending=False)

    df = df.sort_values("latent_rank").reset_index(drop=True)

    df.to_csv(
        cfg.SAVE_DIR / "diagnostic_search_objective_agreement.csv",
        index=False,
    )

    rho_psnr, _ = spearmanr(df["latent_rank"], df["psnr_rank"])
    rho_ssim, _ = spearmanr(df["latent_rank"], df["ssim_rank"])

    print(df.to_string(index=False))

    print("\nSpearman rho (latent-MSE rank vs PSNR rank): "
          f"{rho_psnr:.3f}")
    print("Spearman rho (latent-MSE rank vs SSIM rank): "
          f"{rho_ssim:.3f}")
    print("(rho close to 1.0 = objectives agree; near 0 or negative = they diverge)")

    best_by_latent = df.iloc[0]["name"]
    best_by_psnr = df.sort_values("psnr_rank").iloc[0]["name"]
    best_by_ssim = df.sort_values("ssim_rank").iloc[0]["name"]

    print(f"\nBest by latent MSE : {best_by_latent}")
    print(f"Best by PSNR        : {best_by_psnr}")
    print(f"Best by SSIM        : {best_by_ssim}")

    if best_by_latent != best_by_psnr or best_by_latent != best_by_ssim:
        print(
            "\n*** The candidate selected by latent MSE is NOT the "
            "same candidate that scores best on the image-space "
            "metrics you report. cfg.SELECT_BEST_BY controls which "
            "one actually gets used for the final evaluation run. ***"
        )
    else:
        print(
            "\nThe latent-MSE-selected candidate also wins on PSNR "
            "and SSIM -- the search objective agrees with the "
            "reporting metrics for this run."
        )

    print(
        f"\nFull table saved to: "
        f"{cfg.SAVE_DIR / 'diagnostic_search_objective_agreement.csv'}"
    )
    print("=" * 70)

    return df


# ============================================================
# ADDED: TUNING / EVAL LEAKAGE CHECK
# ============================================================
def check_tuning_eval_leakage(val_dataset, tuning_indices):
    """
    Checks whether the (case, slice) pairs used for DDIM
    hyperparameter tuning overlap with the (case, slice) pairs used
    in the final evaluation (get_validation_cases() x
    cfg.SLICES_TO_EVAL). If they overlap, best_params was selected
    using information from slices later reported on -- the final
    metrics would be optimistic.

    Requires each val_dataset item to expose a case/slice identifier.
    Tries a few common attribute/key names; if none are found, prints
    a warning and returns without asserting anything (rather than
    silently assuming there's no leakage).
    """
    print("\n" + "=" * 70)
    print("TUNING / EVAL LEAKAGE CHECK")
    print("=" * 70)

    def extract_identity(sample):
        # Try dict-style access first, then attribute-style.
        for case_key in ("case", "case_name", "patient", "patient_id"):
            for slice_key in ("slice", "slice_idx", "slice_index"):
                case_val = None
                slice_val = None
                if isinstance(sample, dict):
                    case_val = sample.get(case_key)
                    slice_val = sample.get(slice_key)
                else:
                    case_val = getattr(sample, case_key, None)
                    slice_val = getattr(sample, slice_key, None)
                if case_val is not None and slice_val is not None:
                    return (str(case_val), int(slice_val))
        return None

    tuning_identities = set()
    identity_available = True

    for idx in tuning_indices:
        sample = val_dataset[idx]
        identity = extract_identity(sample)
        if identity is None:
            identity_available = False
            break
        tuning_identities.add(identity)

    if not identity_available:
        print(
            "Could not find a case/slice identifier on val_dataset "
            "items (tried keys/attrs: case, case_name, patient, "
            "patient_id x slice, slice_idx, slice_index)."
        )
        print(
            "Skipping leakage check -- this does NOT mean there is no "
            "leakage, only that it could not be checked automatically. "
            "If PairedLatentDataset can be made to carry this identity, "
            "re-run this check."
        )
        print("=" * 70)
        return

    all_cases, val_cases = get_validation_cases()
    eval_case_names = {c.name for c in val_cases}
    eval_slice_indices = set(cfg.SLICES_TO_EVAL)

    eval_identities = {
        (case_name, slice_idx)
        for case_name in eval_case_names
        for slice_idx in eval_slice_indices
    }

    overlap = tuning_identities & eval_identities

    print(f"Tuning identities found : {len(tuning_identities)}")
    print(f"Eval (case, slice) grid : {len(eval_identities)} "
          f"({len(eval_case_names)} cases x {len(eval_slice_indices)} slices)")
    print(f"Overlap                 : {len(overlap)}")

    if len(overlap) > 0:
        print(
            "\n*** LEAKAGE DETECTED: tuning samples overlap with the "
            "final evaluation set. best_params was selected using "
            "information from slices that are also being reported on "
            "-- treat the final metrics as optimistically biased "
            "until this is fixed (exclude these identities from "
            "tuning-sample selection, or hold out val_pairs_scaled.pt "
            "from the same case split used for final evaluation). ***"
        )
        example = sorted(overlap)[:10]
        print(f"Example overlapping (case, slice) pairs: {example}")
    else:
        print("\nNo overlap detected -- tuning and final evaluation "
              "sets are disjoint on the identities available.")

    print("=" * 70)


# ============================================================
# ADDED: TWO-STAGE COARSE-TO-FINE DDIM SEARCH
# ============================================================
@torch.no_grad()
def _evaluate_candidate_mean_latent_mse(cached_samples, unet, scheduler, guidance, steps, eta):
    """Shared inner loop: mean latent MSE for one (guidance, steps, eta)."""
    sample_mses = []
    for item in cached_samples:
        generated = ddim_sample_with_params(
            condition=item["condition"],
            unet=unet,
            scheduler=scheduler,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            eta=float(eta),
            start_noise=item["noise"],
            verbose=False,
        )
        latent_mse = torch.mean((item["target"] - generated) ** 2).item()
        sample_mses.append(latent_mse)
    return float(np.mean(sample_mses)), float(np.std(sample_mses))


def _build_cached_samples(val_dataset, num_samples, tag):
    """Same caching logic search_best_ddim_parameters used inline,
    factored out so both the coarse and fine stages (and the leakage
    check) can share it and use the same seeded indices for a given
    sample count."""
    indices = select_tuning_samples(val_dataset, num_samples)

    cached_samples = []
    for index in tqdm(indices, desc=f"Preparing {tag} tuning samples"):
        sample = val_dataset[index]
        condition = sample["condition"].unsqueeze(0).to(device).float()
        target = sample["target"].unsqueeze(0).to(device).float()
        noise_seed = stable_seed(cfg.SEED, "tuning", tag, index)
        start_noise = make_fixed_noise_like(target, noise_seed)
        cached_samples.append({
            "index": index, "condition": condition,
            "target": target, "noise": start_noise,
        })
    return cached_samples, indices


@torch.no_grad()
def search_best_ddim_parameters_wide(val_dataset, unet, scheduler):
    """
    Two-stage coarse-to-fine DDIM hyperparameter search.

    Stage 1 (coarse): wide grid over guidance x eta at a cheap step
    count (cfg.COARSE_STEPS), on cfg.NUM_TUNING_SAMPLES pairs. Finds
    which region of the space is actually promising.

    Stage 2 (fine): a focused grid around the top
    cfg.FINE_TOP_K_REGIONS coarse (guidance, eta) results, at full
    step counts (cfg.FINE_STEPS_VALUES) and more tuning samples
    (cfg.NUM_TUNING_SAMPLES_FINE) for a trustworthy final ranking.

    Returns the same shape of dict search_best_ddim_parameters()
    does, so it's a drop-in replacement -- diagnostic 2's image-space
    agreement check and cfg.SELECT_BEST_BY both work unchanged on top
    of this.
    """
    print("\n" + "=" * 70)
    print("WIDE DDIM HYPERPARAMETER SEARCH (coarse-to-fine)")
    print("=" * 70)

    if cfg.RUN_LEAKAGE_CHECK:
        # Uses the coarse-stage sample count/seed for the check --
        # good enough to catch systematic overlap, doesn't need to
        # be re-run per stage.
        _, coarse_indices_for_check = _build_cached_samples(
            val_dataset, cfg.NUM_TUNING_SAMPLES, "leakage_check"
        )
        check_tuning_eval_leakage(val_dataset, coarse_indices_for_check)

    # ---------------- Stage 1: coarse ----------------
    print("\n" + "-" * 70)
    print(f"STAGE 1: coarse sweep -- {len(cfg.COARSE_GUIDANCE_VALUES)} guidance x "
          f"{len(cfg.COARSE_ETA_VALUES)} eta = "
          f"{len(cfg.COARSE_GUIDANCE_VALUES) * len(cfg.COARSE_ETA_VALUES)} candidates, "
          f"{cfg.COARSE_STEPS} steps, {cfg.NUM_TUNING_SAMPLES} samples")
    print("-" * 70)

    coarse_cached_samples, _ = _build_cached_samples(
        val_dataset, cfg.NUM_TUNING_SAMPLES, "coarse"
    )

    coarse_results = []
    for guidance in cfg.COARSE_GUIDANCE_VALUES:
        for eta in cfg.COARSE_ETA_VALUES:
            mean_mse, std_mse = _evaluate_candidate_mean_latent_mse(
                coarse_cached_samples, unet, scheduler,
                guidance, cfg.COARSE_STEPS, eta,
            )
            name = f"coarse_g{guidance}_s{cfg.COARSE_STEPS}_e{eta}"
            coarse_results.append({
                "name": name, "guidance": float(guidance),
                "steps": int(cfg.COARSE_STEPS), "eta": float(eta),
                "mean_latent_mse": mean_mse, "std_latent_mse": std_mse,
                "n": len(coarse_cached_samples),
            })
            print(f"  guidance={guidance:<5} eta={eta:<4} -> "
                  f"mean latent MSE = {mean_mse:.6f} (std {std_mse:.6f})")

    coarse_df = pd.DataFrame(coarse_results).sort_values(
        "mean_latent_mse", ascending=True
    ).reset_index(drop=True)
    coarse_df.to_csv(cfg.SAVE_DIR / "ddim_coarse_search.csv", index=False)

    print(f"\nTop {cfg.FINE_TOP_K_REGIONS} coarse regions:")
    print(coarse_df.head(cfg.FINE_TOP_K_REGIONS).to_string(index=False))

    # ---------------- Stage 2: fine ----------------
    top_regions = coarse_df.head(cfg.FINE_TOP_K_REGIONS)[["guidance", "eta"]].to_dict("records")

    fine_guidance_offsets = np.arange(
        -cfg.FINE_GUIDANCE_RADIUS,
        cfg.FINE_GUIDANCE_RADIUS + 1e-9,
        cfg.FINE_GUIDANCE_STEP,
    )

    fine_candidates = set()
    for region in top_regions:
        base_guidance = region["guidance"]
        for offset in fine_guidance_offsets:
            g = round(float(base_guidance + offset), 3)
            if g <= 0:
                continue
            for eta in cfg.FINE_ETA_VALUES:
                for steps in cfg.FINE_STEPS_VALUES:
                    fine_candidates.add((g, steps, eta))

    fine_candidates = sorted(fine_candidates)

    print("\n" + "-" * 70)
    print(f"STAGE 2: fine sweep -- {len(fine_candidates)} candidates around top "
          f"{cfg.FINE_TOP_K_REGIONS} coarse regions, {cfg.NUM_TUNING_SAMPLES_FINE} samples")
    print("-" * 70)

    fine_cached_samples, _ = _build_cached_samples(
        val_dataset, cfg.NUM_TUNING_SAMPLES_FINE, "fine"
    )

    fine_results = []
    for guidance, steps, eta in tqdm(fine_candidates, desc="Fine search"):
        mean_mse, std_mse = _evaluate_candidate_mean_latent_mse(
            fine_cached_samples, unet, scheduler, guidance, steps, eta,
        )
        name = f"g{guidance}_s{steps}_e{eta}"
        fine_results.append({
            "name": name, "guidance": float(guidance),
            "steps": int(steps), "eta": float(eta),
            "mean_latent_mse": mean_mse, "std_latent_mse": std_mse,
            "n": len(fine_cached_samples),
        })

    fine_df = pd.DataFrame(fine_results).sort_values(
        "mean_latent_mse", ascending=True
    ).reset_index(drop=True)
    fine_df.to_csv(cfg.SAVE_DIR / "ddim_hyperparameter_search.csv", index=False)

    print("\nTop 5 fine candidates:")
    print(fine_df.head(5).to_string(index=False))

    best_by_latent_mse = fine_df.iloc[0].to_dict()

    print("\n" + "=" * 70)
    print("BEST DDIM CONFIGURATION (by latent MSE, wide search)")
    print("=" * 70)
    print(f"Name     : {best_by_latent_mse['name']}")
    print(f"Guidance : {best_by_latent_mse['guidance']}")
    print(f"Steps    : {best_by_latent_mse['steps']}")
    print(f"Eta      : {best_by_latent_mse['eta']}")
    print(f"Mean MSE : {best_by_latent_mse['mean_latent_mse']:.8f}")
    print("=" * 70)

    # Reuse the existing image-space agreement diagnostic / selection
    # logic on top of the fine-stage results, exactly like the
    # original search does -- no duplicated selection logic.
    needs_image_space = (
        cfg.EVALUATE_SEARCH_IN_IMAGE_SPACE
        or cfg.SELECT_BEST_BY in ("psnr", "ssim")
    )

    image_space_df = None
    if needs_image_space:
        image_space_df = report_search_objective_agreement(
            fine_cached_samples, fine_results
        )

    if cfg.SELECT_BEST_BY == "latent_mse":
        chosen = best_by_latent_mse
        chosen_name = str(chosen["name"])
    elif cfg.SELECT_BEST_BY in ("psnr", "ssim"):
        if image_space_df is None:
            raise RuntimeError("SELECT_BEST_BY requires image-space metrics.")
        sort_col = "psnr_rank" if cfg.SELECT_BEST_BY == "psnr" else "ssim_rank"
        chosen_row = image_space_df.sort_values(sort_col).iloc[0]
        chosen_name = str(chosen_row["name"])
        chosen = next(r for r in fine_results if r["name"] == chosen_name)
    else:
        raise ValueError(f"Unknown cfg.SELECT_BEST_BY: {cfg.SELECT_BEST_BY!r}")

    print("\n" + "=" * 70)
    print(f"SELECTED CONFIGURATION (cfg.SELECT_BEST_BY = '{cfg.SELECT_BEST_BY}')")
    print("=" * 70)
    print(f"Name     : {chosen['name']}")
    print(f"Guidance : {chosen['guidance']}")
    print(f"Steps    : {chosen['steps']}")
    print(f"Eta      : {chosen['eta']}")
    if chosen_name != str(best_by_latent_mse["name"]):
        print(f"(Differs from the latent-MSE winner: {best_by_latent_mse['name']})")
    print("=" * 70)

    return {
        "guidance": float(chosen["guidance"]),
        "steps": int(chosen["steps"]),
        "eta": float(chosen["eta"]),
        "name": chosen_name,
        "selected_by": cfg.SELECT_BEST_BY,
        "search_results": fine_results,
        "coarse_results": coarse_results,
    }


@torch.no_grad()
def search_best_ddim_parameters(
    val_dataset,
    unet,
    scheduler,
):
    """
    Fair DDIM hyperparameter search.

    Key improvement over the previous code:
    every candidate is evaluated on the SAME validation pairs and
    with the SAME starting noise for each pair.

    This removes random-noise advantage from the comparison.
    """
    print("\n" + "=" * 70)
    print("DDIM HYPERPARAMETER SEARCH")
    print("=" * 70)

    indices = select_tuning_samples(
        val_dataset,
        cfg.NUM_TUNING_SAMPLES,
    )

    print(f"Tuning samples: {len(indices)}")
    print(
        f"Candidates: {len(cfg.HYPERPARAMETER_COMBINATIONS)}"
    )
    print("=" * 70)

    # Cache validation samples + their fixed noise.
    cached_samples = []

    for sample_number, index in enumerate(
        tqdm(indices, desc="Preparing tuning samples")
    ):
        sample = val_dataset[index]

        condition = (
            sample["condition"]
            .unsqueeze(0)
            .to(device)
            .float()
        )

        target = (
            sample["target"]
            .unsqueeze(0)
            .to(device)
            .float()
        )

        noise_seed = stable_seed(
            cfg.SEED,
            "tuning",
            index,
        )

        start_noise = make_fixed_noise_like(
            target,
            noise_seed,
        )

        cached_samples.append(
            {
                "index": index,
                "condition": condition,
                "target": target,
                "noise": start_noise,
            }
        )

    results = []

    for candidate_id, (
        guidance,
        steps,
        eta,
        name,
    ) in enumerate(cfg.HYPERPARAMETER_COMBINATIONS, start=1):

        print(
            f"\n[{candidate_id}/{len(cfg.HYPERPARAMETER_COMBINATIONS)}] "
            f"{name}"
        )

        sample_mses = []

        iterator = tqdm(
            cached_samples,
            desc=f"  {name}",
            leave=False,
        )

        for item in iterator:
            generated = ddim_sample_with_params(
                condition=item["condition"],
                unet=unet,
                scheduler=scheduler,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                eta=float(eta),
                start_noise=item["noise"],
                verbose=False,
            )

            latent_mse = torch.mean(
                (item["target"] - generated) ** 2
            ).item()

            sample_mses.append(latent_mse)

        mean_mse = float(np.mean(sample_mses))
        std_mse = float(np.std(sample_mses))

        results.append(
            {
                "name": name,
                "guidance": float(guidance),
                "steps": int(steps),
                "eta": float(eta),
                "mean_latent_mse": mean_mse,
                "std_latent_mse": std_mse,
                "n": len(sample_mses),
            }
        )

        print(
            f"  Mean latent MSE: {mean_mse:.8f}"
        )
        print(
            f"  Std latent MSE : {std_mse:.8f}"
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "mean_latent_mse",
        ascending=True,
    ).reset_index(drop=True)

    results_df.to_csv(
        cfg.SAVE_DIR / "ddim_hyperparameter_search.csv",
        index=False,
    )

    best_by_latent_mse = results_df.iloc[0].to_dict()

    print("\n" + "=" * 70)
    print("BEST DDIM CONFIGURATION (by latent MSE)")
    print("=" * 70)
    print(f"Name     : {best_by_latent_mse['name']}")
    print(f"Guidance : {best_by_latent_mse['guidance']}")
    print(f"Steps    : {best_by_latent_mse['steps']}")
    print(f"Eta      : {best_by_latent_mse['eta']}")
    print(
        f"Mean MSE : {best_by_latent_mse['mean_latent_mse']:.8f}"
    )
    print("=" * 70)

    # ADDED: image-space agreement diagnostic. Runs whenever it's
    # explicitly requested, OR whenever SELECT_BEST_BY needs its
    # output to pick a candidate.
    needs_image_space = (
        cfg.EVALUATE_SEARCH_IN_IMAGE_SPACE
        or cfg.SELECT_BEST_BY in ("psnr", "ssim")
    )

    image_space_df = None

    if needs_image_space:
        image_space_df = report_search_objective_agreement(
            cached_samples, results
        )

    # ADDED: pick the actual best_params according to cfg.SELECT_BEST_BY.
    if cfg.SELECT_BEST_BY == "latent_mse":
        chosen = best_by_latent_mse
        chosen_name = str(chosen["name"])

    elif cfg.SELECT_BEST_BY in ("psnr", "ssim"):
        if image_space_df is None:
            raise RuntimeError(
                "SELECT_BEST_BY requires image-space metrics but "
                "image_space_df is None. This should not happen."
            )

        sort_col = "psnr_rank" if cfg.SELECT_BEST_BY == "psnr" else "ssim_rank"
        chosen_row = image_space_df.sort_values(sort_col).iloc[0]
        chosen_name = str(chosen_row["name"])

        # Pull the full candidate record (guidance/steps/eta) from
        # the original results list, matched by name.
        chosen = next(r for r in results if r["name"] == chosen_name)

    else:
        raise ValueError(
            f"Unknown cfg.SELECT_BEST_BY: {cfg.SELECT_BEST_BY!r}. "
            "Must be one of 'latent_mse', 'psnr', 'ssim'."
        )

    print("\n" + "=" * 70)
    print(f"SELECTED CONFIGURATION (cfg.SELECT_BEST_BY = '{cfg.SELECT_BEST_BY}')")
    print("=" * 70)
    print(f"Name     : {chosen['name']}")
    print(f"Guidance : {chosen['guidance']}")
    print(f"Steps    : {chosen['steps']}")
    print(f"Eta      : {chosen['eta']}")
    if chosen_name != str(best_by_latent_mse["name"]):
        print(
            f"(Differs from the latent-MSE winner: "
            f"{best_by_latent_mse['name']})"
        )
    print("=" * 70)

    return {
        "guidance": float(chosen["guidance"]),
        "steps": int(chosen["steps"]),
        "eta": float(chosen["eta"]),
        "name": chosen_name,
        "selected_by": cfg.SELECT_BEST_BY,
        "search_results": results,
    }


# ============================================================
# ADDED: LOAD BEST PARAMS FROM A PREVIOUS SEARCH (SKIP RE-SEARCHING)
# ============================================================
def load_best_params_from_cache():
    """
    Skip re-running the DDIM hyperparameter search (which redoes the
    full sampling + decode work, ~30+ min per your last log). Instead,
    read the CSVs search_best_ddim_parameters()/
    report_search_objective_agreement() already wrote to disk last
    time, and pick a candidate the same way cfg.SELECT_BEST_BY would.

    Falls back to cfg.KNOWN_BEST_CONFIGS (hardcoded from your last
    full search) if the CSVs aren't found -- e.g. first run on a
    fresh machine, or SAVE_DIR was cleared.
    """
    print("\n" + "=" * 70)
    print("USING CACHED SEARCH RESULTS (skipping DDIM search)")
    print("=" * 70)
    print(f"cfg.SELECT_BEST_BY = '{cfg.SELECT_BEST_BY}'")

    latent_csv = cfg.CACHED_SEARCH_CSV
    image_csv = cfg.CACHED_IMAGE_SPACE_CSV

    if cfg.SELECT_BEST_BY == "latent_mse":
        if latent_csv.exists():
            df = pd.read_csv(latent_csv)
            df = df.sort_values(
                "mean_latent_mse", ascending=True
            ).reset_index(drop=True)
            row = df.iloc[0]

            chosen = {
                "guidance": float(row["guidance"]),
                "steps": int(row["steps"]),
                "eta": float(row["eta"]),
                "name": str(row["name"]),
            }

            print(f"Loaded from: {latent_csv}")
        else:
            chosen = dict(cfg.KNOWN_BEST_CONFIGS["latent_mse"])
            print(
                f"{latent_csv} not found -- using hardcoded "
                "KNOWN_BEST_CONFIGS['latent_mse'] fallback."
            )

    elif cfg.SELECT_BEST_BY in ("psnr", "ssim"):
        if image_csv.exists():
            df = pd.read_csv(image_csv)
            sort_col = (
                "psnr_rank" if cfg.SELECT_BEST_BY == "psnr" else "ssim_rank"
            )
            row = df.sort_values(sort_col).iloc[0]

            chosen = {
                "guidance": float(row["guidance"]),
                "steps": int(row["steps"]),
                "eta": float(row["eta"]),
                "name": str(row["name"]),
            }

            print(f"Loaded from: {image_csv}")
        else:
            chosen = dict(cfg.KNOWN_BEST_CONFIGS[cfg.SELECT_BEST_BY])
            print(
                f"{image_csv} not found -- using hardcoded "
                f"KNOWN_BEST_CONFIGS[{cfg.SELECT_BEST_BY!r}] fallback."
            )

    else:
        raise ValueError(
            f"Unknown cfg.SELECT_BEST_BY: {cfg.SELECT_BEST_BY!r}. "
            "Must be one of 'latent_mse', 'psnr', 'ssim'."
        )

    chosen["selected_by"] = cfg.SELECT_BEST_BY
    chosen["search_results"] = None  # not recomputed when using cache

    print(f"Name     : {chosen['name']}")
    print(f"Guidance : {chosen['guidance']}")
    print(f"Steps    : {chosen['steps']}")
    print(f"Eta      : {chosen['eta']}")
    print("=" * 70)

    return chosen


# ============================================================
# CASE SPLIT
# ============================================================
def get_validation_cases():
    all_cases = sorted(
        Path(cfg.DATA_ROOT).glob("BraTS20_Training_*")
    )

    if len(all_cases) == 0:
        raise FileNotFoundError(
            f"No BraTS cases found under:\n{cfg.DATA_ROOT}"
        )

    rng = random.Random(cfg.SEED)
    rng.shuffle(all_cases)

    val_size = max(
        1,
        int(len(all_cases) * cfg.VALIDATION_SPLIT),
    )

    val_cases = all_cases[:val_size]

    return all_cases, val_cases


# ============================================================
# FINAL METRICS
# ============================================================
def compute_final_metrics(metrics):
    import scipy.stats as stats

    final = {}

    standard_keys = [
        "mse",
        "mae",
        "psnr",
        "ssim",
        "latent_mse",
        "noise_mse",
        "vae_mse",
    ]

    if LPIPS_AVAILABLE:
        standard_keys.append("lpips")

    for key in standard_keys:
        if key not in metrics:
            continue

        values = np.asarray(metrics[key], dtype=np.float64)

        if len(values) == 0:
            continue

        mean = float(np.mean(values))
        std = float(np.std(values))

        entry = {
            "mean": mean,
            "std": std,
            "n": len(values),
        }

        if len(values) >= 2:
            try:
                ci = stats.t.interval(
                    0.95,
                    len(values) - 1,
                    loc=mean,
                    scale=stats.sem(values),
                )

                entry["ci_low"] = float(ci[0])
                entry["ci_high"] = float(ci[1])

            except Exception:
                pass

        final[key] = entry

    return final


# ============================================================
# VISUALIZATION
# ============================================================
def save_visualization(
    t1ce,
    real_flair,
    synthetic_flair,
    z_t1ce,
    z_real,
    z_synth,
    case_name,
    slice_idx,
    metrics_dict,
):
    try:
        fig, axes = plt.subplots(
            2,
            4,
            figsize=(20, 10),
        )

        axes[0, 0].imshow(
            t1ce.squeeze().cpu(),
            cmap="gray",
        )
        axes[0, 0].set_title("T1ce")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(
            real_flair.squeeze().cpu(),
            cmap="gray",
        )
        axes[0, 1].set_title("Real FLAIR")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(
            synthetic_flair.squeeze().cpu(),
            cmap="gray",
        )
        axes[0, 2].set_title("Synthetic FLAIR")
        axes[0, 2].axis("off")

        diff = (
            torch.abs(
                real_flair - synthetic_flair
            )
            .squeeze()
            .cpu()
        )

        axes[0, 3].imshow(
            diff,
            cmap="hot",
        )

        axes[0, 3].set_title(
            f"Diff | MSE={metrics_dict['mse']:.5f}"
        )
        axes[0, 3].axis("off")

        axes[1, 0].imshow(
            z_t1ce[0, 0].cpu(),
            cmap="gray",
        )
        axes[1, 0].set_title("T1ce Latent")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(
            z_real[0, 0].cpu(),
            cmap="gray",
        )
        axes[1, 1].set_title("Real FLAIR Latent")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(
            z_synth[0, 0].cpu(),
            cmap="gray",
        )
        axes[1, 2].set_title("Generated FLAIR Latent")
        axes[1, 2].axis("off")

        latent_diff = (
            (z_real - z_synth)
            .abs()[0, 0]
            .cpu()
        )

        axes[1, 3].imshow(
            latent_diff,
            cmap="hot",
        )
        axes[1, 3].set_title(
            f"Latent Diff | MSE={metrics_dict['latent_mse']:.5f}"
        )
        axes[1, 3].axis("off")

        fig.suptitle(
            (
                f"{case_name} | slice {slice_idx} | "
                f"PSNR={metrics_dict['psnr']:.2f} | "
                f"SSIM={metrics_dict['ssim']:.4f}"
            ),
            fontsize=14,
        )

        plt.tight_layout()

        output_path = (
            cfg.VIS_DIR
            / f"{case_name}_slice{slice_idx}_final.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

    except Exception as e:
        print(
            f"Visualization failed for "
            f"{case_name}, slice {slice_idx}: {e}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================
def save_results(
    metrics,
    final_metrics,
    best_params,
):
    data = {
        "case": metrics["case_names"],
        "slice": metrics["slice_indices"],
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "latent_mse": metrics["latent_mse"],
        "noise_mse": metrics["noise_mse"],
        "vae_mse": metrics["vae_mse"],
    }

    if LPIPS_AVAILABLE and len(metrics["lpips"]) > 0:
        data["lpips"] = metrics["lpips"]

    df = pd.DataFrame(data)

    df.to_csv(
        cfg.SAVE_DIR / "detailed_metrics_final.csv",
        index=False,
    )

    # Save selected parameters as JSON.
    with open(
        cfg.SAVE_DIR / "selected_ddim_parameters.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "diffusion_checkpoint": cfg.DIFFUSION_CHECKPOINT,
                "scheduler": "scaled_linear_beta",
                "seed": cfg.SEED,
                "use_cfg": cfg.USE_CFG,
                "guidance": best_params["guidance"],
                "steps": best_params["steps"],
                "eta": best_params["eta"],
                "name": best_params["name"],
            },
            f,
            indent=2,
        )

    with open(
        cfg.SAVE_DIR / "summary_metrics_final.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("=" * 70 + "\n")
        f.write(
            "FINAL QUANTITATIVE METRICS\n"
        )
        f.write(
            "DIFFUSION BEST2 + SCALED_LINEAR_BETA + DDIM\n"
        )
        f.write("=" * 70 + "\n")

        f.write(
            f"Diffusion checkpoint: "
            f"{cfg.DIFFUSION_CHECKPOINT}\n"
        )
        f.write(
            f"Scheduler: scaled_linear_beta\n"
        )
        f.write(
            f"Seed: {cfg.SEED}\n"
        )
        f.write(
            f"CFG enabled: {cfg.USE_CFG}\n"
        )

        f.write(
            f"Guidance scale: "
            f"{best_params['guidance']}\n"
        )
        f.write(
            f"Inference steps: "
            f"{best_params['steps']}\n"
        )
        f.write(
            f"Eta: {best_params['eta']}\n"
        )
        f.write(
            f"Configuration: "
            f"{best_params['name']}\n"
        )

        f.write(
            f"Total evaluated slices: "
            f"{len(metrics['mse'])}\n\n"
        )

        for key, stats in final_metrics.items():
            f.write(
                f"{key.upper()}:\n"
            )

            f.write(
                f"  Mean: "
                f"{stats['mean']:.6f}\n"
            )

            f.write(
                f"  Std:  "
                f"{stats['std']:.6f}\n"
            )

            if (
                "ci_low" in stats
                and "ci_high" in stats
            ):
                f.write(
                    f"  95% CI: "
                    f"[{stats['ci_low']:.6f}, "
                    f"{stats['ci_high']:.6f}]\n"
                )

            f.write(
                f"  N: {stats['n']}\n\n"
            )

        f.write("=" * 70 + "\n")

    print("\n" + "=" * 70)
    print("FINAL QUANTITATIVE METRICS")
    print("=" * 70)

    for key, stats in final_metrics.items():
        if (
            "ci_low" in stats
            and "ci_high" in stats
        ):
            print(
                f"{key.upper():12s}: "
                f"{stats['mean']:.6f} ± "
                f"{stats['std']:.6f} "
                f"(95% CI "
                f"[{stats['ci_low']:.6f}, "
                f"{stats['ci_high']:.6f}])"
            )
        else:
            print(
                f"{key.upper():12s}: "
                f"{stats['mean']:.6f} ± "
                f"{stats['std']:.6f}"
            )

        print(
            f"{'':14s}(n={stats['n']})"
        )

    print("=" * 70)


# ============================================================
# NOISE-MSE TEST
# ============================================================
@torch.no_grad()
def compute_noise_prediction_mse(
    z_flair_scaled,
    z_t1ce_scaled,
    scheduler,
    case_seed,
    timestep_value=500,
):
    """
    Deterministic noise prediction diagnostic.

    A fixed timestep and fixed noise are used so the metric is
    reproducible between executions.
    """
    timestep = torch.tensor(
        [timestep_value],
        device=device,
        dtype=torch.long,
    )

    noise = make_fixed_noise_like(
        z_flair_scaled,
        stable_seed(
            cfg.SEED,
            "noise_mse",
            case_seed,
            timestep_value,
        ),
    )

    noisy = scheduler.add_noise(
        z_flair_scaled,
        noise,
        timestep,
    )

    noise_pred = predict_noise(
        latent=noisy,
        condition=z_t1ce_scaled,
        timestep=timestep_value,
        guidance_scale=1.0,
    )

    return torch.mean(
        (noise_pred - noise) ** 2
    ).item()


# ============================================================
# FINAL EVALUATION
# ============================================================
@torch.no_grad()
def run_evaluation_final(
    best_params,
):
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    print(
        f"Checkpoint : {cfg.DIFFUSION_CHECKPOINT}"
    )
    print(
        "Scheduler  : scaled_linear_beta"
    )
    print(
        f"Guidance   : {best_params['guidance']}"
    )
    print(
        f"Steps      : {best_params['steps']}"
    )
    print(
        f"Eta        : {best_params['eta']}"
    )
    print(
        f"Test cases : {cfg.NUM_TEST_CASES}"
    )
    print(
        f"Slices     : {list(cfg.SLICES_TO_EVAL)}"
    )

    all_cases, val_cases = get_validation_cases()

    test_cases = val_cases

    print(
        f"\nFound {len(all_cases)} total cases"
    )
    print(
        f"Validation cases: {len(val_cases)}"
    )
    print(
        f"Final cases: {len(test_cases)}"
    )

    metrics = {
        "mse": [],
        "mae": [],
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "latent_mse": [],
        "noise_mse": [],
        "vae_mse": [],
        "case_names": [],
        "slice_indices": [],
    }

    total_expected = (
        len(test_cases)
        * len(list(cfg.SLICES_TO_EVAL))
    )

    progress = tqdm(
        total=total_expected,
        desc="Evaluating slices",
    )

    for case_idx, case_path in enumerate(test_cases):
        case_name = case_path.name

        t1ce_file = (
            case_path
            / f"{case_name}_t1ce.nii"
        )

        flair_file = (
            case_path
            / f"{case_name}_flair.nii"
        )

        if (
            not t1ce_file.exists()
            or not flair_file.exists()
        ):
            print(
                f"\nSkipping {case_name}: "
                "T1ce/FLAIR file missing."
            )

            progress.update(
                len(list(cfg.SLICES_TO_EVAL))
            )
            continue

        try:
            t1ce_vol = nib.load(
                t1ce_file
            ).get_fdata()

            flair_vol = nib.load(
                flair_file
            ).get_fdata()

            seg_file = (
                case_path
                / f"{case_name}_seg.nii"
            )

            seg_vol = None

            if seg_file.exists():
                seg_vol = nib.load(
                    seg_file
                ).get_fdata()

        except Exception as e:
            print(
                f"\nSkipping {case_name}: {e}"
            )

            progress.update(
                len(list(cfg.SLICES_TO_EVAL))
            )
            continue

        for slice_idx in cfg.SLICES_TO_EVAL:
            try:
                if (
                    slice_idx >= t1ce_vol.shape[2]
                    or slice_idx >= flair_vol.shape[2]
                ):
                    progress.update(1)
                    continue

                # --------------------------------------------
                # Load preprocessed images
                # --------------------------------------------
                t1ce_tensor = (
                    load_brats_slice(
                        case_path,
                        case_name,
                        "t1ce",
                        slice_idx,
                    ).to(device)
                )

                flair_tensor = (
                    load_brats_slice(
                        case_path,
                        case_name,
                        "flair",
                        slice_idx,
                    ).to(device)
                )

                # --------------------------------------------
                # Segmentation mask
                # --------------------------------------------
                mask_slice = None

                if (
                    seg_vol is not None
                    and slice_idx < seg_vol.shape[2]
                ):
                    mask_slice = (
                        seg_vol[
                            48:208,
                            48:208,
                            slice_idx,
                        ]
                    )

                # --------------------------------------------
                # Skip near-empty slices
                # --------------------------------------------
                if (
                    torch.abs(
                        t1ce_tensor
                    ).mean()
                    < cfg.EMPTY_SLICE_THRESHOLD
                    or torch.abs(
                        flair_tensor
                    ).mean()
                    < cfg.EMPTY_SLICE_THRESHOLD
                ):
                    progress.update(1)
                    continue

                # --------------------------------------------
                # VAE encoding
                # --------------------------------------------
                z_t1ce_mu, _ = vae.encode(
                    t1ce_tensor
                )

                z_flair_mu, _ = vae.encode(
                    flair_tensor
                )

                # --------------------------------------------
                # Latent scaling
                # --------------------------------------------
                z_t1ce_scaled = (
                    z_t1ce_mu - LATENT_MEAN
                ) / (
                    LATENT_STD + 1e-5
                )

                z_flair_scaled = (
                    z_flair_mu - LATENT_MEAN
                ) / (
                    LATENT_STD + 1e-5
                )

                # --------------------------------------------
                # Deterministic noise-prediction diagnostic
                # --------------------------------------------
                case_seed = stable_seed(
                    cfg.SEED,
                    "case",
                    case_name,
                    slice_idx,
                )

                noise_mse = (
                    compute_noise_prediction_mse(
                        z_flair_scaled,
                        z_t1ce_scaled,
                        ddim_scheduler,
                        case_seed,
                    )
                )

                # --------------------------------------------
                # Deterministic DDIM initial noise
                # --------------------------------------------
                start_noise = make_fixed_noise_like(
                    z_flair_scaled,
                    case_seed,
                )

                # --------------------------------------------
                # Generate synthetic FLAIR latent
                # --------------------------------------------
                z_synthetic_scaled = (
                    ddim_sample_with_params(
                        condition=z_t1ce_scaled,
                        unet=unet,
                        scheduler=ddim_scheduler,
                        num_inference_steps=best_params[
                            "steps"
                        ],
                        guidance_scale=best_params[
                            "guidance"
                        ],
                        eta=best_params["eta"],
                        start_noise=start_noise,
                        verbose=(
                            case_idx == 0
                            and slice_idx
                            == list(cfg.SLICES_TO_EVAL)[0]
                        ),
                    )
                )

                # --------------------------------------------
                # Unscale latent
                # --------------------------------------------
                z_synthetic = (
                    z_synthetic_scaled
                    * LATENT_STD
                    + LATENT_MEAN
                )

                # --------------------------------------------
                # Latent error
                # --------------------------------------------
                latent_mse = torch.mean(
                    (
                        z_flair_mu
                        - z_synthetic
                    ) ** 2
                ).item()

                # --------------------------------------------
                # Decode
                # --------------------------------------------
                synthetic_flair = vae.decode(
                    z_synthetic
                )

                flair_recon = vae.decode(
                    z_flair_mu
                )

                # --------------------------------------------
                # VAE reconstruction error
                # --------------------------------------------
                vae_mse = torch.mean(
                    (
                        flair_tensor
                        - flair_recon
                    ) ** 2
                ).item()

                # --------------------------------------------
                # Convert [-1,1] -> [0,1]
                # --------------------------------------------
                synthetic_01 = (
                    torch.clamp(
                        synthetic_flair,
                        -1,
                        1,
                    )
                    + 1
                ) / 2.0

                real_flair_01 = (
                    flair_tensor + 1
                ) / 2.0

                t1ce_01 = (
                    t1ce_tensor + 1
                ) / 2.0

                # Clamp numerical noise.
                synthetic_01 = synthetic_01.clamp(
                    0,
                    1,
                )

                real_flair_01 = real_flair_01.clamp(
                    0,
                    1,
                )

                # --------------------------------------------
                # Image metrics
                # --------------------------------------------
                mse, mae, psnr_val, ssim_val, lpips_val = (
                    compute_metrics(
                        real_flair_01,
                        synthetic_01,
                        compute_lpips=LPIPS_AVAILABLE,
                    )
                )

                # --------------------------------------------
                # Store
                # --------------------------------------------
                metrics["mse"].append(mse)
                metrics["mae"].append(mae)
                metrics["psnr"].append(psnr_val)
                metrics["ssim"].append(ssim_val)
                metrics["latent_mse"].append(
                    latent_mse
                )
                metrics["noise_mse"].append(
                    noise_mse
                )
                metrics["vae_mse"].append(
                    vae_mse
                )

                if LPIPS_AVAILABLE:
                    metrics["lpips"].append(
                        lpips_val
                    )

                metrics["case_names"].append(
                    case_name
                )

                metrics["slice_indices"].append(
                    slice_idx
                )

                # --------------------------------------------
                # Save .pt output
                # --------------------------------------------
                save_dict = {
                    "case": case_name,
                    "slice": slice_idx,
                    "seed": int(case_seed),
                    "real_t1ce": (
                        t1ce_01
                        .squeeze()
                        .cpu()
                        .numpy()
                    ),
                    "real_flair": (
                        real_flair_01
                        .squeeze()
                        .cpu()
                        .numpy()
                    ),
                    "synthetic_flair": (
                        synthetic_01
                        .squeeze()
                        .cpu()
                        .numpy()
                    ),
                    "mask": (
                        mask_slice
                        if mask_slice is not None
                        else np.zeros_like(
                            real_flair_01
                            .squeeze()
                            .cpu()
                            .numpy()
                        )
                    ),
                    "metrics": {
                        "psnr": psnr_val,
                        "ssim": ssim_val,
                        "lpips": (
                            lpips_val
                            if LPIPS_AVAILABLE
                            else 0.0
                        ),
                        "mse": mse,
                        "mae": mae,
                        "latent_mse": latent_mse,
                        "noise_mse": noise_mse,
                        "vae_mse": vae_mse,
                    },
                    "sampling": {
                        "checkpoint": (
                            cfg.DIFFUSION_CHECKPOINT
                        ),
                        "scheduler": (
                            "scaled_linear_beta"
                        ),
                        "guidance": (
                            best_params["guidance"]
                        ),
                        "steps": (
                            best_params["steps"]
                        ),
                        "eta": (
                            best_params["eta"]
                        ),
                    },
                }

                torch.save(
                    save_dict,
                    cfg.PT_DIR
                    / f"{case_name}_slice{slice_idx}.pt",
                )

                # --------------------------------------------
                # Visualization
                # --------------------------------------------
                save_visualization(
                    t1ce=t1ce_tensor,
                    real_flair=real_flair_01,
                    synthetic_flair=synthetic_01,
                    z_t1ce=z_t1ce_mu,
                    z_real=z_flair_mu,
                    z_synth=z_synthetic,
                    case_name=case_name,
                    slice_idx=slice_idx,
                    metrics_dict={
                        "mse": mse,
                        "psnr": psnr_val,
                        "ssim": ssim_val,
                        "latent_mse": latent_mse,
                    },
                )

                # --------------------------------------------
                # Progress summary every 5 processed samples
                # --------------------------------------------
                if (
                    len(metrics["mse"]) % 5
                    == 0
                ):
                    recent_psnr = np.mean(
                        metrics["psnr"][-5:]
                    )

                    recent_ssim = np.mean(
                        metrics["ssim"][-5:]
                    )

                    print(
                        f"\nProcessed "
                        f"{len(metrics['mse'])} slices"
                    )

                    print(
                        f"  Last-5 PSNR: "
                        f"{recent_psnr:.3f}"
                    )

                    print(
                        f"  Last-5 SSIM: "
                        f"{recent_ssim:.4f}"
                    )

            except Exception as e:
                print(
                    f"\nError processing "
                    f"{case_name}, slice {slice_idx}: "
                    f"{e}"
                )

            finally:
                progress.update(1)

    progress.close()

    # --------------------------------------------
    # Final statistics
    # --------------------------------------------
    final_metrics = compute_final_metrics(
        metrics
    )

    # --------------------------------------------
    # Save
    # --------------------------------------------
    save_results(
        metrics=metrics,
        final_metrics=final_metrics,
        best_params=best_params,
    )

    return metrics, final_metrics


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("STARTING FINAL PIPELINE")
    print("=" * 70)

    # --------------------------------------------
    # ADDED: Diagnostic 1 (empty-slice filter check)
    # Runs before anything else touches the model/data pipeline.
    # --------------------------------------------
    if cfg.RUN_EMPTY_SLICE_DIAGNOSTIC:
        diagnose_empty_slice_filter()

    # --------------------------------------------
    # ADDED: cache path -- skip the search AND skip loading the
    # validation latent dataset entirely, since it's only needed
    # for the search.
    # --------------------------------------------
    if cfg.USE_CACHED_SEARCH_RESULTS:
        best_params = load_best_params_from_cache()

    elif cfg.RUN_HYPERPARAMETER_SEARCH:
        # --------------------------------------------
        # Load validation latent dataset
        # --------------------------------------------
        from src.latent_dataset import PairedLatentDataset

        print(
            f"\nLoading validation latent dataset:"
            f"\n{cfg.VAL_LATENTS_PATH}"
        )

        val_dataset = PairedLatentDataset(
            cfg.VAL_LATENTS_PATH
        )

        print(
            f"Validation latent pairs: "
            f"{len(val_dataset)}"
        )

        # ADDED: dispatch to the wide coarse-to-fine search when
        # requested; otherwise unchanged (original narrow fixed-grid
        # search, diagnostic 2 runs inside this call if enabled).
        if cfg.USE_WIDE_SEARCH:
            best_params = search_best_ddim_parameters_wide(
                val_dataset=val_dataset,
                unet=unet,
                scheduler=ddim_scheduler,
            )
        else:
            best_params = search_best_ddim_parameters(
                val_dataset=val_dataset,
                unet=unet,
                scheduler=ddim_scheduler,
            )

    else:
        best_params = {
            "guidance": (
                cfg.DEFAULT_GUIDANCE
            ),
            "steps": (
                cfg.DEFAULT_STEPS
            ),
            "eta": (
                cfg.DEFAULT_ETA
            ),
            "name": "manual_default",
        }

        print(
            "\nHyperparameter search disabled."
        )

        print(
            f"Using guidance="
            f"{best_params['guidance']}, "
            f"steps="
            f"{best_params['steps']}, "
            f"eta="
            f"{best_params['eta']}"
        )

    # --------------------------------------------
    # Final evaluation
    # --------------------------------------------
    metrics, final_metrics = (
        run_evaluation_final(
            best_params
        )
    )

    # --------------------------------------------
    # Final report
    # --------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Checkpoint: "
        f"{cfg.DIFFUSION_CHECKPOINT}"
    )

    print(
        "Scheduler: scaled_linear_beta"
    )

    print(
        f"Selected config: "
        f"{best_params['name']}"
    )

    print(
        f"Guidance: "
        f"{best_params['guidance']}"
    )

    print(
        f"Steps: "
        f"{best_params['steps']}"
    )

    print(
        f"Eta: "
        f"{best_params['eta']}"
    )

    print(
        f"Evaluated slices: "
        f"{len(metrics['mse'])}"
    )

    if "psnr" in final_metrics:
        print(
            f"Final PSNR: "
            f"{final_metrics['psnr']['mean']:.4f}"
        )

    if "ssim" in final_metrics:
        print(
            f"Final SSIM: "
            f"{final_metrics['ssim']['mean']:.4f}"
        )

    if (
        LPIPS_AVAILABLE
        and "lpips" in final_metrics
    ):
        print(
            f"Final LPIPS: "
            f"{final_metrics['lpips']['mean']:.4f}"
        )

    print(
        f"\nResults directory:\n"
        f"{cfg.SAVE_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(
            "\nFATAL ERROR:"
        )
        print(e)

        import traceback

        traceback.print_exc()