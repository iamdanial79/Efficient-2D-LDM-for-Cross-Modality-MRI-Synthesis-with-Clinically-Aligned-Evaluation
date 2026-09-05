import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import random
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
SAVE_DIR = Path(".")
SAVE_DIR.mkdir(exist_ok=True)

# ==========================================
# 1. GENERATOR ARCHITECTURE (must match training exactly)
# ==========================================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = self.conv_block(1, 64, apply_batchnorm=False)
        self.down2 = self.conv_block(64, 128)
        self.down3 = self.conv_block(128, 256)
        self.down4 = self.conv_block(256, 512)
        self.up1 = self.up_block(512, 256)
        self.up2 = self.up_block(512, 128)
        self.up3 = self.up_block(256, 64)
        self.up4 = self.up_block(128, 1, apply_batchnorm=False, act=False)

    def conv_block(self, in_c, out_c, apply_batchnorm=True):
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, padding_mode='reflect'), nn.LeakyReLU(0.2)]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_c))
        return nn.Sequential(*layers)

    def up_block(self, in_c, out_c, apply_batchnorm=True, act=True):
        layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1)]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_c))
        if act:
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4)
        u1 = torch.cat([u1, d3], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        return torch.tanh(self.up4(u3))

# ==========================================
# 2. LOAD TRAINED GENERATOR ONLY
# ==========================================
G = Generator().to(device)
G.load_state_dict(torch.load("pix2pix_generator.pth", map_location=device))
G.eval()
print("✅ Generator loaded, discriminator discarded (not needed for inference)")

# ==========================================
# 3. REBUILD THE SAME VAL SPLIT (identical seed/logic)
# ==========================================
all_cases = sorted(Path(DATA_ROOT).glob("BraTS20_Training_*"))
random.seed(42)
random.shuffle(all_cases)
val_size = int(len(all_cases) * 0.1)
val_cases = all_cases[:val_size]   # FULL validation set, not just first 5

print(f"Running on FULL validation set: {len(val_cases)} patients")

SLICE_START = 70
NUM_SLICES = 30  # matches training slice range (70-99)

# ==========================================
# 4. GENERATE + EVALUATE OVER ENTIRE VAL SET
# ==========================================
metrics = {'mse': [], 'mae': [], 'psnr': [], 'ssim': [], 'case_names': [], 'slice_indices': []}

with torch.no_grad():
    for case_path in tqdm(val_cases, desc="Generating on full val set"):
        case_name = case_path.name
        t1ce_file = case_path / f"{case_name}_t1ce.nii"
        flair_file = case_path / f"{case_name}_flair.nii"
        seg_file = case_path / f"{case_name}_seg.nii"

        if not (t1ce_file.exists() and flair_file.exists()):
            continue

        t1ce_vol = nib.load(t1ce_file).get_fdata()
        flair_vol = nib.load(flair_file).get_fdata()
        seg_vol = nib.load(seg_file).get_fdata() if seg_file.exists() else None

        for i in range(NUM_SLICES):
            slice_idx = SLICE_START + i
            if slice_idx >= t1ce_vol.shape[2]:
                continue

            t1ce_slice = t1ce_vol[48:208, 48:208, slice_idx]
            flair_slice = flair_vol[48:208, 48:208, slice_idx]

            if np.count_nonzero(t1ce_slice) < 500 or np.count_nonzero(flair_slice) < 500:
                continue

            # Same normalization used in training (min-max)
            t1ce_norm = (t1ce_slice - t1ce_slice.min()) / (t1ce_slice.max() - t1ce_slice.min() + 1e-8) * 2 - 1
            t1ce_tensor = torch.from_numpy(t1ce_norm).float().unsqueeze(0).unsqueeze(0).to(device)

            flair_norm_01 = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min() + 1e-8)

            fake_flair = G(t1ce_tensor)
            fake_flair_01 = ((fake_flair.squeeze().cpu().numpy() + 1) / 2)

            # Metrics
            mse_val = mean_squared_error(flair_norm_01, fake_flair_01)
            mae_val = np.mean(np.abs(flair_norm_01 - fake_flair_01))
            try:
                psnr_val = psnr(flair_norm_01, fake_flair_01, data_range=1.0)
            except Exception:
                psnr_val = 0.0
            try:
                ssim_val = ssim(flair_norm_01, fake_flair_01, data_range=1.0, win_size=7)
            except Exception:
                ssim_val = 0.0

            metrics['mse'].append(mse_val)
            metrics['mae'].append(mae_val)
            metrics['psnr'].append(psnr_val)
            metrics['ssim'].append(ssim_val)
            metrics['case_names'].append(case_name)
            metrics['slice_indices'].append(slice_idx)

            seg_slice = seg_vol[48:208, 48:208, slice_idx] if seg_vol is not None else np.zeros_like(flair_norm_01)

            torch.save({
                "real_t1ce": ((t1ce_tensor.squeeze().cpu().numpy() + 1) / 2),
                "real_flair": flair_norm_01,
                "synthetic_flair": fake_flair_01,
                "mask": seg_slice,
            }, SAVE_DIR / f"{case_name}_slice{slice_idx}.pt")

# ==========================================
# 5. SUMMARY STATS
# ==========================================
import scipy.stats as stats
print("\n" + "="*60)
print(f"PIX2PIX FULL VALIDATION RESULTS (n={len(metrics['mse'])})")
print("="*60)
for key in ['mse', 'mae', 'psnr', 'ssim']:
    values = np.array(metrics[key])
    mean, std = values.mean(), values.std()
    ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=stats.sem(values))
    print(f"{key.upper()}: {mean:.4f} +- {std:.4f}  (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])  n={len(values)}")