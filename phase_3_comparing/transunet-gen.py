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
import scipy.stats as stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "D:/datasets/BraTS2020_TrainingData/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
SAVE_DIR = Path("transformer_full_val_eval")
SAVE_DIR.mkdir(exist_ok=True)

# ==========================================
# 1. MODEL ARCHITECTURE (must match training exactly)
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, H * W).permute(0, 2, 1)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.ff(self.norm2(x_flat))
        return x_flat.permute(0, 2, 1).reshape(B, C, H, W)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.GELU()
        )
    def forward(self, x): return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck_conv = ConvBlock(256, 512)
        self.transformer_block = TransformerBlock(dim=512, num_heads=8)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck_conv(self.pool(e3))
        b = self.transformer_block(b)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.tanh(self.final(d1))

# ==========================================
# 2. LOAD TRAINED MODEL ONLY (no optimizer/loss needed)
# ==========================================
model = AttentionUNet().to(device)
model.load_state_dict(torch.load("transformer_baseline_outputs/transformer_unet.pth", map_location=device))
model.eval()
print("✅ Transformer model loaded")

# ==========================================
# 3. REBUILD THE SAME VAL SPLIT (identical seed/logic)
# ==========================================
all_cases = sorted(Path(DATA_ROOT).glob("BraTS20_Training_*"))
random.seed(42)
random.shuffle(all_cases)
val_size = int(len(all_cases) * 0.1)
val_cases = all_cases[:val_size]   # FULL validation set

print(f"Running on FULL validation set: {len(val_cases)} patients")

SLICE_START = 70
NUM_SLICES = 30

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

            pred_flair = model(t1ce_tensor)
            pred_flair_01 = ((pred_flair.squeeze().cpu().numpy() + 1) / 2)

            mse_val = mean_squared_error(flair_norm_01, pred_flair_01)
            mae_val = np.mean(np.abs(flair_norm_01 - pred_flair_01))
            try:
                psnr_val = psnr(flair_norm_01, pred_flair_01, data_range=1.0)
            except Exception:
                psnr_val = 0.0
            try:
                ssim_val = ssim(flair_norm_01, pred_flair_01, data_range=1.0, win_size=7)
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
                "synthetic_flair": pred_flair_01,
                "mask": seg_slice,
            }, SAVE_DIR / f"{case_name}_slice{slice_idx}.pt")

# ==========================================
# 5. SUMMARY STATS
# ==========================================
print("\n" + "="*60)
print(f"TRANSFORMER BASELINE FULL VALIDATION RESULTS (n={len(metrics['mse'])})")
print("="*60)
for key in ['mse', 'mae', 'psnr', 'ssim']:
    values = np.array(metrics[key])
    mean, std = values.mean(), values.std()
    ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=stats.sem(values))
    print(f"{key.upper()}: {mean:.4f} +- {std:.4f}  (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])  n={len(values)}")