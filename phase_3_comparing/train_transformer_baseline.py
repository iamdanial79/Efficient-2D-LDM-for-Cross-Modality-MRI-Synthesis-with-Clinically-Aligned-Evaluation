import os
import gc
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

# ==========================================
# 1. SETUP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
SAVE_DIR = Path("transformer_baseline_outputs")
SAVE_DIR.mkdir(exist_ok=True)

# ==========================================
# 2. DATASET (IDENTICAL)
# ==========================================
class BraTSPairedInMemoryDataset(Dataset):
    def __init__(self, cases, slice_start=70, num_slices=30):
        self.cases = cases
        self.slice_start = slice_start
        self.num_slices = num_slices
        print(f"Transformer: Pre-loading {len(cases)} cases into RAM...")
        self.data = []
        for case_path in tqdm(self.cases):
            case_name = case_path.name
            t1ce_vol = nib.load(case_path / f"{case_name}_t1ce.nii").get_fdata()
            flair_vol = nib.load(case_path / f"{case_name}_flair.nii").get_fdata()
            for i in range(self.num_slices):
                slice_idx = self.slice_start + i
                t1ce_slice = t1ce_vol[:, :, slice_idx][48:208, 48:208]
                flair_slice = flair_vol[:, :, slice_idx][48:208, 48:208]
                t1ce_norm = (t1ce_slice - t1ce_slice.min()) / (t1ce_slice.max() - t1ce_slice.min() + 1e-8)
                flair_norm = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min() + 1e-8)
                self.data.append({
                    "t1ce": torch.from_numpy((t1ce_norm * 2) - 1).float().unsqueeze(0),
                    "flair": torch.from_numpy((flair_norm * 2) - 1).float().unsqueeze(0)
                })
            del t1ce_vol, flair_vol; gc.collect()
        print(f"✅ Transformer Loaded {len(self.data)} pairs!")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

all_cases = sorted(Path(DATA_ROOT).glob("BraTS20_Training_*"))
random.seed(42); random.shuffle(all_cases)
val_size = int(len(all_cases) * 0.1)
val_cases = all_cases[:val_size]
train_cases = all_cases[val_size:]
test_cases = val_cases[:5]

train_dataset = BraTSPairedInMemoryDataset(train_cases)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

# ==========================================
# 3. TRANSFORMER BASELINE (Attention U-Net)
# این مدل دقیقاً مطابق شکل ۱۲ پروپوزال شماست:
# کانولوشن پایه + بلوک‌های اصلاح شده ترنسفورمری
# ==========================================
class TransformerBlock(nn.Module):
    """بلوک ترنسفورمر مکانی (Spatial Transformer Block)"""
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
        # تبدیل به توالی (Sequence)
        x_flat = x.reshape(B, C, H * W).permute(0, 2, 1) 
        
        # Self-Attention
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        x_flat = x_flat + attn_out
        
        # Feed Forward
        x_flat = x_flat + self.ff(self.norm2(x_flat))
        
        # برگشت به فرمت تصویر
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
        # Encoder
        self.enc1 = ConvBlock(1, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck با بلوک ترنسفورمر (مطابق پروپوزال)
        self.bottleneck_conv = ConvBlock(256, 512)
        self.transformer_block = TransformerBlock(dim=512, num_heads=8) 
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck + Transformer
        b = self.bottleneck_conv(self.pool(e3))
        b = self.transformer_block(b) # اعمال ترنسفورمر در گلوگاه
        
        # Decoder با Skip Connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return torch.tanh(self.final(d1))

# ==========================================
# 4. TRAINING LOOP (Much faster than Diffusion)
# ==========================================
model = AttentionUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
L1 = nn.L1Loss()

epochs = 50
print(f"🚀 Training Transformer Baseline (Table 2) for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        t1ce = batch["t1ce"].to(device)
        flair = batch["flair"].to(device)
        
        pred_flair = model(t1ce)
        loss = L1(pred_flair, flair)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_description(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")
        
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()

torch.save(model.state_dict(), SAVE_DIR / "transformer_unet.pth")

# ==========================================
# 5. GENERATE ON VALIDATION SET (Instant)
# ==========================================
print("\nGenerating on the EXACT same 5 validation patients...")
model.eval()

with torch.no_grad():
    for case_path in test_cases:
        case_name = case_path.name
        
        t1ce_vol = nib.load(case_path / f"{case_name}_t1ce.nii").get_fdata()
        flair_vol = nib.load(case_path / f"{case_name}_flair.nii").get_fdata()
        seg_vol = nib.load(case_path / f"{case_name}_seg.nii").get_fdata()
        
        slice_idx = 80
        t1ce_slice = t1ce_vol[48:208, 48:208, slice_idx]
        flair_slice = flair_vol[48:208, 48:208, slice_idx]
        
        t1ce_norm = (t1ce_slice - t1ce_slice.min()) / (t1ce_slice.max() - t1ce_slice.min() + 1e-8) * 2 - 1
        t1ce_tensor = torch.from_numpy(t1ce_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        
        flair_norm = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min() + 1e-8)
        seg_slice = seg_vol[48:208, 48:208, slice_idx]
        
        # Forward pass (1 step, extremely fast)
        fake_flair = model(t1ce_tensor)
        fake_flair_01 = (fake_flair.squeeze().cpu() + 1) / 2
        
        torch.save({
            "real_t1ce": ((t1ce_tensor.squeeze().cpu() + 1) / 2).numpy(),
            "real_flair": flair_norm,
            "synthetic_flair": fake_flair_01.numpy(),
            "mask": seg_slice
        }, SAVE_DIR / f"{case_name}_slice{slice_idx}.pt")

print(f"✅ Transformer Done!")