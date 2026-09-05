import os
import gc
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import numpy as np
import psutil
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. HARDWARE SAFETY & SPLIT SETUP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

DATA_ROOT = "/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
SAVE_DIR = Path("pix2pix_baseline")
SAVE_DIR.mkdir(exist_ok=True)

# ==========================================
# 2. EXACT MATCH TO YOUR LDM SPLIT LOGIC
# ==========================================
print("Calculating exact Train/Val split to match LDM...")
all_cases = sorted(Path(DATA_ROOT).glob("BraTS20_Training_*"))

random.seed(42)  # YOUR EXACT SEED
random.shuffle(all_cases)

val_split = 0.1
val_size = int(len(all_cases) * val_split)

val_cases = all_cases[:val_size]
train_cases = all_cases[val_size:]

# The 5 patients your LDM was tested on:
test_cases = val_cases[:5]

print(f"Total patients: {len(all_cases)}")
print(f"Train patients: {len(train_cases)}")
print(f"Val patients: {len(val_cases)} (Testing on first 5: {[p.name for p in test_cases]})")

# ==========================================
# 3. IN-MEMORY DATASET (Pre-loads into RAM)
# ==========================================
class BraTSPairedInMemoryDataset(Dataset):
    """Loads T1ce and FLAIR pairs into RAM safely."""
    def __init__(self, cases, slice_start=70, num_slices=30):
        self.cases = cases
        self.slice_start = slice_start
        self.num_slices = num_slices
        
        print(f"Pre-loading {len(cases)} cases (T1ce+FLAIR) into RAM...")
        self.data = []

        for case_path in tqdm(self.cases, desc="Loading data"):
            case_name = case_path.name
            t1ce_path = case_path / f"{case_name}_t1ce.nii"
            flair_path = case_path / f"{case_name}_flair.nii"
            
            # Read both volumes
            t1ce_vol = nib.load(t1ce_path).get_fdata()
            flair_vol = nib.load(flair_path).get_fdata()
            
            for i in range(self.num_slices):
                slice_idx = self.slice_start + i
                
                # Process T1ce
                t1ce_slice = t1ce_vol[:, :, slice_idx]
                t1ce_crop = t1ce_slice[48:208, 48:208]  # Center crop 160x160
                t1ce_norm = (t1ce_crop - t1ce_crop.min()) / (t1ce_crop.max() - t1ce_crop.min() + 1e-8)
                t1ce_tensor = torch.from_numpy((t1ce_norm * 2) - 1).float().unsqueeze(0)
                
                # Process FLAIR
                flair_slice = flair_vol[:, :, slice_idx]
                flair_crop = flair_slice[48:208, 48:208]
                flair_norm = (flair_crop - flair_crop.min()) / (flair_crop.max() - flair_crop.min() + 1e-8)
                flair_tensor = torch.from_numpy((flair_norm * 2) - 1).float().unsqueeze(0)
                
                self.data.append({
                    "t1ce": t1ce_tensor,
                    "flair": flair_tensor,
                    "case": case_name,
                    "slice_idx": slice_idx
                })
            
            # Free numpy arrays immediately
            del t1ce_vol, flair_vol
            gc.collect()
                
        print(f"✅ Finished loading {len(self.data)} pairs!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Train on the TRAINING split only (in-memory)
train_dataset = BraTSPairedInMemoryDataset(train_cases, slice_start=70, num_slices=30)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=False)

# ==========================================
# 4. NETWORKS (Unchanged)
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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1)
        )
    
    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

# ==========================================
# 5. TRAINING LOOP
# ==========================================
G = Generator().to(device)
D = Discriminator().to(device)
opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
BCE = nn.BCEWithLogitsLoss()
L1 = nn.L1Loss()

epochs = 50
step = 0
current_batch_size = 2

print(f"\n🚀 Starting FAIR Training on {len(train_cases)} patients for {epochs} epochs...")
print(f"   Total training samples: {len(train_dataset)}")

for epoch in range(epochs):
    G.train()
    D.train()
    loop = tqdm(train_loader, leave=True)
    
    for batch in loop:
        step += 1
        
        # Memory safety check
        if step % 50 == 0:
            if psutil.virtual_memory().percent > 85:
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(3)

        # Extract data from dictionary batch
        t1ce = batch["t1ce"].to(device)
        flair_real = batch["flair"].to(device)
        
        try:
            # Train Discriminator
            fake_flair = G(t1ce)
            D_real = D(t1ce, flair_real)
            D_fake = D(t1ce, fake_flair.detach())
            loss_D = (BCE(D_real, torch.ones_like(D_real)) + BCE(D_fake, torch.zeros_like(D_fake))) * 0.5
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
            
            # Train Generator
            D_fake = D(t1ce, fake_flair)
            loss_G = BCE(D_fake, torch.ones_like(D_fake)) + (L1(fake_flair, flair_real) * 100)
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                if current_batch_size != 1:
                    current_batch_size = 1
                    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
                    print(f"⚠️ Reduced batch size to 1 due to OOM")
                time.sleep(2)
                continue
            else:
                raise e

        loop.set_description(f"Epoch [{epoch+1}/{epochs}] D: {loss_D.item():.3f} G: {loss_G.item():.1f}")

    # End of epoch cleanup
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

torch.save(G.state_dict(), SAVE_DIR / "pix2pix_generator.pth")
print("\n✅ Training Complete!")

# ==========================================
# 6. GENERATE ON EXACT VALIDATION SET
# ==========================================
print("\nGenerating on the EXACT same 5 validation patients as LDM...")
G.eval()

with torch.no_grad():
    for case_path in test_cases:
        case_name = case_path.name
        
        t1ce_vol = nib.load(case_path / f"{case_name}_t1ce.nii").get_fdata()
        flair_vol = nib.load(case_path / f"{case_name}_flair.nii").get_fdata()
        seg_vol = nib.load(case_path / f"{case_name}_seg.nii").get_fdata()
        
        slice_idx = 80  # YOUR EXACT TEST SLICE
        
        t1ce_slice = t1ce_vol[48:208, 48:208, slice_idx]
        flair_slice = flair_vol[48:208, 48:208, slice_idx]
        
        t1ce_norm = (t1ce_slice - t1ce_slice.min()) / (t1ce_slice.max() - t1ce_slice.min() + 1e-8) * 2 - 1
        t1ce_tensor = torch.from_numpy(t1ce_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        
        flair_norm = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min() + 1e-8)
        flair_tensor_01 = torch.from_numpy(flair_norm).float().unsqueeze(0)
        
        seg_tensor = torch.from_numpy(seg_vol[48:208, 48:208, slice_idx]).float().unsqueeze(0)
        
        fake_flair = G(t1ce_tensor)
        fake_flair_01 = (fake_flair.squeeze().cpu() + 1) / 2
        
        torch.save({
            "real_t1ce": ((t1ce_tensor.squeeze().cpu() + 1) / 2).numpy(),
            "real_flair": flair_tensor_01.squeeze().numpy(),
            "synthetic_flair": fake_flair_01.numpy(),
            "mask": seg_tensor.squeeze().numpy()
        }, SAVE_DIR / f"{case_name}_slice{slice_idx}.pt")
        
        print(f"  Saved: {case_name}_slice{slice_idx}.pt")

print(f"\n✅ Done! Fair comparison files saved to '{SAVE_DIR}'.")