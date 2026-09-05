import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from generative.networks.nets import AutoencoderKL
from src.brats_cross_modal_dataset import BraTSPairedInMemoryDataset

DATA_ROOT = ""
device = torch.device("cuda")
batch_size = 64
model_path = "results/vae_mixed/autoencoder_best.pth"
save_root = Path("latent_pairs")
save_root.mkdir(exist_ok=True)

# --- MODEL (exact same arch + checkpoint used during VAE training) ---
autoencoder = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(64, 128, 256),
    latent_channels=8,
    num_res_blocks=2,
    attention_levels=(False, False, True),
    norm_num_groups=32,
).to(device)
checkpoint = torch.load(model_path, map_location=device)

# print(type(checkpoint))

# if isinstance(checkpoint, dict):
#     print(checkpoint.keys())
autoencoder.load_state_dict(checkpoint["model"])
autoencoder.eval()
with torch.no_grad():
    test = autoencoder.encode(torch.randn(1, 1, 160, 160).to(device))
    print(type(test))
    print(len(test))
print("✅ Autoencoder loaded")
with torch.no_grad():
    x = torch.randn(1, 1, 160, 160).to(device)
    recon, _, _ = autoencoder(x)

print("Input :", x.shape)
print("Recon :", recon.shape)
# --- DATASETS (no augmentation, no shuffling — deterministic) ---
train_dataset = BraTSPairedInMemoryDataset(DATA_ROOT, is_train=True)
val_dataset   = BraTSPairedInMemoryDataset(DATA_ROOT, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

# --- ENCODING / NORMALIZATION ---
def encode(loader, output_filename, is_train=True, stats=None):
    all_z_flair, all_z_t1ce, all_pairs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Processing {output_filename.name}"):
            t1ce_imgs = batch["t1ce"].to(device)
            flair_imgs = batch["flair"].to(device)

            # Use ENCODER MEAN (z_mu) ONLY — do NOT sample from the latent
            z_mu_t1ce, _ = autoencoder.encode(t1ce_imgs)
            z_mu_flair, _ = autoencoder.encode(flair_imgs)

            z_t1ce  = z_mu_t1ce.cpu()
            z_flair = z_mu_flair.cpu()
            if len(all_pairs) == 0:
                print("Latent batch shape:", z_t1ce.shape)
            if is_train:
                all_z_flair.append(z_flair)
                all_z_t1ce.append(z_t1ce)

            for i in range(t1ce_imgs.shape[0]):
                all_pairs.append({
                    "z_t1ce":  z_t1ce[i],
                    "z_flair": z_flair[i],
                    "case":    batch["case"][i],
                    "slice_idx": int(batch["slice_idx"][i]),
                })

    if len(all_pairs) == 0:
        print("❌ ERROR: No data was loaded at all! Check your dataset.")
        return None

    # ---- Shared per-channel mean/std across BOTH modalities (train only) ----
    if is_train:
        z_f = torch.cat(all_z_flair, dim=0)
        z_t = torch.cat(all_z_t1ce,  dim=0)
        global_latents = torch.cat([z_f, z_t], dim=0)   # both modalities together

        mean = global_latents.mean(dim=[0, 2, 3])
        std  = global_latents.std(dim=[0, 2, 3])

        stats = {"mean": mean, "std": std}
        torch.save(stats, save_root / "latent_stats.pt")
        print(f"✅ Saved shared per-channel scaling stats to latent_stats.pt")

    mean, std = stats["mean"], stats["std"]
    mean_scale = mean.view(-1, 1, 1)
    std_scale = std.view(-1, 1, 1)

    # ---- Normalize BOTH T1ce and FLAIR with the SAME stats ----
    for pair in all_pairs:
        pair["z_t1ce"]  = (pair["z_t1ce"]  - mean_scale) / (std_scale + 1e-5)
        pair["z_flair"] = (pair["z_flair"] - mean_scale) / (std_scale + 1e-5)

    torch.save(all_pairs, output_filename)
    print(f"✅ Saved {len(all_pairs)} normalized pairs to {output_filename}")

    # ---------------- Sanity-check metrics ----------------
    print("\n" + "=" * 60)
    print(f"🩺 Sanity-check metrics — {output_filename.name}")
    print("=" * 60)

    z_t1ce_all  = torch.stack([p["z_t1ce"]  for p in all_pairs], dim=0)
    z_flair_all = torch.stack([p["z_flair"] for p in all_pairs], dim=0)
    z_all       = torch.cat([z_t1ce_all, z_flair_all], dim=0)

    print(f"Latent shape (per sample): {tuple(all_pairs[0]['z_t1ce'].shape)}")
    print(f"Global mean : {z_all.mean().item():+.6f}   (≈ 0 expected)")
    print(f"Global std  : {z_all.std().item():+.6f}    (≈ 1 expected)")
    print(f"Global min  : {z_all.min().item():+.6f}    (typically ≈ -3..-5)")
    print(f"Global max  : {z_all.max().item():+.6f}    (typically ≈ +3..+5)")

    per_ch_mean = z_all.mean(dim=[0, 2, 3])
    per_ch_std  = z_all.std(dim=[0, 2, 3])
    print(f"\nPer-channel mean: {per_ch_mean.tolist()}")
    print(f"Per-channel std : {per_ch_std.tolist()}")

    # Optional: histogram of latent values to verify ≈ Gaussian
    try:
        plt.figure(figsize=(6, 4))
        plt.hist(z_all.flatten().numpy(), bins=200, density=True, color="steelblue", alpha=0.8)
        plt.title(f"Latent value distribution — {output_filename.name}")
        plt.xlabel("Latent value"); plt.ylabel("Density")
        plt.tight_layout()
        hist_path = save_root / f"latent_hist_{output_filename.stem}.png"
        plt.savefig(hist_path, dpi=120); plt.close()
        print(f"✅ Saved histogram → {hist_path.name}")
    except Exception as e:
        print(f"(Skipping histogram: {e})")

    print("=" * 60 + "\n")
    return stats

# --- RUN ---
print("\nProcessing Training Set...")
stats = encode(train_loader, save_root / "train_pairs_scaled.pt", is_train=True)

print("\nProcessing Validation Set...")
encode(val_loader, save_root / "val_pairs_scaled.pt", is_train=False, stats=stats)

print("\n🎉 Deterministic latent extraction complete!")