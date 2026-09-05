# train_autoencoder_mixed.py
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from pytorch_msssim import ssim
from generative.networks.nets import AutoencoderKL
from src.brats_cross_modal_dataset import BraTSInMemoryDataset
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda")
save_dir = Path("results/vae_mixed")
save_dir.mkdir(parents=True, exist_ok=True)

num_epochs = 100
batch_size = 16      
lr = 1e-4

DATA_ROOT = "/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# --- DATA (The Unpaired Concatenation Strategy) ---
print("Loading T1ce...")
dataset_t1ce_train = BraTSInMemoryDataset(DATA_ROOT, modality="t1ce", is_train=True)
print("Loading FLAIR...")
dataset_flair_train = BraTSInMemoryDataset(DATA_ROOT, modality="flair", is_train=True)

# قاطی کردن دو مودالیته برای ایجاد فضای نهفته مشترک
train_dataset = ConcatDataset([dataset_t1ce_train, dataset_flair_train])

# ولیدیشن هم باید هر دو را ببیند تا ارزیابی دقیق باشد
dataset_t1ce_val = BraTSInMemoryDataset(DATA_ROOT, modality="t1ce", is_train=False)
dataset_flair_val = BraTSInMemoryDataset(DATA_ROOT, modality="flair", is_train=False)
val_dataset = ConcatDataset([dataset_t1ce_val, dataset_flair_val])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# --- MODEL ---
autoencoder = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(64, 128, 256),  # Deep network: 2 downsamples -> 40x40
    latent_channels=8,            # Standard baseline, upgrade only if needed
    num_res_blocks=2,
    attention_levels=(False, False, True),
    norm_num_groups=32,
).to(device)

# ==========================================
# CRITICAL VERIFICATION STEP (As Recommended)
# ==========================================
print("\n" + "="*50)
print("VERIFYING LATENT SPACE DIMENSIONS...")
with torch.no_grad():
    dummy_input = torch.randn(1, 1, 160, 160).to(device)
    # We only pass it through the encoder to see the compressed shape
    _,z_mu, z_log_var = autoencoder(dummy_input)
    latent_shape = z_mu.shape
    
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Latent Shape: {latent_shape}")
    
    # Calculate actual compression factor
    h_in, w_in = 160, 160
    h_lat, w_lat = latent_shape[2], latent_shape[3]
    spatial_compression = (h_in / h_lat)
    print(f"Spatial Compression Factor: {spatial_compression}x")
    
    if h_lat == 40 and w_lat == 40:
        print("✅ SUCCESS: Latent space is 40x40. Perfect for Medical LDM!")
    else:
        print("⚠️ WARNING: Latent space is NOT 40x40. Investigate architecture!")
print("="*50 + "\n")
# ==========================================

optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6,
)
scaler = GradScaler(enabled=True)

history = []

best_checkpoint = save_dir / "autoencoder_best.pth"
latest_checkpoint = save_dir / "autoencoder_latest.pth"

start_epoch = 0
best_val_loss = float("inf")

# Prefer latest checkpoint
if latest_checkpoint.exists():
    checkpoint_path = latest_checkpoint

# Otherwise fall back to best checkpoint
elif best_checkpoint.exists():
    checkpoint_path = best_checkpoint

# Otherwise train from scratch
else:
    checkpoint_path = None

if checkpoint_path is not None:
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    autoencoder.load_state_dict(checkpoint["model"])

    # Optimizer (if available)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Scheduler (if available)
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    # AMP scaler (if available)
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"✅ Resuming from epoch {start_epoch}")
    print(f"✅ Best validation loss: {best_val_loss:.6f}")

else:
    print("No checkpoint found. Starting from scratch.")



print(f"\n🚀 Starting MIXED MODALITY VAE Training...")
log_path = save_dir / "training_log.csv"

if log_path.exists():
    history = pd.read_csv(log_path).to_dict("records")
else:
    history = []
# --- TRAIN LOOP ---
for epoch in range(start_epoch,num_epochs):
    autoencoder.train()
    train_loss_sum = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in pbar:
        images = batch["source"].to(device) 
        optimizer.zero_grad()

        with autocast("cuda"):
            reconstruction, z_mu, z_log_var = autoencoder(images)
            
            loss_l1 = F.l1_loss(reconstruction, images)
            loss_ms_ssim = 1 - ssim(reconstruction, images, data_range=2.0, size_average=True)
            recon_loss = 0.9 * loss_l1 + 0.1 * loss_ms_ssim

            kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp(), dim=[1, 2, 3])
            latent_size = z_mu.numel() // images.shape[0]
            kl_loss = kl_loss.mean() / latent_size 

            loss = recon_loss + (1e-5 * kl_loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        train_loss_sum += loss.item()
        pbar.set_postfix(L1=f"{loss_l1.item():.4f}", MSSSIM=f"{loss_ms_ssim.item():.4f}")

    scheduler.step()
    train_loss = train_loss_sum / len(train_loader)
    torch.save({
    "epoch": epoch,
    "best_val_loss": best_val_loss,
    "model": autoencoder.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict(),
                }, latest_checkpoint)

    # --- VALIDATION & VISUALIZATION ---
    if (epoch + 1) % 5 == 0 or epoch == 0:
        autoencoder.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch_idx,batch in enumerate(val_loader):
                images = batch["source"].to(device)

                reconstruction, z_mu, z_log_var = autoencoder(images)

                l1 = F.l1_loss(reconstruction, images)
                ms_ssim_val = 1 - ssim(reconstruction, images, data_range=2.0, size_average=True)
                recon_loss = 0.9 * l1 + 0.1 * ms_ssim_val

                kl_loss = -0.5 * torch.sum(
                    1 + z_log_var - z_mu.pow(2) - z_log_var.exp(),
                    dim=[1,2,3]
                )
                latent_size = z_mu.numel() // images.shape[0]
                kl_loss = kl_loss.mean() / latent_size

                val_loss = recon_loss + (1e-5 * kl_loss)
                val_loss_sum += val_loss.item()

                if batch_idx == 0: 
                    case_name = batch["case"][0]
                    slice_idx = batch["slice_idx"][0].item()
                    
                    img_orig = images[0, 0].cpu().numpy()
                    img_recon = reconstruction[0, 0].cpu().numpy()
                    
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(img_orig, cmap='gray', vmin=-1, vmax=1)
                    axes[0].set_title(f"Original\n{case_name} - Slice {slice_idx}")
                    axes[0].axis('off')
                    
                    axes[1].imshow(img_recon, cmap='gray', vmin=-1, vmax=1)
                    axes[1].set_title("VAE Reconstruction")
                    axes[1].axis('off')
                    
                    plt.savefig(save_dir / f"epoch{epoch+1}_{case_name}_s{slice_idx}.png")
                    plt.close()

        avg_val_loss = val_loss_sum / len(val_loader)
        print(f"\n✅ Epoch {epoch+1} | Val Loss: {avg_val_loss:.6f}\n")

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": avg_val_loss, "lr": scheduler.get_last_lr()[0]})
        pd.DataFrame(history).to_csv(save_dir / "training_log.csv", index=False)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            torch.save({
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "model": autoencoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            }, best_checkpoint)

            print(f"💾 Saved new best model (Val Loss = {best_val_loss:.6f})")

print("\n🎉 Mixed VAE Training Finished!")
torch.save({
    "epoch": epoch,
    "best_val_loss": best_val_loss,
    "model": autoencoder.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict(),
}, save_dir / "autoencoder_final.pth")