import os
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR

# ==========================================
# 1. IMPORTS
# ==========================================
from generative.networks.nets import DiffusionModelUNet
from src.latent_dataset import PairedLatentDataset
from generative.networks.schedulers import DDPMScheduler
import pandas as pd

# ==========================================
# CONFIG & SETUP
# ==========================================
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda")
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
save_dir = Path("diffusion")
save_dir.mkdir(parents=True, exist_ok=True)

# Diffusion Config
num_train_timesteps = 1000
batch_size = 64
lr = 1e-4
num_epochs = 50
cfg_dropout_prob = 0.1  

# ==========================================
# FIXED EMA CLASS
# ==========================================
class EMA:
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def apply_shadow(self):
        self.backup = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

# ==========================================
# BETA SCHEDULE
# ==========================================
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    schedule="scaled_linear_beta", # for the test this code once runned with linear_beta and it results where saved 
    beta_start=0.00085,
    beta_end=0.012,
)

alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

# ==========================================
# DATA
# ==========================================
train_dataset = PairedLatentDataset("latent_pairs/train_pairs_scaled.pt")
val_dataset = PairedLatentDataset("latent_pairs/val_pairs_scaled.pt")
sample = train_dataset[0]

print("Condition:", sample["condition"].shape)
print("Target   :", sample["target"].shape)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

# ==========================================
# MODEL (Corrected in_channels and out_channels)
# ==========================================
# Latent shape is (8, 40, 40).
# in_channels = 8 (noisy flair) + 8 (t1ce condition) = 16
# out_channels = 8 (predicted noise)
unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=16,
    out_channels=8,
    num_channels=(64, 128, 256),
    attention_levels=(False, True, True),
    num_head_channels=(32, 64, 128),
    num_res_blocks=3,
).to(device)
with torch.no_grad():
    x = torch.randn(2, 16, 40, 40, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    y = unet(x, t)
    print("UNet output:", y.shape)
optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-5)
lr_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6,
)
scaler = GradScaler(enabled=True)

best_val_loss = float("inf")
ema = EMA(unet, decay=0.999)
epochs_no_improve = 0
patience = 5  # Evaluated every 5 epochs
history = []

print(f"🚀 Starting Diffusion Training on Per-Channel Latents for {num_epochs} epochs...")

# ==========================================
# TRAIN & VAL LOOP
# ==========================================
for epoch in range(num_epochs):
    unet.train()
    train_loss_sum = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in pbar:
        # FIX 1: Replaced batch["z_t1ce"] and batch["z_flair"] with dataset keys
        z_t1ce = batch["condition"].to(device)
        z_flair = batch["target"].to(device)

        optimizer.zero_grad()

        with autocast("cuda"):
            noise = torch.randn_like(z_flair)
            timesteps = torch.randint(0, num_train_timesteps, (z_flair.shape[0],), device=device).long()

            alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

            noisy_z_flair = sqrt_alpha_t * z_flair + sqrt_one_minus_alpha_t * noise

            # Classifier-free guidance dropout (per-sample)
            cfg_mask = torch.rand(z_t1ce.shape[0], device=device) < cfg_dropout_prob
            z_t1ce_input = torch.where(cfg_mask.view(-1, 1, 1, 1), torch.zeros_like(z_t1ce), z_t1ce)

            model_input = torch.cat([noisy_z_flair, z_t1ce_input], dim=1)

            noise_pred = unet(model_input, timesteps)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Watch for NaNs or exploding gradients
        if not torch.isfinite(loss):
            raise RuntimeError("Training diverged: loss became NaN or Inf.")

        scaler.scale(loss).backward()
        
        # Gradient clipping at 1.0 before optimizer step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        ema.update()

        train_loss_sum += loss.item()
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

    avg_train_loss = train_loss_sum / len(train_loader)
    lr_scheduler.step()

    # Validation every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        unet.eval()
        val_loss_sum = 0.0

        # FIX 3: Apply EMA shadow weights for validation and saving
        ema.apply_shadow()

        with torch.no_grad():
            for batch in val_loader:
                z_t1ce = batch["condition"].to(device)
                z_flair = batch["target"].to(device)
                
                noise = torch.randn_like(z_flair)
                timesteps = torch.randint(0, num_train_timesteps, (z_flair.shape[0],), device=device).long()

                alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                noisy_z_flair = torch.sqrt(alpha_t) * z_flair + torch.sqrt(1.0 - alpha_t) * noise

                # No CFG dropout during validation
                model_input = torch.cat([noisy_z_flair, z_t1ce], dim=1)
                
                with autocast("cuda"):
                    noise_pred = unet(model_input, timesteps)
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    
                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        history.append(
                            {
                                "epoch": epoch + 1,
                                "train_loss": avg_train_loss,
                                "val_loss": avg_val_loss,
                                "lr": current_lr,
                            }
                        )

        pd.DataFrame(history).to_csv(
            save_dir / "training_log.csv",
            index=False,
        )

        tqdm.write(f"✅ Epoch {epoch+1} | Train MSE: {avg_train_loss:.5f} | Val MSE (EMA): {avg_val_loss:.5f} | LR: {current_lr:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # FIX 4: Save FULL checkpoint (EMA weights are currently applied to the model)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": unet.state_dict(),
                "ema_state_dict": ema.shadow,
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
            torch.save(checkpoint, save_dir / "diffusion_best.pth")
            tqdm.write("🎉 Saved new best diffusion model (EMA weights + full checkpoint)!")
            
            epochs_no_improve = 0
        else:
            epochs_no_improve += 5 
            tqdm.write(f"⏳ No improvement for {epochs_no_improve} epochs. Best Val Loss: {best_val_loss:.5f}")

        # FIX 3: Restore raw weights to continue training properly

        ema.restore()
        torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                save_dir / "diffusion_last.pth",
                )
        unet.train()

    if epochs_no_improve >= patience:
        tqdm.write(f"⏹️ Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.5f}")
        break

print("\n🔥 Diffusion Training Complete! Next step: Evaluate via reverse diffusion + VAE decoding.")