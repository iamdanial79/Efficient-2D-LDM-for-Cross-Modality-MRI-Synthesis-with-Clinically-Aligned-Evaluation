from pathlib import Path
import random

import torch
from torch.utils.data import Dataset
import torchio as tio
import numpy as np
import nibabel as nib 
import random

class BraTSCrossModalDataset(Dataset):

    def __init__(
        self,
        root_dir,
        source_modality="flair",
        target_modality="t1ce",
        transform=None,
        is_train=True,
        val_split=0.1,
        seed=42,
        slice_start=70,
        num_slices=30,
    ):

        self.root_dir = Path(root_dir)

        self.cases = sorted(
            self.root_dir.glob("BraTS20_Training_*")
        )

        print(f"Found {len(self.cases)} cases")

        random.seed(seed)
        random.shuffle(self.cases)

        val_size = int(len(self.cases) * val_split)

        if is_train:
            self.cases = self.cases[val_size:]
        else:
            self.cases = self.cases[:val_size]

        print(
            f"{'Train' if is_train else 'Validation'} "
            f"cases: {len(self.cases)}"
        )

        self.source_modality = source_modality.lower()
        self.target_modality = target_modality.lower()

        self.slice_start = slice_start
        self.num_slices = num_slices

        self.transform = transform or tio.Compose([
            tio.RescaleIntensity(
                out_min_max=(-1, 1)
            ),
            tio.CropOrPad((160, 160, 1)),
        ])

    def __len__(self):

        return len(self.cases) * self.num_slices

    def __getitem__(self, idx):

        case_idx = idx // self.num_slices
        slice_offset = idx % self.num_slices

        case_path = self.cases[case_idx]
        case_name = case_path.name

        slice_idx = self.slice_start + slice_offset

        source_path = (
            case_path /
            f"{case_name}_{self.source_modality}.nii"
        )

        source_volume = tio.ScalarImage(source_path)

        # NEW: If source and target are the same, don't read the file twice!
        if self.source_modality == self.target_modality:
            target_volume = source_volume
        else:
            target_path = (
                case_path /
                f"{case_name}_{self.target_modality}.nii"
            )
            target_volume = tio.ScalarImage(target_path)

        source_slice = source_volume.data[:, :, :, slice_idx:slice_idx + 1]
        target_slice = target_volume.data[:, :, :, slice_idx:slice_idx + 1]
        
        source_subject = tio.Subject(
            image=tio.ScalarImage(
                tensor=source_slice
            )
        )

        target_subject = tio.Subject(
            image=tio.ScalarImage(
                tensor=target_slice
            )
        )

        source_subject = self.transform(
            source_subject
        )

        target_subject = self.transform(
            target_subject
        )

        source_image = (
            source_subject.image.data
            .squeeze(-1)
        )

        target_image = (
            target_subject.image.data
            .squeeze(-1)
        )

        return {
            "source": source_image,
            "target": target_image,
            "case": case_name,
            "slice_idx": slice_idx,
        }
    



class BraTSInMemoryDataset(Dataset):
    def __init__(self, root_dir, modality="flair", is_train=True, val_split=0.1, seed=42, slice_start=70, num_slices=30):
        self.root_dir = Path(root_dir)
        self.cases = sorted(self.root_dir.glob("BraTS20_Training_*"))
        
        rng = random.Random(seed)
        rng.shuffle(self.cases)
        
        val_size = int(len(self.cases) * val_split)
        self.cases = self.cases[val_size:] if is_train else self.cases[:val_size]
        
        self.transform = None
        if is_train:
            self.transform = tio.Compose([
                tio.RandomAffine(
                    scales=(0.95, 1.05),
                    degrees=5,
                    translation=3,
                    p=0.5,
                ),
                tio.RandomGamma(p=0.3),
                tio.RandomNoise(p=0.2),
            ])
            
        print(f"{'Pre-loading Train' if is_train else 'Pre-loading Val'} {modality} into RAM (Dynamic Augmentation Pipeline)...")
        
        
        self.data = []

        for case_path in self.cases:
            case_name = case_path.name
            img_path = case_path / f"{case_name}_{modality}.nii"
            
            if not img_path.exists():
                continue
                
            # 1. Load -> float32
            vol_nii = nib.load(img_path).get_fdata().astype(np.float32)
            
            for i in range(num_slices):
                slice_idx = slice_start + i
                slice_2d = vol_nii[:, :, slice_idx] 
                
                # 2. Center Crop (Pure Numpy)
                crop = slice_2d[48:208, 48:208] 
                
                # 3. Skip Empty Slices
                if np.count_nonzero(crop) < 500:
                    continue
                
                # 4. Percentile Normalization (Robust)
                low = np.percentile(crop, 1)
                high = np.percentile(crop, 99)
                crop = np.clip(crop, low, high)
                crop = (crop - low) / (high - low + 1e-8)
                crop = crop * 2 - 1  # Map to [-1, 1]
                
                # 5. To Tensor (No Augmentation Here!)
                tensor = torch.from_numpy(crop).float().unsqueeze(0) # Shape: (1, 160, 160)
                
                self.data.append({"source": tensor,
                    "case": case_name,
                    "slice_idx": slice_idx,})
                
        print(f"✅ Finished loading {len(self.data)} ORIGINAL {modality} slices into RAM!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tensor = sample["source"]

        if self.transform is not None:
            tensor_4d = tensor.unsqueeze(-1)
            img_tio = tio.ScalarImage(tensor=tensor_4d)
            tensor = self.transform(img_tio).data.squeeze(-1)

        return {
            "source": tensor,
            "case": sample["case"],
            "slice_idx": sample["slice_idx"],
        }
    


class BraTSPairedInMemoryDataset(Dataset):
    """Loads T1ce and FLAIR pairs into RAM safely.
    
    NOTE: Normalization now matches BraTSInMemoryDataset (percentile-clipped),
    since that is the distribution the VAE was actually trained on.
    """
    def __init__(self, root_dir, is_train=True, val_split=0.1, seed=42, slice_start=70, num_slices=30):
        self.root_dir = Path(root_dir)
        self.cases = sorted(self.root_dir.glob("BraTS20_Training_*"))
        
        random.seed(seed)
        random.shuffle(self.cases)
        val_size = int(len(self.cases) * val_split)
        self.cases = self.cases[val_size:] if is_train else self.cases[:val_size]
        
        print(f"{'Pre-loading Train' if is_train else 'Pre-loading Val'} PAIRED (T1ce+FLAIR) into RAM...")
        self.data = []

        for case_path in self.cases:
            case_name = case_path.name
            t1ce_path = case_path / f"{case_name}_t1ce.nii"
            flair_path = case_path / f"{case_name}_flair.nii"

            if not t1ce_path.exists() or not flair_path.exists():
                continue
            
            # Read both volumes
            t1ce_vol = nib.load(t1ce_path).get_fdata()
            flair_vol = nib.load(flair_path).get_fdata()
            
            for i in range(num_slices):
                slice_idx = slice_start + i

                if slice_idx >= t1ce_vol.shape[2] or slice_idx >= flair_vol.shape[2]:
                    continue
                
                # ---- Process T1ce (percentile normalization, matches VAE training) ----
                t1ce_slice = t1ce_vol[:, :, slice_idx]
                t1ce_crop = t1ce_slice[48:208, 48:208]  # Center crop 160x160

                # Skip near-empty slices (same guard as BraTSInMemoryDataset)
                if np.count_nonzero(t1ce_crop) < 500:
                    continue

                low_t1ce = np.percentile(t1ce_crop, 1)
                high_t1ce = np.percentile(t1ce_crop, 99)
                t1ce_crop = np.clip(t1ce_crop, low_t1ce, high_t1ce)
                t1ce_norm = (t1ce_crop - low_t1ce) / (high_t1ce - low_t1ce + 1e-8)
                t1ce_tensor = torch.from_numpy(t1ce_norm * 2 - 1).float().unsqueeze(0)  # Map to [-1, 1]
                
                # ---- Process FLAIR (percentile normalization, matches VAE training) ----
                flair_slice = flair_vol[:, :, slice_idx]
                flair_crop = flair_slice[48:208, 48:208]

                if np.count_nonzero(flair_crop) < 500:
                    continue

                low_flair = np.percentile(flair_crop, 1)
                high_flair = np.percentile(flair_crop, 99)
                flair_crop = np.clip(flair_crop, low_flair, high_flair)
                flair_norm = (flair_crop - low_flair) / (high_flair - low_flair + 1e-8)
                flair_tensor = torch.from_numpy(flair_norm * 2 - 1).float().unsqueeze(0)
                
                self.data.append({
                    "t1ce": t1ce_tensor,
                    "flair": flair_tensor,
                    "case": case_name,
                    "slice_idx": slice_idx
                })
                
        print(f"✅ Finished loading {len(self.data)} pairs!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


