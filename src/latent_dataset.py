import torch
from torch.utils.data import Dataset


class PairedLatentDataset(Dataset):
    def __init__(self, pt_file_path):
        print(f"Loading latent pairs from {pt_file_path}...")

        self.data = torch.load(
            pt_file_path,
            map_location="cpu",
            weights_only=False,
        )

        print(f"✅ Loaded {len(self.data)} latent pairs.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        return {
            # Condition (INPUT)
            "condition": item["z_t1ce"].float(),

            # Ground truth (TARGET)
            "target": item["z_flair"].float(),

            # Metadata
            "case": item["case"],
            "slice_idx": item["slice_idx"],
        }