from torch.utils.data import DataLoader

from brats_cross_modal_dataset import (
    BraTSCrossModalDataset
)
import matplotlib.pyplot as plt

dataset = BraTSCrossModalDataset(
    root_dir="BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
    source_modality="flair",
    target_modality="t1ce",
    is_train=True,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

batch = next(iter(loader))

print(batch["source"].shape)
print(batch["target"].shape)

print(batch["source"].dtype)
print(batch["target"].dtype)

print(batch["case"][0])
print(batch["slice_idx"][0])



source = batch["source"][0, 0].numpy()
target = batch["target"][0, 0].numpy()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(source, cmap="gray")
plt.title("FLAIR")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(target, cmap="gray")
plt.title("T1CE")
plt.axis("off")

plt.tight_layout()
plt.savefig("paired_example.png")