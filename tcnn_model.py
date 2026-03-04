import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

class DDDSequenceDataset(Dataset):
    def __init__(self, prebuilt_file, base_path="", transform=None):
        """
        base_path: directory to prepend to all frame paths
        """
        self.base_path = base_path
        self.transform = transform
        self.sequence_frame_paths_and_label = []

        if not os.path.exists(prebuilt_file):
            raise FileNotFoundError(f"Prebuilt file not found: {prebuilt_file}")

        with open(prebuilt_file, "r") as f:
            for line in f:
                paths, label = line.strip().split("|")
                frame_paths = paths.split(",")
                self.sequence_frame_paths_and_label.append((frame_paths, int(label)))

        print(f"[Dataset] Loaded {len(self.sequence_frame_paths_and_label)} sequences from {prebuilt_file}")

    def __len__(self):
        return len(self.sequence_frame_paths_and_label)

    def __getitem__(self, idx):
        relative_paths, label = self.sequence_frame_paths_and_label[idx]
        images = []

        for rel_path in relative_paths:
            full_path = os.path.join(self.base_path, rel_path)
            if not os.path.exists(full_path):
                img = torch.zeros((1, 64, 64), dtype=torch.float)
            else:
                img = read_image(full_path, mode=ImageReadMode.GRAY).float() / 255.0

            if self.transform:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images)  # (T, C, H, W)
        return images, torch.tensor(label, dtype=torch.float)