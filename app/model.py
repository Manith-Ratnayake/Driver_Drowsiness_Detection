import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.video import r3d_18 
import tqdm
 
 
class VideoDataset(Dataset):

    def __init__(self, root_dir, frames_per_video=16):
        self.samples = []
        self.frames_per_video = frames_per_video

        self.transform = transforms.Compose([
            transforms.CenterCrop((250, 250)),
            transforms.ToTensor()
        ])

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(root_dir, cls)
            for video in os.listdir(cls_path):
                self.samples.append(
                    (os.path.join(cls_path, video), self.class_to_idx[cls])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = sorted(os.listdir(video_path))[:self.frames_per_video]

        images = []
        for f in frames:
            img = Image.open(os.path.join(video_path, f)).convert("RGB")
            images.append(self.transform(img))

        # [C, T, H, W] -> R3D expects [B, C, T, H, W] later in DataLoader
        video_tensor = torch.stack(images, dim=1)
        return video_tensor, label

 
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available. Please run on a machine with a GPU.")
 
device = torch.device("cuda")

dataset = VideoDataset("dataset/")
loader = DataLoader(dataset, batch_size=8, shuffle=True)  

 
num_classes = 2
model = r3d_18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

epochs = 50
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
 


patience = 5
best_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0

    
    loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
    
    for videos, labels in loop:
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())  

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epoch(s)")

    if patience_counter >= patience:
        print("Early stopping triggered!")
        break