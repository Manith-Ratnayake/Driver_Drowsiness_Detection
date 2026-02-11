import os
from tqdm import tqdm
import numpy as np
import random
from accelerate import Accelerator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torchvision.models import vit_l_16, ViT_L_16_Weights


from torch.utils.data import DataLoader, random_split
import argparse
from datetime import datetime



seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



parser = argparse.ArgumentParser(description="ViT-B Training Script")
parser.add_argument(
    "--epoch",
    type=int,
    required=False,
    default=0,
    help="Resume from checkpoint epoch number",
)
args = parser.parse_args()
epochs_resume_no = args.epoch



img_size = 224

transform = v2.Compose([
    v2.CenterCrop((img_size, img_size)),
    v2.ToTensor(),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])



dataset = ImageFolder("/kaggle/working/train_data", transform=transform)

train_ratio = 0.7
val_ratio   = 0.15
test_ratio  = 0.15

dataset_size = len(dataset)
train_size = int(train_ratio * dataset_size)
val_size   = int(val_ratio * dataset_size)
test_size  = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(seed)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(train_dataset)}")
print(f"Val:   {len(val_dataset)}")
print(f"Test:  {len(test_dataset)}")


accelerator = Accelerator()
device = accelerator.device

model = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

num_features = model.heads.head.in_features  # 768
model.heads.head = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)


for param in model.heads.head.parameters():
    param.requires_grad = True



criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4,            # ViT prefers lower LR
    weight_decay=0.01
)



model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader, test_loader
)



epochs = 51
start_epoch = 1

steps_per_epoch = len(train_loader)

scheduler = OneCycleLR(
    optimizer,
    max_lr=3e-4,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    pct_start=0.3,
    anneal_strategy="cos",
    div_factor=25,
    final_div_factor=1e4,
)



ckpt_path = f"/kaggle/working/ViT_B_16_epoch_{epochs_resume_no}.pth"

if ckpt_path and os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    accelerator.unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")



for epoch in range(start_epoch, epochs):
    model.train()
    total, correct, train_loss = 0, 0, 0.0

    for images, labels in tqdm(train_loader, disable=not accelerator.is_local_main_process):
        outputs = model(images)
        loss = criterion(outputs, labels)

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss /= total
    train_acc = 100 * correct / total




    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = 100 * val_correct / val_total

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        log = (
            f"{datetime.now()} | Epoch {epoch} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}%"
        )
        print(log)

        with open("training_details_ViT_B16.txt", "a") as f:
            f.write(log + "\n")

        accelerator.save(
            {
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            f"ViT_B_16_epoch_{epoch}.pth"
        )


model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_loss /= test_total
test_acc = 100 * test_correct / test_total

if accelerator.is_main_process:
    print(f"Final Test | Loss {test_loss:.4f} Acc {test_acc:.2f}%")
