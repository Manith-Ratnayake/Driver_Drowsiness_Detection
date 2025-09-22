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
from torchvision.models.densenet import DenseNet201_Weights, densenet201

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




parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--epoch", type=int, required=False, default=0, help="To resume : Please provide last checkpoint epoch number not the wanted checkpointed")
args = parser.parse_args()
epochs_resume_no = args.epoch








desired_height = 350
desired_width  = 350

transform = v2.Compose([
    v2.CenterCrop((desired_height, desired_width)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


dataset = ImageFolder("/kaggle/working/train_data", transform=transform)



train_ratio = 0.7
val_ratio   = 0.15
test_ratio  = 0.15

dataset_size     = len(dataset)
train_size       = int(train_ratio * dataset_size)
validation_size  = int(val_ratio * dataset_size)
test_size        = dataset_size - train_size - validation_size 


train_dataset, validation_dataset, test_dataset = random_split(
    dataset,
    [train_size, validation_size , test_size],
    generator=torch.Generator().manual_seed(seed)  
)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train size: {len(train_dataset)} ({len(train_dataset)/len(dataset)*100:.2f}%)")
print(f"Validation size: {len(validation_dataset)} ({len(validation_dataset)/len(dataset)*100:.2f}%)")
print(f"Test size: {len(test_dataset)} ({len(test_dataset)/len(dataset)*100:.2f}%)")





accelerator = Accelerator()
device = accelerator.device


model = densenet201(weights=DenseNet201_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        for param in module.parameters():
            param.requires_grad = True


num_features = model.classifier.in_features
model.classifier = nn.Sequential( # 1920 
    nn.Flatten(1), 

    nn.Linear(num_features, 256), 
    nn.ReLU(),           
    nn.Dropout(0.4),

    nn.Linear(256, 32),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(32, 2)
)






criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), weight_decay=0.01)


model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
model, optimizer, train_loader, val_loader, test_loader
)


start_epoch = 1
epochs = 101
best_val_loss = float("inf")
patience, counter = 5, 0


steps_per_epoch = len(train_loader)

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.001,          
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    pct_start=0.3,        
    anneal_strategy='cos',
    div_factor=25,        
    final_div_factor=1e4  
)

path = f"/kaggle/working/DenseNet201_epoch_{epochs_resume_no}.pth"


if path and os.path.exists(path):
    checkpoint = torch.load(path)

    model_to_load = accelerator.unwrap_model(model)
    model_to_load.load_state_dict(checkpoint["model"])
    
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"] + 1  
    print(f"Resuming training from epoch {start_epoch}")



for epoch in range(start_epoch, epochs):
    model.train()
    total, correct, train_loss = 0, 0, 0.0

    for images, labels in tqdm(train_loader, disable=not accelerator.is_local_main_process):
        images, labels = images, labels
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()        
        optimizer.zero_grad()

        train_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)


    train_loss /= total
    train_acc = 100 * correct / total

    model.eval()
    validation_loss, validation_correct, validation_total = 0.0, 0, 0


    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images, labels
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            validation_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            validation_correct += (predicted == labels).sum().item()
            validation_total += labels.size(0)



    validation_loss /= validation_total
    validation_accuracy = 100 * validation_correct / validation_total


    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:    
        log_line = (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Epoch {epoch}: Train Loss {train_loss:.4f}, Acc {train_acc:.2f}% | Validation Loss {validation_loss:.4f}, Acc {validation_accuracy:.2f}%")
        print(log_line)        


        with open("training_details_DenseNet201_exp2.txt", "a") as f:
             f.write(log_line + "\n")


        accelerator.save(
            {
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),  
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy
            },
            f"DenseNet201_epoch_{epoch}.pth",
        )

        



model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0


with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images, labels
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)


test_loss /= test_total
test_acc = 100 * test_correct / test_total


if accelerator.is_main_process:
    log_test_line = f"Final Test: Loss {test_loss:.4f}, Acc {test_acc:.2f}%"
    print(log_test_line)

    with open("training_details_DenseNet201_exp2.txt", "a") as f:
        f.write(log_test_line + "\n")