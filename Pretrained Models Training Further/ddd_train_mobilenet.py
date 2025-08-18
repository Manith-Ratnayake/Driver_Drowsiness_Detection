import os
import torch
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from torch.utils.data import Subset
from collections import defaultdict
import os

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP









#from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),  # Scales to [0,1] and converts to float32
    v2.Normalize([0.5]*3, [0.5]*3)
])


dataset = ImageFolder("/kaggle/input/nthuddd2/train_data", transform=transform)

# Get actual class names
class_names = dataset.classes  # e.g., ['Drowsy', 'Not Drowsy']

# Initialize nested counters
person_class_glasses_count = defaultdict(lambda: {class_names[0]: {"glasses":0, "noglasses":0},
                                                   class_names[1]: {"glasses":0, "noglasses":0}})

for idx, (path, label) in enumerate(dataset.samples):
    filename = os.path.basename(path)
    person_id = filename.split("_")[0]
    glasses_status = filename.split("_")[1]  # 'glasses' or 'noglasses'
    class_name = dataset.classes[label]      # exact class name from ImageFolder

    person_class_glasses_count[person_id][class_name][glasses_status] += 1

# Print results
for person, class_dict in person_class_glasses_count.items():
    print(f"Person {person}:")
    for cls, glasses_dict in class_dict.items():
        print(f"  {cls}: glasses = {glasses_dict['glasses']}, noglasses = {glasses_dict['noglasses']}")

















# Define which persons go into train and test
test_ids = ["002"]           # Person 2 as test
train_ids = [pid for pid in person_class_glasses_count.keys() if pid not in test_ids]  # all others

# Get indices for train and test
train_indices = [i for i, (path, _) in enumerate(dataset.samples) if os.path.basename(path).split("_")[0] in train_ids]
test_indices  = [i for i, (path, _) in enumerate(dataset.samples) if os.path.basename(path).split("_")[0] in test_ids]

# Create Subsets


train_dataset = Subset(dataset, train_indices)
test_dataset  = Subset(dataset, test_indices)

total_size = len(dataset)
train_size = len(train_dataset)
test_size  = len(test_dataset)

print(f"Train size: {train_size} ({train_size/total_size*100:.2f}%)")
print(f"Test size: {test_size} ({test_size/total_size*100:.2f}%)")















from torch.utils.data import DataLoader
import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,   
    shuffle=True    
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,   
    shuffle=False    
)






def train(rank, world_size, train_dataset, test_dataset):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    device = torch.device(f'cuda:{rank}')   

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 🔹 Model
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)

    # weights = EfficientNet_V2_S_Weights.DEFAULT
    # model = efficientnet_v2_s(weights=weights)
    # model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # 🔹 Loss and optimizer (no schedulers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    best_val_loss = float('inf')
    patience = 5
    counter = 0

    # 🔹 Resume from checkpoint if exists
    start_epoch = 1


    checkpoint_path = "/kaggle/working/efficientnet_v2_s_7.pth"
    #checkpoint_path = "/kaggle/working/mobilenet_v3_small.pth"

    if rank == 0 and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        counter = checkpoint["counter"] 
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total, correct = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        if rank == 0:
            print(f"Epoch {epoch}: Train Acc: {100*correct/total:.2f}%")

            # 🔹 Evaluate
            model.eval()
            val_loss = 0
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= len(test_loader)
            val_acc = 100 * val_correct / val_total
            print(f"Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.2f}%")

            # 🔹 Save checkpoint
            #checkpoint_path = f"efficientnet_v2_s_{epoch}.pth"
            checkpoint_path = f"mobilenet_v3_small_{epoch}.pth"


            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "counter": counter
            }
            torch.save(checkpoint, checkpoint_path)

      


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, train_dataset, test_dataset), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
