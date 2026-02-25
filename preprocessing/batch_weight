from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

all_labels    = ...
class_weights = [...]
dataset       = ...

sample_weights = [class_weights[label] for label in all_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
