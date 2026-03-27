import numpy as np
import torch
import random


class WeightedSequenceBatchSource():
    def __init__(
        self,
        sequence_frame_paths,
        sequence_labels,
        batch_size,
        frames_per_window,
        shuffle=True,
        seed=42,
        shard_id=0,
        num_shards=1,
    ):
        self.sequence_frame_paths = sequence_frame_paths
        self.sequence_labels = sequence_labels
        self.batch_size = batch_size
        self.frames_per_window = frames_per_window
        self.shuffle = shuffle
        self.seed = seed
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.rng = random.Random(seed + shard_id)

        # ---- SHARD SPLIT ----
        dataset_indices = list(range(len(self.sequence_labels)))
        self.shard_indices = dataset_indices[self.shard_id::self.num_shards]

        shard_labels = [self.sequence_labels[i] for i in self.shard_indices]

        # ---- COMPUTE CLASS WEIGHTS ----
        labels_tensor = torch.tensor(shard_labels, dtype=torch.long)
        class_counts = torch.bincount(labels_tensor)

        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels_tensor]

        self.sample_weights = sample_weights.numpy()

        # ---- ENCODE PATHS (same as your code) ----
        self.sequence_frame_paths_encoded = [
            [np.frombuffer(path.encode("utf-8"), dtype=np.uint8) for path in seq_paths]
            for seq_paths in self.sequence_frame_paths
        ]

        self.sequence_repeated_labels = [
            [np.array([label], dtype=np.float32) for _ in range(self.frames_per_window)]
            for label in self.sequence_labels
        ]

        self.num_samples = len(self.shard_indices)
        self.cursor = 0
        self.sampled_indices = []

        self.reset()

    def reset(self):
        # ---- WEIGHTED SAMPLING WITH REPLACEMENT ----
        self.sampled_indices = self.rng.choices(
            self.shard_indices,
            weights=self.sample_weights,
            k=self.num_samples
        )
        self.cursor = 0

    def __len__(self):
        return len(self.sampled_indices) // self.batch_size

    def __call__(self):
        if self.cursor >= len(self.sampled_indices):
            raise StopIteration

        batch_indices = self.sampled_indices[
            self.cursor:self.cursor + self.batch_size
        ]
        self.cursor += len(batch_indices)

        flat_paths = [path for idx in batch_indices for path in self.sequence_frame_paths_encoded[idx]]
        flat_labels = [label for idx in batch_indices for label in self.sequence_repeated_labels[idx]]

        return flat_paths, flat_labels