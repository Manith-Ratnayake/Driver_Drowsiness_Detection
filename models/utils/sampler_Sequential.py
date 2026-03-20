import math
import random
import numpy as np


class SequentialBatchSource():
    def __init__(
        self,
        sequence_frame_paths,
        sequence_labels,
        batch_size,
        frames_per_window,
        drop_last=True,
        shard_id=0,
        num_shards=1,
    ):
        self.sequence_frame_paths = sequence_frame_paths
        self.sequence_labels = sequence_labels
        self.batch_size = batch_size
        self.frames_per_window = frames_per_window
        self.drop_last = drop_last
        self.shard_id = shard_id
        self.num_shards = num_shards

        dataset_window_indices = list(range(len(self.sequence_labels)))
        self.shard_window_indices = dataset_window_indices[self.shard_id::self.num_shards]

        if self.drop_last:
            usable = (len(self.shard_window_indices) // self.batch_size) * self.batch_size
            self.shard_window_indices = self.shard_window_indices[:usable]
            self.num_batches = len(self.shard_window_indices) // self.batch_size
        else:
            self.num_batches = math.ceil(len(self.shard_window_indices) / self.batch_size)

        self.sequence_frame_paths_encoded = [
            [np.frombuffer(path.encode("utf-8"), dtype=np.uint8) for path in seq_paths]
            for seq_paths in self.sequence_frame_paths
        ]

        self.sequence_repeated_labels = [
            [np.array([label], dtype=np.float32) for _ in range(self.frames_per_window)]
            for label in self.sequence_labels
        ]

        self.cursor = 0

    def reset(self):
        self.cursor = 0

    def __len__(self):
        return self.num_batches

    def __call__(self):
        if self.cursor >= len(self.shard_window_indices):
            raise StopIteration

        batch_indices = self.shard_window_indices[
            self.cursor:self.cursor + self.batch_size
        ]

        self.cursor += self.batch_size

        if len(batch_indices) < self.batch_size and self.drop_last:
            raise StopIteration

        flat_paths = [path for idx in batch_indices for path in self.sequence_frame_paths_encoded[idx]]
        flat_labels = [label for idx in batch_indices  for label in self.sequence_repeated_labels[idx]]
        
        return flat_paths, flat_labels
