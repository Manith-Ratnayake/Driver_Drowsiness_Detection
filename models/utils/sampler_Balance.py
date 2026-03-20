import math
import random
import numpy as np


class BalancedSequenceBatchSource():
    def __init__(
        self,
        sequence_frame_paths,
        sequence_labels,
        batch_size,
        frames_per_window,
        min_class_fraction=0.3,
        shuffle=True,
        drop_last=True,
        seed=42,
        shard_id=0,
        num_shards=1,
    ):
        self.sequence_frame_paths = sequence_frame_paths
        self.sequence_labels = sequence_labels
        self.batch_size = batch_size
        self.frames_per_window = frames_per_window
        self.min_class_fraction = min_class_fraction
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.rng = random.Random(seed + shard_id)

        dataset_window_indices = list(range(len(self.sequence_labels)))
        self.shard_window_indices = dataset_window_indices[
            self.shard_id::self.num_shards
        ]

        self.positive_indices = [
            idx for idx in self.shard_window_indices
            if int(self.sequence_labels[idx]) == 1
        ]
        self.negative_indices = [
            idx for idx in self.shard_window_indices
            if int(self.sequence_labels[idx]) == 0
        ]

       
        self.min_per_class = math.ceil(self.batch_size * self.min_class_fraction)
        if 2 * self.min_per_class > self.batch_size:
            raise ValueError("min_class_fraction is too large for this batch_size.")

        self.local_num_sequences = len(self.shard_window_indices)

        if self.drop_last:
            self.num_batches = self.local_num_sequences // self.batch_size
        else:
            self.num_batches = math.ceil(self.local_num_sequences / self.batch_size)

        self.sequence_frame_paths_encoded = [
            [np.frombuffer(path.encode("utf-8"), dtype=np.uint8) for path in seq_paths]
            for seq_paths in self.sequence_frame_paths
        ]

        self.sequence_repeated_labels = [
            [np.array([label], dtype=np.float32) for _ in range(self.frames_per_window)]
            for label in self.sequence_labels
        ]

        self.epoch_indices = []
        self.cursor = 0
        self.reset()

    def _sample_with_replacement_if_needed(self, candidate_indices, count):
        if count <= len(candidate_indices):
            return self.rng.sample(candidate_indices, count)

        sampled = self.rng.sample(candidate_indices, len(candidate_indices))
        sampled += [
            self.rng.choice(candidate_indices)
            for _ in range(count - len(candidate_indices))
        ]
        return sampled

    def _build_local_epoch_indices(self):
        epoch_indices = []

        remaining_samples = self.local_num_sequences

        for batch_idx in range(self.num_batches):
            current_batch_size = min(self.batch_size, remaining_samples)

            if current_batch_size <= 0:
                break

            if current_batch_size < self.batch_size and self.drop_last:
                break

            if current_batch_size == 1:
                if len(self.positive_indices) >= len(self.negative_indices):
                    batch_indices = [self.rng.choice(self.positive_indices)]
                else:
                    batch_indices = [self.rng.choice(self.negative_indices)]
            else:
                current_min_per_class = min(
                    self.min_per_class,
                    current_batch_size // 2
                )

                pos_count = self.rng.randint(
                    current_min_per_class,
                    current_batch_size - current_min_per_class,
                )
                neg_count = current_batch_size - pos_count

                batch_pos = self._sample_with_replacement_if_needed(
                    self.positive_indices,
                    pos_count,
                )
                batch_neg = self._sample_with_replacement_if_needed(
                    self.negative_indices,
                    neg_count,
                )

                batch_indices = batch_pos + batch_neg

            if self.shuffle:
                self.rng.shuffle(batch_indices)

            epoch_indices.extend(batch_indices)
            remaining_samples -= current_batch_size

        return epoch_indices

    def reset(self):
        self.epoch_indices = self._build_local_epoch_indices()
        self.cursor = 0

    def __len__(self):
        return self.num_batches

    def __call__(self):
        if self.cursor >= len(self.epoch_indices):
            raise StopIteration

        batch_indices = self.epoch_indices[
            self.cursor:self.cursor + self.batch_size
        ]
        self.cursor += len(batch_indices)

        if len(batch_indices) < self.batch_size and self.drop_last:
            raise StopIteration

        flat_paths = [path for idx in batch_indices for path in self.sequence_frame_paths_encoded[idx]]
        flat_labels = [label for idx in batch_indices  for label in self.sequence_repeated_labels[idx]]

        return flat_paths, flat_labels
