import math
import random
import numpy as np


class BalancedEpochSequenceBatchSource():
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

    def _take_random(self, pool, count):
        taken = pool[:count]
        del pool[:count]
        return taken

    def _build_local_epoch_indices(self):
        pos_pool = self.positive_indices.copy()
        neg_pool = self.negative_indices.copy()

        if self.shuffle:
            self.rng.shuffle(pos_pool)
            self.rng.shuffle(neg_pool)

        epoch_indices = []
        used_count = 0

        while len(pos_pool) + len(neg_pool) > 0:
            remaining_total = len(pos_pool) + len(neg_pool)

            if self.drop_last and remaining_total < self.batch_size:
                break

            current_batch_size = min(self.batch_size, remaining_total)

            if current_batch_size == 1:
                if len(pos_pool) > 0:
                    batch_indices = self._take_random(pos_pool, 1)
                else:
                    batch_indices = self._take_random(neg_pool, 1)
            else:
                current_min_per_class = min(
                    self.min_per_class,
                    current_batch_size // 2
                )

                feasible_pos_min = max(
                    current_min_per_class,
                    current_batch_size - len(neg_pool)
                )
                feasible_pos_max = min(
                    len(pos_pool),
                    current_batch_size - current_min_per_class
                )

                if feasible_pos_min <= feasible_pos_max:
                    pos_count = self.rng.randint(feasible_pos_min, feasible_pos_max)
                else:
                    # Cannot satisfy min_class_fraction with remaining unused samples,
                    # so use the most balanced feasible split without replacement.
                    pos_count = min(len(pos_pool), current_batch_size)
                    neg_needed = current_batch_size - pos_count
                    if neg_needed > len(neg_pool):
                        neg_needed = len(neg_pool)
                        pos_count = current_batch_size - neg_needed

                neg_count = current_batch_size - pos_count

                batch_pos = self._take_random(pos_pool, pos_count)
                batch_neg = self._take_random(neg_pool, neg_count)

                batch_indices = batch_pos + batch_neg

            if self.shuffle:
                self.rng.shuffle(batch_indices)

            epoch_indices.extend(batch_indices)
            used_count += len(batch_indices)

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