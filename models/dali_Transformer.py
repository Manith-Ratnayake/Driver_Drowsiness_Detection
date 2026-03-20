%%file train.py

from accelerate import Accelerator
gradient_accumulation_steps = 1
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

device_id = accelerator.local_process_index
num_shards = accelerator.num_processes
shard_id = accelerator.local_process_index

import wandb
wandb.login(key="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")


import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torchvision.ops import sigmoid_focal_loss

from accelerate.utils import set_seed
import evaluate
from nvidia.dali import fn, types, pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel

console = Console()

seed = 42
set_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class CNNEncoder(nn.Module):

  def __init__(self, backbone_model, encoder_output_dim, encoder_dropout, finetune_start_layer=0):
        super().__init__()
        backbone_first_conv_block = backbone_model.features[0][0]
        backbone_model.features[0][0] = nn.Conv2d(
                    in_channels = 1,
                    out_channels = backbone_first_conv_block.out_channels,
                    kernel_size = backbone_first_conv_block.kernel_size,
                    stride = backbone_first_conv_block.stride,
                    padding = backbone_first_conv_block.padding,
                    bias=False)


        with torch.no_grad():
            backbone_model.features[0][0].weight.copy_(backbone_first_conv_block.weight.mean(dim=1, keepdim=True))

        self.features = backbone_model.features

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        backbone_output_dim = backbone_model.classifier[0].in_features

        self.fc = nn.Sequential(
            nn.Linear(backbone_output_dim, encoder_output_dim),
            nn.LayerNorm(encoder_output_dim),
            nn.GELU(),
            nn.Dropout(encoder_dropout),
        )

        for param in self.features.parameters():
             param.requires_grad = False

        for param in self.features[finetune_start_layer:].parameters():
            param.requires_grad = True


  def forward(self, x):
      x = self.features(x)
      x = self.pool(x)
      x = self.fc(x)
      return x



class CNNTemporalTransformer(nn.Module):

    def __init__(self, backbone_model, encoder_output_dim, encoder_dropout,
                 temporal_dropout, num_heads=4, num_layers=2, max_len=32):

        super().__init__()

        self.encoder = CNNEncoder(
            backbone_model=backbone_model,
            encoder_output_dim=encoder_output_dim,
            encoder_dropout=encoder_dropout
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, encoder_output_dim))

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=encoder_output_dim,
            nhead=num_heads,
            dropout=temporal_dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(encoder_output_dim, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        features = self.encoder(x)
        features = features.view(B, T, -1)

        features = features + self.pos_embedding[:, :T, :]
        transformed = self.transformer(features)
        last_timestep = transformed[:, -1, :]
        logits = self.classifier(last_timestep)

        return logits.squeeze(1)



ENV_PATHS = {
    "Kaggle": {
        "train_1": "/kaggle/input/ddd-training1-face",
        "train_2": "/kaggle/input/ddd-training2-face",
        "val": "/kaggle/input/ddd-validation-face",
    },
    "Colab": {
        "train_1": "/root/.cache/kagglehub/datasets/manith04/ddd-training1-face/versions/1/",
        "train_2": "/root/.cache/kagglehub/datasets/manith04/ddd-training2-face/versions/1/",
        "val": "/root/.cache/kagglehub/datasets/manith04/ddd-validation-face/versions/1/",
    },
}

env = os.environ
env_name = "Kaggle" if "KAGGLE_URL_BASE" in env else "Colab" if "COLAB_GPU" in env else "Local"


dataset_roots = ENV_PATHS[env_name]
train_dir_1 = dataset_roots["train_1"]
train_dir_2 = dataset_roots["train_2"]
val_dir = dataset_roots["val"]



def create_metrics():
    metrics = {"accuracy": evaluate.load("accuracy"),
               "precision": evaluate.load("precision"),
               "recall": evaluate.load("recall"),
               "f1": evaluate.load("f1"),}
    return metrics


def update_metrics(metrics, preds, labels):
    for metric in metrics.values():
        metric.add_batch(predictions=preds, references=labels)


def compute_metrics(metrics):
    return {k: v.compute()[k] for k, v in metrics.items()}


class DDDSequenceIndex():
    def __init__(
        self,
        root_dir,
        time_window_in_seconds,
        frames_per_window,
        time_window_stride_by_seconds,
    ):
        self.root_dir = root_dir
        self.time_window_in_seconds = time_window_in_seconds
        self.frames_per_window = frames_per_window
        self.time_window_stride_by_seconds = time_window_stride_by_seconds

        self.total_positive = 0
        self.total_negative = 0
        self.sequence_frame_paths = []
        self.sequence_labels = []

        self._build_sequences()

        print(f"Total sequences: {len(self.sequence_frame_paths)}")
        print(f"Positive sequences: {self.total_positive}")
        print(f"Negative sequences: {self.total_negative}")


    def _build_sequences(self):

        for subject_entry in os.scandir(self.root_dir):
            for scenario_entry in os.scandir(subject_entry.path):

                scenario_name = scenario_entry.name
                video_fps = 15 if "night" in scenario_name.lower() else 30

                for video_type_entry in os.scandir(scenario_entry.path):
                    self._process_windows_in_video(video_type_entry.path, video_fps)



    def _process_windows_in_video(
        self,
        video_path,
        video_fps
    ):

        video_frame_list = sorted(entry.name for entry in os.scandir(video_path))
        total_frames_in_video = len(video_frame_list)
        available_frames_per_window = round(self.time_window_in_seconds * video_fps)
        
        stride_between_windows_in_frames = max(1, round(self.time_window_stride_by_seconds * video_fps))
        total_windows_in_video = (total_frames_in_video - available_frames_per_window) // stride_between_windows_in_frames + 1
        frames_space_in_window = (available_frames_per_window - 1) / (self.frames_per_window - 1)

        selected_frames_indices_in_window = [
            round(frame_position * frames_space_in_window)
            for frame_position in range(self.frames_per_window)
        ]

        last_frame_offset = selected_frames_indices_in_window[-1]

        frame_paths = [os.path.join(video_path, name)
                      for name in video_frame_list]

        frame_labels = [int(name.rsplit("_", 1)[-1].split(".")[0])
                        for name in video_frame_list]

        append_paths = self.sequence_frame_paths.append
        append_label = self.sequence_labels.append

        total_positive = self.total_positive
        total_negative = self.total_negative


        # Create sequences
        for window_index in range(total_windows_in_video):
            window_start_frame = window_index * stride_between_windows_in_frames
            
            last_frame_index = (window_start_frame + last_frame_offset)
            if last_frame_index >= total_frames_in_video:
              break
            
            frame_indices = [window_start_frame + frame_offset for frame_offset in selected_frames_indices_in_window]

            # Remove Duplicates
            for frame_index in range(1, len(frame_indices)):
                if frame_indices[frame_index] <= frame_indices[frame_index - 1]:
                    frame_indices[frame_index] = frame_indices[frame_index - 1] + 1

            if frame_indices[-1] >= total_frames_in_video:
              break

            paths = [frame_paths[frame_index] 
                    for frame_index in frame_indices]

            label = frame_labels[frame_indices[-1]]

            if label == 1:
                total_positive += 1
            else:
                total_negative += 1

            append_paths(paths)
            append_label(label)

        self.total_positive = total_positive
        self.total_negative = total_negative





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




@pipeline_def
def train_sequence_pipeline(batch_source, image_height, image_width, frames_per_window):
    filepaths, labels = fn.external_source(
        source=batch_source,
        num_outputs=2,
        batch=True,
        parallel=False
    )

    encoded = fn.io.file.read(filepaths)
    images = fn.experimental.decoders.image(
        encoded,
        device="mixed",
        output_type=types.GRAY
    )

    images = fn.resize(images, resize_x=image_width, resize_y=image_height)

    mirror = fn.random.coin_flip(probability=0.5)
    images = fn.flip(images, horizontal=mirror)

    angle = fn.random.uniform(range=(-10.0, 10.0))
    images = fn.rotate(images, angle=angle, fill_value=0, keep_size=True)

    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[127.5],
        std=[127.5],
    )

    labels = fn.cast(labels, dtype=types.FLOAT)
    return images, labels




@pipeline_def
def val_sequence_pipeline(batch_source, image_height, image_width, frames_per_window):
    filepaths, labels = fn.external_source(
        source=batch_source,
        num_outputs=2,
        batch=True,
        parallel=False
    )

    encoded = fn.io.file.read(filepaths)
    images = fn.experimental.decoders.image(
        encoded,
        device="mixed",
        output_type=types.GRAY
    )

    images = fn.resize(images, resize_x=image_width, resize_y=image_height)

    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[127.5],
        std=[127.5],
    )

    labels = fn.cast(labels, dtype=types.FLOAT)
    return images, labels





class DALISequenceLoader():

    def __init__(self, iterator, batch_source, frames_per_window):
        self.iterator = iterator
        self.batch_source = batch_source
        self.frames_per_window = frames_per_window

    def __iter__(self):
        return iter(self.iterator)

    def __len__(self):
        return len(self.batch_source)

    def reset(self):
        self.batch_source.reset()
        self.iterator.reset()






def create_dali_dataloaders(
    accelerator,
    batch_size,
    time_window_in_seconds,
    frames_per_window,
    time_window_stride_by_seconds,
    prefetch_queue_depth,
    image_height,
    image_width,
):


    device_id = accelerator.local_process_index
    shard_id = accelerator.local_process_index
    num_shards = accelerator.num_processes

    train_index_1 = DDDSequenceIndex(
        root_dir=train_dir_1,
        time_window_in_seconds=time_window_in_seconds,
        frames_per_window=frames_per_window,
        time_window_stride_by_seconds=time_window_stride_by_seconds,
    )

    train_index_2 = DDDSequenceIndex(
        root_dir=train_dir_2,
        time_window_in_seconds=time_window_in_seconds,
        frames_per_window=frames_per_window,
        time_window_stride_by_seconds=time_window_stride_by_seconds,
    )

    val_index = DDDSequenceIndex(
        root_dir=val_dir,
        time_window_in_seconds=time_window_in_seconds,
        frames_per_window=frames_per_window,
        time_window_stride_by_seconds=time_window_stride_by_seconds,
    )

    train_sequence_frame_paths = (
        train_index_1.sequence_frame_paths + train_index_2.sequence_frame_paths
    )
    train_sequence_labels = (
        train_index_1.sequence_labels + train_index_2.sequence_labels
    )

    total_positive = train_index_1.total_positive + train_index_2.total_positive
    total_negative = train_index_1.total_negative + train_index_2.total_negative

    train_source = WeightedSequenceBatchSource(
        sequence_frame_paths=train_sequence_frame_paths,
        sequence_labels=train_sequence_labels,
        batch_size=batch_size,
        frames_per_window=frames_per_window,
        shuffle=True,
        drop_last=True,
        seed=seed + shard_id,
        shard_id=shard_id,
        num_shards=num_shards,
    )

    val_source = SequentialBatchSource(
        sequence_frame_paths=val_index.sequence_frame_paths,
        sequence_labels=val_index.sequence_labels,
        batch_size=batch_size,
        frames_per_window=frames_per_window,
        drop_last=False,
        shard_id=shard_id,
        num_shards=num_shards,
    )

    train_pipe = train_sequence_pipeline(
        batch_source=train_source,
        image_height=image_height,
        image_width=image_width,
        frames_per_window=frames_per_window,
        batch_size=batch_size * frames_per_window,
        num_threads=4,
        device_id=device_id,
        seed=seed + shard_id,
        prefetch_queue_depth=prefetch_queue_depth,

    )

    val_pipe = val_sequence_pipeline(
        batch_source=val_source,
        image_height=image_height,
        image_width=image_width,
        frames_per_window=frames_per_window,
        batch_size=batch_size * frames_per_window,
        num_threads=4,
        device_id=device_id,
        seed=seed + shard_id,
        prefetch_queue_depth=prefetch_queue_depth,

    )

    train_pipe.build()
    val_pipe.build()

    train_iter = DALIGenericIterator(
        [train_pipe],
        output_map=["frames", "labels"],
        size=-1,
        auto_reset=False,
        prepare_first_batch=True,
    )

    val_iter = DALIGenericIterator(
        [val_pipe],
        output_map=["frames", "labels"],
        size=-1,
        auto_reset=False,
        prepare_first_batch=True,
    )

    train_loader = DALISequenceLoader(train_iter, train_source, frames_per_window)
    val_loader = DALISequenceLoader(val_iter, val_source, frames_per_window)
    return train_loader, val_loader, total_positive, total_negative





def get_criterion(
    loss_type,
    total_positive,
    total_negative,
    device,
):
    total_samples = total_positive + total_negative
    loss_type = loss_type.lower()

    if loss_type == "focal":
        alpha = -1
        gamma = 2.0  

        def criterion(outputs, labels):
            return sigmoid_focal_loss(inputs=outputs, targets=labels.float(), alpha=alpha, gamma=gamma, reduction="mean")

        return criterion

    elif loss_type == "bce":
        positive_weight_value = 1.0
        positive_weight = torch.tensor(positive_weight_value, dtype=torch.float32, device=device)
        return torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")




def setup_training(accelerator, checkpoint_dir, extra_path, log_file, device):
    best_val_loss = float("inf")
    start_epoch = 0
    best_val_accuracy = 0.0

    if os.path.isdir(checkpoint_dir) and os.listdir(checkpoint_dir):
        accelerator.print("🔄 Loading checkpoint...")
        accelerator.load_state(checkpoint_dir)

        if os.path.exists(extra_path):
            extra = torch.load(extra_path, map_location=device)
            start_epoch = extra.get("epoch", 0)
            best_val_loss = extra.get("best_val_loss", float("inf"))
            best_val_accuracy = extra.get("best_val_accuracy", 0.0)
            accelerator.print(f"Resuming from epoch {start_epoch}")
        else:
            accelerator.print("⚡ No extra_state.pt found, starting from scratch.")

    if accelerator.is_main_process and start_epoch == 0:
        with open(log_file, "w") as f:
            f.write(
                "epoch,train_loss,train_acc,train_precision,train_recall,train_f1,"
                "val_loss,val_acc,val_precision,val_recall,val_f1\n"
            )

    return start_epoch, best_val_loss, best_val_accuracy




def run_validation(model, val_loader, accelerator, device, criterion, frames_per_window):
    model.eval()
    val_metrics = create_metrics()
    val_loss = 0.0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Validation", disable=not accelerator.is_main_process)

        for data in val_bar:

            batch = data[0]
            frames = batch["frames"]
            labels = batch["labels"]

            flat_batch_size = frames.shape[0]
            real_batch_size = flat_batch_size // frames_per_window

            frames = frames.view(real_batch_size, frames_per_window, *frames.shape[1:])
            labels = labels.view(real_batch_size, frames_per_window, -1)[:, 0, 0]

            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1)

            outputs = model(frames)
            loss = criterion(outputs, labels)

            gathered_loss = accelerator.gather_for_metrics(loss.detach().unsqueeze(0))
            val_loss += gathered_loss.mean().item()

            preds = (torch.sigmoid(outputs) > 0.5).int()
            preds, labels_int = accelerator.gather_for_metrics((preds, labels.int()))
            update_metrics(val_metrics, preds, labels_int)

    val_results = compute_metrics(val_metrics)
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, val_results




def log_epoch_results(epoch, avg_train_loss, train_results, avg_val_loss, val_results, log_file):
    print(
        f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f} | "
        f"Acc: {train_results['accuracy']:.4f} | "
        f"Precision: {train_results['precision']:.4f} | "
        f"Recall: {train_results['recall']:.4f} | "
        f"F1: {train_results['f1']:.4f}"
    )
    print(
        f"Epoch {epoch + 1} - Val   Loss: {avg_val_loss:.4f} | "
        f"Acc: {val_results['accuracy']:.4f} | "
        f"Precision: {val_results['precision']:.4f} | "
        f"Recall: {val_results['recall']:.4f} | "
        f"F1: {val_results['f1']:.4f}"
    )

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "train_accuracy": train_results["accuracy"],
        "train_precision": train_results["precision"],
        "train_recall": train_results["recall"],
        "train_f1": train_results["f1"],
        "val_loss": avg_val_loss,
        "val_accuracy": val_results["accuracy"],
        "val_precision": val_results["precision"],
        "val_recall": val_results["recall"],
        "val_f1": val_results["f1"],
    })

    with open(log_file, "a") as f:
      f.write(
          f"{epoch + 1},{avg_train_loss:.3f},{train_results['accuracy']:.3f},"
          f"{train_results['precision']:.3f},{train_results['recall']:.3f},{train_results['f1']:.3f},"
          f"{avg_val_loss:.3f},{val_results['accuracy']:.3f},{val_results['precision']:.3f},"
          f"{val_results['recall']:.3f},{val_results['f1']:.3f}\n"
      )



def save_best_models(
    model,
    accelerator,
    epoch,
    avg_val_loss,
    val_results,
    best_val_loss,
    best_val_accuracy,
    best_loss_model_path,
    best_acc_model_path,
):
    unwrapped_model = accelerator.unwrap_model(model)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": unwrapped_model.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
            },
            best_loss_model_path,
        )

        artifact = wandb.Artifact(f"{run_id}-best-loss-model", type="model")
        artifact.add_file(best_loss_model_path, name=f"{run_id}/best_loss_model/model.pth")
        wandb.log_artifact(artifact)

        print("⭐ Best val loss model updated!")

    if val_results["accuracy"] > best_val_accuracy:
        best_val_accuracy = val_results["accuracy"]

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": unwrapped_model.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
            },
            best_acc_model_path,
        )

        artifact = wandb.Artifact("models", type="model")
        artifact.add_file(best_acc_model_path, name=f"{run_id}/best_acc_model/model.pth")
        wandb.log_artifact(artifact)

        print("⭐ Best val accuracy model updated!")

    return best_val_loss, best_val_accuracy




epochs = 180
checkpoint_dir = "checkpoint"
log_file = "metrics_log.txt"
extra_path = os.path.join(checkpoint_dir, "extra_state.pt")
best_loss_model_path = "best_val_loss_model.pth"
best_acc_model_path = "best_val_acc_model.pth"


batch_size = 32
time_window_in_seconds = 8
frames_per_window = 16
time_window_stride_by_seconds = 5

prefetch_queue_depth = 20
image_height=224,
image_width=224,


train_loader, val_loader, total_positive, total_negative = create_dali_dataloaders(
    accelerator=accelerator,
    batch_size=batch_size,
    time_window_in_seconds=time_window_in_seconds,
    frames_per_window=frames_per_window,
    time_window_stride_by_seconds=time_window_stride_by_seconds,
    image_height=image_height,
    image_width=image_width,
    prefetch_queue_depth=prefetch_queue_depth,
)


EXPERIMENT_NO = 5
run_id = f"exp_{EXPERIMENT_NO}"
console.print(Panel.fit(f"[bold yellow]Running Experiment:[/bold yellow]\n[bold green]{run_id}[/bold green]",border_style="bright_blue", ))


from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
weights = MobileNet_V3_Small_Weights.DEFAULT
backbone = mobilenet_v3_small(weights=weights)


encoder_output_dim = 256
encoder_dropout = 0.3
temporal_dropout = 0.4
num_heads = 4
num_layers = 2
max_len = 32

model = CNNTemporalTransformer(
    backbone_model=backbone,
    encoder_dropout = encoder_dropout,
    temporal_dropout = temporal_dropout,
    encoder_output_dim = encoder_output_dim,
    num_heads = num_heads,
    num_layers = num_layers,
    max_len = max_len,
)

loss_type = "focal"
encoder_backbone_lr = 1e-4
encoder_head_lr = 1e-3
temporal_lr = 1e-3
weight_decay = 1e-4


optimizer = torch.optim.AdamW([
    {"params": model.encoder.features.parameters(), "lr": encoder_backbone_lr},
    {"params": model.encoder.fc.parameters(), "lr": encoder_head_lr},
    {"params": model.transformer.parameters(), "lr": temporal_lr},
    {"params": [model.pos_embedding], "lr": temporal_lr},
    {"params": model.classifier.parameters(), "lr": temporal_lr},
], weight_decay=1e-4)


scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)



model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
device = accelerator.device

criterion = get_criterion(
    loss_type=loss_type,
    total_positive=total_positive,
    total_negative=total_negative,
    device=device
)

start_epoch, best_val_loss, best_val_accuracy = setup_training(
    accelerator=accelerator,
    checkpoint_dir=checkpoint_dir,
    extra_path=extra_path,
    log_file=log_file,
    device=device
)



if accelerator.is_main_process:
    wandb.init(
        project="NTHUDD-FACE",
        id=run_id,
        name=run_id,
        resume="allow",
        config = {
          "epochs": epochs,
          "batch_size": batch_size,
          "train_batch_strategy": "weighted_sampling_with_replacement",
          "graident_accumulation": gradient_accumulation_steps,
            
          "time_window_in_seconds": time_window_in_seconds,
          "frames_per_window": frames_per_window,
          "time_window_stride_by_seconds": time_window_stride_by_seconds,
          
          "encoder_dropout": encoder_dropout,
          "temporal_dropout": temporal_dropout,
          "ecoder_backbone_lr": encoder_backbone_lr,
          "encoder_head_lr": encoder_head_lr,
          "temporal_lr": temporal_lr,
          "loss_type": loss_type,
          "optimizer" : "AdamW",
          "scheduler": "CosineAnnealingWarmRestarts",
          
          "encoder_output_dim": encoder_output_dim,
          "num_heads": num_heads,
          "num_layers": num_layers,
          "max_len": max_len,
      }
    )

    wandb.save("./train.ipynb")




optimizer.zero_grad(set_to_none=True)

for epoch in range(start_epoch, epochs):
    model.train()

    if accelerator.is_main_process:
        print(f"\nEpoch {epoch + 1}/{epochs}")

    train_metrics = create_metrics()
    train_loss = 0.0
    num_train_batches = 0

    train_bar = tqdm(
        train_loader,
        desc="Training",
        disable=not accelerator.is_main_process
    )

    for batch_idx, data in enumerate(train_bar):
        batch = data[0]
        frames = batch["frames"]
        labels = batch["labels"]

        flat_batch_size = frames.shape[0]
        real_batch_size = flat_batch_size // frames_per_window

        frames = frames.view(
            real_batch_size,
            frames_per_window,
            *frames.shape[1:]
        )
        labels = labels.view(
            real_batch_size,
            frames_per_window,
            -1
        )[:, 0, 0]

        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1).float()

        with accelerator.accumulate(model):
            outputs = model(frames).view(-1)
            loss = criterion(outputs, labels)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step(epoch + batch_idx / max(len(train_loader), 1))
                optimizer.zero_grad(set_to_none=True)

        gathered_loss = accelerator.gather_for_metrics(loss.detach().unsqueeze(0))
        train_loss += gathered_loss.mean().item()
        num_train_batches += 1

        probs = torch.sigmoid(outputs.detach())
        preds = (probs > 0.5).int()
        preds, labels_int = accelerator.gather_for_metrics((preds, labels.int()))
        update_metrics(train_metrics, preds, labels_int)

    train_loader.reset()

    avg_train_loss = train_loss / max(num_train_batches, 1)
    train_results = compute_metrics(train_metrics)

    avg_val_loss, val_results = run_validation(
        model=model,
        val_loader=val_loader,
        accelerator=accelerator,
        device=device,
        criterion=criterion,
        frames_per_window=frames_per_window,
    )

    val_loader.reset()

    if accelerator.is_main_process:
        log_epoch_results(
            epoch=epoch,
            avg_train_loss=avg_train_loss,
            train_results=train_results,
            avg_val_loss=avg_val_loss,
            val_results=val_results,
            log_file=log_file,
        )

        best_val_loss, best_val_accuracy = save_best_models(
            model=model,
            accelerator=accelerator,
            epoch=epoch,
            avg_val_loss=avg_val_loss,
            val_results=val_results,
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_accuracy,
            best_loss_model_path=best_loss_model_path,
            best_acc_model_path=best_acc_model_path,
        )

    accelerator.wait_for_everyone()
    accelerator.save_state(checkpoint_dir)

    if accelerator.is_main_process:
        torch.save(
            {
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
            },
            extra_path,
        )

if accelerator.is_main_process:
    wandb.finish()
