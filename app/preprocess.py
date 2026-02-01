import os
import cv2
from pathlib import Path
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================
DATASET_ROOT = r"C:\Users\Manith\Desktop\MResarch\data"
OUTPUT_ROOT  = r"C:\Users\Manith\Desktop\MResarch\processed_dataset"

TRAIN_SUBJECTS = ["001", "002"]
VAL_SUBJECTS   = ["005", "006"]

# ===============================
# CREATE OUTPUT FOLDERS
# ===============================
for split in ["train", "val"]:
    for cls in ["drowsy", "non_drowsy"]:
        Path(os.path.join(OUTPUT_ROOT, split, cls)).mkdir(
            parents=True, exist_ok=True
        )

# ===============================
# HELPER: FIND LABEL FILE
# ===============================
def find_drowsiness_label(scenario_path, video_name):
    base = os.path.splitext(video_name)[0]

    for f in os.listdir(scenario_path):
        if base in f and "drowsiness" in f.lower() and f.endswith(".txt"):
            return os.path.join(scenario_path, f)

    return None

# ===============================
# PROCESS SUBJECT
# ===============================
def process_subject(subject_id, split):
    subject_path = os.path.join(DATASET_ROOT, subject_id)

    for scenario in os.listdir(subject_path):
        scenario_path = os.path.join(subject_path, scenario)
        if not os.path.isdir(scenario_path):
            continue

        for file in os.listdir(scenario_path):
            if not file.lower().endswith(".avi"):
                continue

            video_path = os.path.join(scenario_path, file)
            label_path = find_drowsiness_label(scenario_path, file)

            if label_path is None:
                print(f"❌ Missing label for {video_path}")
                continue

            # Load labels
            with open(label_path, "r") as f:
                labels = [int(x.strip()) for x in f.readlines()]

            cap = cv2.VideoCapture(video_path)
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= len(labels):
                    break

                label = labels[frame_idx]
                cls = "drowsy" if label == 1 else "non_drowsy"

                out_name = (
                    f"{subject_id}_{scenario}_{file[:-4]}_"
                    f"{frame_idx:05d}.jpg"
                )

                out_path = os.path.join(
                    OUTPUT_ROOT, split, cls, out_name
                )

                cv2.imwrite(out_path, frame)
                frame_idx += 1

            cap.release()

# ===============================
# RUN
# ===============================
print("Processing TRAIN...")
for sid in tqdm(TRAIN_SUBJECTS):
    process_subject(sid, "train")

print("DONE")