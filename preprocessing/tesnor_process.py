import os
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np

def save_frames_as_tensor(root_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Get all jpg files
    all_files = []
    for subject in os.listdir(root_dir):
        subj_path = os.path.join(root_dir, subject)
        for scenario in os.listdir(subj_path):
            scen_path = os.path.join(subj_path, scenario)
            files = [os.path.join(scen_path, f)
                     for f in os.listdir(scen_path) if f.lower().endswith(".jpg")]
            all_files.extend(files)

    # single progress bar for all frames
    for img_path in tqdm(all_files, desc="Processing Frames"):
        try:
            # prepare output path
            parts = img_path.split(os.sep)
            subject, scenario, fname = parts[-3], parts[-2], parts[-1]
            out_scen = os.path.join(out_dir, subject, scenario)
            os.makedirs(out_scen, exist_ok=True)

            img = Image.open(img_path).convert("L")
            img_array = np.array(img, dtype=np.float32)/255.0
            tensor = torch.from_numpy(img_array)
            torch.save(tensor, os.path.join(out_scen, fname.replace(".jpg",".pt")))
        except Exception as e:
            print(f"Failed: {img_path}, {e}")
            continue

root_dir = "/root/.cache/kagglehub/datasets/manith04/ddd-processed-validation-frames-type-1/versions/1"
out_dir  = "/content/validation/"

save_frames_as_tensor(root_dir, out_dir)