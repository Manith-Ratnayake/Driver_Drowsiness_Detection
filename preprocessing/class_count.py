from collections import Counter
import os

all_labels = []
root_dir = ""

for subject in os.listdir(root_dir):
    subject_path = os.path.join(root_dir, subject)
    if not os.path.isdir(subject_path):
        continue
    for scenario in os.listdir(subject_path):
        scenario_path = os.path.join(subject_path, scenario)
        if not os.path.isdir(scenario_path):
            continue
        frame_files = sorted([f for f in os.listdir(scenario_path) if f.endswith(".jpg")])
        for fname in frame_files:
            label = int(fname.rsplit("_", 1)[-1].split(".")[0])
            all_labels.append(label)

label_counts = Counter(all_labels)
print(label_counts)