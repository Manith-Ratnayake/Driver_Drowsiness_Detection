import os

class DDDSequenceBuilder:
    def __init__(self, data_root_dir, time_window_in_seconds,
                 total_frames_in_window, time_window_right_shift_by_seconds,
                 save_file=None):
        self.data_root_dir = data_root_dir
        self.time_window_in_seconds = time_window_in_seconds
        self.total_frames_in_window = total_frames_in_window
        self.time_window_right_shift_by_seconds = time_window_right_shift_by_seconds

        os.makedirs("prebuilt", exist_ok=True)
        self.save_file = save_file or f"prebuilt/{time_window_in_seconds}_{total_frames_in_window}_{time_window_right_shift_by_seconds}.txt"
        self.sequence_frame_paths_and_label = []  # descriptive

    def build(self):
        for subject in os.listdir(self.data_root_dir):
            subject_path = os.path.join(self.data_root_dir, subject)

            for scenario in os.listdir(subject_path):
                video_path = os.path.join(subject_path, scenario)
                frame_files = sorted(f for f in os.listdir(video_path) if f.lower().endswith(".jpg"))
                total_frames = len(frame_files)
                if total_frames == 0:
                    continue

                fps = 15 if scenario in ["night_noglasses", "night_glasses"] else 30
                window_frame_count = min(total_frames, int(fps * self.time_window_in_seconds))
                stride = max(1, int(round(self.time_window_right_shift_by_seconds * fps)))
                num_windows = max(1, (total_frames - window_frame_count) // stride + 1)
                frame_time_interval = self.time_window_in_seconds / self.total_frames_in_window

                for w in range(num_windows):
                    frame_paths = []

                    for i in range(self.total_frames_in_window):
                        t = w * self.time_window_right_shift_by_seconds + i * frame_time_interval
                        idx = min(int(round(t * fps)), total_frames - 1)
                        fname = frame_files[idx]

                        # store **relative path** only
                        relative_path = os.path.join(subject, scenario, fname)
                        frame_paths.append(relative_path)

                    # Only save last frame label
                    last_frame_fname = frame_files[min(
                        int(round((w*self.time_window_right_shift_by_seconds + (self.total_frames_in_window-1)*frame_time_interval)*fps)),
                        total_frames-1
                    )]
                    last_label = int(last_frame_fname.rsplit("_", 1)[-1].split(".")[0])

                    self.sequence_frame_paths_and_label.append((frame_paths, last_label))

        self._save()
        print(f"Prebuilt sequences saved to {self.save_file} ({len(self.sequence_frame_paths_and_label)} sequences).")

    def _save(self):
        with open(self.save_file, "w") as f:
            for frame_paths, label in self.sequence_frame_paths_and_label:
                f.write(f"{','.join(frame_paths)}|{label}\n")