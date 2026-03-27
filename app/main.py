import sys
import cv2
import math
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



model_path = "face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.2,
    min_face_presence_confidence=0.2,
    min_tracking_confidence=0.2
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)



LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

LEFT_EYE_CENTER_POINTS = [33, 133]
RIGHT_EYE_CENTER_POINTS = [362, 263]
NOSE_TIP = 1
CHIN = 152



def landmark_to_xy(landmark, frame_shape):
    h, w = frame_shape[:2]
    return np.array([landmark.x * w, landmark.y * h], dtype=np.float32)


def calculate_EAR(landmarks, eye_indices, frame_shape):
    h, w = frame_shape[:2]
    pts = np.array(
        [[landmarks[idx].x * w, landmarks[idx].y * h] for idx in eye_indices],
        dtype=np.float32
    )

    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])

    if C == 0:
        return 0.0

    return float((A + B) / (2.0 * C))


def calculate_MAR(landmarks, mouth_indices, frame_shape):
    h, w = frame_shape[:2]
    pts = np.array(
        [[landmarks[idx].x * w, landmarks[idx].y * h] for idx in mouth_indices],
        dtype=np.float32
    )

    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[3] - pts[7])
    D = np.linalg.norm(pts[0] - pts[6])

    if D == 0:
        return 0.0

    return float((A + B + C) / (2.0 * D))


def calculate_head_roll(landmarks, frame_shape):
    left_eye_outer = landmark_to_xy(landmarks[LEFT_EYE_CENTER_POINTS[0]], frame_shape)
    left_eye_inner = landmark_to_xy(landmarks[LEFT_EYE_CENTER_POINTS[1]], frame_shape)
    right_eye_outer = landmark_to_xy(landmarks[RIGHT_EYE_CENTER_POINTS[0]], frame_shape)
    right_eye_inner = landmark_to_xy(landmarks[RIGHT_EYE_CENTER_POINTS[1]], frame_shape)

    left_center = (left_eye_outer + left_eye_inner) / 2.0
    right_center = (right_eye_outer + right_eye_inner) / 2.0

    dx = right_center[0] - left_center[0]
    dy = right_center[1] - left_center[1]

    angle_deg = math.degrees(math.atan2(dy, dx))
    return float(angle_deg)


def classify_eye_output(left_ear, right_ear, ear_threshold=0.25):
    avg_ear = (left_ear + right_ear) / 2.0

    if avg_ear < ear_threshold:
        return "DROWSY EYES", avg_ear
    return "EYES OPEN", avg_ear


def classify_mouth_output(mar, mar_threshold=0.60):
    if mar > mar_threshold:
        return "YAWNING", mar
    return "MOUTH NORMAL", mar


def classify_head_output(head_roll_deg, roll_threshold=10.0):
    if head_roll_deg > roll_threshold:
        return "HEAD TILTED RIGHT", head_roll_deg
    if head_roll_deg < -roll_threshold:
        return "HEAD TILTED LEFT", head_roll_deg
    return "HEAD STRAIGHT", head_roll_deg



class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax.set_title("EAR / MAR over time")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Value")
        self.ax.set_ylim(0, 1.0)
        self.ax.grid(True)

        self.line_left_ear, = self.ax.plot([], [], label="Left EAR", color="blue")
        self.line_right_ear, = self.ax.plot([], [], label="Right EAR", color="green")
        self.line_mar, = self.ax.plot([], [], label="MAR", color="red")

        self.ax.legend()

        self.x_data = []
        self.left_ear_data = []
        self.right_ear_data = []
        self.mar_data = []

    def update_plot(self, frame_idx, left_ear, right_ear, mar):
        self.x_data.append(frame_idx)
        self.left_ear_data.append(left_ear)
        self.right_ear_data.append(right_ear)
        self.mar_data.append(mar)

        self.line_left_ear.set_data(self.x_data, self.left_ear_data)
        self.line_right_ear.set_data(self.x_data, self.right_ear_data)
        self.line_mar.set_data(self.x_data, self.mar_data)

        self.ax.set_xlim(0, max(1, frame_idx))
        self.ax.figure.canvas.draw()

    def reset_plot(self):
        self.x_data.clear()
        self.left_ear_data.clear()
        self.right_ear_data.clear()
        self.mar_data.clear()

        self.ax.cla()
        self.ax.set_title("EAR / MAR over time")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Value")
        self.ax.set_ylim(0, 1.0)
        self.ax.grid(True)

        self.line_left_ear, = self.ax.plot([], [], label="Left EAR", color="blue")
        self.line_right_ear, = self.ax.plot([], [], label="Right EAR", color="green")
        self.line_mar, = self.ax.plot([], [], label="MAR", color="red")

        self.ax.legend()
        self.draw()



class App(QWidget):
    def __init__(self, video_path="video1.avi"):
        super().__init__()

        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setFixedSize(640, 480)

        self.stats_label = QLabel("Numerical outputs")
        self.stats_label.setAlignment(Qt.AlignTop)
        self.stats_label.setStyleSheet("font-size: 16px;")

        self.model_output_label = QLabel("Model outputs")
        self.model_output_label.setAlignment(Qt.AlignTop)
        self.model_output_label.setStyleSheet(
            "font-size: 16px; padding: 8px; border: 1px solid gray;"
        )

        self.drowsy_label = QLabel("NOT DROWSY")
        self.drowsy_label.setAlignment(Qt.AlignCenter)
        self.drowsy_label.setFixedHeight(50)
        self.drowsy_label.setStyleSheet(
            "background-color: green; color: white; font-size: 18px;"
        )

        self.play_btn = QPushButton("Pause")
        self.replay_btn = QPushButton("Replay")
        self.play_btn.clicked.connect(self.toggle_play)
        self.replay_btn.clicked.connect(self.replay_video)

        self.plot_canvas = PlotCanvas(self, width=5, height=4)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.stats_label)
        right_layout.addWidget(self.model_output_label)
        right_layout.addWidget(self.drowsy_label)
        right_layout.addWidget(self.plot_canvas)
        right_layout.addStretch()

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_btn)
        button_layout.addWidget(self.replay_btn)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label)
        left_layout.addLayout(button_layout)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("Error: Cannot open video")
            sys.exit()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.playing = True
        self.frame_idx = 0

        self.ear_threshold = 0.25
        self.mar_threshold = 0.60
        self.roll_threshold = 10.0

        self.setWindowTitle("Head + Eye + Mouth Output Monitor")
        self.setFixedSize(1250, 650)

    def run_landmarker(self, frame_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        return face_landmarker.detect_for_video(mp_image, timestamp_ms)

    def update_frame(self):
        if not self.playing:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.run_landmarker(frame_rgb)

        left_ear, right_ear, mar, head_roll = 0.0, 0.0, 0.0, 0.0
        stats_text = "Face not detected"
        model_text = (
            "Eye model output   : N/A\n"
            "Mouth model output : N/A\n"
            "Head model output  : N/A"
        )

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_ear = calculate_EAR(landmarks, LEFT_EYE, frame.shape)
            right_ear = calculate_EAR(landmarks, RIGHT_EYE, frame.shape)
            mar = calculate_MAR(landmarks, MOUTH, frame.shape)
            head_roll = calculate_head_roll(landmarks, frame.shape)

            eye_state, avg_ear = classify_eye_output(
                left_ear=left_ear,
                right_ear=right_ear,
                ear_threshold=self.ear_threshold
            )
            mouth_state, mouth_score = classify_mouth_output(
                mar=mar,
                mar_threshold=self.mar_threshold
            )
            head_state, head_score = classify_head_output(
                head_roll_deg=head_roll,
                roll_threshold=self.roll_threshold
            )

            stats_text = (
                f"Left EAR   : {left_ear:.3f}\n"
                f"Right EAR  : {right_ear:.3f}\n"
                f"Avg EAR    : {avg_ear:.3f}\n"
                f"MAR        : {mouth_score:.3f}\n"
                f"Head Roll  : {head_score:.2f}°"
            )

            model_text = (
                f"Eye model output   : {eye_state}\n"
                f"Mouth model output : {mouth_state}\n"
                f"Head model output  : {head_state}"
            )

            # overall decision
            if eye_state == "DROWSY EYES" or mouth_state == "YAWNING":
                self.drowsy_label.setText("DROWSY")
                self.drowsy_label.setStyleSheet(
                    "background-color: red; color: white; font-size: 18px;"
                )
            else:
                self.drowsy_label.setText("NOT DROWSY")
                self.drowsy_label.setStyleSheet(
                    "background-color: green; color: white; font-size: 18px;"
                )

            # optional landmark drawing
            h, w = frame.shape[:2]
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame_rgb, (x, y), 1, (0, 255, 0), -1)

        else:
            self.drowsy_label.setText("NO FACE")
            self.drowsy_label.setStyleSheet(
                "background-color: orange; color: white; font-size: 18px;"
            )

        self.stats_label.setText(stats_text)
        self.model_output_label.setText(model_text)

        self.plot_canvas.update_plot(self.frame_idx, left_ear, right_ear, mar)
        self.frame_idx += 1

        h, w, ch = frame_rgb.shape
        img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))


    def toggle_play(self):
        if self.playing:
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            self.timer.start(30)
            self.play_btn.setText("Pause")
        self.playing = not self.playing


    def replay_video(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_idx = 0
        self.plot_canvas.reset_plot()

        self.stats_label.setText("Numerical outputs")
        self.model_output_label.setText("Model outputs")
        self.drowsy_label.setText("NOT DROWSY")
        self.drowsy_label.setStyleSheet(
            "background-color: green; color: white; font-size: 18px;"
        )

        self.timer.start(30)
        self.playing = True
        self.play_btn.setText("Pause")


if __name__ == "__main__":
    video_path = "video1.avi"
    app = QApplication(sys.argv)
    window = App(video_path)
    window.show()
    sys.exit(app.exec_())