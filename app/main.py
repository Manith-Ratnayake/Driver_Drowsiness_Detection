import sys
import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- MediaPipe setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# --- Landmarks ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

# --- Calculate EAR ---
def calculate_EAR(landmarks, eye_indices, frame_shape):
    h, w = frame_shape[:2]
    pts = np.array([[int(landmarks[idx].x * w), int(landmarks[idx].y * h)] for idx in eye_indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)

# --- Calculate MAR ---
def calculate_MAR(landmarks, mouth_indices, frame_shape):
    h, w = frame_shape[:2]
    pts = np.array([[int(landmarks[idx].x * w), int(landmarks[idx].y * h)] for idx in mouth_indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[3] - pts[7])
    D = np.linalg.norm(pts[0] - pts[6])
    return (A + B + C) / (2.0 * D)

# --- Matplotlib canvas ---
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.set_title("EAR / MAR over time")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Value")
        self.ax.set_ylim(0, 0.5)
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

        self.ax.set_xlim(0, max(1, self.x_data[-1]))
        self.ax.figure.canvas.draw()

# --- Main PyQt app ---
class App(QWidget):
    def __init__(self, video_path="video1.avi"):
        super().__init__()

        # Video display
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setFixedSize(640, 480)

        # Stats display
        self.stats_label = QLabel("EAR/MAR values")
        self.stats_label.setAlignment(Qt.AlignTop)
        self.stats_label.setStyleSheet("font-size: 16px;")

        # Drowsiness indicator
        self.drowsy_label = QLabel("NOT DROWSY")
        self.drowsy_label.setAlignment(Qt.AlignCenter)
        self.drowsy_label.setFixedHeight(50)
        self.drowsy_label.setStyleSheet("background-color: green; color: white; font-size: 18px;")

        # Buttons
        self.play_btn = QPushButton("Pause")
        self.replay_btn = QPushButton("Replay")
        self.play_btn.clicked.connect(self.toggle_play)
        self.replay_btn.clicked.connect(self.replay_video)

        # Matplotlib canvas
        self.plot_canvas = PlotCanvas(self, width=5, height=4)

        # Right side layout (stats + drowsy + plot)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.stats_label)
        right_layout.addWidget(self.drowsy_label)
        right_layout.addWidget(self.plot_canvas)
        right_layout.addStretch()

        # Buttons layout below video
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_btn)
        button_layout.addWidget(self.replay_btn)

        # Left side layout (video + buttons)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label)
        left_layout.addLayout(button_layout)

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("Error: Cannot open video")
            sys.exit()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.playing = True
        self.frame_idx = 0
        self.ear_threshold = 0.25  # EAR threshold for drowsiness
        self.setWindowTitle("EAR + MAR + Drowsiness Monitor")
        self.setFixedSize(1200, 600)

    def update_frame(self):
        if not self.playing:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        left_ear, right_ear, mar = 0.0, 0.0, 0.0
        stats_text = "Face not detected"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = calculate_EAR(landmarks, LEFT_EYE, frame.shape)
            right_ear = calculate_EAR(landmarks, RIGHT_EYE, frame.shape)
            mar = calculate_MAR(landmarks, MOUTH, frame.shape)
            stats_text = f"Left EAR: {left_ear:.3f}\nRight EAR: {right_ear:.3f}\nMAR: {mar:.3f}"

        self.stats_label.setText(stats_text)

        # --- Update drowsiness indicator ---
        avg_ear = (left_ear + right_ear) / 2.0
        if avg_ear < self.ear_threshold:
            self.drowsy_label.setText("DROWSY")
            self.drowsy_label.setStyleSheet("background-color: red; color: white; font-size: 18px;")
        else:
            self.drowsy_label.setText("NOT DROWSY")
            self.drowsy_label.setStyleSheet("background-color: green; color: white; font-size: 18px;")

        # Update plot
        self.plot_canvas.update_plot(self.frame_idx, left_ear, right_ear, mar)
        self.frame_idx += 1

        # Display video
        h, w, ch = frame_rgb.shape
        img = QImage(frame_rgb.data, w, h, ch*w, QImage.Format_RGB888)
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
        self.plot_canvas.x_data.clear()
        self.plot_canvas.left_ear_data.clear()
        self.plot_canvas.right_ear_data.clear()
        self.plot_canvas.mar_data.clear()
        self.plot_canvas.ax.cla()
        self.plot_canvas.ax.set_title("EAR / MAR over time")
        self.plot_canvas.ax.set_xlabel("Frame")
        self.plot_canvas.ax.set_ylabel("Value")
        self.plot_canvas.ax.set_ylim(0, 0.5)
        self.plot_canvas.ax.grid(True)
        self.plot_canvas.line_left_ear, = self.plot_canvas.ax.plot([], [], label="Left EAR", color="blue")
        self.plot_canvas.line_right_ear, = self.plot_canvas.ax.plot([], [], label="Right EAR", color="green")
        self.plot_canvas.line_mar, = self.plot_canvas.ax.plot([], [], label="MAR", color="red")
        self.plot_canvas.ax.legend()
        self.timer.start(30)
        self.playing = True
        self.play_btn.setText("Pause")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App("video1.avi")  # replace with your video path
    window.show()
    sys.exit(app.exec_())
