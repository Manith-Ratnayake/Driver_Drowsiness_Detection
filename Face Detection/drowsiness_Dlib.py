import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import face_recognition
from scipy.spatial import distance
import warnings
warnings.filterwarnings("ignore")

# === EAR & MAR Thresholds ===
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.6

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

def process_image(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    eye_flag = mouth_flag = False

    for face_location in face_locations:
        landmarks = face_recognition.face_landmarks(rgb_frame, [face_location])[0]

        if 'left_eye' in landmarks and 'right_eye' in landmarks and 'bottom_lip' in landmarks:
            left_eye = np.array(landmarks['left_eye'])
            right_eye = np.array(landmarks['right_eye'])
            mouth = np.array(landmarks['bottom_lip'])

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            mar = mouth_aspect_ratio(mouth)

            if ear < EYE_AR_THRESH:
                eye_flag = True
            if mar > MOUTH_AR_THRESH:
                mouth_flag = True

            # Draw landmarks
            for (x, y) in np.concatenate([left_eye, right_eye, mouth]):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    return frame, eye_flag, mouth_flag

# === User Prompt ===
mode = input("Enter 1 for image mode, 2 for webcam video mode: ")

# === IMAGE MODE ===
if mode == '1':
    img_path = "driver.png"
    if not os.path.exists(img_path):
        raise FileNotFoundError("Image not found. Please check the path.")

    image = cv2.imread(img_path)
    image = cv2.resize(image, (640, 480))

    _, eye_flag, mouth_flag = process_image(image)

    # Show result
    if eye_flag or mouth_flag:
        cv2.putText(image, "DROWSY", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        cv2.putText(image, "AWAKE", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Drowsiness Detection - Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === VIDEO MODE ===
elif mode == '2':
    video_cap = cv2.VideoCapture(0)
    if not video_cap.isOpened():
        raise RuntimeError("Webcam not found or cannot be opened.")

    print("Press 'q' or ESC to quit video mode.")

    count = score = 0

    while True:
        success, frame = video_cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        count += 1

        if count % 5 == 0:
            frame, eye_flag, mouth_flag = process_image(frame)

            if eye_flag or mouth_flag:
                score += 1
            else:
                score -= 1
                if score < 0:
                    score = 0

        # Display score
        cv2.putText(frame, f"Score: {score}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if score >= 5:
            cv2.putText(frame, "DROWSY", (frame.shape[1] - 180, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow("Drowsiness Detection - Video", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # ESC
            break

    video_cap.release()
    cv2.destroyAllWindows()

else:
    print("Invalid choice. Enter 1 or 2.")
