import cv2
import numpy as np
import mediapipe as mp

# Load image
image = cv2.imread("driver1.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

results = face_mesh.process(image_rgb)

if results.multi_face_landmarks:
    landmarks = results.multi_face_landmarks[0].landmark

    # 3D model points of selected facial landmarks
    model_points = np.array([
        [0.0, 0.0, 0.0],             # Nose tip
        [0.0, -63.6, -12.5],         # Chin
        [-43.3, 32.7, -26.0],        # Left eye left corner
        [43.3, 32.7, -26.0],         # Right eye right corner
        [-28.9, -28.9, -24.1],       # Left Mouth corner
        [28.9, -28.9, -24.1]         # Right mouth corner
    ])

    # 2D image points
    image_points = np.array([
        [landmarks[1].x * image.shape[1], landmarks[1].y * image.shape[0]],      # Nose tip
        [landmarks[152].x * image.shape[1], landmarks[152].y * image.shape[0]],  # Chin
        [landmarks[33].x * image.shape[1], landmarks[33].y * image.shape[0]],    # Left eye left corner
        [landmarks[263].x * image.shape[1], landmarks[263].y * image.shape[0]],  # Right eye right corner
        [landmarks[78].x * image.shape[1], landmarks[78].y * image.shape[0]],    # Left mouth corner
        [landmarks[308].x * image.shape[1], landmarks[308].y * image.shape[0]]   # Right mouth corner
    ], dtype="double")

    # Camera matrix
    focal_length = image.shape[1]
    center = (image.shape[1] / 2, image.shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # No distortion

    # SolvePnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    # Project 3D axis points
    axis_length = 60
    axis_points = np.float32([
        [axis_length, 0, 0],  # X axis
        [0, axis_length, 0],  # Y axis
        [0, 0, axis_length]   # Z axis
    ])
    imgpts, _ = cv2.projectPoints(axis_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    p_origin = (int(image_points[0][0]), int(image_points[0][1]))

    # Draw axes
    cv2.line(image, p_origin, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 2)  # X - Red
    cv2.line(image, p_origin, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 2)  # Y - Green
    cv2.line(image, p_origin, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 2)  # Z - Blue

    # Offset for labels so they don't cover the lines
    offset = 20
    cv2.putText(image, 'X', (int(imgpts[0][0][0] + (offset - 15)), int(imgpts[0][0][1] - (offset - 27))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(image, 'Y', (int(imgpts[1][0][0] - (offset- 13)), int(imgpts[1][0][1] - (offset- 10))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(image, 'Z', (int(imgpts[2][0][0] - offset),int(imgpts[2][0][1] + (offset - 10))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # Get Euler angles
    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rmat[0,0]*rmat[0,0] + rmat[1,0]*rmat[1,0])
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(-rmat[2,0], sy)
        yaw = np.arctan2(rmat[1,0], rmat[0,0])
        roll = np.arctan2(rmat[2,1], rmat[2,2])
    else:
        pitch = np.arctan2(-rmat[2,0], sy)
        yaw = np.arctan2(-rmat[0,1], rmat[1,1])
        roll = 0

    print(f"Pitch: {np.degrees(pitch):.2f}, Yaw: {np.degrees(yaw):.2f}, Roll: {np.degrees(roll):.2f}")

    cv2.imshow("Head Pose", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No face detected.")
