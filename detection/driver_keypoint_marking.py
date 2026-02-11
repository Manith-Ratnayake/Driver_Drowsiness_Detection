import mediapipe as mp
import cv2
import numpy as np
import os

print("Mediapipe version:", mp.__version__)
print("cv2 version:", cv2.__version__)

# Eye & lips landmarks
# LEFT_EYE_OUTER = [ 
#     130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25
# ]


# RIGHT_EYE_OUTER = [
#     286, 258, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341
#     ]

LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 
    321, 375, 291, 308, 324, 318, 402, 
    317, 14, 87, 178, 88, 95, 185, 40, 
    39, 37, 0, 267, 269, 270, 409, 291
]

#LEFT_EYE = [
#     160, 159, 158, 157, 173, 133 , 155, 154, 153, 145, 144, 163, 7, 33
# ]

# RIGHT_EYE = [
#     384, 385, 386, 387, 388, 466, 263,  249, 390, 373, 374, 380, 381, 382, 369
# ]


LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]



mp_face_mesh = mp.solutions.face_mesh

def get_landmark(image):
    """Run mediapipe landmark detection."""
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        rgb_image.flags.writeable = False
        result = face_mesh.process(rgb_image)
        return result

def draw_outer_contours(image, landmarks):
    """Draw polygons for eyes and lips."""
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    h, w, _ = image.shape

    for face in landmarks.multi_face_landmarks:
        def draw_polygon(indices, color):
            pts = np.array([(int(face.landmark[i].x * w), int(face.landmark[i].y * h)) for i in indices])
            if len(pts) > 2:  # convex hull requires at least 3 points
                hull = cv2.convexHull(pts)
                cv2.polylines(image, [hull], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

        # Draw polygons
        draw_polygon(LEFT_EYE, (0, 255, 0))
        draw_polygon(RIGHT_EYE, (0, 255, 0))
        draw_polygon(LIPS_OUTER, (0, 255, 0))

    return image


def process_folder(input_folder, output_folder, log_file):
    """Process all images in folder and save annotated results."""
    os.makedirs(output_folder, exist_ok=True)

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)

        if not (image_path.lower().endswith(".jpg") or image_path.lower().endswith(".png")):
            continue

        #print(f"Processing {image_path} ...")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Detect landmarks
        result = get_landmark(image=image)

        if result.multi_face_landmarks:
            # Draw contours
            annotated_image = draw_outer_contours(image.copy(), result)
            save_path = os.path.join(output_folder, image_name)
            cv2.imwrite(save_path, annotated_image)
        else:
            print(f"❌ No eyes detected in {image_path}")
            with open(log_file, "a") as f:
                f.write(image_path + "\n")


if __name__ == "__main__":

    log_file = "no_eye.txt"
    open(log_file, "w").close()
    process_folder("drowsy", "marked_drowsy", log_file)
    process_folder("notdrowsy", "marked_notdrowsy", log_file)
    print("✅ Processing complete.")
