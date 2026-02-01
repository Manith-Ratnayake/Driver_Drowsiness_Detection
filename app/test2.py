
import cv2

# Path to your video
video_path = "video1.avi"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    # Show frame in a window
    cv2.imshow("Video", frame)

    # Wait 30 ms and check if 'q' is pressed to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
