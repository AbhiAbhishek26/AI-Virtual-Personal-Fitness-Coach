```python
import cv2
import mediapipe as mp
import numpy as np


# Calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Unable to access webcam.")
            break

        # Flip the frame like a mirror
        frame = cv2.flip(frame, 1)

        # Get frame dimensions
        height, width, _ = frame.shape

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect pose
        results = pose.process(rgb_frame)

        # Draw body landmarks
        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Get landmarks
            landmarks = results.pose_landmarks.landmark

            # Right hip
            hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            # Right knee
            knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

            # Right ankle
            ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            # Convert coordinates to pixels
            hip_point = [hip.x * width, hip.y * height]
            knee_point = [knee.x * width, knee.y * height]
            ankle_point = [ankle.x * width, ankle.y * height]

            # Calculate knee angle
            angle = calculate_angle(
                hip_point,
                knee_point,
                ankle_point
            )

            # Display knee angle
            cv2.putText(
                frame,
                "Knee Angle: " + str(int(angle)),
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # Display video
        cv2.imshow(
            "AI Fitness Coach - Knee Angle",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release webcam
cap.release()
cv2.destroyAllWindows()
```
