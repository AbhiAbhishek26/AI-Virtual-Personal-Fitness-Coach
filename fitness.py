import cv2
import mediapipe as mp
import numpy as np
import time


# =========================================================
# MEDIAPIPE INITIALIZATION
# =========================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# =========================================================
# ANGLE CALCULATION
# =========================================================

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


# =========================================================
# SQUAT COUNTER
# =========================================================

class SquatCounter:

    def __init__(self):

        self.counter = 0
        self.stage = None

    def update(self, landmarks):

        required = [
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value
        ]

        # Check landmark visibility
        for landmark_id in required:

            if landmarks[landmark_id].visibility < 0.5:

                return None, self.counter, self.stage

        # Hip
        hip = [
            landmarks[
                mp_pose.PoseLandmark.RIGHT_HIP.value
            ].x,
            landmarks[
                mp_pose.PoseLandmark.RIGHT_HIP.value
            ].y
        ]

        # Knee
        knee = [
            landmarks[
                mp_pose.PoseLandmark.RIGHT_KNEE.value
            ].x,
            landmarks[
                mp_pose.PoseLandmark.RIGHT_KNEE.value
            ].y
        ]

        # Ankle
        ankle = [
            landmarks[
                mp_pose.PoseLandmark.RIGHT_ANKLE.value
            ].x,
            landmarks[
                mp_pose.PoseLandmark.RIGHT_ANKLE.value
            ].y
        ]

        # Calculate knee angle
        angle = calculate_angle(
            hip,
            knee,
            ankle
        )

        # Down
        if angle < 90:

            self.stage = "down"

        # Up + count rep
        if angle > 160 and self.stage == "down":

            self.stage = "up"
            self.counter += 1

        return angle, self.counter, self.stage


# =========================================================
# BICEP CURL COUNTER
# =========================================================

class BicepCurlCounter:

    def __init__(self):

        self.counter = 0
        self.stage = None

    def update(self, landmarks):

        required = [
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value
        ]

        # Check visibility
        for landmark_id in required:

            if landmarks[landmark_id].visibility < 0.5:

                return None, self.counter, self.stage

        # Shoulder
        shoulder = [
            landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            ].x,
            landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            ].y
        ]

        # Elbow
        elbow = [
            landmarks[
                mp_pose.PoseLandmark.RIGHT_ELBOW.value
            ].x,
            landmarks[
                mp_pose.PoseLandmark.RIGHT_ELBOW.value
            ].y
        ]

        # Wrist
        wrist = [
            landmarks[
                mp_pose.PoseLandmark.RIGHT_WRIST.value
            ].x,
            landmarks[
                mp_pose.PoseLandmark.RIGHT_WRIST.value
            ].y
        ]

        # Calculate elbow angle
        angle = calculate_angle(
            shoulder,
            elbow,
            wrist
        )

        # Arm curled
        if angle < 50:

            self.stage = "up"

        # Arm extended + count
        if angle > 160 and self.stage == "up":

            self.stage = "down"
            self.counter += 1

        return angle, self.counter, self.stage


# =========================================================
# MAIN FITNESS FUNCTION
# =========================================================

def run_fitness():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")
        return

    squat_counter = SquatCounter()

    curl_counter = BicepCurlCounter()

    start_time = time.time()

    # Default exercise
    exercise = "squat"

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:

                print("ERROR: Could not read webcam.")
                break

            # Flip camera
            frame = cv2.flip(frame, 1)

            # Convert BGR → RGB
            image_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Process pose
            results = pose.process(image_rgb)

            # Convert RGB → BGR
            image = cv2.cvtColor(
                image_rgb,
                cv2.COLOR_RGB2BGR
            )

            # Default values

            angle = None
            reps = 0
            stage = None
            feedback = "No body detected"

            # =================================================
            # POSE DETECTED
            # =================================================

            if results.pose_landmarks:

                landmarks = results.pose_landmarks.landmark

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # =================================================
                # SQUAT
                # =================================================

                if exercise == "squat":

                    angle, reps, stage = squat_counter.update(
                        landmarks
                    )

                    if angle is None:

                        feedback = "Move into camera view"

                    elif angle < 90:

                        feedback = "Good Depth"

                    elif angle < 120:

                        feedback = "Go Lower"

                    elif angle > 160:

                        feedback = "Stand Straight"

                    else:

                        feedback = "Keep Going"

                # =================================================
                # BICEP CURL
                # =================================================

                elif exercise == "curl":

                    angle, reps, stage = curl_counter.update(
                        landmarks
                    )

                    if angle is None:

                        feedback = "Move into camera view"

                    elif angle < 50:

                        feedback = "Good Curl"

                    elif angle > 160:

                        feedback = "Extend Arm"

                    else:

                        feedback = "Keep Curling"

                # =================================================
                # DISPLAY ANGLE
                # =================================================

                if angle is not None:

                    cv2.putText(
                        image,
                        f"Angle: {int(angle)}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                # =================================================
                # DISPLAY REPS
                # =================================================

                cv2.putText(
                    image,
                    f"Reps: {reps}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

                # =================================================
                # DISPLAY STAGE
                # =================================================

                cv2.putText(
                    image,
                    f"Stage: {stage}",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                # =================================================
                # DISPLAY FEEDBACK
                # =================================================

                cv2.putText(
                    image,
                    f"Feedback: {feedback}",
                    (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    image,
                    "No body detected",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # =================================================
            # TIMER
            # =================================================

            elapsed_time = int(
                time.time() - start_time
            )

            minutes = elapsed_time // 60
            seconds = elapsed_time % 60

            cv2.putText(
                image,
                f"Time: {minutes:02d}:{seconds:02d}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # =================================================
            # CURRENT EXERCISE
            # =================================================

            cv2.putText(
                image,
                f"Exercise: {exercise.upper()}",
                (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # =================================================
            # INSTRUCTIONS
            # =================================================

            cv2.putText(
                image,
                "Press S = Squat | C = Curl | Q = Quit",
                (20, image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # =================================================
            # SHOW CAMERA
            # =================================================

            cv2.imshow(
                "AI Virtual Personal Fitness Coach",
                image
            )

            # =================================================
            # KEYBOARD CONTROLS
            # =================================================

            key = cv2.waitKey(10) & 0xFF

            if key == ord("q"):

                break

            elif key == ord("s"):

                exercise = "squat"

            elif key == ord("c"):

                exercise = "curl"

    # =========================================================
    # RELEASE CAMERA
    # =========================================================

    cap.release()
    cv2.destroyAllWindows()


# =========================================================
# START PROGRAM
# =========================================================

if __name__ == "__main__":

    run_fitness()