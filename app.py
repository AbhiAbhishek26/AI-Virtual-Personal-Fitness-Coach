import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import av
import threading
import csv
import os
import json
import pandas as pd
from datetime import datetime

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Virtual Personal Fitness Coach",
    page_icon="🏋️",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

WORKOUT_FILE = os.path.join(
    BASE_DIR,
    "current_workout.json"
)

HISTORY_FILE = os.path.join(
    BASE_DIR,
    "workout_logs.csv"
)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# CALCULATE ANGLE
# ============================================================

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = (
        np.arctan2(
            c[1] - b[1],
            c[0] - b[0]
        )
        -
        np.arctan2(
            a[1] - b[1],
            a[0] - b[0]
        )
    )

    angle = np.abs(
        radians * 180.0 / np.pi
    )

    if angle > 180:
        angle = 360 - angle

    return angle


# ============================================================
# CURRENT WORKOUT FILE
# ============================================================

def write_current_workout(
    exercise,
    reps,
    angle,
    stage,
    feedback,
    duration
):

    data = {
        "exercise": exercise,
        "reps": reps,
        "angle": angle,
        "stage": stage,
        "feedback": feedback,
        "duration": duration
    }

    temp_file = WORKOUT_FILE + ".tmp"

    try:

        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file
            )

        os.replace(
            temp_file,
            WORKOUT_FILE
        )

    except Exception:
        pass


def read_current_workout():

    if not os.path.exists(
        WORKOUT_FILE
    ):
        return None

    try:

        with open(
            WORKOUT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return None


def clear_current_workout():

    try:

        if os.path.exists(
            WORKOUT_FILE
        ):

            os.remove(
                WORKOUT_FILE
            )

    except Exception:
        pass


# ============================================================
# SAVE WORKOUT
# ============================================================

def save_workout(
    exercise,
    reps,
    duration
):

    try:

        file_exists = os.path.exists(
            HISTORY_FILE
        )

        with open(
            HISTORY_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(
                file
            )

            if not file_exists:

                writer.writerow([
                    "Date",
                    "Exercise",
                    "Reps",
                    "Duration",
                    "Feedback"
                ])

            writer.writerow([
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                exercise,
                reps,
                duration,
                "Workout completed"
            ])

        return True

    except Exception as e:

        st.error(
            f"Error saving workout: {e}"
        )

        return False


# ============================================================
# READ WORKOUT HISTORY
# ============================================================

def read_workout_history():

    if not os.path.exists(
        HISTORY_FILE
    ):

        return None

    try:

        history = pd.read_csv(
            HISTORY_FILE
        )

        return history

    except Exception as e:

        st.error(
            f"Error reading workout history: {e}"
        )

        return None


# ============================================================
# SQUAT COUNTER
# ============================================================

class SquatCounter:

    def __init__(self):

        self.counter = 0
        self.stage = "up"

    def update(
        self,
        landmarks
    ):

        hip_id = (
            mp_pose.PoseLandmark.RIGHT_HIP.value
        )

        knee_id = (
            mp_pose.PoseLandmark.RIGHT_KNEE.value
        )

        ankle_id = (
            mp_pose.PoseLandmark.RIGHT_ANKLE.value
        )

        required = [
            hip_id,
            knee_id,
            ankle_id
        ]

        for landmark_id in required:

            if (
                landmarks[
                    landmark_id
                ].visibility < 0.5
            ):

                return (
                    None,
                    self.counter,
                    self.stage
                )

        hip = [
            landmarks[hip_id].x,
            landmarks[hip_id].y
        ]

        knee = [
            landmarks[knee_id].x,
            landmarks[knee_id].y
        ]

        ankle = [
            landmarks[ankle_id].x,
            landmarks[ankle_id].y
        ]

        angle = calculate_angle(
            hip,
            knee,
            ankle
        )

        if angle < 100:

            self.stage = "down"

        if (
            angle > 160
            and self.stage == "down"
        ):

            self.stage = "up"

            self.counter += 1

        return (
            angle,
            self.counter,
            self.stage
        )


# ============================================================
# BICEP CURL COUNTER
# ============================================================

class BicepCurlCounter:

    def __init__(self):

        self.counter = 0
        self.stage = "down"

    def update(
        self,
        landmarks
    ):

        shoulder_id = (
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        )

        elbow_id = (
            mp_pose.PoseLandmark.RIGHT_ELBOW.value
        )

        wrist_id = (
            mp_pose.PoseLandmark.RIGHT_WRIST.value
        )

        required = [
            shoulder_id,
            elbow_id,
            wrist_id
        ]

        for landmark_id in required:

            if (
                landmarks[
                    landmark_id
                ].visibility < 0.5
            ):

                return (
                    None,
                    self.counter,
                    self.stage
                )

        shoulder = [
            landmarks[shoulder_id].x,
            landmarks[shoulder_id].y
        ]

        elbow = [
            landmarks[elbow_id].x,
            landmarks[elbow_id].y
        ]

        wrist = [
            landmarks[wrist_id].x,
            landmarks[wrist_id].y
        ]

        angle = calculate_angle(
            shoulder,
            elbow,
            wrist
        )

        if angle < 60:

            self.stage = "up"

        if (
            angle > 150
            and self.stage == "up"
        ):

            self.stage = "down"

            self.counter += 1

        return (
            angle,
            self.counter,
            self.stage
        )


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class FitnessVideoProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.exercise = "squat"

        self.exercise_name = "Squat"

        self.squat_counter = (
            SquatCounter()
        )

        self.curl_counter = (
            BicepCurlCounter()
        )

        self.start_time = time.time()

        self.lock = threading.Lock()

        self.current_reps = 0

        self.current_angle = 0

        self.current_stage = "Ready"

        self.current_feedback = (
            "Start workout"
        )

        self.elapsed_time = 0

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(
        self,
        frame
    ):

        image = frame.to_ndarray(
            format="bgr24"
        )

        image = cv2.flip(
            image,
            1
        )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.pose.process(
            rgb
        )

        angle = None

        reps = 0

        stage = None

        feedback = "No body detected"

        # ====================================================
        # BODY DETECTED
        # ====================================================

        if results.pose_landmarks:

            landmarks = (
                results.pose_landmarks.landmark
            )

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # =================================================
            # SQUAT
            # =================================================

            if self.exercise == "squat":

                (
                    angle,
                    reps,
                    stage
                ) = self.squat_counter.update(
                    landmarks
                )

                if angle is None:

                    feedback = (
                        "Move into camera view"
                    )

                elif angle < 100:

                    feedback = "Good Depth"

                elif angle < 130:

                    feedback = "Go Lower"

                elif angle > 160:

                    feedback = "Stand Straight"

                else:

                    feedback = "Keep Going"

            # =================================================
            # BICEP CURL
            # =================================================

            elif self.exercise == "curl":

                (
                    angle,
                    reps,
                    stage
                ) = self.curl_counter.update(
                    landmarks
                )

                if angle is None:

                    feedback = (
                        "Move your right arm into view"
                    )

                elif angle < 60:

                    feedback = "Good Curl"

                elif angle > 150:

                    feedback = "Extend Arm"

                else:

                    feedback = "Keep Curling"

        else:

            feedback = "No body detected"

        # ====================================================
        # TIMER
        # ====================================================

        elapsed_time = int(
            time.time()
            -
            self.start_time
        )

        current_angle = (
            int(angle)
            if angle is not None
            else 0
        )

        current_stage = (
            stage
            if stage is not None
            else "Not detected"
        )

        with self.lock:

            self.current_reps = reps

            self.current_angle = (
                current_angle
            )

            self.current_stage = (
                current_stage
            )

            self.current_feedback = (
                feedback
            )

            self.elapsed_time = (
                elapsed_time
            )

        # ====================================================
        # SAVE CURRENT WORKOUT
        # ====================================================

        write_current_workout(
            self.exercise_name,
            reps,
            current_angle,
            current_stage,
            feedback,
            elapsed_time
        )

        # ====================================================
        # CAMERA DISPLAY
        # ====================================================

        cv2.putText(
            image,
            f"Exercise: {self.exercise.upper()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            f"Reps: {reps}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        cv2.putText(
            image,
            f"Angle: {current_angle}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            f"Stage: {current_stage}",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            f"Feedback: {feedback}",
            (20, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        minutes = (
            elapsed_time // 60
        )

        seconds = (
            elapsed_time % 60
        )

        cv2.putText(
            image,
            f"Time: {minutes:02d}:{seconds:02d}",
            (20, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "🏋️ AI Virtual Personal Fitness Coach"
)

st.write(
    "AI-powered real-time workout monitoring "
    "using MediaPipe and Computer Vision."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Workout Settings"
)

exercise = st.sidebar.selectbox(
    "Choose Exercise",
    [
        "Squat",
        "Bicep Curl"
    ]
)


if exercise == "Squat":

    st.sidebar.info(
        "🏋️ SQUAT MODE\n\n"
        "Keep your full body visible.\n\n"
        "Go down below approximately 100°.\n\n"
        "Then stand completely straight.\n\n"
        "Down + Up = 1 repetition."
    )

else:

    st.sidebar.info(
        "💪 BICEP CURL MODE\n\n"
        "Keep your right arm visible.\n\n"
        "Start with your arm extended.\n\n"
        "Curl until the elbow angle is below 60°.\n\n"
        "Extend your arm again.\n\n"
        "Curl + Extend = 1 repetition."
    )


# ============================================================
# INSTRUCTIONS
# ============================================================

st.header(
    "📋 Instructions"
)

col1, col2 = st.columns(2)

with col1:

    st.write(
        "1️⃣ Select an exercise."
    )

    st.write(
        "2️⃣ Click START."
    )

    st.write(
        "3️⃣ Allow camera access."
    )


with col2:

    st.write(
        "4️⃣ Keep your body visible."
    )

    st.write(
        "5️⃣ Perform the movement slowly."
    )

    st.write(
        "6️⃣ Watch the AI feedback."
    )


# ============================================================
# VIDEO PROCESSOR CREATION
# ============================================================

def create_video_processor():

    clear_current_workout()

    processor = (
        FitnessVideoProcessor()
    )

    if exercise == "Bicep Curl":

        processor.exercise = "curl"

        processor.exercise_name = (
            "Bicep Curl"
        )

    else:

        processor.exercise = "squat"

        processor.exercise_name = (
            "Squat"
        )

    return processor


# ============================================================
# LIVE CAMERA
# ============================================================

st.header(
    "📹 Live Workout Camera"
)

ctx = webrtc_streamer(
    key=f"fitness-coach-{exercise}",
    video_processor_factory=(
        create_video_processor
    ),
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


# ============================================================
# READ CURRENT WORKOUT
# ============================================================

workout = read_current_workout()


if workout is None:

    reps_value = 0

    angle_value = 0

    stage_value = "Ready"

    feedback_value = "Start workout"

    time_value = 0

    workout_exercise = exercise

else:

    reps_value = workout.get(
        "reps",
        0
    )

    angle_value = workout.get(
        "angle",
        0
    )

    stage_value = workout.get(
        "stage",
        "Ready"
    )

    feedback_value = workout.get(
        "feedback",
        "Start workout"
    )

    time_value = workout.get(
        "duration",
        0
    )

    workout_exercise = workout.get(
        "exercise",
        exercise
    )


# ============================================================
# LIVE DASHBOARD
# ============================================================

st.header(
    "📊 Live Workout Dashboard"
)

d1, d2, d3, d4 = st.columns(4)


with d1:

    st.metric(
        "🔢 Repetitions",
        reps_value
    )


with d2:

    st.metric(
        "📐 Joint Angle",
        f"{angle_value}°"
    )


with d3:

    st.metric(
        "🔄 Stage",
        stage_value
    )


with d4:

    minutes = (
        time_value // 60
    )

    seconds = (
        time_value % 60
    )

    st.metric(
        "⏱️ Workout Time",
        f"{minutes:02d}:{seconds:02d}"
    )


# ============================================================
# AI FEEDBACK
# ============================================================

st.subheader(
    "🤖 AI Feedback"
)


if ctx.state.playing:

    if feedback_value in [
        "Good Depth",
        "Good Curl"
    ]:

        st.success(
            "✅ " + feedback_value
        )

    elif feedback_value in [
        "Go Lower",
        "Extend Arm",
        "Move into camera view",
        "Move your right arm into view",
        "No body detected"
    ]:

        st.warning(
            "⚠️ " + feedback_value
        )

    else:

        st.info(
            "💡 " + feedback_value
        )

else:

    if time_value > 0:

        st.success(
            "✅ Workout completed!"
        )

    else:

        st.info(
            "Start the camera to begin."
        )


# ============================================================
# WORKOUT STATUS
# ============================================================

st.subheader(
    "🏃 Workout Status"
)


if ctx.state.playing:

    st.success(
        f"🟢 AI is monitoring your "
        f"{exercise.lower()}."
    )

else:

    if time_value > 0:

        st.success(
            "🟢 Workout completed."
        )

    else:

        st.info(
            "🔵 Camera is stopped."
        )


# ============================================================
# SAVE WORKOUT
# ============================================================

st.markdown("---")

st.header(
    "💾 Save Workout"
)


if (
    not ctx.state.playing
    and time_value > 0
):

    st.write(
        "Your completed workout is ready to save."
    )

    s1, s2, s3 = st.columns(3)


    with s1:

        st.write(
            f"**Exercise:** "
            f"{workout_exercise}"
        )


    with s2:

        st.write(
            f"**Repetitions:** "
            f"{reps_value}"
        )


    with s3:

        st.write(
            f"**Duration:** "
            f"{time_value} seconds"
        )


    if st.button(
        "💾 Save Workout Session",
        use_container_width=True
    ):

        saved = save_workout(
            workout_exercise,
            reps_value,
            time_value
        )

        if saved:

            st.success(
                "✅ Workout saved successfully!"
            )

            st.info(
                "📚 Your workout has been "
                "added to Workout History."
            )

            clear_current_workout()


elif ctx.state.playing:

    st.info(
        "🟢 Complete your workout "
        "and click STOP."
    )

else:

    st.info(
        "Complete a workout session first."
    )


# ============================================================
# WORKOUT HISTORY
# ============================================================

st.markdown("---")

st.header(
    "📚 Workout History"
)

history = read_workout_history()


if (
    history is not None
    and not history.empty
):

    history = history.dropna(
        how="all"
    )

    st.success(
        f"✅ {len(history)} "
        f"workout session(s) recorded."
    )

    # --------------------------------------------------------
    # HISTORY TABLE
    # --------------------------------------------------------

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # WORKOUT SUMMARY
    # ========================================================

    st.subheader(
        "📈 Workout Summary"
    )


    total_sessions = len(
        history
    )


    total_reps = int(
        pd.to_numeric(
            history["Reps"],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )


    total_time = int(
        pd.to_numeric(
            history["Duration"],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )


    h1, h2, h3 = st.columns(3)


    with h1:

        st.metric(
            "Total Sessions",
            total_sessions
        )


    with h2:

        st.metric(
            "Total Repetitions",
            total_reps
        )


    with h3:

        st.metric(
            "Total Workout Time",
            f"{total_time} sec"
        )


    # ========================================================
    # WORKOUT PROGRESS
    # ========================================================

    st.subheader(
        "📊 Workout Progress"
    )


    chart_data = history.copy()


    chart_data["Reps"] = pd.to_numeric(
        chart_data["Reps"],
        errors="coerce"
    ).fillna(0)


    chart_data["Duration"] = pd.to_numeric(
        chart_data["Duration"],
        errors="coerce"
    ).fillna(0)


    st.write(
        "### 🔢 Repetitions per Workout"
    )

    st.line_chart(
        chart_data["Reps"]
    )


    st.write(
        "### ⏱️ Workout Duration"
    )

    st.bar_chart(
        chart_data["Duration"]
    )


    # ========================================================
    # EXERCISE-WISE SUMMARY
    # ========================================================

    st.subheader(
        "🏋️ Exercise-wise Summary"
    )


    exercise_summary = (
        chart_data
        .groupby("Exercise")
        .agg(
            Sessions=("Exercise", "count"),
            Total_Reps=("Reps", "sum"),
            Total_Duration=("Duration", "sum")
        )
        .reset_index()
    )


    exercise_summary[
        "Total_Reps"
    ] = exercise_summary[
        "Total_Reps"
    ].astype(int)


    exercise_summary[
        "Total_Duration"
    ] = exercise_summary[
        "Total_Duration"
    ].astype(int)


    st.dataframe(
        exercise_summary,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # AI PERFORMANCE REPORT
    # ========================================================

    st.markdown("---")

    st.header(
        "🤖 AI Performance Report"
    )

    st.write(
        "The system analyzes your recorded workout "
        "history and generates a simple performance report."
    )


    # --------------------------------------------------------
    # PERFORMANCE CALCULATIONS
    # --------------------------------------------------------

    average_reps = (
        total_reps / total_sessions
        if total_sessions > 0
        else 0
    )


    average_duration = (
        total_time / total_sessions
        if total_sessions > 0
        else 0
    )


    # Find most practiced exercise
    exercise_counts = (
        chart_data["Exercise"]
        .value_counts()
    )


    if not exercise_counts.empty:

        most_practiced = (
            exercise_counts.index[0]
        )

        most_practiced_count = int(
            exercise_counts.iloc[0]
        )

    else:

        most_practiced = "None"

        most_practiced_count = 0


    # --------------------------------------------------------
    # REPORT METRICS
    # --------------------------------------------------------

    r1, r2, r3, r4 = st.columns(4)


    with r1:

        st.metric(
            "🏋️ Sessions",
            total_sessions
        )


    with r2:

        st.metric(
            "🔢 Avg Reps",
            f"{average_reps:.1f}"
        )


    with r3:

        st.metric(
            "⏱️ Avg Duration",
            f"{average_duration:.1f} sec"
        )


    with r4:

        st.metric(
            "💪 Main Exercise",
            most_practiced
        )


    # ========================================================
    # AUTOMATIC PERFORMANCE FEEDBACK
    # ========================================================

    st.subheader(
        "💡 Performance Feedback"
    )


    if average_reps >= 15:

        performance_message = (
            "Your recorded sessions show a "
            "relatively high repetition count. "
            "Continue focusing on controlled movement "
            "and consistent form."
        )

    elif average_reps >= 8:

        performance_message = (
            "Your recorded sessions show a "
            "moderate repetition count. "
            "Focus on maintaining consistent movement "
            "throughout each repetition."
        )

    else:

        performance_message = (
            "Your recorded sessions contain a "
            "lower repetition count. "
            "Focus on controlled repetitions and "
            "gradually building consistency."
        )


    st.info(
        "🤖 " + performance_message
    )


    # ========================================================
    # DURATION FEEDBACK
    # ========================================================

    if average_duration < 30:

        duration_message = (
            "Your recorded sessions are relatively short. "
            "You can gradually increase workout duration "
            "as appropriate for your training routine."
        )

    elif average_duration < 120:

        duration_message = (
            "Your recorded sessions have a moderate duration. "
            "Continue maintaining a consistent workout routine."
        )

    else:

        duration_message = (
            "Your recorded sessions have a longer duration. "
            "Continue monitoring your form and take appropriate "
            "rest periods during training."
        )


    st.info(
        "⏱️ " + duration_message
    )


    # ========================================================
    # SUGGESTED IMPROVEMENT
    # ========================================================

    st.subheader(
        "🎯 Suggested Improvement"
    )


    if total_sessions == 1:

        suggestion = (
            "Record more workout sessions to build a "
            "larger history. This will allow the system "
            "to show more meaningful progress trends."
        )

    elif total_sessions < 5:

        suggestion = (
            "Continue recording regular workout sessions. "
            "As more sessions are added, your progress "
            "charts and exercise statistics will become "
            "more informative."
        )

    else:

        suggestion = (
            "Review your repetition and duration trends "
            "regularly and focus on maintaining consistent "
            "exercise technique."
        )


    st.success(
        "🎯 " + suggestion
    )


    # ========================================================
    # AI REPORT SUMMARY
    # ========================================================

    st.subheader(
        "📋 Complete Performance Report"
    )


    report_col1, report_col2 = st.columns(2)


    with report_col1:

        st.write(
            f"**🏋️ Total Workout Sessions:** "
            f"{total_sessions}"
        )

        st.write(
            f"**🔢 Total Repetitions:** "
            f"{total_reps}"
        )

        st.write(
            f"**📊 Average Repetitions/Session:** "
            f"{average_reps:.1f}"
        )


    with report_col2:

        st.write(
            f"**⏱️ Total Workout Time:** "
            f"{total_time} seconds"
        )

        st.write(
            f"**⏱️ Average Session Duration:** "
            f"{average_duration:.1f} seconds"
        )

        st.write(
            f"**💪 Most Practiced Exercise:** "
            f"{most_practiced}"
        )

else:

    st.info(
        "📭 No workout sessions recorded yet."
    )


# ============================================================
# DEVELOPER INFORMATION
# ============================================================

with st.expander(
    "📁 Developer Information"
):

    st.write(
        "**Workout history file:**"
    )

    st.code(
        HISTORY_FILE
    )


    if os.path.exists(
        HISTORY_FILE
    ):

        st.success(
            "✅ workout_logs.csv exists."
        )

    else:

        st.warning(
            "⚠️ workout_logs.csv does not exist yet."
        )


# ============================================================
# TECHNOLOGIES USED
# ============================================================

st.markdown("---")

st.subheader(
    "🛠️ Technologies Used"
)

t1, t2, t3, t4 = st.columns(4)


with t1:

    st.write(
        "🐍 Python"
    )


with t2:

    st.write(
        "👁️ OpenCV"
    )


with t3:

    st.write(
        "🧍 MediaPipe"
    )


with t4:

    st.write(
        "🌐 Streamlit"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Virtual Personal Fitness Coach | "
    "Python • OpenCV • MediaPipe • "
    "Streamlit • WebRTC"
)