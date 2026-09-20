# 🏋️ AI Virtual Personal Fitness Coach

An AI-powered virtual fitness coach that uses computer vision and pose estimation to detect workout movements, count repetitions, provide real-time feedback, track workout duration, and maintain workout history.

The application is built using Python, OpenCV, MediaPipe, and Streamlit.

---

## 🎯 Project Objective

The main objective of this project is to develop a virtual personal fitness coach that can analyze a user's workout through a webcam.

The system uses human body landmarks and joint angles to identify exercise movements and automatically count repetitions.

It also provides workout analytics and performance feedback to help users monitor their exercise sessions.

---

## ✨ Features

- 📷 Real-time webcam workout detection
- 🧍 Human pose detection using MediaPipe
- 🏋️ Squat detection and repetition counting
- 💪 Bicep curl detection and repetition counting
- 📐 Joint-angle calculation
- 🔄 Automatic exercise stage detection
- 🔢 Automatic repetition counter
- 🤖 Real-time workout feedback
- ⏱️ Workout duration timer
- 💾 Save workout sessions
- 📊 Workout history
- 📈 Repetition charts
- 📊 Workout duration charts
- 📝 Exercise-wise workout summary
- 🤖 AI Performance Report
- 📋 CSV-based workout logging
- 🌐 Streamlit web interface
- 🎥 Live camera streaming using WebRTC

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Computer Vision
- OpenCV
- MediaPipe

### Web Application
- Streamlit
- Streamlit-WebRTC

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib

### Other Libraries
- aiortc
- av
- CSV
- JSON

---

## 🧠 How the System Works

The application follows these main steps:

```text
Webcam
   ↓
Video Frame Capture
   ↓
MediaPipe Pose Detection
   ↓
Body Landmark Detection
   ↓
Joint Angle Calculation
   ↓
Exercise Stage Detection
   ↓
Repetition Counting
   ↓
Real-Time Feedback
   ↓
Workout Data Storage
   ↓
Workout History & Analytics