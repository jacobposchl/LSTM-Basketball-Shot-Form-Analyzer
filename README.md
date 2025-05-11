# 🏀 FormAI - Basketball Shot Form Analyzer

FormAI is a full-stack AI-powered platform that helps basketball players improve their shooting form using real-time video analysis and machine learning. It uses pose estimation and gesture recognition to track shot quality and gives users live and post-analysis feedback through a mobile app.

> 📌 **Patent Status:** Non-provisional patent filed and currently pending approval.

---

## 📱 Mobile App (Client)

- Record shooting sessions (jump shots, 3-pointers, free throws)
- Save user profiles and track shooting progress
- Leaderboard to compare form scores
- Upload videos to a Firebase + Google Cloud Platform (GCP) backend
- Users provide feedback via thumb gestures (👍 = good form, 👎 = bad form)

---

## ☁️ Cloud Infrastructure

### 🔹 Firebase
- Stores:
  - `job_id` for shot sessions
  - LSTM-based prediction `reports` for each video

### 🔹 GCP Bucket
- Stores uploaded videos from the mobile app
- Triggered on upload via Pub/Sub topic

### 🔹 GCP VM (Processing Backend)
- Hosts the full data processing pipeline
- Subscribes to Pub/Sub to process new shot videos

---

## 🧠 AI Pipeline

### `main.py`
- Extracts joint landmarks using:
  - **MediaPipe** for full-body + hand tracking
  - **YOLO** for ball detection
- Detects key shot phases:
  - Shot start, peak, and release
- Calculates kinematic features:
  - Joint angles, velocity, acceleration
- Creates structured datasets for ML training:
  - Frames × Feature Matrix

### `data_processing.py`
- Cleans and normalizes extracted feature data
- Prepares data for time series classification

### `lstm_model.py`
- Runs inside a **Docker container** (separate dependencies)
- Trains and evaluates an LSTM model to classify form quality
- Uses thumbs up/down gesture data as ground truth labels
- Uploads results back to Firebase `reports`

> 📉 **Current Model Performance:**  
> Achieves an AUC of ~0.6 on a small dataset of ~200 total labeled shots (100 per class). The limited dataset size is currently the largest bottleneck to performance.

---

## 🧪 Technologies Used

- **Python**, **Docker**, **Firebase**, **GCP (Pub/Sub, VM, Storage)**
- **MediaPipe**, **YOLO**, **LSTM (Keras)**
- **OpenCV**, **NumPy**, **Pandas**, **scikit-learn**

---

## 🚀 Future Plans

- Replace MediaPipe with higher-accuracy pose detectors (e.g., MMPose or STDPose)
- Add live form correction feedback
- Improve gesture detection robustness
- Expand to multi-user collaboration and cloud model training
- Collect more labeled shot data to boost model performance

---

## 📝 License

This project is currently under active development. Licensing details will be provided in the future.

---

## 🙋‍♂️ Author

Jacob Poschl  
[LinkedIn](https://www.linkedin.com/in/jposchl) • [GitHub](https://github.com/jposchl)

---
