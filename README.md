# LSTM-Basketball-Shot-Form-Analyzer

## Overview
The **Basketball Shot Form Analyzer** is an ongoing project aimed at using artificial intelligence and machine learning to analyze and classify basketball shooting forms. This tool is designed to help players, coaches, and enthusiasts evaluate their shooting technique by identifying key movement patterns, biomechanical metrics, and shot quality. The ultimate goal is to provide feedback on whether a player's form is optimal or needs improvement.

This project is still in its early stages. The foundational code for creating datasets from videos of basketball shots has just been completed, and further work will focus on training models and refining analysis techniques.

---

## Current Features
- **Accurate Dataset Generation:**
  - Processes basketball shot videos to extract biomechanical features.
  - Tracks player and ball movements using MediaPipe's pose detection framework and YOLOv5 API.
  - Computes:
    - Joint positions relative to the hip center.
    - Velocities and accelerations of key joints.
    - Angles between joints (e.g., shoulder-elbow-wrist).
  - Outputs data in a structured format as a CSV file for use in machine learning models.

- **Visualization:**
  - Displays the detected skeleton overlaid on the video during processing.
  - Highlights key joints and provides real-time feedback on feature extraction.

---

## How It Works
### Dataset Generation
1. **Input:**
   - A video file of a basketball shot (e.g., a jump shot).
   - The video should show the player clearly, with sufficient lighting and resolution for pose detection.
   - Video should be from the front, visibly seeing both arms of the player. I've noticed that maxium accuracy comes from an around 45 degree angle from the front.

2. **Pose Detection:**
   - Uses MediaPipe to detect and track the player's joints and body movements frame by frame.

3. **Feature Extraction:**
   - Joint Positions: Positions of key joints (e.g., shoulder, elbow, wrist) relative to the skeleton's center.
   - Velocities & Accelerations: Derived by calculating changes in joint positions over time.
   - Angles: Computes angles at key joints (e.g., shoulder-elbow-wrist) to capture form dynamics.

4. **Dataset Output:**
   - A CSV file containing:
     - Frame-by-frame data for joint positions, velocities, accelerations, angles, and ball position. All of these positions are relative to the center of the player's hip, insuring  maximum accuracy.
     - Labels indicating shot quality, good or bad (currently set manually).

### Future Work
- **Form Start and End Time Detection:**
  - Automate the identification of the beginning and end of a shot using temporal analysis of joint and ball movements.
  - This would allow the LSTM model to accurately fit to the data.

- **Model Training:**
  - Train LSTM models on the generated datasets to classify shots as "good" or "bad" based on biomechanical patterns.

- **Feedback System:**
  - Develop an interface that provides real-time feedback and suggestions for improving shooting form.

---

## Current Limitations

- **Limited Data:**
  - The project is still building a robust dataset. More labeled videos are needed to improve model training. Right now a dataset of an estimated 100 videos minimum are required for an accurate reading.
  - Something that sets this project apart from other's is the model is learning how you would improve your form tailored to you, specifically.
  - This is for those who are trying to improve the form they have worked on for a long time.
  - The goal of this model is to help you differentiate the small deviations in your form that separates a miss from a make

- **Pose Detection Accuracy:**
  - MediaPipe sometimes struggles with fast movements or occlusions. Future iterations may incorporate BlazePose or other advanced frameworks for better accuracy.

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/LSTM-Basketball-Shot-Form-Analyzer.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dataset generation script:
   ```bash
   python DataGenerator.py
   ```
   - Replace the input video path in the script with your own video file.
   - The output dataset will be saved as `dataset.csv` in the project directory.

4. View real-time visualization during processing to ensure pose detection is accurate.

---

## Roadmap
1. **Phase 1: FIX DATA**
   - Refine dataset generation and automate shot start detection.
   - Impute missing values.
   - Adjust the length of different lengthed videos.
2. **Phase 2: CREATE MODEL**
   - Train and validate LSTM models for shot classification.
3. **Phase 3: CREATE INTERFACE**
   - Develop a user-friendly interface for analyzing and providing feedback on basketball shots.
4. **Phase 4: UPGRADE**
   - Expand to include more advanced metrics, such as player fatigue analysis and injury prevention insights.

---

## Contributing
Contributions are welcome! If you'd like to contribute to this project:
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Submit a pull request with a detailed explanation of your changes.

---

## License
This project is currently under the **MIT License**, allowing for use and modification with proper attribution. However, the codebase is subject to change as the project evolves.

---

## Acknowledgments
- **MediaPipe:** For providing an excellent pose detection framework.
- **OpenCV:** For video processing and visualization tools.
- **Basketball Coaches and Players:** For inspiring the idea of improving shooting form through AI.

---

## Contact
For questions, suggestions, or collaboration inquiries, feel free to reach out:
- **Email:** jake.poschl@gmail.com
- **GitHub Issues:** [Open an Issue](https://github.com/yourusername/basketball-shot-form-analyzer/issues)

---

Thank you for your interest in the Basketball Shot Form Analyzer! This is just the beginning, and there's a lot more to come.
