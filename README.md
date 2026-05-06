# Eye Contact and Engagement Analysis

## Overview
This project implements a real-time computer vision system to monitor and analyze user engagement during online interactions. By combining **Deep Learning (ResNet-18)** with **Geometric Iris Tracking (MediaPipe)**, the system provides a robust "Hybrid AI" approach to detect whether a user is attentive, distracted, or disengaged.

## Key Features
- **Hybrid Inference Engine**: Integrates a ResNet-18 classification model with a MediaPipe-based iris tracking system to handle "domain shift" between training data and real-world webcam environments.
- **Engagement States**: Classifies behavior into three actionable states:
  - **Attentive**: Centered gaze and head orientation.
  - **Distracted (Eyes)**: Pupil deviation detected via geometric rules.
  - **Disengaged**: Head turned or tilted down, detected via global context.
- **High Performance**: Optimized using ResNet-18 to achieve a balance between 99.9% test accuracy and real-time inference speed.
- **Iris Tracking**: Real-time Gaze Ratio calculation (horizontal pupil position) to catch subtle distractions that global head-pose models might miss.

## Technical Architecture
1. **Model Training**: Trained on the Columbia Gaze Dataset. Comparisons between ResNet-18 and ResNet-50 showed that lighter models are superior for datasets under 10k images to avoid overfitting.
2. **Iris Landmarks**: Utilizes MediaPipe Face Mesh landmarks (468-477) to calculate the "Gaze Ratio."
3. **Logic Layer**: A rule-based system (Pupil Ratio < 0.42 or > 0.58) acts as an immediate flag for distraction, overriding the neural network when eye movement is the primary indicator.

## Requirements
- Python 3.8+
- OpenCV
- MediaPipe
- PyTorch / TorchVision
- NumPy
- Matplotlib

## Repository Structure
- `L22-6707+L22-6765.ipynb`: End-to-end implementation including geometric labeling, model training, and the hybrid inference loop.
- `Kaggle Model.ipynb`: Comparative analysis and benchmarking of different Deep Learning architectures.
- `DS Deliverable 2.docx`: Technical project report and documentation.

## How to Run
1. Install dependencies: `pip install opencv-python mediapipe torch torchvision numpy`
2. Open `L22-6707+L22-6765.ipynb`.
3. Run the `run_hybrid_inference()` function to start the webcam-based engagement analysis.

## Contributors
- **Muhammad Mubeen** (L22-6707)
- **Partner** (L22-6765)
