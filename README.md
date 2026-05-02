# 🚀 AI-Driven CNC Tool Wear Detection Pipeline

## 💡 The Vision
In modern manufacturing, tool wear is a massive bottleneck. Unnoticed wear kills product quality, spikes scrap rates, and tanks machine uptime. Traditional monitoring? Too manual and way too expensive. 

This project disrupts that paradigm by leveraging Machine Learning (ML) and Deep Learning (DL) to build a fully automated tool wear monitoring system using raw CNC sensor data. 

## 📊 The Dataset & Data Engineering
We are leveraging the robust "CNC Milling Machine Tool Wear" dataset from the University of Michigan SMART Lab.

*   **Raw Data:** 18 distinct machining experiments captured at an insane 100 ms sampling rate.
*   **Sensors Tracked:** X, Y, Z axes and spindle positions, velocities, accelerations, plus power and current output.
*   **The ETL Pipeline:** We built a custom script (`src/merge_experiments.py`) to fuse 18 separate CSVs into a single, clean `merged_data.csv`. (Note: This is already pre-generated in the `data/` folder for your convenience).
*   **Feature Engineering:** We tossed out the noisy data (`Feedrate != 50`, `Position != 198`) and isolated the active cutting phases ("Layer 1-3 Up/Down"), leaving us with 12,943 pristine, perfectly balanced data points.

## 🧠 The Architecture & Models

### The Baseline: Classic ML (2D Snapshot)
We started with standard static models in `src/baseline_models.py`.
*   **Logistic Regression:** Hit ~55% accuracy. It proved that the sensor-to-wear relationship is definitely not linear.
*   **Random Forest:** Bumped up to ~75% accuracy. Feature importance analysis showed that power output, current, acceleration, and velocity were the heavy hitters. Good, but we can do better.

### The Game Changer: 1D-CNN Deep Learning (3D Time-Series)
To smash past that 75% ceiling, we built a Deep Learning model (`src/nn_model_timeseries.py`) that actually understands the flow of time.
*   **Sliding Window:** We used a 50-timestep (5-second) sliding window to capture temporal trends.
*   **Group Split Validation:** Random shuffling leads to massive data leakage. To prove true generalization, we split the data strictly by Experiment ID (14 for training, 4 totally isolated for testing).
*   **The Architecture:** Dual Conv1D layers + MaxPooling1D for feature extraction, followed by Flatten and Dense layers, with Dropout to crush overfitting.

### 🏆 The Results
The CNN absolutely crushed it.
*   **Accuracy:** **~99.6%** on the unseen test set.
*   **Confusion Matrix:** 0 False Positives and only 10 False Negatives out of over 2400 test cases.
*   **Inference Speed:** Clocks in at under 50ms, making it perfect for real-time, in-cycle production monitoring.

## 🛠️ Quick Start Guide

### Prerequisites
You'll need a Python 3.8+ environment.

### Installation
1. Clone the repo and ensure `merged_data.csv` is chilling in the `data/CNC_Milling_Data/` directory.
2. Install the stack:
   ```bash
   pip install -r requirements.txt
