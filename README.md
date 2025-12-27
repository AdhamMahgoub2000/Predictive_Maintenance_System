<<<<<<< HEAD
# Predictive_Maintenance_System
=======
Aircraft Engine Failure Imminence Classification (CNN-LSTM)

- Project Overview

This project implements a deep learning–based health state classification system for aircraft engines using multivariate time-series sensor data. The goal is to classify engine failure risk rather than predict exact Remaining Useful Life (RUL), which is more robust and actionable in real-world predictive maintenance systems.

The system is built using the NASA CMAPSS Turbofan Engine Degradation Dataset and follows industry-standard practices in data preprocessing, windowed time-series modeling, and deep learning architecture design.

⸻

- Problem Statement

Given historical sensor readings from aircraft engines operating under varying conditions, classify the current health state of an engine into one of three categories:

Class	Description	Risk Level
0	Healthy	Low
1	Degrading	Medium
2	Failure Imminent	High ⚠️

This formulation prioritizes early fault detection and safety-critical decision making.

⸻
- Dataset Description

NASA CMAPSS Turbofan Engine Dataset

Each dataset consists of multiple multivariate time series, where each time series represents the full (or partial) operational life of a single engine.

Dataset Variants

Dataset	Train Engines	Test Engines	Operating Conditions	Fault Modes
FD001	100	100	1	1 (HPC degradation)
FD002	260	259	6	1 (HPC degradation)
FD003	100	100	1	2 (HPC + Fan degradation)
FD004	248	249	6	2 (HPC + Fan degradation)

This notebook currently focuses on FD001 as a baseline.

⸻

- Data Format

Each row represents one operational cycle of an engine and contains 26 columns:
	1.	Engine ID
	2.	Cycle number
3–5. Operational settings
6–26. Sensor measurements

⸻

- Exploratory Data Analysis (EDA)

The following preprocessing and EDA steps were completed:
	•	Assigned correct column names
	•	Engine-wise lifecycle visualization
	•	Correlation analysis
	•	Identified and removed globally constant sensors:
	•	sensor_5, sensor_16, sensor_18, sensor_19
	•	Retained 16 informative sensors for modeling

Flat sensor signals were only removed if globally constant, as sensors that activate near failure are informative.

⸻

- Label Engineering (Classification)

The original RUL values were converted into discrete health classes:

def rul_to_class(rul):
    if rul > 50:
        return 0  # Healthy
    elif rul > 20:
        return 1  # Degrading
    else:
        return 2  # Failure Imminent

This mapping is used consistently across training, testing, and inference.

⸻

- Time-Series Windowing

To enable deep learning on temporal data, a sliding window approach was used:
	•	Window size: 50 cycles
	•	Step size: 1 cycle
	•	One window → one classification label

Training Data
	•	Sliding windows generated across full engine lifecycles
	•	Resulting shape:

X_train: (15631, 50, 16)
y_train: (15631,)

Test Data
	•	Only the last 50 cycles per engine are used
	•	One prediction per engine
	•	Engines with fewer than 50 cycles are left-padded

This follows the official CMAPSS evaluation protocol.

⸻

- Important Design Decisions
	•	Classification chosen over regression for robustness and actionability
	•	Failure-imminent class (Class 2) treated as highest-risk
	•	Padding applied only after normalization
	•	Test set kept untouched for final evaluation

⸻

- Model Architecture (Planned)

The model will use a CNN-LSTM hybrid architecture:
	•	1D CNN: Local temporal feature extraction
	•	LSTM: Long-term degradation modeling
	•	Fully connected layers + Softmax: Health state classification

Loss function:
	•	CrossEntropyLoss

⸻

- Next Steps
	•	Feature normalization
	•	PyTorch Dataset & DataLoader implementation
	•	CNN-LSTM model training
	•	Evaluation (Accuracy, F1-score, Confusion Matrix)
	•	Cross-dataset generalization (FD002–FD004)
	•	Deployment with FastAPI + Docker

⸻

📚 Reference

A. Saxena, K. Goebel, D. Simon, and N. Eklund, Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation, PHM 2008.

⸻

👤 Author

Adham Mahgoub

Mechanical Engineer | Machine Learning Engineer

⸻
>>>>>>> 4bfd42c (Next->ModelTuning)
