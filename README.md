Aircraft Engine Failure Imminence Classification (CNN-LSTM)

- Project Overview

This project implements a deep learning–based health state classification system for aircraft engines using multivariate time-series sensor data. The goal is to classify engine failure risk rather than predict exact Remaining Useful Life (RUL), which is more robust and actionable in real-world predictive maintenance systems.

The system is built using the NASA CMAPSS Turbofan Engine Degradation Dataset and follows industry-standard practices in data preprocessing, windowed time-series modeling, and deep learning architecture design.

⸻

- Problem Statement

Given historical sensor readings from aircraft engines operating under varying conditions, classify the current health state of an engine into one of two categories:

Class	Description	Risk Level
0	Normal	Low
1	Failure Imminent	High ⚠️

This binary classification formulation prioritizes early fault detection and safety-critical decision making, using a failure threshold of 30 remaining cycles.

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
	3–5.	Operational settings (included in model)
	6–26.	Sensor measurements (21 sensors)

⸻

- Exploratory Data Analysis (EDA)

The following preprocessing and EDA steps were completed:
	•	Assigned correct column names
	•	Engine-wise lifecycle visualization
	•	Correlation analysis
	•	Identified and removed globally constant sensors:
	•	sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19
	•	Retained 14 informative sensors plus 3 operational settings (17 features total) for modeling

Flat sensor signals were only removed if globally constant, as sensors that activate near failure are informative.

⸻

- Label Engineering (Classification)

The original RUL values were converted into binary health classes:

def convert_to_binary(df, failure_threshold=30):
    df['label'] = (df['RUL'] <= failure_threshold).astype(int)
    return df

Class 0: Normal (RUL > 30 cycles)
Class 1: Failure Imminent (RUL ≤ 30 cycles)

This binary classification mapping is used consistently across training, testing, and inference.

⸻

- Time-Series Windowing

To enable deep learning on temporal data, a sliding window approach was used:
	•	Window size: 20 cycles
	•	Step size: 1 cycle
	•	One window → one classification label

Training Data
	•	Sliding windows generated across full engine lifecycles
	•	Features include 3 operational settings and 14 sensor measurements (17 features per timestep)
	•	Resulting shape: (N, 20, 17) where N is the number of windows

Test Data
	•	Only the last 20 cycles per engine are used
	•	One prediction per engine
	•	Engines with fewer than 20 cycles are excluded

This follows the official CMAPSS evaluation protocol.

⸻

- Important Design Decisions
	•	Binary classification chosen over regression for robustness and actionability
	•	Failure-imminent class (Class 1) treated as highest-risk
	•	MinMaxScaler applied to sensor columns only; operational settings kept unscaled
	•	Test set kept untouched for final evaluation
	•	Batch size: 32, learning rate: 0.0001, dropout: 0.3

⸻

- Model Architecture

The implemented model uses a CNN-LSTM hybrid architecture:
	•	1D CNN: Two Conv1d layers (64 channels, kernel size 3) for local temporal feature extraction
	•	LSTM: Single-layer LSTM (128 hidden units) for long-term degradation modeling
	•	Fully connected layers: 128 → 64 → 2 classes with ReLU activations and dropout
	•	Softmax: Health state classification (binary)

Configuration:
	•	Input size: 17 features (3 operational settings + 14 sensors)
	•	CNN channels: 64
	•	LSTM hidden size: 128
	•	LSTM layers: 1
	•	Dropout rates: 0.3 (CNN, LSTM, FC)
	•	Loss function: CrossEntropyLoss
	•	Optimizer: Adam (learning rate: 0.0001)

⸻

- Usage

Training

To train the model:

python train.py

This will load the training data, preprocess it, train the CNN-LSTM model, and save the trained model and scaler to disk.

API Server

To start the FastAPI prediction server:

python predict.py

The API will be available at http://localhost:8080 with interactive documentation at http://localhost:8080/docs.

API Endpoint

POST /predict

Request body (JSON):
	•	features: List of 340 floats (20 timesteps × 17 features)
	•	Feature order: [op_set_1, op_set_2, op_set_3, sensor_2, sensor_3, ..., sensor_21] for each timestep

Response:
	•	predicted_class: 0 (Normal) or 1 (Failure Imminent)
	•	confidence: Prediction confidence score (0–1)

Example Usage

See example_predict.py for complete examples including:
	•	Loading data from test files
	•	Making batch predictions
	•	Simulating engine degradation
	•	Direct model usage (without API)

⸻

- Implementation Status

Completed:
	•	Feature normalization (MinMaxScaler on sensor columns)
	•	PyTorch Dataset & DataLoader implementation
	•	CNN-LSTM model training and evaluation
	•	Model serialization (predictive_maintenance_model.pth, scaler.pkl)
	•	FastAPI REST API deployment (predict.py)
	•	Example prediction scripts and test suite

Files:
	•	train.py: Model training script
	•	predict.py: FastAPI server for predictions
	•	example_predict.py: Usage examples and demonstrations
	•	test_predict.py: Unit tests for API endpoints

Future Enhancements:
	•	Cross-dataset generalization (FD002–FD004)
	•	Docker containerization
	•	Model performance metrics dashboard

⸻

📚 Reference

A. Saxena, K. Goebel, D. Simon, and N. Eklund, Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation, PHM 2008.

⸻

👤 Author

Adham Mahgoub

Mechanical Engineer | Machine Learning Engineer

⸻
