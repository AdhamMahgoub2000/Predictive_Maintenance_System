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

- Quick Start

```bash
# Train the model
python train.py

# Start the API server
python predict.py

# Test the health endpoint
curl http://localhost:8080/health

# Make a prediction (see example_predict.py for details)
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @request_example.json
```

For Docker deployment:
```bash
docker build -t predictive-maintenance-engine .
docker run -p 8080:8080 predictive-maintenance-engine
```

For Kubernetes deployment:
```bash
kubectl apply -k k8s/
```

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

API Endpoints

GET /health

Health check endpoint for monitoring and Kubernetes probes.

Response:
	•	status: "healthy"
	•	service: "predictive-maintenance-api"

POST /predict

Prediction endpoint for engine health classification.

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

- Docker Containerization

The application is fully containerized using Docker, making it easy to deploy and run in any environment.

Building the Docker Image

To build the Docker image:

docker build -t predictive-maintenance-engine .

The Dockerfile uses Python 3.13 slim as the base image, installs dependencies using uv, and sets up the FastAPI server for predictions.

Running the Container

To run the containerized FastAPI server:

docker run -p 8080:8080 predictive-maintenance-engine

This will:
	•	Start the FastAPI server on port 8080
	•	Make the API available at http://localhost:8080
	•	Serve the interactive API documentation at http://localhost:8080/docs
	•	Provide health check endpoint at http://localhost:8080/health

The container includes:
	•	All Python dependencies (managed by uv)
	•	The trained model (predictive_maintenance_model.pth)
	•	The feature scaler (scaler.pkl)
	•	The FastAPI prediction server

Note: The Docker image is optimized for inference. Training data and notebooks are excluded via .dockerignore to keep the image size minimal.

⸻

- Kubernetes Deployment

The application can be deployed to Kubernetes for production use with high availability, scalability, and automatic recovery.

Prerequisites

	•	Kubernetes cluster (v1.20+)
	•	kubectl configured to access your cluster
	•	Docker image built and pushed to a container registry (or available locally)

Building and Pushing the Docker Image

First, build and tag the Docker image:

docker build -t predictive-maintenance-engine:latest .

If using a container registry (e.g., Docker Hub, GCR, ECR), tag and push the image:

docker tag predictive-maintenance-engine:latest <registry>/predictive-maintenance-engine:latest
docker push <registry>/predictive-maintenance-engine:latest

Update the image reference in k8s/deployment.yaml if using a remote registry.

Deploying to Kubernetes

Option 1: Using kubectl directly

Deploy all resources:

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml  # Optional, if you have an ingress controller

Option 2: Using Kustomize

Deploy using kustomize (recommended):

kubectl apply -k k8s/

This will create all resources in the predictive-maintenance namespace.

Verifying the Deployment

Check deployment status:

kubectl get deployments -n predictive-maintenance
kubectl get pods -n predictive-maintenance
kubectl get services -n predictive-maintenance

View logs:

kubectl logs -f deployment/predictive-maintenance-api -n predictive-maintenance

Accessing the API

ClusterIP Service (internal access):

kubectl port-forward -n predictive-maintenance service/predictive-maintenance-api 8080:80

Then access the API at http://localhost:8080

Ingress (external access):

If you've configured the ingress, update your /etc/hosts or DNS to point predictive-maintenance.local to your ingress controller's IP, then access:

http://predictive-maintenance.local/docs

Scaling

Scale the deployment:

kubectl scale deployment predictive-maintenance-api -n predictive-maintenance --replicas=3

Or update the replicas field in k8s/deployment.yaml and reapply.

Configuration

The deployment includes:

	•	2 replicas for high availability
	•	Resource limits: 1Gi memory, 500m CPU
	•	Liveness and readiness probes using `/health` endpoint
	•	Automatic restart on failure
	•	Health check intervals: 10s (liveness), 5s (readiness)

To customize, edit k8s/deployment.yaml before deploying.

Cleaning Up

Remove all resources:

kubectl delete -k k8s/

Or delete individually:

kubectl delete -f k8s/

⸻

- Implementation Status

Completed:
	•	Feature normalization (MinMaxScaler on sensor columns)
	•	PyTorch Dataset & DataLoader implementation
	•	CNN-LSTM model training and evaluation
	•	Model serialization (predictive_maintenance_model.pth, scaler.pkl)
	•	FastAPI REST API deployment (predict.py)
	•	Health check endpoint (/health) for monitoring and Kubernetes probes
	•	Example prediction scripts and test suite
	•	Docker containerization for easy deployment
	•	Kubernetes deployment manifests for production with health checks

Files:
	•	train.py: Model training script
	•	predict.py: FastAPI server for predictions
	•	example_predict.py: Usage examples and demonstrations
	•	Dockerfile: Container configuration for deployment
	•	k8s/: Kubernetes deployment manifests (deployment, service, ingress, namespace)

Project Status: ✅ Complete

The project is production-ready with:
	•	Trained CNN-LSTM model for engine failure prediction
	•	RESTful API with health monitoring
	•	Docker containerization
	•	Kubernetes deployment configuration
	•	Comprehensive documentation and examples

Future Enhancements:
	•	Cross-dataset generalization (FD002–FD004)
	•	Model performance metrics dashboard
	•	Real-time monitoring and alerting integration
	•	Model versioning and A/B testing support

⸻

📚 Reference

A. Saxena, K. Goebel, D. Simon, and N. Eklund, Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation, PHM 2008.

⸻

👤 Author

Adham Mahgoub

Mechanical Engineer | Machine Learning Engineer

⸻
