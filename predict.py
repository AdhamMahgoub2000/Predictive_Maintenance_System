import torch
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from train import PredictiveMaintenance

# ======================================================
# Configuration (MUST match training)
# ======================================================
WINDOW_SIZE = 20
INPUT_SIZE = 17          # 3 operational settings + 14 sensors (21 - 7 constant sensors)
CNN_CHANNELS = 64
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
NUM_CLASSES = 2
DROPOUT = 0.3

MODEL_PATH = "predictive_maintenance_model.pth"
SCALER_PATH = "scaler.pkl"

# Sensors removed during training
CONSTANT_SENSORS = ['sensor_1','sensor_6','sensor_5','sensor_10',
                    'sensor_16','sensor_18','sensor_19']

# ======================================================
# Request schema
# ======================================================
class EngineData(BaseModel):
    # 20 timesteps × 17 features = 340
    features: conlist(float, min_length=340, max_length=340)

# ======================================================
# Load scaler
# ======================================================
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ======================================================
# Load model
# ======================================================
model = PredictiveMaintenance(
    INPUT_SIZE,
    CNN_CHANNELS,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    NUM_CLASSES,
    cnn_dropout=DROPOUT,
    lstm_dropout=DROPOUT,
    fc_dropout=DROPOUT
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

print("✅ Model & scaler loaded successfully")

# ======================================================
# FastAPI app
# ======================================================
app = FastAPI(title="Predictive Maintenance API")

@app.get("/health")
def health():
    """
    Health check endpoint for Kubernetes probes.
    Returns 200 OK if the service is ready to serve requests.
    """
    return {"status": "healthy", "service": "predictive-maintenance-api"}

@app.post("/predict")
def predict(data: EngineData):
    try:
        # Convert to numpy
        x = np.array(data.features, dtype=np.float32)

        # Reshape → (20, 17)
        x = x.reshape(WINDOW_SIZE, INPUT_SIZE)

        # Scale only sensor columns (indices 3-16), not operational settings (indices 0-2)
        # The scaler was fitted only on sensor columns during training
        sensor_indices = list(range(3, 17))  # 14 sensor columns
        op_set_indices = list(range(3))  # 3 operational settings
        
        # Scale sensor columns
        x_sensors = scaler.transform(x[:, sensor_indices])
        
        # Combine: operational settings (unchanged) + scaled sensors
        x_scaled = np.zeros_like(x)
        x_scaled[:, op_set_indices] = x[:, op_set_indices]  # Keep operational settings as-is
        x_scaled[:, sensor_indices] = x_sensors  # Use scaled sensors

        # Convert to tensor → (1, 20, 17)
        x = torch.tensor(x_scaled).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        return {
            "predicted_class": int(pred_class),
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ======================================================
# Run server
# ======================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)