from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import json
import time
import os

MODEL_VERSION = os.getenv("MODEL_VERSION", "isolation_forest_v1")
THRESHOLD     = float(os.getenv("ALERT_THRESHOLD", "-0.1"))  # IF score threshold

print(f"Loading model: {MODEL_VERSION}...")
model = joblib.load(f"model_registry/{MODEL_VERSION}.pkl")

with open("model_registry/feature_cols_v1.json") as f:
    feature_cols = json.load(f)

print(f"Model loaded. Features: {feature_cols}")

app = FastAPI(
    title="WaterRich Anomaly Detection API",
    description="Real-time water quality anomaly detection for Florida waterways",
    version="1.0.0",
)

# schemas
class SensorReading(BaseModel):
    sensor_id:        str
    timestamp:        str
    pH:               Optional[float] = None
    dissolved_oxygen: Optional[float] = None
    turbidity:        Optional[float] = None
    temperature:      Optional[float] = None
    conductivity:     Optional[float] = None

class PredictionResponse(BaseModel):
    sensor_id:      str
    timestamp:      str
    model_version:  str
    anomaly_score:  float
    is_alert:       bool
    latency_ms:     float
    features_used:  list

# endpoints
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_version": MODEL_VERSION,
        "features":      feature_cols,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    t0 = time.time()

    feature_values = []
    features_used  = []
    for col in feature_cols:
        val = getattr(reading, col, None)
        if val is None:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required feature: {col}"
            )
        feature_values.append(val)
        features_used.append(col)

    # Run inference
    score    = float(model.score_samples([feature_values])[0])
    is_alert = score < THRESHOLD

    latency_ms = (time.time() - t0) * 1000

    return PredictionResponse(
        sensor_id     = reading.sensor_id,
        timestamp     = reading.timestamp,
        model_version = MODEL_VERSION,
        anomaly_score = score,
        is_alert      = is_alert,
        latency_ms    = round(latency_ms, 2),
        features_used = features_used,
    )

@app.get("/predictions/latest")
def latest_predictions():
    """Placeholder — returns model metadata. Wire to DB in production."""
    return {
        "model_version": MODEL_VERSION,
        "status":        "live",
        "message":       "POST to /predict with a sensor reading to get anomaly score",
    }

@app.get("/alerts/active")
def active_alerts():
    """Placeholder — returns threshold config. Wire to DB in production."""
    return {
        "alert_threshold": THRESHOLD,
        "model_version":   MODEL_VERSION,
        "message":         "Alerts are written to WaterRich.alerts.active Kafka topic",
    }
