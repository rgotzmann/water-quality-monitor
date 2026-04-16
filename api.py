"""
api.py — WaterRich Anomaly Detection API
Features: /metrics, A/B split, provenance logging, hot-swap, root endpoint.
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import joblib
import json
import time
import os
import uuid
import hashlib
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone

REGISTRY_DIR    = os.getenv("REGISTRY_DIR",    "model_registry")
MODEL_VERSION   = os.getenv("MODEL_VERSION",   "isolation_forest_v1")
MODEL_VERSION_B = os.getenv("MODEL_VERSION_B", "lof_v1")          # A/B model B
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "-0.35"))
AB_SPLIT        = float(os.getenv("AB_SPLIT", "0.5"))             # 50/50 split
GIT_SHA         = os.getenv("GIT_SHA", "unknown")
IMAGE_DIGEST    = os.getenv("IMAGE_DIGEST", "unknown")
DATA_SNAPSHOT   = os.getenv("DATA_SNAPSHOT_ID", "snapshots/readings/year=2026")

_models = {}
_model_lock = threading.Lock()


def load_model(version: str):
    paths = [
        f"{REGISTRY_DIR}/{version}.pkl",
        f"{REGISTRY_DIR}/{version}/{version}.pkl",
    ]
    for path in paths:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(f"Model not found: {version}")

def get_model(version: str):
    with _model_lock:
        if version not in _models:
            print(f"[api] Loading model: {version}")
            _models[version] = load_model(version)
        return _models[version]


with open(f"{REGISTRY_DIR}/feature_cols_v1.json") as f:
    FEATURE_COLS = json.load(f)

# Pre-load both models at startup
get_model(MODEL_VERSION)
try:
    get_model(MODEL_VERSION_B)
except FileNotFoundError:
    print(f"[api] Model B ({MODEL_VERSION_B}) not found — A/B disabled")
    MODEL_VERSION_B = MODEL_VERSION

_metrics = {
    "requests_total":    0,
    "errors_total":      0,
    "alerts_total":      0,
    "latencies_ms":      deque(maxlen=1000),
    "ab_counts":         defaultdict(int),
    "ab_alerts":         defaultdict(int),
    "start_time":        time.time(),
}

# ── App
app = FastAPI(
    title="WaterRich Anomaly Detection API",
    description="Real-time water quality anomaly detection — Florida waterways",
    version="2.0.0",
)


# ── Schemas
class SensorReading(BaseModel):
    sensor_id:        str
    timestamp:        str
    pH:               Optional[float] = None
    dissolved_oxygen: Optional[float] = None
    turbidity:        Optional[float] = None
    temperature:      Optional[float] = None
    conductivity:     Optional[float] = None


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    request_id:        str
    sensor_id:         str
    timestamp:         str
    model_version:     str
    data_snapshot_id:  str
    pipeline_git_sha:  str
    container_digest:  str
    anomaly_score:     float
    is_alert:          bool
    latency_ms:        float
    ab_group:          str


# ── Helpers
def assign_ab_group(sensor_id: str) -> str:
    """Deterministic A/B assignment based on sensor_id hash."""
    h = int(hashlib.md5(sensor_id.encode()).hexdigest(), 16)
    return "A" if (h % 100) < int(AB_SPLIT * 100) else "B"


def run_inference(model_version: str, feature_values: list) -> float:
    model = get_model(model_version)
    is_lof = "lof" in model_version
    X = [feature_values]
    import numpy as np
    score = float(model.score_samples(
        np.array(X) if is_lof else X
    )[0])
    return score


# ── Endpoints
@app.get("/")
def root():
    uptime_s = int(time.time() - _metrics["start_time"])
    return {
        "project":   "WaterRich — Real-Time Water Quality Monitor",
        "team":      "WaterRich",
        "status":    "live",
        "model_a":   MODEL_VERSION,
        "model_b":   MODEL_VERSION_B,
        "uptime_s":  uptime_s,
        "endpoints": {
            "predict":  "POST /predict",
            "health":   "GET /health",
            "metrics":  "GET /metrics",
            "switch":   "POST /switch?model=<version>",
            "docs":     "GET /docs",
        },
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_a":      MODEL_VERSION,
        "model_b":      MODEL_VERSION_B,
        "git_sha":      GIT_SHA,
        "image_digest": IMAGE_DIGEST,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }

@app.get("/debug/models")
def debug_models():
    import glob
    files = glob.glob(f"{REGISTRY_DIR}/**/*.pkl", recursive=True)
    return {"pkl_files": files, "registry_dir": REGISTRY_DIR}

@app.get("/metrics")
def metrics():
    """Prometheus-style metrics for monitoring."""
    lats = list(_metrics["latencies_ms"])
    import numpy as np
    p50 = round(float(np.percentile(lats, 50)), 2) if lats else 0
    p95 = round(float(np.percentile(lats, 95)), 2) if lats else 0
    p99 = round(float(np.percentile(lats, 99)), 2) if lats else 0
    uptime = time.time() - _metrics["start_time"]
    error_rate = (
        _metrics["errors_total"] / _metrics["requests_total"]
        if _metrics["requests_total"] > 0 else 0.0
    )
    alert_rate = (
        _metrics["alerts_total"] / _metrics["requests_total"]
        if _metrics["requests_total"] > 0 else 0.0
    )
    return {
        "requests_total":   _metrics["requests_total"],
        "errors_total":     _metrics["errors_total"],
        "alerts_total":     _metrics["alerts_total"],
        "error_rate":       round(error_rate, 4),
        "alert_rate":       round(alert_rate, 4),
        "latency_p50_ms":   p50,
        "latency_p95_ms":   p95,
        "latency_p99_ms":   p99,
        "uptime_seconds":   round(uptime, 1),
        "ab_counts":        dict(_metrics["ab_counts"]),
        "ab_alerts":        dict(_metrics["ab_alerts"]),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    t0 = time.time()
    _metrics["requests_total"] += 1

    # A/B group assignment
    ab_group = assign_ab_group(reading.sensor_id)
    model_version = MODEL_VERSION if ab_group == "A" else MODEL_VERSION_B
    _metrics["ab_counts"][ab_group] += 1

    # Build feature vector
    feature_values = []
    for col in FEATURE_COLS:
        val = getattr(reading, col, None)
        if val is None:
            _metrics["errors_total"] += 1
            raise HTTPException(status_code=422,
                                detail=f"Missing required feature: {col}")
        feature_values.append(val)

    # Inference
    try:
        score = run_inference(model_version, feature_values)
    except Exception as e:
        _metrics["errors_total"] += 1
        raise HTTPException(status_code=500, detail=str(e))

    is_alert = score < ALERT_THRESHOLD
    latency_ms = (time.time() - t0) * 1000

    # Update metrics
    _metrics["latencies_ms"].append(latency_ms)
    if is_alert:
        _metrics["alerts_total"] += 1
        _metrics["ab_alerts"][ab_group] += 1

    # Provenance trace
    request_id = str(uuid.uuid4())

    return PredictionResponse(
        request_id       = request_id,
        sensor_id        = reading.sensor_id,
        timestamp        = reading.timestamp,
        model_version    = model_version,
        data_snapshot_id = DATA_SNAPSHOT,
        pipeline_git_sha = GIT_SHA,
        container_digest = IMAGE_DIGEST,
        anomaly_score    = score,
        is_alert         = is_alert,
        latency_ms       = round(latency_ms, 2),
        ab_group         = ab_group,
    )


@app.post("/switch")
def switch_model(model: str = Query(..., description="Model version to activate")):
    """Hot-swap the active model without restarting the container."""
    global MODEL_VERSION
    try:
        get_model(model)   # pre-load into cache
        MODEL_VERSION = model
        return {"status": "ok", "active_model": MODEL_VERSION}
    except FileNotFoundError:
        raise HTTPException(status_code=404,
                            detail=f"Model not found: {model}")


@app.get("/predictions/latest")
def latest_predictions():
    return {"model_version": MODEL_VERSION, "status": "live",
            "message": "POST to /predict with a sensor reading"}


@app.get("/alerts/active")
def active_alerts():
    return {"alert_threshold": ALERT_THRESHOLD,
            "model_version": MODEL_VERSION,
            "alerts_total": _metrics["alerts_total"]}
