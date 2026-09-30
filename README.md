# WaterRich — Real-Time Water Quality Monitor

A real-time water quality monitoring system using Apache Kafka and machine learning to detect anomalies in Florida waterway sensor data. Streams USGS historical readings through a Kafka pipeline, applies ML anomaly detection, and surfaces alerts within seconds of a sensor reading crossing a threshold.
=======
# Real-Time Water Quality Monitor

**Live API:** https://water-quality-monitor-swmo.onrender.com  
**Repo:** https://github.com/rgotzmann/water-quality-monitor

---

## Overview

Traditional water quality monitoring relies on infrequent manual sampling. WaterRich provides continuous, real-time monitoring of key water quality parameters using machine learning to detect anomalies and trigger alerts automatically.

```
USGS API / Sensors
      ↓
Kafka Producer (event_generator.py)
      ↓
Kafka Topic: WaterRich.readings.raw
      ↓
Stream Ingestor (stream_ingestor.py) — pandera schema validation → parquet snapshots
      ↓
FastAPI (api.py) — Isolation Forest + LOF A/B split
      ↓
Kafka Topic: WaterRich.alerts.active
      ↓
Alert System / Dashboard
```

---

## Monitored Parameters

| Parameter        | Healthy Range       | Why It Matters                         |
|------------------|---------------------|----------------------------------------|
| pH               | 5.5 – 9.5           | Acidity — fish die outside this range  |
| Dissolved Oxygen | > 3.0 mg/L          | Low = dead zones, pollution            |
| Turbidity        | < 1 NTU (drinking)  | Cloudiness — indicates runoff/sediment |
| Temperature      | Varies by season    | Affects oxygen levels & aquatic life   |
| Conductivity     | < 500 µS/cm         | Detects salt or chemical pollution     |

---

## ML Models

Two unsupervised anomaly detection models are trained on historical USGS readings using a chronological split (no temporal leakage):

| Model                | ROC-AUC | PR-AUC | Latency p50 | Size    | Role                  |
|----------------------|---------|--------|-------------|---------|-----------------------|
| Isolation Forest     | 0.7242  | 0.3685 | 1.36 ms     | 0.97 MB | Primary (A/B Group A) |
| Local Outlier Factor | 0.4735  | 0.1430 | 16.70 ms    | 0.28 MB | Challenger (A/B Group B) |

Isolation Forest is the deployed model. A/B traffic is split deterministically by `sensor_id` hash. Results are written to `WaterRich.reco_responses` and analysed via `scripts/ab_analysis.py`.

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
```

---

## Project Structure

```
water-quality-monitor/
├── src/
│   ├── ingest.py          # Kafka consumer + pandera schema validation
│   ├── transform.py       # Chronological split, feature engineering
│   ├── train.py           # Model training, evaluation, serialization
│   ├── drift.py           # PSI drift detection per feature
│   └── online_eval.py     # KPI computation from Kafka logs
├── tests/
│   └── test_pipeline.py   # 27 unit tests, 76% coverage
├── scripts/
│   ├── probe.py           # Periodic API probe → Kafka topics
│   ├── retrain.py         # Automated retraining with versioning
│   ├── ab_analysis.py     # Two-proportion z-test on A/B results
│   └── availability.py    # Availability calculation over time window
├── .github/workflows/
│   └── ci.yml             # Lint → Test → Build → Push → Deploy → Retrain
├── data/
│   └── usgs_water_quality.csv   # USGS Palm Beach County data (gitignored)
├── model_registry/
│   ├── isolation_forest_v1.pkl
│   ├── lof_v1.pkl
│   ├── feature_cols_v1.json
│   ├── latest.json
│   └── v1.0/ v1.1/ v1.2/ v1.3/  # Versioned model artifacts
├── api.py                 # FastAPI — /predict, /metrics, /switch, /health
├── pipeline.py            # Unified runner: --stage train|eval|drift|all
├── event_generator.py     # Kafka producer — replays USGS CSV
├── stream_ingestor.py     # Kafka consumer — validates + snapshots
├── trainer.py             # Legacy single-file trainer
├── Dockerfile             # Multi-stage build, non-root user
├── requirements.txt
├── .env.example           # All config as environment variables
├── setup.cfg              # Coverage exclusion rules
└── RUNBOOK.md             # Alert rules, SLOs, rollback procedures
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker Desktop
- Conda (recommended)

### 1. Create Environment

```bash
conda create -n waterrich python=3.11 -y
conda activate waterrich
pip install -r requirements.txt
```

### 2. Start Kafka

```bash
docker run -d --name kafka -p 9092:9092 apache/kafka:latest
```

### 3. Create Kafka Topics

```bash
for topic in readings.raw predictions.out alerts.active reco_requests reco_responses; do
  docker exec -it kafka /opt/kafka/bin/kafka-topics.sh \
    --create --topic WaterRich.$topic \
    --bootstrap-server localhost:9092 \
    --partitions 3 --replication-factor 1
done
```

### 4. Download Data

Go to [waterqualitydata.us](https://waterqualitydata.us) and download:
- **Location:** Florida → Palm Beach County
- **Data profile:** Sample Results (physical/chemical metadata)
- **Format:** CSV

Save as `data/usgs_water_quality.csv`.

### 5. Train Models

```bash
python pipeline.py --stage train
```

### 6. Run the Pipeline

```bash
# Terminal 1 — Stream ingestor (consumer + schema validation)
python stream_ingestor.py

# Terminal 2 — Event generator (producer — replays USGS data)
python event_generator.py

# Terminal 3 — View live alerts
docker exec -it kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic WaterRich.alerts.active \
  --from-beginning
```

### 7. Start the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8080
```

Visit `http://localhost:8080/docs` for the interactive Swagger UI.

---

## API Endpoints

| Endpoint         | Method | Description                           | SLO Latency  |
|------------------|--------|---------------------------------------|--------------|
| `/`              | GET    | Project status + uptime               | —            |
| `/health`        | GET    | Model versions, git sha, image digest | < 50 ms p99  |
| `/predict`       | POST   | Anomaly score + provenance trace      | < 200 ms p99 |
| `/metrics`       | GET    | p95 latency, error rate, A/B counts   | < 50 ms p99  |
| `/switch`        | POST   | Zero-downtime model hot-swap          | —            |
| `/alerts/active` | GET    | Current alert threshold + count       | < 500 ms p99 |

### Example: POST /predict

```bash
curl -X POST https://water-quality-monitor-swmo.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id":        "USGS-FL-007",
    "timestamp":        "2026-04-15",
    "pH":               4.2,
    "dissolved_oxygen": 1.8,
    "turbidity":        45.0,
    "temperature":      31.0,
    "conductivity":     1200.0
  }'
```

Response includes `request_id`, `model_version`, `anomaly_score`, `is_alert`, `ab_group`, `pipeline_git_sha`, and `container_digest` for full provenance tracing.

---

## Configuration

All config is via environment variables. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key variables:

| Variable          | Default                       | Description              |
|-------------------|-------------------------------|--------------------------|
| `KAFKA_BOOTSTRAP` | `localhost:9092`              | Kafka broker address     |
| `MODEL_VERSION`   | `isolation_forest_v1`         | Active model             |
| `ALERT_THRESHOLD` | `-0.35`                       | Anomaly score threshold  |
| `CONTAMINATION`   | `0.05`                        | Expected anomaly rate    |
| `DATA_CSV`        | `data/usgs_water_quality.csv` | Training data path       |

---

## CI/CD

GitHub Actions runs on every push to `main`:

1. **Lint** — flake8
2. **Test** — pytest (27 tests, 76% coverage, >= 70% gate)
3. **Build & Push** — Docker image to `ghcr.io/rgotzmann/waterrich-api:latest`
4. **Deploy** — Render deploy via API
5. **Retrain** — Daily at 02:00 UTC, auto-commits new model version

### Automated Retraining

```bash
python scripts/retrain.py         # creates model_registry/vX.Y/
```

Hot-swap to new version without restart:

```bash
curl -X POST "https://water-quality-monitor-swmo.onrender.com/switch?model=isolation_forest_v1.3"
```

---

## Testing

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Test coverage by module:

| Module             | Coverage |
|--------------------|----------|
| src/train.py       | 98%      |
| src/drift.py       | 75%      |
| src/online_eval.py | 82%      |
| src/transform.py   | 61%      |
| **Total**          | **76%**  |

---

## Data Sources

- **[USGS Water Quality Portal](https://waterqualitydata.us)** — 23,222 sample results from 207 Palm Beach County sites
- **[EPA Water Quality Portal](https://www.epa.gov/waterdata)** — National water monitoring data
- **[OpenAQ](https://openaq.org)** — Global environmental sensor data

---

## Tech Stack

| Component          | Technology                           |
|--------------------|--------------------------------------|
| Streaming          | Apache Kafka (KRaft, no Zookeeper)   |
| ML                 | scikit-learn — Isolation Forest + LOF|
| Schema validation  | pandera                              |
| API                | FastAPI + uvicorn                    |
| Language           | Python 3.11                          |
| Containerization   | Docker (multi-stage)                 |
| CI/CD              | GitHub Actions                       |
| Cloud deploy       | Render                               |
| Data               | USGS Water Quality Portal            |

---

## Monitoring & Ops

- **Live metrics:** `GET /metrics` — p50/p95/p99 latency, error rate, alert rate, A/B counts
- **Drift detection:** `python pipeline.py --stage drift` — PSI per feature
- **Availability:** `python scripts/availability.py --hours 216`
- **Runbook:** see `RUNBOOK.md` for alert rules and rollback procedures

---

## License

MIT License — see `LICENSE` for details.
