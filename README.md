# water-quality-monitor

# 💧 Real-Time Water Quality Monitor

A real-time water quality monitoring system using Apache Kafka and machine learning to detect anomalies in local waterway sensor data.

---

## Overview

This system streams water quality sensor readings through a Kafka pipeline, applies an ML anomaly detection model, and surfaces alerts when readings fall outside healthy parameters. It uses publicly available USGS waterway data and is designed to be extended with physical sensors.

---

## Architecture


Data Source (USGS API / Sensors) ->
        
  Kafka Producer ->
        
Kafka Topic: water.quality.readings ->
        
  Kafka Consumer ->
        
  ML Anomaly Detector (Isolation Forest) ->
        
Kafka Topic: water.quality.alerts ->
        
  Alert System


---

## Monitored Parameters

| Parameter          | Healthy Range      | Why It Matters                        |
|--------------------|--------------------|---------------------------------------|
| pH                 | 6.5 – 8.5          | Acidity — fish die outside this range |
| Dissolved Oxygen   | > 6 mg/L           | Low = dead zones, pollution           |
| Turbidity          | < 1 NTU (drinking) | Cloudiness — indicates runoff/sediment|
| Temperature        | Varies by season   | Affects oxygen levels & aquatic life  |
| Conductivity / TDS | < 500 µS/cm        | Detects salt or chemical pollution    |

---

## Getting Started

### Prerequisites

- Python 3.9+
- Docker Desktop (for Kafka)
- USGS API key

### 1. Start Kafka

```bash
docker run -d --name kafka -p 9092:9092 apache/kafka:latest
```

### 2. Create Kafka Topics

```bash
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh \
  --create --topic water.quality.readings \
  --bootstrap-server localhost:9092 \
  --partitions 1 --replication-factor 1

docker exec -it kafka /opt/kafka/bin/kafka-topics.sh \
  --create --topic water.quality.alerts \
  --bootstrap-server localhost:9092 \
  --partitions 1 --replication-factor 1
```

### 3. Install Dependencies

```bash
pip install confluent-kafka pandas scikit-learn joblib requests
```

### 4. Train the Model

```bash
python trainer.py
```

### 5. Run the Pipeline (3 concurrent terminals)

```bash
# Terminal 1 — Consumer / Anomaly Detector
python stream_ingestor.py

# Terminal 2 — Producer / Data Stream
python event_generator.py

# Terminal 3 — View live alerts
docker exec -it kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic water.quality.alerts \
  --from-beginning
```

---

## Project Structure

```
project/
├── data/
│   └── water_quality.csv        # USGS data goes here
├── model_registry/              # Auto-created by trainer.py
│   ├── isolation_forest_v1.pkl
│   └── feature_cols_v1.json
├── event_generator.py           # Kafka producer — streams readings
├── stream_ingestor.py           # Kafka consumer — runs ML inference
├── trainer.py                   # Trains anomaly detection model
└── README.md
```

---

## Data Sources

- **[USGS Water Quality Portal](https://waterqualitydata.us)** — Historical readings from US waterways (search by state/county)
- **[EPA Water Quality Portal](https://www.epa.gov/waterdata)** — National water monitoring data
- **[OpenAQ](https://openaq.org)** — Global environmental sensor data

---

## ML Model

The system uses **Isolation Forest**, an unsupervised anomaly detection algorithm which can process multivariate sensor data (pH, dissolved oxygen, etc). It is trained on historical "normal" readings to achieve baseline and flags deviations in real time.

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.05)  
model.fit(historical_readings)
```

---

## Future Work

- [ ] Physical sensor integration (Raspberry Pi + Atlas Scientific)
- [ ] Grafana dashboard for live visualisation
- [ ] SMS/email alerts for critical anomalies
- [ ] Multi-location support across waterway network
- [ ] Seasonal baseline recalibration

---

## Tech Stack

| Component        | Technology                      |
|------------------|---------------------------------|
| Streaming        | Apache Kafka                    |
| ML               | scikit-learn (Isolation Forest) |
| Language         | Python 3.9+                     |
| Data             | USGS Water Quality Portal       |
| Containerization | Docker                          |

---

## 📄 License

MIT License — see `LICENSE` for details.
