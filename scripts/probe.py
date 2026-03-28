"""
WaterRich — API Probe Script
Hits /predict with real sensor readings and writes results to Kafka.
Run manually or via GitHub Actions cron (*/15 * * * *).
"""

import requests
import json
import time
import random
import os
from datetime import datetime
from confluent_kafka import Producer

API_URL       = os.getenv("API_URL", "https://water-quality-monitor-swmo.onrender.com")
BOOTSTRAP     = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
REQ_TOPIC     = "WaterRich.reco_requests"
RESP_TOPIC    = "WaterRich.reco_responses"
N_PROBES      = int(os.getenv("N_PROBES", "10"))  # probes per run

SAMPLE_READINGS = [
    # Normal Florida waterway readings
    {"sensor_id": "USGS-FL-001", "pH": 7.2, "dissolved_oxygen": 8.1,
     "turbidity": 1.2, "temperature": 24.5, "conductivity": 450.0},
    {"sensor_id": "USGS-FL-002", "pH": 6.8, "dissolved_oxygen": 7.5,
     "turbidity": 2.1, "temperature": 26.0, "conductivity": 380.0},
    {"sensor_id": "USGS-FL-003", "pH": 7.5, "dissolved_oxygen": 9.0,
     "turbidity": 0.8, "temperature": 23.0, "conductivity": 510.0},
    {"sensor_id": "USGS-FL-004", "pH": 6.5, "dissolved_oxygen": 6.8,
     "turbidity": 3.5, "temperature": 27.5, "conductivity": 620.0},
    {"sensor_id": "USGS-FL-005", "pH": 7.8, "dissolved_oxygen": 8.8,
     "turbidity": 1.5, "temperature": 22.0, "conductivity": 290.0},
    # Anomalous readings (pollution events)
    {"sensor_id": "USGS-FL-006", "pH": 4.2, "dissolved_oxygen": 2.1,
     "turbidity": 45.0, "temperature": 31.0, "conductivity": 1200.0},
    {"sensor_id": "USGS-FL-007", "pH": 10.5, "dissolved_oxygen": 1.5,
     "turbidity": 88.0, "temperature": 34.0, "conductivity": 2500.0},
    {"sensor_id": "USGS-FL-008", "pH": 5.1, "dissolved_oxygen": 2.8,
     "turbidity": 22.0, "temperature": 29.5, "conductivity": 950.0},
]

producer = Producer({"bootstrap.servers": BOOTSTRAP})

def delivery_report(err, msg):
    if err:
        print(f"  [KAFKA ERROR] {err}")

print(f"[WaterRich Probe] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[WaterRich Probe] API: {API_URL}")
print(f"[WaterRich Probe] Running {N_PROBES} probes...\n")

results = {
    "total":        0,
    "success":      0,
    "errors":       0,
    "alerts":       0,
    "personalized": 0,
    "latencies_ms": [],
}

for i in range(N_PROBES):
    reading = random.choice(SAMPLE_READINGS)
    reading["timestamp"] = datetime.now().isoformat()

    # Write request to Kafka
    producer.produce(
        REQ_TOPIC,
        key=reading["sensor_id"],
        value=json.dumps({"probe_id": i, "request": reading}),
        callback=delivery_report,
    )
    producer.poll(0)

    # Hit the API
    t0 = time.time()
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=reading,
            timeout=10,
        )
        latency_ms = (time.time() - t0) * 1000
        results["latencies_ms"].append(latency_ms)
        results["total"] += 1

        if response.status_code == 200:
            data = response.json()
            results["success"] += 1

            # Count as personalised if model (not fallback) responded
            if data.get("model_version") in ("isolation_forest_v1", "lof_v1"):
                results["personalized"] += 1

            if data.get("is_alert"):
                results["alerts"] += 1

            status = "ALERT" if data.get("is_alert") else "ok"
            print(f"  Probe {i+1:02d} | sensor={reading['sensor_id']} | "
                  f"score={data.get('anomaly_score', 0):.4f} | "
                  f"latency={latency_ms:.1f}ms | {status}")

            # Write response to Kafka
            producer.produce(
                RESP_TOPIC,
                key=reading["sensor_id"],
                value=json.dumps({
                    "probe_id":      i,
                    "model_version": data.get("model_version"),
                    "anomaly_score": data.get("anomaly_score"),
                    "is_alert":      data.get("is_alert"),
                    "latency_ms":    latency_ms,
                    "timestamp":     reading["timestamp"],
                }),
                callback=delivery_report,
            )
            producer.poll(0)

        else:
            results["errors"] += 1
            print(f"  Probe {i+1:02d} | ERROR {response.status_code}")

    except Exception as e:
        results["errors"] += 1
        results["total"]  += 1
        print(f"  Probe {i+1:02d} | EXCEPTION: {e}")

    time.sleep(0.5)

producer.flush()

import numpy as np

pct_success     = 100 * results["success"] / results["total"] if results["total"] else 0
pct_personalized = 100 * results["personalized"] / results["success"] if results["success"] else 0
p50 = np.percentile(results["latencies_ms"], 50) if results["latencies_ms"] else 0
p99 = np.percentile(results["latencies_ms"], 99) if results["latencies_ms"] else 0

print(f"""
╔══════════════════════════════════════════════╗
  WaterRich Probe Summary
  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
╠══════════════════════════════════════════════╣
  Total probes         : {results['total']}
  Successful (2xx)     : {results['success']}  ({pct_success:.1f}%)
  Errors               : {results['errors']}
  Alerts triggered     : {results['alerts']}
  % Personalised       : {pct_personalized:.1f}%
  Latency p50          : {p50:.1f} ms
  Latency p99          : {p99:.1f} ms
╚══════════════════════════════════════════════╝
""")

with open("probe_results.json", "w") as f:
    json.dump({
        "timestamp":       datetime.now().isoformat(),
        "total":           results["total"],
        "success":         results["success"],
        "errors":          results["errors"],
        "alerts":          results["alerts"],
        "pct_success":     round(pct_success, 1),
        "pct_personalized":round(pct_personalized, 1),
        "latency_p50_ms":  round(p50, 1),
        "latency_p99_ms":  round(p99, 1),
    }, f, indent=2)
print("Results saved -> probe_results.json")
