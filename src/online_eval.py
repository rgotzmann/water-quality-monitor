"""
src/online_eval.py — Online KPI computation from Kafka reco_responses logs.

Proxy success metric: "alert confirmed within 30 minutes"
  - A prediction is a SUCCESS if is_alert=True AND within 30 min
    a second reading from the same sensor also scores anomalous.
  - This simulates "recommended item watched within N minutes" for water quality:
    a triggered alert is validated if the sensor keeps reading abnormally.

KPIs computed:
  1. Alert Confirmation Rate (ACR@30min) — % of alerts confirmed by follow-up reading
  2. False Alert Rate             — % of alerts NOT confirmed (potential false positives)
  3. Mean time to first alert     — latency from anomalous reading to alert
  4. % Personalised responses     — model (not fallback) responses / total
  5. Throughput                   — messages per minute in observation window
"""
import json

import pandas as pd
import numpy as np
from confluent_kafka import Consumer
from datetime import timedelta

CONFIRMATION_WINDOW_MIN = 30   # minutes

def load_responses_from_kafka(bootstrap: str, topic: str, # pragma: no cover
                               max_msgs: int = 10000) -> pd.DataFrame:
    """Consume all available messages from reco_responses topic."""
    consumer = Consumer({
        "bootstrap.servers":  bootstrap,
        "group.id":           "WaterRich-eval-v1",
        "auto.offset.reset":  "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([topic])
    records = []
    empty_polls = 0
    while len(records) < max_msgs and empty_polls < 5:
        msg = consumer.poll(2.0)
        if msg is None:
            empty_polls += 1
            continue
        if msg.error():
            continue
        try:
            records.append(json.loads(msg.value().decode()))
        except Exception:
            continue
    consumer.close()
    return pd.DataFrame(records) if records else pd.DataFrame()

def load_responses_from_file(path: str) -> pd.DataFrame:
    """Load probe_results.json or a JSONL log file for offline analysis."""
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame([data])
    # JSONL
    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(records)

def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute all online KPIs from a responses DataFrame."""
    if df.empty:
        return {"error": "No data available"}

    total = len(df)

    # Parse timestamps
    df = df.copy()
    df["ts"] = pd.to_datetime(df.get("timestamp", pd.NaT), errors="coerce")
    df = df.sort_values("ts")

    # % Personalised
    model_col = "model_version" if "model_version" in df.columns else None
    if model_col:
        personalised = df[model_col].isin(
            ["isolation_forest_v1", "lof_v1"]
        ).sum()
        pct_personalised = round(100 * personalised / total, 1)
    else:
        pct_personalised = None

    # Alert stats
    alert_col = "is_alert" if "is_alert" in df.columns else None
    if alert_col:
        n_alerts = int(df[alert_col].sum())
        alert_rate = round(100 * n_alerts / total, 1)
    else:
        n_alerts, alert_rate = 0, 0.0

    # Alert Confirmation Rate (ACR@30min)
    # For each alert, check if same sensor has another alert within 30 min
    acr = None
    if alert_col and "sensor_id" in df.columns and not df["ts"].isna().all():
        alerts_df = df[df[alert_col] ].copy()
        if len(alerts_df) > 0:
            confirmed = 0
            for _, row in alerts_df.iterrows():
                window_end = row["ts"] + timedelta(minutes=CONFIRMATION_WINDOW_MIN)
                follow_ups = df[
                    (df["sensor_id"] == row.get("sensor_id")) &
                    (df["ts"] > row["ts"]) &
                    (df["ts"] <= window_end) &
                    (df.get(alert_col, False) )
                ]
                if len(follow_ups) > 0:
                    confirmed += 1
            acr = round(100 * confirmed / len(alerts_df), 1)

    # Latency stats
    latency_col = "latency_ms" if "latency_ms" in df.columns else None
    if latency_col:
        lats = df[latency_col].dropna()
        lat_p50 = round(float(np.percentile(lats, 50)), 1) if len(lats) else None
        lat_p99 = round(float(np.percentile(lats, 99)), 1) if len(lats) else None
    else:
        lat_p50 = lat_p99 = None

    # Throughput (msgs/min over observation window)
    if not df["ts"].isna().all():
        window_min = (df["ts"].max() - df["ts"].min()).total_seconds() / 60
        throughput = round(total / window_min, 1) if window_min > 0 else None
    else:
        throughput = None

    return {
        "total_responses":       total,
        "pct_personalised":      pct_personalised,
        "n_alerts":              n_alerts,
        "alert_rate_pct":        alert_rate,
        "acr_30min_pct":         acr,
        "false_alert_rate_pct":  round(100 - acr, 1) if acr is not None else None,
        "latency_p50_ms":        lat_p50,
        "latency_p99_ms":        lat_p99,
        "throughput_msgs_per_min": throughput,
        "observation_window":    {
            "start": str(df["ts"].min()),
            "end":   str(df["ts"].max()),
        } if not df["ts"].isna().all() else None,
    }

def print_kpi_report(kpis: dict):
    print("\n" + "=" * 55)
    print("  WaterRich — Online KPI Report")
    print("=" * 55)
    print(f"  Total responses           : {kpis.get('total_responses')}")
    print(f"  % Personalised            : {kpis.get('pct_personalised')}%")
    print(f"  Alerts triggered          : {kpis.get('n_alerts')} ({kpis.get('alert_rate_pct')}%)")
    print(f"  Alert Confirmation @30min : {kpis.get('acr_30min_pct')}%")
    print(f"  False alert rate          : {kpis.get('false_alert_rate_pct')}%")
    print(f"  Latency p50               : {kpis.get('latency_p50_ms')} ms")
    print(f"  Latency p99               : {kpis.get('latency_p99_ms')} ms")
    print(f"  Throughput                : {kpis.get('throughput_msgs_per_min')} msg/min")
    if kpis.get("observation_window"):
        w = kpis["observation_window"]
        print(f"  Window                    : {w['start']} → {w['end']}")
    print("=" * 55)
