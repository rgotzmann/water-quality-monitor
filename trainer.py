import pandas as pd
import numpy as np
import glob
import json
import time
import os
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)

FEATURE_COLS   = ["pH", "dissolved_oxygen", "turbidity", "temperature", "conductivity"]
CONTAMINATION  = 0.05   # expected anomaly rate ~5%
REGISTRY_DIR   = "model_registry"
DATA_CSV       = "data/resultphyschem.csv"  # fallback if no snapshots yet
SNAPSHOT_GLOB  = "data/snapshots/readings/**/*.parquet"

os.makedirs(REGISTRY_DIR, exist_ok=True)

print("WaterRich — Model Trainer")

snapshot_files = glob.glob(SNAPSHOT_GLOB, recursive=True)

if snapshot_files:
    print(f"\nLoading {len(snapshot_files)} parquet snapshot(s)...")
    df = pd.concat([pd.read_parquet(f) for f in snapshot_files], ignore_index=True)
    print(f"Loaded {len(df)} records from snapshots.")
else:
    print("\nNo snapshots found — loading and pivoting raw CSV...")
    raw = pd.read_csv(DATA_CSV, low_memory=False)

    PARAM_MAP = {
        "pH":                   "pH",
        "dissolved_oxygen":     "Dissolved oxygen (DO)",
        "turbidity":            "Turbidity",
        "temperature":          "Temperature, water",
        "conductivity":         "Specific conductance",
    }
    filtered = raw[raw["CharacteristicName"].isin(PARAM_MAP.values())][
        ["MonitoringLocationIdentifier", "ActivityStartDate",
         "ActivityLocation/LatitudeMeasure", "ActivityLocation/LongitudeMeasure",
         "CharacteristicName", "ResultMeasureValue"]
    ].copy()
    filtered["ResultMeasureValue"] = pd.to_numeric(
        filtered["ResultMeasureValue"], errors="coerce"
    )
    reverse_map = {v: k for k, v in PARAM_MAP.items()}
    filtered["param"] = filtered["CharacteristicName"].map(reverse_map)
    df = filtered.pivot_table(
        index=["MonitoringLocationIdentifier", "ActivityStartDate",
               "ActivityLocation/LatitudeMeasure", "ActivityLocation/LongitudeMeasure"],
        columns="param",
        values="ResultMeasureValue",
        aggfunc="mean",
    ).reset_index()
    df.columns.name = None
    df = df.rename(columns={
        "MonitoringLocationIdentifier":       "sensor_id",
        "ActivityStartDate":                  "timestamp",
        "ActivityLocation/LatitudeMeasure":   "latitude",
        "ActivityLocation/LongitudeMeasure":  "longitude",
    })
    print(f"Pivoted to {len(df)} multi-parameter records.")

# ── Feature matrix ────────────────────────────────────────────────────────────
available_cols = [c for c in FEATURE_COLS if c in df.columns]
missing_cols   = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    print(f"Warning: columns not found and will be skipped: {missing_cols}")

X_full = df[available_cols].dropna()
print(f"\nClean feature matrix: {len(X_full)} rows × {len(available_cols)} features")
print(f"Features used: {available_cols}")

with open(f"{REGISTRY_DIR}/feature_cols_v1.json", "w") as f:
    json.dump(available_cols, f)
print(f"Feature schema saved → {REGISTRY_DIR}/feature_cols_v1.json")

# ── Synthetic labels for evaluation ──────────────────────────────────────────
# Since the data is unlabelled, inject known anomalies as ground-truth:
# pH outside 6.5–8.5 OR dissolved oxygen < 6.0 mg/L = anomaly (label=1)
labels = pd.Series(0, index=X_full.index)
if "pH" in X_full.columns:
    labels[(X_full["pH"] < 5.5) | (X_full["pH"] > 9.5)] = 1
if "dissolved_oxygen" in X_full.columns:
    labels[X_full["dissolved_oxygen"] < 3.0] = 1

n_anomalies = labels.sum()
print(f"\nGround-truth anomalies (rule-based labels): {n_anomalies} / {len(labels)} "
      f"({100*n_anomalies/len(labels):.1f}%)")

# Train/test split (stratified on labels)
X_train, X_test, y_train, y_test = train_test_split(
    X_full, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# ════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Isolation Forest
# ════════════════════════════════════════════════════════════════════════════
print("MODEL 1 — Isolation Forest")

start = time.time()
iso = IsolationForest(
    n_estimators=100,
    contamination=CONTAMINATION,
    random_state=42,
    n_jobs=-1,
)
iso.fit(X_train)
iso_train_time = time.time() - start
print(f"Training time : {iso_train_time:.2f} s")

# Inference latency benchmark (1000 single-row predictions)
latencies = []
for _ in range(1000):
    sample = X_test.sample(1).values
    t0 = time.time()
    iso.predict(sample)
    latencies.append((time.time() - t0) * 1000)

iso_p50 = np.percentile(latencies, 50)
iso_p99 = np.percentile(latencies, 99)

# Evaluation — scores: higher = more normal in IF, so negate for anomaly probability
iso_scores  = -iso.score_samples(X_test)          # higher = more anomalous
iso_preds   = (iso.predict(X_test) == -1).astype(int)

iso_roc_auc = roc_auc_score(y_test, iso_scores)
iso_pr_auc  = average_precision_score(y_test, iso_scores)
iso_anomaly_rate = iso_preds.mean() * 100

# Model size
joblib.dump(iso, f"{REGISTRY_DIR}/isolation_forest_v1.pkl")
iso_size_mb = os.path.getsize(f"{REGISTRY_DIR}/isolation_forest_v1.pkl") / 1e6

print(f"ROC-AUC       : {iso_roc_auc:.4f}")
print(f"PR-AUC        : {iso_pr_auc:.4f}")
print(f"Anomaly rate  : {iso_anomaly_rate:.1f}%")
print(f"Latency p50   : {iso_p50:.2f} ms")
print(f"Latency p99   : {iso_p99:.2f} ms")
print(f"Model size    : {iso_size_mb:.2f} MB")
print(f"Model saved   → {REGISTRY_DIR}/isolation_forest_v1.pkl")

# ════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Local Outlier Factor
# ════════════════════════════════════════════════════════════════════════════
print("MODEL 2 — Local Outlier Factor (LOF)")

start = time.time()
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=CONTAMINATION,
    novelty=True,      # novelty=True required for predict() on new data
    n_jobs=-1,
)
lof.fit(X_train.values)
lof_train_time = time.time() - start
print(f"Training time : {lof_train_time:.2f} s")

# Inference latency benchmark
latencies = []
for _ in range(1000):
    sample = X_test.sample(1)
    t0 = time.time()
    lof.predict(sample)
    latencies.append((time.time() - t0) * 1000)
lof_p50 = np.percentile(latencies, 50)
lof_p99 = np.percentile(latencies, 99)

# Evaluation
lof_scores  = -lof.score_samples(X_test)
lof_preds   = (lof.predict(X_test) == -1).astype(int)

lof_roc_auc = roc_auc_score(y_test, lof_scores)
lof_pr_auc  = average_precision_score(y_test, lof_scores)
lof_anomaly_rate = lof_preds.mean() * 100

joblib.dump(lof, f"{REGISTRY_DIR}/lof_v1.pkl")
lof_size_mb = os.path.getsize(f"{REGISTRY_DIR}/lof_v1.pkl") / 1e6

print(f"ROC-AUC       : {lof_roc_auc:.4f}")
print(f"PR-AUC        : {lof_pr_auc:.4f}")
print(f"Anomaly rate  : {lof_anomaly_rate:.1f}%")
print(f"Latency p50   : {lof_p50:.2f} ms")
print(f"Latency p99   : {lof_p99:.2f} ms")
print(f"Model size    : {lof_size_mb:.2f} MB")
print(f"Model saved   → {REGISTRY_DIR}/lof_v1.pkl")

print("\n" + "=" * 60)
print("SUMMARY — Copy these values into your report")
print("=" * 60)
print(f"{'Metric':<25} {'Isolation Forest':>18} {'LOF':>18}")
print("-" * 62)
print(f"{'PR-AUC':<25} {iso_pr_auc:>18.4f} {lof_pr_auc:>18.4f}")
print(f"{'ROC-AUC':<25} {iso_roc_auc:>18.4f} {lof_roc_auc:>18.4f}")
print(f"{'Anomaly Rate (%)':<25} {iso_anomaly_rate:>17.1f}% {lof_anomaly_rate:>17.1f}%")
print(f"{'Train Time (s)':<25} {iso_train_time:>18.2f} {lof_train_time:>18.2f}")
print(f"{'Latency p50 (ms)':<25} {iso_p50:>18.2f} {lof_p50:>18.2f}")
print(f"{'Latency p99 (ms)':<25} {iso_p99:>18.2f} {lof_p99:>18.2f}")
print(f"{'Model Size (MB)':<25} {iso_size_mb:>18.2f} {lof_size_mb:>18.2f}")
print("=" * 60)

results = {
    "isolation_forest": {
        "pr_auc": round(iso_pr_auc, 4),
        "roc_auc": round(iso_roc_auc, 4),
        "anomaly_rate": round(iso_anomaly_rate, 1),
        "train_time_s": round(iso_train_time, 2),
        "latency_p50_ms": round(iso_p50, 2),
        "latency_p99_ms": round(iso_p99, 2),
        "model_size_mb": round(iso_size_mb, 2),
    },
    "lof": {
        "pr_auc": round(lof_pr_auc, 4),
        "roc_auc": round(lof_roc_auc, 4),
        "anomaly_rate": round(lof_anomaly_rate, 1),
        "train_time_s": round(lof_train_time, 2),
        "latency_p50_ms": round(lof_p50, 2),
        "latency_p99_ms": round(lof_p99, 2),
        "model_size_mb": round(lof_size_mb, 2),
    },
}
with open(f"{REGISTRY_DIR}/eval_results_v1.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {REGISTRY_DIR}/eval_results_v1.json")
print("\nDone.")
