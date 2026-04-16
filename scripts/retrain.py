"""
scripts/retrain.py — Automated retraining with semantic versioning.
Trains both models, evaluates, publishes to model_registry/vX.Y/,
and writes metadata for hot-swap and provenance.

Usage:
  python scripts/retrain.py               # auto-increment patch version
  python scripts/retrain.py --major       # bump major version
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

REGISTRY_DIR   = os.getenv("REGISTRY_DIR",   "model_registry")
SNAPSHOT_GLOB  = os.getenv("SNAPSHOT_GLOB",  "snapshots/readings/**/*.parquet")
DATA_CSV       = os.getenv("DATA_CSV",        "data/usgs_water_quality.csv")
CONTAMINATION  = float(os.getenv("CONTAMINATION", "0.05"))
FEATURE_COLS   = ["pH", "dissolved_oxygen", "turbidity",
                  "temperature", "conductivity"]
PARAM_MAP = {
    "pH":               "pH",
    "dissolved_oxygen": "Dissolved oxygen (DO)",
    "turbidity":        "Turbidity",
    "temperature":      "Temperature, water",
    "conductivity":     "Specific conductance",
}


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def next_version(registry_dir: str, bump_major: bool = False) -> str:
    """Find highest existing vX.Y and increment."""
    versions = []
    for d in glob.glob(f"{registry_dir}/v*"):
        base = os.path.basename(d)
        if base.startswith("v") and "." in base:
            try:
                parts = base[1:].split(".")
                versions.append((int(parts[0]), int(parts[1])))
            except ValueError:
                continue
    if not versions:
        return "v1.0"
    major, minor = max(versions)
    if bump_major:
        return f"v{major + 1}.0"
    return f"v{major}.{minor + 1}"


def load_data() -> pd.DataFrame:
    files = glob.glob(SNAPSHOT_GLOB, recursive=True)
    if files:
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        print(f"[retrain] Loaded {len(df)} rows from {len(files)} snapshots")
        return df
    print("[retrain] No snapshots — loading CSV...")
    raw = pd.read_csv(DATA_CSV, low_memory=False)
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
               "ActivityLocation/LatitudeMeasure",
               "ActivityLocation/LongitudeMeasure"],
        columns="param", values="ResultMeasureValue", aggfunc="mean",
    ).reset_index()
    df.columns.name = None
    return df


def main(bump_major: bool = False):
    version = next_version(REGISTRY_DIR, bump_major)
    version_dir = f"{REGISTRY_DIR}/{version}"
    os.makedirs(version_dir, exist_ok=True)
    print(f"\n[retrain] === WaterRich Retraining — {version} ===")
    print(f"[retrain] Output dir: {version_dir}")

    df = load_data()
    available = [c for c in FEATURE_COLS if c in df.columns]
    X_full = df[available].dropna()

    # Chronological split
    split_idx = int(len(X_full) * 0.8)
    X_train = X_full.iloc[:split_idx]
    X_test  = X_full.iloc[split_idx:]

    labels = pd.Series(0, index=X_full.index)
    if "pH" in X_full.columns:
        labels[(X_full["pH"] < 5.5) | (X_full["pH"] > 9.5)] = 1
    if "dissolved_oxygen" in X_full.columns:
        labels[X_full["dissolved_oxygen"] < 3.0] = 1
    y_test = labels.iloc[split_idx:]

    git_sha   = get_git_sha()
    trained_at = datetime.utcnow().isoformat() + "Z"

    # Save feature schema
    with open(f"{version_dir}/feature_cols.json", "w") as f:
        json.dump(available, f)

    results = {}
    for name, ModelClass, kwargs in [
        ("isolation_forest", IsolationForest,
         {"n_estimators": 100, "contamination": CONTAMINATION,
          "random_state": 42, "n_jobs": -1}),
        ("lof",              LocalOutlierFactor,
         {"n_neighbors": 20, "contamination": CONTAMINATION,
          "novelty": True, "n_jobs": -1}),
    ]:
        print(f"\n[retrain] Training {name}...")
        t0 = time.time()
        model = ModelClass(**kwargs)
        is_lof = name == "lof"
        model.fit(X_train.values if is_lof else X_train)
        train_time = round(time.time() - t0, 2)

        scores = -model.score_samples(
            X_test.values if is_lof else X_test
        )
        roc = round(float(roc_auc_score(y_test, scores)), 4) \
            if y_test.sum() > 0 else None
        pr  = round(float(average_precision_score(y_test, scores)), 4) \
            if y_test.sum() > 0 else None

        model_name = f"{name}_{version}"
        pkl_path   = f"{version_dir}/{model_name}.pkl"
        joblib.dump(model, pkl_path)
        size_mb = round(os.path.getsize(pkl_path) / 1e6, 2)

        # Also write to root registry for backward compat
        joblib.dump(model, f"{REGISTRY_DIR}/{name}_v1.pkl")

        meta = {
            "model_name":      model_name,
            "version":         version,
            "trained_at":      trained_at,
            "git_sha":         git_sha,
            "training_rows":   len(X_train),
            "feature_cols":    available,
            "contamination":   CONTAMINATION,
            "roc_auc":         roc,
            "pr_auc":          pr,
            "train_time_s":    train_time,
            "size_mb":         size_mb,
        }
        with open(f"{version_dir}/{model_name}_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        results[name] = meta
        print(f"[retrain] {name}: ROC-AUC={roc}  PR-AUC={pr}  "
              f"time={train_time}s  size={size_mb}MB")

    # Write version manifest
    manifest = {
        "version":    version,
        "trained_at": trained_at,
        "git_sha":    git_sha,
        "models":     results,
    }
    manifest_path = f"{version_dir}/manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Update latest pointer
    with open(f"{REGISTRY_DIR}/latest.json", "w") as f:
        json.dump({"version": version, "updated_at": trained_at}, f)

    print(f"\n[retrain] Done. Version {version} published.")
    print(f"[retrain] Manifest: {manifest_path}")
    print(f"[retrain] To activate: POST /switch?model=isolation_forest_{version}")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--major", action="store_true",
                        help="Bump major version instead of minor")
    parser.add_argument("--skip-if-no-data", action="store_true")
    args = parser.parse_args()

    if args.skip_if_no_data:
        files = glob.glob(SNAPSHOT_GLOB, recursive=True)
        if not files and not os.path.exists(DATA_CSV):
            print("[retrain] No data available — skipping.")
            sys.exit(0)

    main(bump_major=args.major) 
