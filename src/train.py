"""
src/train.py — Train anomaly detection models + serialize with metadata.
"""
import os
import json
import time
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, average_precision_score

def train_isolation_forest(X_train: pd.DataFrame,
                            contamination: float = 0.05) -> IsolationForest:
    model = IsolationForest(n_estimators=100, contamination=contamination,
                             random_state=42, n_jobs=-1)
    model.fit(X_train)
    return model

def train_lof(X_train: pd.DataFrame,
              contamination: float = 0.05) -> LocalOutlierFactor:
    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination,
                                novelty=True, n_jobs=-1)
    model.fit(X_train.values)
    return model

def evaluate_model(model, X_test: pd.DataFrame,
                   y_test: pd.Series, model_name: str) -> dict:
    is_lof = isinstance(model, LocalOutlierFactor)
    X = X_test.values if is_lof else X_test
    scores  = -model.score_samples(X)
    preds   = (model.predict(X) == -1).astype(int)

    metrics = {
        "model":        model_name,
        "roc_auc":      round(float(roc_auc_score(y_test, scores)), 4),
        "pr_auc":       round(float(average_precision_score(y_test, scores)), 4),
        "anomaly_rate": round(float(preds.mean() * 100), 1),
    }

    # Subpopulation analysis by sensor quartile
    quartiles = pd.qcut(np.arange(len(X_test)), q=4,
                        labels=["Q1","Q2","Q3","Q4"])
    sub = {}
    for q in ["Q1","Q2","Q3","Q4"]:
        mask = quartiles == q
        if mask.sum() < 5:
            continue
        sub[q] = {
            "roc_auc": round(float(roc_auc_score(
                y_test[mask], scores[mask])), 4) if y_test[mask].sum() > 0 else None,
            "n": int(mask.sum()),
        }
    metrics["subpopulation"] = sub
    return metrics

def benchmark_latency(model, X_test: pd.DataFrame, n: int = 1000) -> dict:
    is_lof = isinstance(model, LocalOutlierFactor)
    latencies = []
    for _ in range(n):
        sample = X_test.sample(1)
        X = sample.values if is_lof else sample
        t0 = time.perf_counter()
        model.predict(X)
        latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
    }

def serialize_model(model, model_name: str, metrics: dict,
                    latency: dict, feature_cols: list,
                    registry_dir: str, git_commit: str = "unknown"):
    os.makedirs(registry_dir, exist_ok=True)
    path = f"{registry_dir}/{model_name}.pkl"
    joblib.dump(model, path)
    size_mb = round(os.path.getsize(path) / 1e6, 2)

    metadata = {
        "model_name":    model_name,
        "version":       "v1",
        "trained_at":    pd.Timestamp.now().isoformat(),
        "feature_cols":  feature_cols,
        "git_commit":    git_commit,
        "size_mb":       size_mb,
        **metrics,
        **latency,
    }
    meta_path = f"{registry_dir}/{model_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[train] Saved {model_name} → {path}  ({size_mb} MB)")
    print(f"[train]   ROC-AUC={metrics['roc_auc']}  "
          f"PR-AUC={metrics['pr_auc']}  "
          f"p50={latency['p50_ms']}ms")
    return metadata
