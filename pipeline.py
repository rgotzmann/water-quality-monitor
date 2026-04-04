"""
pipeline.py — End-to-end pipeline runner.
All config from environment variables or .env file.
Usage:
  python pipeline.py --stage train
  python pipeline.py --stage eval
  python pipeline.py --stage drift
  python pipeline.py --stage all
"""
import argparse
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

CFG = {
    "BOOTSTRAP":        os.getenv("KAFKA_BOOTSTRAP",    "localhost:9092"),
    "INPUT_TOPIC":      os.getenv("INPUT_TOPIC",        "WaterRich.readings.raw"),
    "ALERT_TOPIC":      os.getenv("ALERT_TOPIC",        "WaterRich.alerts.active"),
    "RESP_TOPIC":       os.getenv("RESP_TOPIC",         "WaterRich.reco_responses"),
    "SNAPSHOT_DIR":     os.getenv("SNAPSHOT_DIR",       "snapshots/readings"),
    "SNAPSHOT_GLOB":    os.getenv("SNAPSHOT_GLOB",      "snapshots/readings/**/*.parquet"),
    "DATA_CSV":         os.getenv("DATA_CSV",           "data/usgs_water_quality.csv"),
    "REGISTRY_DIR":     os.getenv("REGISTRY_DIR",       "model_registry"),
    "CONTAMINATION":    float(os.getenv("CONTAMINATION","0.05")),
    "TEST_FRAC":        float(os.getenv("TEST_FRAC",    "0.2")),
    "MODEL_VERSION":    os.getenv("MODEL_VERSION",      "isolation_forest_v1"),
    "ALERT_THRESHOLD":  float(os.getenv("ALERT_THRESHOLD", "-0.1")),
}

def stage_train():
    from src.transform import (load_snapshots, load_csv, chronological_split,
                                get_feature_matrix, make_labels)
    from src.train import (train_isolation_forest, train_lof,
                            evaluate_model, benchmark_latency, serialize_model)

    print("\n[pipeline] === TRAIN STAGE ===")

    # Load data
    import glob
    files = glob.glob(CFG["SNAPSHOT_GLOB"], recursive=True)
    if files:
        df = load_snapshots(CFG["SNAPSHOT_GLOB"])
    else:
        df = load_csv(CFG["DATA_CSV"])

    # Chronological split (no leakage)
    train_df, test_df = chronological_split(df, test_frac=CFG["TEST_FRAC"])

    X_train = get_feature_matrix(train_df).dropna()
    X_test  = get_feature_matrix(test_df).dropna()
    y_train = make_labels(X_train)
    y_test  = make_labels(X_test)

    feature_cols = list(X_train.columns)

    # Save feature schema
    os.makedirs(CFG["REGISTRY_DIR"], exist_ok=True)
    with open(f"{CFG['REGISTRY_DIR']}/feature_cols_v1.json", "w") as f:
        json.dump(feature_cols, f)

    git_commit = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip()

    all_results = []
    for name, train_fn in [
        ("isolation_forest_v1", train_isolation_forest),
        ("lof_v1",              train_lof),
    ]:
        print(f"\n[pipeline] Training {name}...")
        model   = train_fn(X_train, CFG["CONTAMINATION"])
        metrics = evaluate_model(model, X_test, y_test, name)
        latency = benchmark_latency(model, X_test)
        meta    = serialize_model(model, name, metrics, latency,
                                   feature_cols, CFG["REGISTRY_DIR"], git_commit)
        all_results.append(meta)

    # Save combined results
    with open(f"{CFG['REGISTRY_DIR']}/eval_results_v1.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[pipeline] Results → {CFG['REGISTRY_DIR']}/eval_results_v1.json")

def stage_eval():
    from src.online_eval import load_responses_from_kafka, compute_kpis, print_kpi_report
    import json

    print("\n[pipeline] === ONLINE EVAL STAGE ===")
    df   = load_responses_from_kafka(CFG["BOOTSTRAP"], CFG["RESP_TOPIC"])
    kpis = compute_kpis(df)
    print_kpi_report(kpis)
    with open("online_kpis.json", "w") as f:
        json.dump(kpis, f, indent=2)
    print("[pipeline] KPIs saved → online_kpis.json")

def stage_drift():
    from src.transform import load_snapshots, load_csv, chronological_split, get_feature_matrix
    from src.drift import check_drift, save_drift_report
    import glob

    print("\n[pipeline] === DRIFT STAGE ===")
    files = glob.glob(CFG["SNAPSHOT_GLOB"], recursive=True)
    df = load_snapshots(CFG["SNAPSHOT_GLOB"]) if files else load_csv(CFG["DATA_CSV"])
    baseline_df, current_df = chronological_split(df, test_frac=0.2)
    X_baseline = get_feature_matrix(baseline_df).dropna()
    X_current  = get_feature_matrix(current_df).dropna()
    report = check_drift(X_baseline, X_current)
    save_drift_report(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WaterRich Pipeline Runner")
    parser.add_argument("--stage", choices=["train","eval","drift","all"],
                        default="all")
    args = parser.parse_args()

    if args.stage in ("train", "all"):
        stage_train()
    if args.stage in ("drift", "all"):
        stage_drift()
    if args.stage in ("eval", "all"):
        stage_eval()

    print("\n[pipeline] Done.")
