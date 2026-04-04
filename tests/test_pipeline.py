"""
tests/test_pipeline.py — Unit tests for WaterRich pipeline modules.
Run: pytest tests/ -v --cov=src --cov-report=term-missing
"""
import pytest
import pandas as pd
import numpy as np

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingest import validate_event
from src.transform import make_labels, chronological_split, get_feature_matrix
from src.drift import compute_psi, check_drift
from src.online_eval import compute_kpis

# fixture
@pytest.fixture
def valid_event():
    return {
        "sensor_id":        "USGS-FL-001",
        "timestamp":        "2026-01-01T10:00:00",
        "pH":               7.2,
        "dissolved_oxygen": 8.1,
        "turbidity":        1.5,
        "temperature":      24.0,
        "conductivity":     450.0,
    }

@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "sensor_id":        [f"USGS-FL-{i:03d}" for i in range(n)],
        "timestamp":        pd.date_range("2024-01-01", periods=n, freq="1h"),
        "pH":               np.random.uniform(6.0, 9.0, n),
        "dissolved_oxygen": np.random.uniform(4.0, 12.0, n),
        "turbidity":        np.random.uniform(0.5, 5.0, n),
        "temperature":      np.random.uniform(20.0, 30.0, n),
        "conductivity":     np.random.uniform(200.0, 800.0, n),
    })

# schema validation tests
class TestSchemaValidation:
    def test_valid_event_passes(self, valid_event):
        is_valid, reason = validate_event(valid_event)
        assert is_valid, f"Expected valid but got: {reason}"

    def test_missing_sensor_id_fails(self, valid_event):
        del valid_event["sensor_id"]
        is_valid, reason = validate_event(valid_event)
        assert not is_valid
        assert "sensor_id" in reason

    def test_missing_pH_fails(self, valid_event):
        del valid_event["pH"]
        is_valid, reason = validate_event(valid_event)
        assert not is_valid

    def test_missing_dissolved_oxygen_fails(self, valid_event):
        del valid_event["dissolved_oxygen"]
        is_valid, reason = validate_event(valid_event)
        assert not is_valid

    def test_invalid_pH_range_fails(self, valid_event):
        valid_event["pH"] = 15.0   # pH > 14 is impossible
        is_valid, reason = validate_event(valid_event)
        assert not is_valid

    def test_negative_conductivity_fails(self, valid_event):
        valid_event["conductivity"] = -10.0
        is_valid, reason = validate_event(valid_event)
        assert not is_valid

    def test_extra_fields_allowed(self, valid_event):
        valid_event["extra_field"] = "should_be_ignored"
        is_valid, _ = validate_event(valid_event)
        assert is_valid

# transform tests
class TestTransform:
    def test_chronological_split_no_leakage(self, sample_df):
        train, test = chronological_split(sample_df, test_frac=0.2)
        assert len(train) + len(test) == len(sample_df)
        # No temporal leakage: all train timestamps < all test timestamps
        assert train["timestamp"].max() <= test["timestamp"].min()

    def test_chronological_split_size(self, sample_df):
        train, test = chronological_split(sample_df, test_frac=0.2)
        assert len(test) == pytest.approx(len(sample_df) * 0.2, abs=2)

    def test_make_labels_normal_readings(self):
        df = pd.DataFrame({
            "pH":               [7.0, 7.5, 6.5],
            "dissolved_oxygen": [8.0, 7.0, 6.0],
        })
        labels = make_labels(df)
        assert labels.sum() == 0  # all normal

    def test_make_labels_anomalous_pH(self):
        df = pd.DataFrame({
            "pH":               [4.0, 11.0, 7.0],  # first two anomalous
            "dissolved_oxygen": [8.0,  8.0, 8.0],
        })
        labels = make_labels(df)
        assert labels[0] == 1
        assert labels[1] == 1
        assert labels[2] == 0

    def test_make_labels_low_DO(self):
        df = pd.DataFrame({
            "pH":               [7.0, 7.0],
            "dissolved_oxygen": [1.5, 8.0],  # first anomalous
        })
        labels = make_labels(df)
        assert labels[0] == 1
        assert labels[1] == 0

    def test_get_feature_matrix_drops_missing_cols(self, sample_df):
        X = get_feature_matrix(sample_df)
        expected_cols = ["pH", "dissolved_oxygen", "turbidity",
                          "temperature", "conductivity"]
        assert list(X.columns) == expected_cols

# drift detection tests
class TestDrift:
    def test_psi_identical_distributions(self):
        np.random.seed(0)
        data = np.random.normal(7.0, 0.5, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.1

    def test_psi_shifted_distribution(self):
        np.random.seed(0)
        baseline = np.random.normal(7.0, 0.5, 1000)
        shifted  = np.random.normal(10.0, 0.5, 1000)  # large shift
        psi = compute_psi(baseline, shifted)
        assert psi > 0.2

    def test_check_drift_no_drift(self, sample_df):
        train, test = chronological_split(sample_df, test_frac=0.2)
        report = check_drift(train, test)
        assert "any_drift" in report

    def test_check_drift_detects_shift(self, sample_df):
        baseline = sample_df.copy()
        drifted  = sample_df.copy()
        drifted["pH"] = drifted["pH"] + 4.0  # large shift
        report = check_drift(baseline, drifted)
        assert report["pH"]["drifted"] is True

# online eval tests
class TestOnlineEval:
    def test_compute_kpis_empty_df(self):
        kpis = compute_kpis(pd.DataFrame())
        assert "error" in kpis

    def test_compute_kpis_basic(self):
        df = pd.DataFrame([
            {"sensor_id": "S1", "timestamp": "2026-01-01T10:00:00",
             "model_version": "isolation_forest_v1", "is_alert": False,
             "latency_ms": 5.0},
            {"sensor_id": "S1", "timestamp": "2026-01-01T10:01:00",
             "model_version": "isolation_forest_v1", "is_alert": True,
             "latency_ms": 6.0},
            {"sensor_id": "S2", "timestamp": "2026-01-01T10:02:00",
             "model_version": "isolation_forest_v1", "is_alert": False,
             "latency_ms": 4.5},
        ])
        kpis = compute_kpis(df)
        assert kpis["total_responses"] == 3
        assert kpis["pct_personalised"] == 100.0
        assert kpis["n_alerts"] == 1

    def test_pct_personalised_fallback(self):
        df = pd.DataFrame([
            {"sensor_id": "S1", "timestamp": "2026-01-01T10:00:00",
             "model_version": "fallback", "is_alert": False, "latency_ms": 1.0},
            {"sensor_id": "S1", "timestamp": "2026-01-01T10:01:00",
             "model_version": "isolation_forest_v1", "is_alert": False,
             "latency_ms": 2.0},
        ])
        kpis = compute_kpis(df)
        assert kpis["pct_personalised"] == 50.0

# train tests
class TestTrain:
    def test_train_isolation_forest(self, sample_df):
        from src.train import train_isolation_forest
        from src.transform import get_feature_matrix
        X = get_feature_matrix(sample_df).dropna()
        model = train_isolation_forest(X)
        assert hasattr(model, "predict")
        assert hasattr(model, "score_samples")

    def test_train_lof(self, sample_df):
        from src.train import train_lof
        from src.transform import get_feature_matrix
        X = get_feature_matrix(sample_df).dropna()
        model = train_lof(X)
        assert hasattr(model, "predict")

    def test_evaluate_model_isolation_forest(self, sample_df):
        from src.train import train_isolation_forest, evaluate_model
        from src.transform import get_feature_matrix, make_labels, chronological_split
        train_df, test_df = chronological_split(sample_df)
        test_df = test_df.copy()
        test_df.iloc[0, test_df.columns.get_loc("pH")] = 4.0
        X_train = get_feature_matrix(train_df).dropna()
        X_test  = get_feature_matrix(test_df).dropna()
        y_test  = make_labels(X_test)
        model   = train_isolation_forest(X_train)
        metrics = evaluate_model(model, X_test, y_test, "isolation_forest_v1")
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "anomaly_rate" in metrics
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1

    def test_benchmark_latency(self, sample_df):
        from src.train import train_isolation_forest, benchmark_latency
        from src.transform import get_feature_matrix
        X = get_feature_matrix(sample_df).dropna()
        model = train_isolation_forest(X)
        latency = benchmark_latency(model, X, n=10)
        assert "p50_ms" in latency
        assert "p99_ms" in latency
        assert latency["p50_ms"] > 0

    def test_serialize_model(self, sample_df, tmp_path):
        from src.train import train_isolation_forest, evaluate_model, serialize_model
        from src.transform import get_feature_matrix, make_labels, chronological_split
        train_df, test_df = chronological_split(sample_df)
        test_df = test_df.copy()
        test_df.iloc[0, test_df.columns.get_loc("pH")] = 4.0
        X_train = get_feature_matrix(train_df).dropna()
        X_test  = get_feature_matrix(test_df).dropna()
        y_test  = make_labels(X_test)
        model   = train_isolation_forest(X_train)
        metrics = evaluate_model(model, X_test, y_test, "isolation_forest_v1")
        latency = {"p50_ms": 1.0, "p99_ms": 2.0}
        meta    = serialize_model(model, "isolation_forest_v1", metrics,
                                  latency, list(X_train.columns),
                                  str(tmp_path))
        assert os.path.exists(f"{tmp_path}/isolation_forest_v1.pkl")
        assert os.path.exists(f"{tmp_path}/isolation_forest_v1_metadata.json")
        import math
        roc = meta["roc_auc"]
        assert roc is None or isinstance(roc, float)

# extra online eval tests to get accuracy above 70%
class TestOnlineEvalExtra:
    def test_latency_computed(self):
        from src.online_eval import compute_kpis
        df = pd.DataFrame([
            {"sensor_id": "S1", "timestamp": "2026-01-01T10:00:00",
             "model_version": "isolation_forest_v1",
             "is_alert": False, "latency_ms": 5.0},
            {"sensor_id": "S1", "timestamp": "2026-01-01T10:01:00",
             "model_version": "isolation_forest_v1",
             "is_alert": False, "latency_ms": 10.0},
        ])
        kpis = compute_kpis(df)
        assert kpis["latency_p50_ms"] is not None
        assert kpis["latency_p99_ms"] is not None

    def test_alert_rate(self):
        from src.online_eval import compute_kpis
        df = pd.DataFrame([
            {"sensor_id": "S1", "timestamp": "2026-01-01T10:00:00",
             "model_version": "isolation_forest_v1",
             "is_alert": True,  "latency_ms": 5.0},
            {"sensor_id": "S1", "timestamp": "2026-01-01T10:01:00",
             "model_version": "isolation_forest_v1",
             "is_alert": False, "latency_ms": 5.0},
            {"sensor_id": "S1", "timestamp": "2026-01-01T10:02:00",
             "model_version": "isolation_forest_v1",
             "is_alert": False, "latency_ms": 5.0},
            {"sensor_id": "S1", "timestamp": "2026-01-01T10:03:00",
             "model_version": "isolation_forest_v1",
             "is_alert": False, "latency_ms": 5.0},
        ])
        kpis = compute_kpis(df)
        assert kpis["alert_rate_pct"] == 25.0


class TestTransformExtra:
    def test_subpopulation_split(self, sample_df):
        from src.transform import subpopulation_split
        groups = subpopulation_split(sample_df)
        assert len(groups) > 0
        for key, grp in groups.items():
            assert len(grp) > 0

    def test_subpopulation_split_no_sensor_col(self, sample_df):
        from src.transform import subpopulation_split
        df = sample_df.drop(columns=["sensor_id"])
        groups = subpopulation_split(df)
        assert "all" in groups

    def test_load_snapshots_missing_raises(self):
        from src.transform import load_snapshots
        import pytest
        with pytest.raises(FileNotFoundError):
            load_snapshots("nonexistent/**/*.parquet")


class TestOnlineEvalExtra2:
    def test_load_responses_from_file_json(self, tmp_path):
        from src.online_eval import load_responses_from_file, compute_kpis
        import json
        records = [
            {"sensor_id": "S1", "timestamp": "2026-01-01T10:00:00",
             "model_version": "isolation_forest_v1",
             "is_alert": False, "latency_ms": 5.0},
            {"sensor_id": "S1", "timestamp": "2026-01-01T10:01:00",
             "model_version": "isolation_forest_v1",
             "is_alert": True, "latency_ms": 6.0},
        ]
        path = tmp_path / "responses.json"
        with open(path, "w") as f:
            json.dump(records, f)
        df = load_responses_from_file(str(path))
        assert len(df) == 2
        kpis = compute_kpis(df)
        assert kpis["total_responses"] == 2

    def test_print_kpi_report(self, capsys):
        from src.online_eval import print_kpi_report
        kpis = {
            "total_responses": 10,
            "pct_personalised": 100.0,
            "n_alerts": 2,
            "alert_rate_pct": 20.0,
            "acr_30min_pct": 50.0,
            "false_alert_rate_pct": 50.0,
            "latency_p50_ms": 5.0,
            "latency_p99_ms": 10.0,
            "throughput_msgs_per_min": 60.0,
            "observation_window": {
                "start": "2026-01-01",
                "end":   "2026-01-02",
            },
        }
        print_kpi_report(kpis)
        captured = capsys.readouterr()
        assert "WaterRich" in captured.out
        assert "100.0%" in captured.out
