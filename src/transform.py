"""
src/transform.py — Feature engineering and chronological train/test split.
No leakage: split is done on ActivityStartDate before any fitting.
"""
import pandas as pd

import glob

FEATURE_COLS = ["pH", "dissolved_oxygen", "turbidity", "temperature", "conductivity"]

PARAM_MAP = {
    "pH":               "pH",
    "dissolved_oxygen": "Dissolved oxygen (DO)",
    "turbidity":        "Turbidity",
    "temperature":      "Temperature, water",
    "conductivity":     "Specific conductance",
}

def load_snapshots(snapshot_glob: str) -> pd.DataFrame:
    files = glob.glob(snapshot_glob, recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet files found: {snapshot_glob}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)

def load_csv(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, low_memory=False)
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
        columns="param", values="ResultMeasureValue", aggfunc="mean",
    ).reset_index()
    df.columns.name = None
    df = df.rename(columns={
        "MonitoringLocationIdentifier":     "sensor_id",
        "ActivityStartDate":                "timestamp",
        "ActivityLocation/LatitudeMeasure": "latitude",
        "ActivityLocation/LongitudeMeasure":"longitude",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)

def make_labels(df: pd.DataFrame) -> pd.Series:
    """Rule-based ground truth for Florida waterways."""
    labels = pd.Series(0, index=df.index)
    if "pH" in df.columns:
        labels[(df["pH"] < 5.5) | (df["pH"] > 9.5)] = 1
    if "dissolved_oxygen" in df.columns:
        labels[df["dissolved_oxygen"] < 3.0] = 1
    return labels

def chronological_split(df: pd.DataFrame,
                         test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split — no leakage.
    Training set = first (1-test_frac) of time-ordered rows
    Test set = last test_frac of time-ordered rows
    """
    split_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    print(f"[transform] Chronological split: train={len(train)} "
          f"({train['timestamp'].min()} → {train['timestamp'].max()})")
    print(f"[transform]                       test={len(test)}  "
          f"({test['timestamp'].min()} → {test['timestamp'].max()})")
    return train, test

def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available].copy()

def subpopulation_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split test set by sensor_id county prefix for subpopulation analysis."""
    if "sensor_id" not in df.columns:
        return {"all": df}
    groups = {}
    for sid, grp in df.groupby("sensor_id"):
        groups[str(sid)] = grp
    return groups

def test_load_csv_pivot(self, tmp_path):
    from src.transform import load_csv
    import pandas as pd

    # minimal USGS-style CSV
    data = {
        "MonitoringLocationIdentifier": ["S1","S1","S1","S2","S2"],
        "ActivityStartDate":            ["2024-01-01"]*5,
        "ActivityLocation/LatitudeMeasure":  [26.0]*5,
        "ActivityLocation/LongitudeMeasure": [-80.0]*5,
        "CharacteristicName": [
            "pH", "Dissolved oxygen (DO)", "Turbidity",
            "pH", "Dissolved oxygen (DO)"
        ],
        "ResultMeasureValue": [7.2, 8.1, 1.5, 6.8, 7.5],
    }
    csv_path = tmp_path / "test.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    df = load_csv(str(csv_path))
    assert "pH" in df.columns
    assert "dissolved_oxygen" in df.columns
    assert len(df) > 0
