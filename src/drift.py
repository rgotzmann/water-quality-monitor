"""
src/drift.py — Distribution drift detection on incoming sensor readings.
Uses Population Stability Index (PSI) per feature.
"""
import numpy as np
import pandas as pd
import json



FEATURE_COLS = ["pH", "dissolved_oxygen", "turbidity", "temperature", "conductivity"]
PSI_THRESHOLD = 0.2   # PSI > 0.2 = significant drift
N_BINS        = 10

def compute_psi(expected: np.ndarray, actual: np.ndarray,
                n_bins: int = N_BINS) -> float:
    """
    Population Stability Index.
    PSI < 0.1  : no drift
    PSI 0.1-0.2: moderate drift
    PSI > 0.2  : significant drift — trigger retraining
    """
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    def safe_hist(data):
        counts, _ = np.histogram(data, bins=bins)
        freq = counts / len(data)
        freq = np.where(freq == 0, 1e-4, freq)
        return freq

    exp_freq = safe_hist(expected)
    act_freq = safe_hist(actual)
    psi = np.sum((act_freq - exp_freq) * np.log(act_freq / exp_freq))
    return round(float(psi), 4)

def check_drift(baseline_df: pd.DataFrame,
                current_df: pd.DataFrame,
                threshold: float = PSI_THRESHOLD) -> dict:
    """Compare current window to baseline. Returns per-feature PSI + alert flag."""
    results = {}
    any_drift = False

    for col in FEATURE_COLS:
        if col not in baseline_df.columns or col not in current_df.columns:
            continue
        psi = compute_psi(
            baseline_df[col].dropna().values,
            current_df[col].dropna().values,
        )
        drifted = psi > threshold
        if drifted:
            any_drift = True
        results[col] = {"psi": psi, "drifted": drifted}

    results["any_drift"]  = any_drift
    results["threshold"]  = threshold
    results["n_baseline"] = len(baseline_df)
    results["n_current"]  = len(current_df)
    return results

def save_drift_report(report: dict, output_path: str = "drift_report.json"):
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[drift] Report saved → {output_path}")
    if report.get("any_drift"):
        print("[drift] DRIFT DETECTED — consider retraining.")
    else:
        print("[drift] No significant drift detected.")
    for feat, info in report.items():
        if isinstance(info, dict) and "psi" in info:
            flag = "⚠️" if info["drifted"] else "✓"
            print(f"  {flag} {feat}: PSI={info['psi']}")
