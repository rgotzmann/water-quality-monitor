"""
scripts/availability.py — Compute API availability over a time window.
Reads probe_results.json logs and calculates uptime percentage.

Usage:
  python scripts/availability.py --hours 72
  python scripts/availability.py --hours 216   # 72h before + 144h after = 216h total
"""
import argparse
import glob
import json
import os
from datetime import datetime, timedelta


def load_probe_logs(log_dir: str = ".") -> list:
    """Load all probe_results*.json files."""
    records = []
    for path in glob.glob(f"{log_dir}/probe_results*.json"):
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                records.extend(data)
            else:
                records.append(data)
    return records


def compute_availability(records: list, window_hours: int = 216) -> dict:
    """
    Availability = successful probes / total probes in window.
    A probe is successful if pct_success == 100.0 or success count > 0.
    """
    if not records:
        return {"error": "No probe records found"}

    now = datetime.utcnow()
    cutoff = now - timedelta(hours=window_hours)

    window_records = []
    for r in records:
        ts_str = r.get("timestamp")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", ""))
                if ts >= cutoff:
                    window_records.append(r)
            except ValueError:
                window_records.append(r)
        else:
            window_records.append(r)

    if not window_records:
        window_records = records  # use all if no timestamps

    total_runs   = sum(r.get("total", 0) for r in window_records)
    total_success = sum(r.get("success", 0) for r in window_records)
    total_errors  = sum(r.get("errors", 0) for r in window_records)

    availability = round(100 * total_success / total_runs, 2) \
        if total_runs > 0 else 0.0

    return {
        "window_hours":    window_hours,
        "probe_runs":      len(window_records),
        "total_requests":  total_runs,
        "successful":      total_success,
        "errors":          total_errors,
        "availability_pct": availability,
        "slo_target_pct":  70.0,
        "slo_met":         availability >= 70.0,
    }


def print_availability_report(result: dict):
    met = "met" if result.get("slo_met") else "not met"
    print("\n" + "=" * 52)
    print("  WaterRich — Availability Report")
    print("=" * 52)
    print(f"  Window:          {result['window_hours']}h")
    print(f"  Probe runs:      {result['probe_runs']}")
    print(f"  Total requests:  {result['total_requests']}")
    print(f"  Successful:      {result['successful']}")
    print(f"  Errors:          {result['errors']}")
    print(f"  Availability:    {result['availability_pct']}%")
    print(f"  SLO (>=70%):     {met}")
    print("=" * 52)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=216)
    parser.add_argument("--logdir", default=".")
    args = parser.parse_args()

    records = load_probe_logs(args.logdir)
    result  = compute_availability(records, args.hours)
    print_availability_report(result)

    with open("availability_report.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved -> availability_report.json")
