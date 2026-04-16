"""
scripts/ab_analysis.py — A/B experiment analysis.
Reads /metrics endpoint, computes two-proportion z-test on alert rates,
and makes a deploy/rollback decision.

Usage:
  python scripts/ab_analysis.py --api https://water-quality-monitor-swmo.onrender.com
  python scripts/ab_analysis.py --file probe_results.json
"""
import argparse
import json
import math
import requests


def two_proportion_z_test(n_a: int, k_a: int,
                           n_b: int, k_b: int) -> tuple:
    """
    Two-proportion z-test.
    H0: alert_rate_A == alert_rate_B
    Returns (z_stat, p_value, ci_lower, ci_upper)
    """
    if n_a == 0 or n_b == 0:
        return None, None, None, None

    p_a = k_a / n_a
    p_b = k_b / n_b
    p_pool = (k_a + k_b) / (n_a + n_b)

    se = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    if se == 0:
        return 0.0, 1.0, 0.0, 0.0

    z = (p_a - p_b) / se

    # Two-tailed p-value approximation
    p_value = 2 * (1 - normal_cdf(abs(z)))

    # 95% CI for difference p_a - p_b
    se_diff = math.sqrt(p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b)
    ci_lower = round((p_a - p_b) - 1.96 * se_diff, 4)
    ci_upper = round((p_a - p_b) + 1.96 * se_diff, 4)

    return round(z, 4), round(p_value, 4), ci_lower, ci_upper


def normal_cdf(x: float) -> float:
    """Approximation of standard normal CDF."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def bootstrap_ci(samples_a: list, samples_b: list,
                 n_boot: int = 10000, alpha: float = 0.05) -> tuple:
    """Bootstrap CI for difference in means (latency)."""
    import random
    diffs = []
    for _ in range(n_boot):
        mean_a = sum(random.choices(samples_a, k=len(samples_a))) / len(samples_a)
        mean_b = sum(random.choices(samples_b, k=len(samples_b))) / len(samples_b)
        diffs.append(mean_a - mean_b)
    diffs.sort()
    lo = diffs[int(n_boot * alpha/2)]
    hi = diffs[int(n_boot * (1 - alpha/2))]
    return round(lo, 2), round(hi, 2)


def fetch_metrics(api_url: str) -> dict:
    resp = requests.get(f"{api_url}/metrics", timeout=10)
    resp.raise_for_status()
    return resp.json()


def analyse(metrics: dict) -> dict:
    ab_counts = metrics.get("ab_counts", {})
    ab_alerts = metrics.get("ab_alerts", {})

    n_a = ab_counts.get("A", 0)
    n_b = ab_counts.get("B", 0)
    k_a = ab_alerts.get("A", 0)
    k_b = ab_alerts.get("B", 0)

    p_a = round(k_a / n_a, 4) if n_a > 0 else None
    p_b = round(k_b / n_b, 4) if n_b > 0 else None

    z, p_val, ci_lo, ci_hi = two_proportion_z_test(n_a, k_a, n_b, k_b)

    # Decision rule
    ALPHA = 0.05
    if p_val is None:
        decision = "INSUFFICIENT_DATA"
    elif p_val < ALPHA and p_b is not None and p_a is not None:
        if p_b < p_a:
            decision = "DEPLOY_B"    # B has significantly lower alert rate
        else:
            decision = "KEEP_A"      # A is better
    else:
        decision = "NO_SIGNIFICANT_DIFFERENCE"

    result = {
        "group_A": {
            "model":      metrics.get("model_a", "isolation_forest_v1"),
            "n":          n_a,
            "alerts":     k_a,
            "alert_rate": p_a,
        },
        "group_B": {
            "model":      metrics.get("model_b", "lof_v1"),
            "n":          n_b,
            "alerts":     k_b,
            "alert_rate": p_b,
        },
        "z_statistic":  z,
        "p_value":      p_val,
        "ci_95":        [ci_lo, ci_hi],
        "alpha":        ALPHA,
        "decision":     decision,
        "latency_p95_ms": metrics.get("latency_p95_ms"),
        "error_rate":     metrics.get("error_rate"),
    }
    return result


def print_report(result: dict):
    print("  WaterRich — A/B Experiment Report")
    a = result["group_A"]
    b = result["group_B"]
    print(f"  Group A ({a['model']})")
    print(f"    n={a['n']}  alerts={a['alerts']}  "
          f"alert_rate={a['alert_rate']}")
    print(f"  Group B ({b['model']})")
    print(f"    n={b['n']}  alerts={b['alerts']}  "
          f"alert_rate={b['alert_rate']}")
    print(f"\n  Two-proportion z-test:")
    print(f"    z = {result['z_statistic']}")
    print(f"    p = {result['p_value']}")
    print(f"    95% CI for (A-B): {result['ci_95']}")
    print(f"\n  Latency p95: {result['latency_p95_ms']} ms")
    print(f"  Error rate:  {result['error_rate']}")
    print(f"\n  ➜  DECISION: {result['decision']}")
    print("=" * 58)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api",  default="https://water-quality-monitor-swmo.onrender.com")
    parser.add_argument("--file", default=None)
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            metrics = json.load(f)
    else:
        print(f"Fetching metrics from {args.api}...")
        metrics = fetch_metrics(args.api)

    result = analyse(metrics)
    print_report(result)

    with open("ab_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nResults saved -> ab_results.json")
