# WaterRich — Monitoring Runbook

## SLOs

| SLO | Target | Metric |
|-----|--------|--------|
| Availability | ≥ 70% | `successful_probes / total_probes` over 216h window |
| Latency p95 | < 500 ms | `/metrics` → `latency_p95_ms` |
| Error rate | < 5% | `/metrics` → `error_rate` |
| Alert precision | > 50% ACR@30 | Confirmed alerts / total alerts |

---

## Alert Rules

### Alert 1 — High Error Rate
- **Condition:** `error_rate > 0.05` for 5 consecutive probes
- **Severity:** Critical
- **Action:**
  1. Check `/health` — is the API responding?
  2. Check Render logs for exceptions
  3. Rollback: `POST /switch?model=isolation_forest_v1`
  4. If Render is down: redeploy via `curl -X POST $RENDER_DEPLOY_URL`

### Alert 2 — High Latency
- **Condition:** `latency_p95_ms > 500`
- **Severity:** Warning
- **Action:**
  1. Check `/metrics` for request volume spike
  2. Check Render service CPU/memory
  3. If model B (LOF) is active, switch to model A (faster): `POST /switch?model=isolation_forest_v1`

### Alert 3 — 100% Alert Rate
- **Condition:** `alert_rate > 0.95` for 3 consecutive probes
- **Severity:** Warning (likely threshold miscalibration)
- **Action:**
  1. Check `ALERT_THRESHOLD` env var on Render
  2. Expected range: `-0.30` to `-0.40`
  3. Update via Render dashboard → Environment → `ALERT_THRESHOLD=-0.35`

### Alert 4 — Model Drift
- **Condition:** PSI > 0.2 on any feature (from `drift_report.json`)
- **Severity:** Warning
- **Action:**
  1. Run `python scripts/retrain.py` to produce new version
  2. Check new model metrics in `model_registry/vX.Y/manifest.json`
  3. Hot-swap: `POST /switch?model=isolation_forest_vX.Y`

---

## Dashboard Links

- **Live API:** https://water-quality-monitor-swmo.onrender.com
- **Metrics endpoint:** https://water-quality-monitor-swmo.onrender.com/metrics
- **Swagger docs:** https://water-quality-monitor-swmo.onrender.com/docs
- **GitHub Actions:** https://github.com/rgotzmann/water-quality-monitor/actions
- **Render dashboard:** https://dashboard.render.com

---

## Model Update Procedure

1. Run retrain: `python scripts/retrain.py`
2. Check metrics in `model_registry/vX.Y/manifest.json`
3. If PR-AUC improves: `POST /switch?model=isolation_forest_vX.Y`
4. Monitor `/metrics` for 30 min
5. If error rate spikes: rollback with `POST /switch?model=isolation_forest_v1`

---

## Availability Calculation

```bash
python scripts/availability.py --hours 216
```

Target: ≥ 70% over the 216h window (72h before + 144h after submission).
