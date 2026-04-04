"""
src/ingest.py — Kafka consumer + schema validation
Reads from WaterRich.readings.raw, validates with pandera, buffers to parquet.
"""
import json
import os
import time
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check
from confluent_kafka import Consumer, Producer

# Schema (pandera)
READING_SCHEMA = DataFrameSchema({
    "sensor_id":        Column(str,   nullable=False),
    "timestamp":        Column(str,   nullable=False),
    "pH":               Column(float, Check.in_range(0, 14),    nullable=True),
    "dissolved_oxygen": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
    "turbidity":        Column(float, Check.greater_than_or_equal_to(0), nullable=True),
    "temperature":      Column(float, Check.in_range(-5, 50),   nullable=True),
    "conductivity":     Column(float, Check.greater_than_or_equal_to(0), nullable=True),
}, coerce=True)

REQUIRED_FIELDS = {"sensor_id", "timestamp", "pH", "dissolved_oxygen"}

def validate_event(data: dict) -> tuple[bool, str]:
    """Returns (is_valid, reason). Checks required fields then pandera schema."""
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        return False, f"Missing fields: {missing}"
    try:
        df = pd.DataFrame([data])
        READING_SCHEMA.validate(df, lazy=True)
        return True, "ok"
    except pa.errors.SchemaErrors as e:
        return False, str(e.failure_cases.to_dict())

def make_consumer(bootstrap: str, group_id: str, topic: str) -> Consumer:
    c = Consumer({
        "bootstrap.servers":  bootstrap,
        "group.id":           group_id,
        "auto.offset.reset":  "earliest",
        "enable.auto.commit": True,
    })
    c.subscribe([topic])
    return c

def run_ingestor(bootstrap: str, input_topic: str, alert_topic: str, # pragma: no cover
                 snapshot_dir: str, batch_size: int = 500):
    consumer = make_consumer(bootstrap, "WaterRich-ingestor-v1", input_topic)
    producer = Producer({"bootstrap.servers": bootstrap})
    buffer, count, invalid = [], 0, 0
    os.makedirs(snapshot_dir, exist_ok=True)

    print(f"[ingest] Listening on {input_topic} ...")
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"[ingest] ERROR: {msg.error()}")
                continue
            try:
                data = json.loads(msg.value().decode())
            except json.JSONDecodeError:
                invalid += 1
                continue

            valid, reason = validate_event(data)
            if not valid:
                print(f"[ingest] INVALID: {reason}")
                invalid += 1
                continue

            buffer.append(data)
            count += 1

            # Threshold alerts
            ph = data.get("pH")
            do = data.get("dissolved_oxygen")
            if (ph and (float(ph) < 5.5 or float(ph) > 9.5)) or \
               (do and float(do) < 3.0):
                producer.produce(alert_topic, key=data["sensor_id"],
                                 value=json.dumps({**data, "alert_type": "threshold"}))
                producer.poll(0)

            if len(buffer) >= batch_size:
                _write_snapshot(buffer, snapshot_dir)
                buffer = []

            if count % 200 == 0:
                print(f"[ingest] consumed={count} invalid={invalid}")

    except KeyboardInterrupt:
        if buffer:
            _write_snapshot(buffer, snapshot_dir)
    finally:
        consumer.close()
        producer.flush()

def _write_snapshot(records: list, snapshot_dir: str):
    df  = pd.DataFrame(records)
    now = pd.Timestamp.now()
    path = f"{snapshot_dir}/year={now.year}/month={now.month:02d}/day={now.day:02d}/"
    os.makedirs(path, exist_ok=True)
    fname = f"{path}WaterRich_{now.strftime('%H%M%S%f')}.parquet"
    df.to_parquet(fname, index=False)
    print(f"[ingest] snapshot → {fname}  ({len(records)} rows)")

def test_write_snapshot(tmp_path):
    from src.ingest import _write_snapshot
    records = [
        {"sensor_id": "S1", "timestamp": "2026-01-01",
         "pH": 7.0, "dissolved_oxygen": 8.0}
    ]
    _write_snapshot(records, str(tmp_path))
    import glob
    files = glob.glob(str(tmp_path) + "/**/*.parquet", recursive=True)
    assert len(files) == 1
