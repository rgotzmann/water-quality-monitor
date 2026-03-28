from confluent_kafka import Consumer, Producer
import json
import pandas as pd
import os
import time

BOOTSTRAP     = "localhost:9092"
INPUT_TOPIC   = "WaterRich.readings.raw"
ALERT_TOPIC   = "WaterRich.alerts.active"
GROUP_ID      = "WaterRich-ingestor-v1"
BATCH_SIZE    = 500
SNAPSHOT_DIR  = "snapshots/readings"

REQUIRED_FIELDS = {"sensor_id", "timestamp", "pH", "dissolved_oxygen"}

consumer = Consumer({
    "bootstrap.servers":  BOOTSTRAP,
    "group.id":           GROUP_ID,
    "auto.offset.reset":  "earliest",
    "enable.auto.commit": True,
})

producer = Producer({"bootstrap.servers": BOOTSTRAP})

consumer.subscribe([INPUT_TOPIC])

buffer       = []
count        = 0
invalid      = 0
start_time   = time.time()

print(f"[WaterRich Ingestor] Listening on {INPUT_TOPIC} ...")
print(f"[WaterRich Ingestor] Writing snapshots to {SNAPSHOT_DIR}/")
print("Press Ctrl+C to stop.\n")

def write_snapshot(records):
    """Write a batch of records to a partitioned parquet file."""
    df   = pd.DataFrame(records)
    now  = pd.Timestamp.now()
    path = (f"{SNAPSHOT_DIR}/"
            f"year={now.year}/month={now.month:02d}/day={now.day:02d}/")
    os.makedirs(path, exist_ok=True)
    fname = f"{path}WaterRich_{now.strftime('%H%M%S')}.parquet"
    df.to_parquet(fname, index=False)
    print(f"[SNAPSHOT] Wrote {len(records)} records → {fname}")

def delivery_report(err, msg):
    if err:
        print(f"[ALERT] Delivery failed: {err}")

try:
    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue
        if msg.error():
            print(f"[ERROR] Consumer error: {msg.error()}")
            continue

        try:
            data = json.loads(msg.value().decode("utf-8"))
        except json.JSONDecodeError as e:
            print(f"[INVALID] JSON decode error: {e}")
            invalid += 1
            continue

        missing = REQUIRED_FIELDS - set(data.keys())
        if missing:
            print(f"[INVALID] Missing fields {missing} — skipping")
            invalid += 1
            continue

        buffer.append(data)
        count += 1

        ph = data.get("pH")
        do = data.get("dissolved_oxygen")
        alert = None

        if ph is not None and ph != "nan":
            try:
                ph_val = float(ph)
                if ph_val < 6.5 or ph_val > 8.5:
                    alert = {**data, "alert_type": "pH_out_of_range",
                             "alert_value": ph_val}
            except (ValueError, TypeError):
                pass

        if do is not None and do != "nan":
            try:
                do_val = float(do)
                if do_val < 6.0:
                    alert = {**data, "alert_type": "low_dissolved_oxygen",
                             "alert_value": do_val}
            except (ValueError, TypeError):
                pass

        if alert:
            producer.produce(
                ALERT_TOPIC,
                key=str(data.get("sensor_id", "unknown")),
                value=json.dumps(alert, default=str),
                callback=delivery_report,
            )
            producer.poll(0)

        if len(buffer) >= BATCH_SIZE:
            write_snapshot(buffer)
            buffer = []

        if count % 200 == 0:
            elapsed = time.time() - start_time
            rate    = count / elapsed if elapsed > 0 else 0
            print(f"[STATS] consumed={count} invalid={invalid} "
                  f"rate={rate:.1f} msg/s")

except KeyboardInterrupt:
    print("\n[WaterRich Ingestor] Stopping...")

finally:
    # Write any remaining buffered events
    if buffer:
        write_snapshot(buffer)
        print(f"[SNAPSHOT] Flushed final {len(buffer)} records.")

    consumer.close()
    producer.flush()
    print(f"[DONE] Total consumed: {count} | Invalid: {invalid}")
