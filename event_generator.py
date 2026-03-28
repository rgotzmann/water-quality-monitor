import pandas as pd, json, time
from confluent_kafka import Producer

df = pd.read_csv("resultphyschem.csv", low_memory=False)

PARAM_MAP = {
    "pH":                    "pH",
    "dissolved_oxygen":      "Dissolved oxygen (DO)",
    "turbidity":             "Turbidity",
    "temperature":           "Temperature, water",
    "conductivity":          "Specific conductance",
}

df_filtered = df[df["CharacteristicName"].isin(PARAM_MAP.values())][
    ["MonitoringLocationIdentifier", "ActivityStartDate",
     "ActivityLocation/LatitudeMeasure", "ActivityLocation/LongitudeMeasure",
     "CharacteristicName", "ResultMeasureValue"]
].copy()

df_filtered["ResultMeasureValue"] = pd.to_numeric(
    df_filtered["ResultMeasureValue"], errors="coerce"
)

reverse_map = {v: k for k, v in PARAM_MAP.items()}
df_filtered["param"] = df_filtered["CharacteristicName"].map(reverse_map)

df_pivot = df_filtered.pivot_table(
    index=["MonitoringLocationIdentifier", "ActivityStartDate",
           "ActivityLocation/LatitudeMeasure", "ActivityLocation/LongitudeMeasure"],
    columns="param",
    values="ResultMeasureValue",
    aggfunc="mean"
).reset_index()

df_pivot.columns.name = None
df_pivot = df_pivot.dropna(subset=["pH", "dissolved_oxygen"])
df_pivot = df_pivot.reset_index(drop=True)  # ensure index starts at 0

print(f"Ready to stream {len(df_pivot)} events from {df_pivot['MonitoringLocationIdentifier'].nunique()} sites")

producer = Producer({"bootstrap.servers": "localhost:9092"})
REPLAY_SPEED = 100

for idx, row in df_pivot.iterrows():
    event = {
        "sensor_id":        row["MonitoringLocationIdentifier"],
        "timestamp":        row["ActivityStartDate"],
        "latitude":         row.get("ActivityLocation/LatitudeMeasure"),
        "longitude":        row.get("ActivityLocation/LongitudeMeasure"),
        "pH":               row.get("pH"),
        "dissolved_oxygen": row.get("dissolved_oxygen"),
        "turbidity":        row.get("turbidity"),
        "temperature":      row.get("temperature"),
        "conductivity":     row.get("conductivity"),
    }
    producer.produce(
        "WaterRich.readings.raw",
        key=str(row["MonitoringLocationIdentifier"]),
        value=json.dumps(event, default=str)
    )
    producer.poll(0)
    if idx % 200 == 0:
        print(f"Sent {idx} events...")
    time.sleep(0.001)

producer.flush()
print("Streaming complete.")
