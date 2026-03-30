import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Reproducible synthetic "raw / unredacted" dataset
rng = np.random.default_rng(20260328)

stations = [
    ("Shatin Pumping Station North Gauge", 114.2213, 22.3918, "Shatin SPS", "JN-104"),
    ("Central Interceptor Rain Gauge 2", 114.1738, 22.2819, "Central Interceptor", "MH-022"),
    ("Eastern Trunk Sewer Gauge 5", 114.2451, 22.2749, "Eastern Trunk Sewer", "TS-315"),
    ("Kowloon South Diversion Chamber Gauge", 114.1772, 22.3041, "Kowloon South Diversion Chamber", "KD-087"),
    ("Tsuen Wan Coastal Interceptor Gauge", 114.1135, 22.3701, "Tsuen Wan Coastal Interceptor", "TW-156"),
    ("Tuen Mun North Pump Sump Gauge", 113.9761, 22.4015, "Tuen Mun North Pump Sump", "TM-203"),
    ("Sha Tin Main Trunk Gauge 3", 114.2088, 22.3874, "Sha Tin Main Trunk", "ST-143"),
    ("Kwun Tong Outfall Monitoring Gauge", 114.2314, 22.3097, "Kwun Tong Outfall", "KT-266"),
    ("Yuen Long Box Culvert Gauge", 114.0415, 22.4452, "Yuen Long Box Culvert", "YL-052"),
    ("Tai Po Riverside Interceptor Gauge", 114.1704, 22.4528, "Tai Po Riverside Interceptor", "TP-094"),
    ("Aberdeen Preliminary Treatment Gauge", 114.1549, 22.2477, "Aberdeen PTW", "AB-031"),
    ("Chai Wan Trunk Sewer Gauge 1", 114.2438, 22.2656, "Chai Wan Trunk Sewer", "CW-118"),
]

start = datetime(2026, 3, 24, 0, 0, 0)
periods = 96  # 24 hours at 5-min intervals

def rain_profile(t_index, station_index):
    # Multi-peak rainfall pattern over a day, plus station-specific scaling/noise
    x = t_index
    peak1 = 4.5 * np.exp(-((x - 30) / 10) ** 2)
    peak2 = 7.0 * np.exp(-((x - 58) / 8) ** 2)
    peak3 = 2.8 * np.exp(-((x - 78) / 12) ** 2)
    base = 0.15 + 0.06 * np.sin((x + station_index * 2) / 8.0)
    station_factor = 0.85 + 0.35 * np.sin((station_index + 1) / 3.0)
    noise = rng.normal(0, 0.22)
    val = max(0.0, (peak1 + peak2 + peak3 + base) * station_factor + noise)
    return round(val, 1)

rows = []
for s_idx, (station_name, lon, lat, facility, node_id) in enumerate(stations):
    for t_idx in range(periods):
        timestamp = start + timedelta(minutes=5 * t_idx)
        rows.append({
            "station_name": station_name,
            "longitude": lon,
            "latitude": lat,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "rainfall_mm": rain_profile(t_idx, s_idx),
            "related_facility": facility,
            "node_id": node_id,
        })

df = pd.DataFrame(rows)

# Save file
out_path = Path("undessensitized_water_system_data_1152_rows.csv")
df.to_csv(out_path, index=False, encoding="utf-8")

# Show a small preview
preview = df.head(12).copy()
preview
