import csv
import json
from openai import OpenAI

INPUT_CSV   = "S1/data/undessensitized_water_system_data_1152_rows.csv"
OUTPUT_CSV  = "S1/results/desensitized_output.csv"
OUTPUT_JSON = "S1/results/desensitized_output.json"

client = OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", ## Replace with your actual API key
    base_url="https://api.deepseek.com",
)

# ── Read raw CSV ──────────────────────────────────────────────────────────────
with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    raw_csv = f.read()

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a data desensitization expert for water system monitoring data.
Apply ALL of the following rules to the raw CSV data provided by the user:

1. Remove precise location: delete longitude and latitude fields entirely.
2. Anonymize station names: replace real names with sequential IDs (Station_A, Station_B, ...).
   The same real station name must always map to the same anonymous ID.
3. Assign region: infer a region label from the station name.
   Use only: "East Zone", "Central Zone", "North Zone", "South Zone", "West Zone".
4. Reduce time precision: truncate timestamps to hourly format (YYYY-MM-DD HH:00).
5. Aggregate rainfall: for each (station_id, hour), sum all rainfall_mm values
   and add small random noise (±0.3 mm max, keep non-negative). Round to 1 decimal place.
6. Classify rainfall level by hourly total:
   >= 16 mm → "Heavy Rain" | >= 8 mm → "Moderate Rain" | >= 2 mm → "Light Rain" | else → "Trace"
7. Remove sensitive fields: delete related_facility and node_id.
8. Output fields (in this order): station_id, region, time, hourly_rainfall_mm, rainfall_level

Return ONLY a valid JSON array — no markdown, no explanation, no extra text.
Each element must have exactly the 5 output fields listed above."""

USER_PROMPT = f"""Desensitize the following raw CSV data and return the result as a JSON array:

{raw_csv}"""

# ── Call LLM ──────────────────────────────────────────────────────────────────
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT},
    ],
    stream=False,  # disable streaming output and wait for the full result
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)

content = response.choices[0].message.content.strip()

# Handle wrapped ```json ... ``` blocks
if content.startswith("```"):
    content = "\n".join(content.splitlines()[1:])
    if content.rstrip().endswith("```"):
        content = content.rstrip()[:-3]

# ── Parse and save ────────────────────────────────────────────────────────────
records = json.loads(content)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["station_id", "region", "time", "hourly_rainfall_mm", "rainfall_level"])
    writer.writeheader()
    writer.writerows(records)

print(f"Done: total {len(records)} desensitized records")
print(f"  CSV  → {OUTPUT_CSV}")
print(f"  JSON → {OUTPUT_JSON}")
