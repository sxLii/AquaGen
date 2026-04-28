"""
S2.py - Synthetic Data Generation via LLM (Approach 2)
Workflow:
    1. Read real SWMM output CSV and extract statistical summaries + representative samples
    2. First LLM call: learn data patterns and output structured generation rules (JSON)
    3. Second LLM call: synthesize 288 new rows based on rules (1 day, 5-minute interval)
    4. Save synthesized data
"""

import csv
import json
import statistics
from openai import OpenAI

INPUT_CSV    = "S2/data/swmm_subset_Tank1_CSO8_C14.csv"
OUTPUT_CSV   = "S2/results/synthetic_output.csv"
OUTPUT_JSON  = "S2/results/synthetic_output.json"
SYNTH_ROWS   = 288   # 1 day x 12 steps/hour x 24 hours

client = OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", ## Replace with your actual API key
    base_url="https://api.deepseek.com",
)

# ── Read raw data and extract statistical summary ─────────────────────────────
with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

cols = ["Tank1_Depth", "CSO8_Depth", "C14_Flow"]
vals = {c: [float(r[c]) for r in rows] for c in cols}

def _stats(v):
    s = sorted(v)
    n = len(s)
    return {"min": round(min(v),4), "max": round(max(v),4),
            "mean": round(statistics.mean(v),4), "std": round(statistics.stdev(v),4),
            "p25": round(s[n//4],4), "p50": round(s[n//2],4), "p75": round(s[3*n//4],4)}

def _corr(x, y):
    n = len(x); mx, my = sum(x)/n, sum(y)/n
    num = sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))
    den = (sum((xi-mx)**2 for xi in x)*sum((yi-my)**2 for yi in y))**0.5
    return round(num/den, 4) if den else 0

data_profile = {
    "source": "SWMM simulation output, Jan-2000, 5-min intervals, 8927 rows",
    "columns": {c: _stats(vals[c]) for c in cols},
    "correlations": {
        "Tank1_Depth vs C14_Flow":  _corr(vals["Tank1_Depth"], vals["C14_Flow"]),
        "Tank1_Depth vs CSO8_Depth": _corr(vals["Tank1_Depth"], vals["CSO8_Depth"]),
        "CSO8_Depth vs C14_Flow":   _corr(vals["CSO8_Depth"],  vals["C14_Flow"]),
    },
    "representative_samples": [
        {k: (float(rows[i][k]) if k != "Date_time" else rows[i][k]) for k in rows[0]}
        for i in range(0, len(rows), 900)
    ],
}

# ── First call: LLM learns data patterns and returns generation rules ─────────
LEARN_SYSTEM = """You are a hydrological data analyst.
Given a statistical profile and representative samples of SWMM simulation output,
extract concise data generation rules that capture:
- value ranges and distributions for each variable
- inter-variable relationships and physical constraints
- temporal patterns (diurnal cycles, gradual trends, event-driven spikes)
- physical constraints (non-negativity, upper bounds, mass balance hints)

Return ONLY a valid JSON object with key "rules" containing a list of rule strings.
No markdown, no extra text."""

LEARN_USER = f"""Analyze the following real data profile and extract generation rules:

{json.dumps(data_profile, indent=2)}"""

print("Step 1: LLM learning data patterns...")
learn_response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": LEARN_SYSTEM},
        {"role": "user",   "content": LEARN_USER},
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)

learn_content = learn_response.choices[0].message.content.strip()
if learn_content.startswith("```"):
    learn_content = "\n".join(learn_content.splitlines()[1:])
    if learn_content.rstrip().endswith("```"):
        learn_content = learn_content.rstrip()[:-3]

learned_rules = json.loads(learn_content)
print(f"  Learned {len(learned_rules['rules'])} rules")
for r in learned_rules["rules"]:
    print(f"  • {r}")

# ── Second call: LLM synthesizes data using learned rules ─────────────────────
SYNTH_SYSTEM = """You are a synthetic hydrological data generator.
Given a set of learned data generation rules, generate a realistic synthetic dataset
in CSV format with exactly the columns: Date_time, Tank1_Depth, CSO8_Depth, C14_Flow.

Requirements:
- Date_time format: DD-Mon-YYYY HH:MM:00  (e.g. 01-Feb-2000 00:00:00)
- Time starts at 01-Feb-2000 00:00:00, step = 5 minutes
- Strictly follow all provided rules
- Values must be physically plausible (non-negative, within observed bounds)
- Tank1_Depth and C14_Flow must maintain strong positive correlation
- Include realistic temporal variation (not flat, not random noise)

Return ONLY the raw CSV text (header + data rows). No markdown, no explanation."""

SYNTH_USER = f"""Generate exactly {SYNTH_ROWS} rows of synthetic data using these learned rules:

{json.dumps(learned_rules, indent=2)}"""

print(f"\nStep 2: LLM synthesizing {SYNTH_ROWS} rows...")
synth_response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": SYNTH_SYSTEM},
        {"role": "user",   "content": SYNTH_USER},
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)


synth_content = synth_response.choices[0].message.content.strip()
if synth_content.startswith("```"):
    synth_content = "\n".join(synth_content.splitlines()[1:])
    if synth_content.rstrip().endswith("```"):
        synth_content = synth_content.rstrip()[:-3]

# ── Parse and save ────────────────────────────────────────────────────────────
synth_rows = list(csv.DictReader(synth_content.splitlines()))

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Date_time", "Tank1_Depth", "CSO8_Depth", "C14_Flow"])
    writer.writeheader()
    writer.writerows(synth_rows)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(synth_rows, f, ensure_ascii=False, indent=2)

print(f"Done: total {len(synth_rows)} synthetic records")
print(f"  CSV  → {OUTPUT_CSV}")
print(f"  JSON → {OUTPUT_JSON}")
