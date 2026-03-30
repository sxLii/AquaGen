import pathlib
import pandas as pd
from pyswmm import Simulation, Nodes, Links

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = pathlib.Path(__file__).parent
INP_PATH        = SCRIPT_DIR / "Astlingen_SWMM.inp"
OUT_DEPTH_CSV   = SCRIPT_DIR / "swmm_water_level.csv"
OUT_FLOW_CSV    = SCRIPT_DIR / "swmm_flow.csv"
OUT_SUBSET_CSV  = SCRIPT_DIR / "swmm_subset_Tank1_CSO8_C14.csv"
REPORT_STEP     = "5min"   # must match REPORT_STEP in [OPTIONS] (00:05:00)

# ── Run simulation ─────────────────────────────────────────────────────────────
print(f"Running SWMM: {INP_PATH}")

times      = []
depth_data = {}   # node_id -> list of depths  (from Nodes)
flow_data  = {}   # link_id -> list of flows    (from Links)

with Simulation(str(INP_PATH)) as sim:
    nodes = list(Nodes(sim))
    links = list(Links(sim))
    for node in nodes:
        depth_data[node.nodeid] = []
    for link in links:
        flow_data[link.linkid] = []

    for step in sim:
        times.append(sim.current_time)
        for node in nodes:
            depth_data[node.nodeid].append(node.depth)
        for link in links:
            flow_data[link.linkid].append(link.flow)

print(f"  Routing steps recorded: {len(times)}")
print(f"  Period: {times[0]}  →  {times[-1]}")
print(f"  Nodes:  {len(depth_data)}  |  Links: {len(flow_data)}")

# ── Build full-resolution DataFrame (shared helper inline) ────────────────────
_dt = pd.to_datetime(times)

def _resample_df(series_dict):
    df = pd.DataFrame(series_dict, index=_dt)
    df = df.resample(REPORT_STEP).last().dropna(how="all").reset_index()
    df.insert(0, "Date_time", df["index"].dt.strftime("%d-%b-%Y %H:%M:%S"))
    return df.drop(columns=["index"])

# ── All-nodes depth CSV ────────────────────────────────────────────────────────
df_depth = _resample_df({f"node_{nid}_Depth": v for nid, v in depth_data.items()})

# ── All-links flow CSV ─────────────────────────────────────────────────────────
df_flow = _resample_df({f"link_{lid}_Flow": v for lid, v in flow_data.items()})

# ── Subset CSV: T1 water level (node), CSO8 water level (node), C14 flow (link) ─
_subset_candidates = {
    "Tank1_Depth": depth_data.get("T1"),
    "CSO8_Depth":  depth_data.get("CSO8"),
    "C14_Flow":    flow_data.get("C14"),
}
for _col, _vals in _subset_candidates.items():
    if _vals is None:
        print(f"  WARNING: '{_col}' not found in simulation, skipped.")
df_subset = _resample_df({k: v for k, v in _subset_candidates.items() if v is not None})

# ── Save ───────────────────────────────────────────────────────────────────────
df_depth.to_csv(OUT_DEPTH_CSV,   index=False)
df_flow.to_csv(OUT_FLOW_CSV,     index=False)
df_subset.to_csv(OUT_SUBSET_CSV, index=False)

print(f"\nResampled to {REPORT_STEP} intervals → {len(df_depth)} rows")
for label, df, path in [
    ("Water Level (all nodes)", df_depth,  OUT_DEPTH_CSV),
    ("Flow       (all links)",  df_flow,   OUT_FLOW_CSV),
    ("Subset    [Tank1_Depth(node), CSO8_Depth(node), C14_Flow(link)]", df_subset, OUT_SUBSET_CSV),
]:
    print(f"\n{label}")
    print(f"  Saved  → {path}")
    print(f"  Shape  : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
