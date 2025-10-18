#!/usr/bin/env python3
"""
Artillery Multi-Report Dashboard Generator
Merges multiple Artillery JSON reports into a single, interactive HTML dashboard.
"""

import json
import re
import argparse
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import yaml
from pathlib import Path
import numpy as np
import sys
import os

# ---------------- CONFIG ----------------
TIME_AXIS_MODE = "elapsed"  # "elapsed" or "datetime"
PERCENTILES = ["p90", "p95", "p99"]
SHOW_GRAPH_RPS = True  # Default visibility
SHOW_GRAPH_P90 = True  # Default visibility
SHOW_GRAPH_COMBINED = True  # Default visibility
SHOW_GRAPH_CODES = True
SHOW_GRAPH_DONUT = True
FILTER_START_SEC = 0
FILTER_END_SEC = None

# When consolidating multiple files, you may enable smoothing to reduce jitter.
SMOOTH_PERCENTILES_ON_MERGE = True  # set False to disable smoothing on merged pxx traces
SMOOTH_WINDOW = 3  # simple moving window size (odd integer)

# Default SLA/TPL values (used only if SLA JSON missing or endpoint not in SLA)
DEFAULT_SLA = 500  # ms
DEFAULT_TPH = 0

# ---------------- CLI ARGUMENTS ----------------
parser = argparse.ArgumentParser(description="Merge up to N Artillery JSON reports into one HTML dashboard.")
parser.add_argument("--json", required=True, help="Comma-separated list of Artillery JSON report files (or single file).")
parser.add_argument("--yaml", required=False, help="Optional Artillery YAML config file (for phases).")
parser.add_argument("--sla", type=str, default=None, help="Optional SLA JSON path. JSON should map transaction-> [sla_ms, expected_tph].")
parser.add_argument("--output", required=True, help="Output HTML file path.")
args = parser.parse_args()

# ---------------- INPUT FILES ----------------
# Build list of input JSON report paths (allow comma separated list)
if args.json:
    INPUT_JSONS = [p.strip() for p in args.json.split(",") if p.strip()]
else:
    INPUT_JSONS = []

INPUT_YAML = args.yaml
INPUT_SLA = args.sla
OUTPUT_HTML = args.output

# Validate JSON existence: print full resolved path for missing files
existing_files = []
missing_files = []
for f in INPUT_JSONS:
    p = Path(f).resolve()
    if p.exists():
        existing_files.append(str(p))
    else:
        missing_files.append(str(p))

# If none exist -> fail; if some missing -> warn and continue
if not existing_files:
    print("❌ Error: None of the provided JSON files exist. Exiting.")
    sys.exit(1)

if missing_files:
    print("⚠️ Warning: Some JSON files were not found and will be skipped:")
    for f in missing_files:
        print(f"    - File not found: {f}")

# Update INPUT_JSONS to absolute existing paths
INPUT_JSONS = existing_files
print(f"✅ Proceeding with {len(INPUT_JSONS)} existing JSON file(s).")

# ---------------- LOAD SLA JSON (Optional) ----------------
sla_data = {}
SLA_ENABLED = False
if INPUT_SLA:
    sla_path = Path(INPUT_SLA).resolve()
    if not sla_path.exists():
        print(f"⚠️ Warning: SLA file '{sla_path}' not found. SLA will be disabled.")
    else:
        try:
            with open(sla_path, "r", encoding="utf-8") as f:
                sla_data = json.load(f) or {}
            SLA_ENABLED = bool(sla_data)
            print(f"✅ Loaded SLA definitions from {sla_path} ({len(sla_data)} entries).")
        except Exception as e:
            print(f"⚠️ Warning: Could not parse SLA JSON '{sla_path}': {e}")
            sla_data = {}
            SLA_ENABLED = False
else:
    print("ℹ️ No SLA file provided. SLA columns will be omitted.")

# ---------------- LOAD YAML (phases) if present ----------------
config_yaml = {}
if INPUT_YAML:
    yaml_p = Path(INPUT_YAML)
    if yaml_p.exists():
        try:
            with open(yaml_p, "r", encoding="utf-8") as f:
                config_yaml = yaml.safe_load(f) or {}
            print(f"✅ Loaded YAML config from {yaml_p}")
        except Exception as e:
            print(f"⚠️ Could not read YAML config {yaml_p}: {e}")
            config_yaml = {}
    else:
        print(f"⚠️ YAML config not found at {INPUT_YAML}. Phases disabled.")

# ---------------- LOAD JSON INTERMEDIATE SNAPSHOTS ----------------
all_intermediate = []
all_agg_counters = defaultdict(int)
all_agg_summaries = defaultdict(lambda: {"count": 0, "mean": 0, "min": float("inf"), "max": 0, "p90": 0, "p95": 0, "p99": 0})

for json_file in INPUT_JSONS:
    p = Path(json_file)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load {p}: {e}")
        continue

    # collect intermediate
    intermediate_list = data.get("intermediate", [])
    if intermediate_list:
        # annotate each snapshot with a source file (helpful for debugging)
        for snap in intermediate_list:
            snap["_source_file"] = str(p)
        all_intermediate.extend(intermediate_list)

    # aggregate counters & summaries (safely combine)
    for k, v in data.get("aggregate", {}).get("counters", {}).items():
        all_agg_counters[k] += v
    for k, s in data.get("aggregate", {}).get("summaries", {}).items():
        existing = all_agg_summaries[k]
        n1, n2 = existing.get("count", 0), s.get("count", 0)
        if n1 + n2 > 0:
            existing["mean"] = (existing["mean"] * n1 + s.get("mean", 0) * n2) / (n1 + n2)
            existing["min"] = min(existing["min"], s.get("min", existing["min"]))
            existing["max"] = max(existing["max"], s.get("max", existing["max"]))
            for p_key in PERCENTILES:
                # Use max of percentile values for consolidation (conservative)
                existing[p_key] = max(existing.get(p_key, 0), s.get(p_key, 0))
            existing["count"] = n1 + n2

agg_counters = all_agg_counters
agg_summaries = all_agg_summaries
intermediate = all_intermediate

# If after loading everything we still have no snapshots -> abort (can't plot)
if not intermediate:
    print("❌ No snapshots found in provided JSON files. Exiting.")
    sys.exit(1)

# ---------------- TIME AXIS (Single vs Multi-file Consolidation) ----------------
if len(INPUT_JSONS) > 1:
    # Consolidated mode: normalize timestamps across multiple runs
    print("🧩 Consolidated mode: aligning snapshots from multiple JSONs...")
    time_points = []
    for snap in intermediate:
        t = snap.get("firstMetricAt") or snap.get("lastMetricAt")
        if t:
            # convert milliseconds -> datetime, rounding to nearest second for robust comparison
            try:
                dt = datetime.fromtimestamp(int(t) / 1000.0).replace(microsecond=0)
                time_points.append(dt)
            except Exception:
                pass

    time_points = sorted(list({tp for tp in time_points}))

    if not time_points:
        print("❌ Consolidated mode: no valid timestamps discovered. Exiting.")
        sys.exit(1)

    intermediate = sorted(intermediate, key=lambda s: s.get("firstMetricAt", s.get("lastMetricAt", 0)))
    
    # durations: ensure positive and reasonable
    durations = []
    for i in range(len(time_points)):
        if i < len(time_points) - 1:
            dt = (time_points[i + 1] - time_points[i]).total_seconds()
            durations.append(max(1.0, dt))
        else:
            durations.append(10.0)  # terminal interval fallback

    start_time = time_points[0]
    consolidated_mode = True
    print(f"✅ Consolidated alignment: {len(time_points)} unique time points from snapshots.")
else:
    # Single-run original behavior
    valid_intermediate = [snap for snap in intermediate if snap.get("firstMetricAt")]
    if not valid_intermediate:
        print("❌ Single run mode: No valid timestamps found. Exiting.")
        sys.exit(1)
    intermediate = valid_intermediate
    
    time_points = [datetime.fromtimestamp(snap["firstMetricAt"] / 1000.0) for snap in intermediate]
    start_time = time_points[0]
    # durations as in your original code (avoid small or zero intervals)
    durations = [
        max(1.0, (time_points[i + 1] - time_points[i]).total_seconds())
        if i < len(time_points) - 1 else
        10.0
        for i in range(len(time_points))
    ]
    filtered_indices_all = list(range(len(time_points))) # Needed for single-run mode mapping
    consolidated_mode = False

# ---------------- FILTER TIME RANGE ----------------
if FILTER_END_SEC is None:
    FILTER_END_SEC = float("inf")

elapsed_seconds = [(t - start_time).total_seconds() for t in time_points]
filtered_indices = [i for i, t in enumerate(elapsed_seconds) if FILTER_START_SEC <= t <= FILTER_END_SEC]
if not filtered_indices:
    raise SystemExit("❌ No data points fall within the selected filter range.")

# reduce time_points and durations to the filtered window
time_points = [time_points[i] for i in filtered_indices]
durations = [durations[i] for i in filtered_indices]
filtered_elapsed_seconds = [(t - time_points[0]).total_seconds() for t in time_points]

# x_labels: use ISO datetimes when TIME_AXIS_MODE == "datetime", else elapsed seconds (float)
x_labels = [
    t.isoformat(timespec="seconds") if TIME_AXIS_MODE == "datetime" else round((t - time_points[0]).total_seconds(), 2)
    for t in time_points
]

# ---------------- DISCOVER ENDPOINTS ----------------
def discover_endpoints(counters, summaries):
    eps = set()
    for k in counters.keys():
        if k.startswith("plugins.metrics-by-endpoint."):
            part = k.replace("plugins.metrics-by-endpoint.", "")
            if ".codes." in part:
                eps.add(part.split(".codes.")[0])
            elif ".errors." in part:
                eps.add(part.split(".errors.")[0])
    for k in summaries.keys():
        if k.startswith("plugins.metrics-by-endpoint.response_time."):
            eps.add(k.replace("plugins.metrics-by-endpoint.response_time.", ""))
    return sorted(eps)

endpoints = discover_endpoints(agg_counters, agg_summaries)

# ---------------- DATA CONTAINERS ----------------
n_points = len(time_points)
ep_avg = {ep: [0] * n_points for ep in endpoints}
ep_pxx = {ep: {p: [0] * n_points for p in PERCENTILES} for ep in endpoints}
ep_codes = {ep: defaultdict(lambda: [0] * n_points) for ep in endpoints}
ep_counts = {ep: [0] * n_points for ep in endpoints}
ep_pass_counts = {ep: [0] * n_points for ep in endpoints}
ep_total_codes = defaultdict(lambda: defaultdict(int)) # For detailed error table


# ---------------- FILL SNAPSHOTS ----------------
for idx_orig, snap in enumerate(intermediate):
    t_ms = snap.get("firstMetricAt") or snap.get("lastMetricAt")
    if not t_ms:
        continue
    try:
        snap_dt = datetime.fromtimestamp(int(t_ms) / 1000.0).replace(microsecond=0)
    except Exception:
        continue

    matched_idx = -1
    if consolidated_mode:
        try:
            matched_idx = time_points.index(snap_dt)
        except ValueError:
            deltas = [abs((snap_dt - tp).total_seconds()) for tp in time_points]
            if deltas and min(deltas) < 1.0:
                matched_idx = int(np.argmin(deltas))
    else:
        try:
            if idx_orig in filtered_indices_all:
                matched_idx = filtered_indices.index(idx_orig)
        except ValueError:
            continue

    if matched_idx == -1:
        continue

    sums = snap.get("summaries", {})
    counters = snap.get("counters", {})
    for key, summary in sums.items():
        if key.startswith("plugins.metrics-by-endpoint.response_time."):
            ep = key.replace("plugins.metrics-by-endpoint.response_time.", "")
            if ep in endpoints:
                ep_avg[ep][matched_idx] = summary.get("mean", 0)
                for p in PERCENTILES:
                    ep_pxx[ep][p][matched_idx] = summary.get(p, 0)
    for k, v in counters.items():
        if k.startswith("plugins.metrics-by-endpoint."):
            part = k.replace("plugins.metrics-by-endpoint.", "")
            if ".codes." in part:
                ep_part, code = part.split(".codes.")
                if ep_part in endpoints:
                    ep_counts[ep_part][matched_idx] += v
                    ep_codes[ep_part][code][matched_idx] = v
                    ep_total_codes[ep_part][code] += v # for error table
                    if code.isdigit() and code.startswith("2"):
                        ep_pass_counts[ep_part][matched_idx] += v
            elif ".errors." in part:
                ep_part, code = part.split(".errors.")
                code_key = f"ERR:{code}"
                if ep_part in endpoints:
                    ep_counts[ep_part][matched_idx] += v
                    ep_codes[ep_part][code_key][matched_idx] = v
                    ep_total_codes[ep_part][code_key] += v # for error table

# ---------------- OPTIONAL: smoothing percentiles across merged snapshots ----------------
def smooth_series(arr, window):
    if window <= 1:
        return arr
    a = np.array(arr, dtype=float)
    pad = window // 2
    a_p = np.pad(a, pad_width=pad, mode="edge")
    kernel = np.ones(window) / window
    sm = np.convolve(a_p, kernel, mode="valid")
    return sm.tolist()

if consolidated_mode and SMOOTH_PERCENTILES_ON_MERGE:
    print(f"🔧 Applying smoothing to percentiles (window={SMOOTH_WINDOW}) for consolidated run.")
    for ep in endpoints:
        for p in PERCENTILES:
            ep_pxx[ep][p] = smooth_series(ep_pxx[ep][p], SMOOTH_WINDOW)
        ep_avg[ep] = smooth_series(ep_avg[ep], SMOOTH_WINDOW)

# ---------------- CALCULATE RPS & OVERALL METRICS ----------------
ep_rps = {ep: [0] * n_points for ep in endpoints}
ep_pass_rps = {ep: [0] * n_points for ep in endpoints}
for ep in endpoints:
    for i in range(n_points):
        dur = durations[i] if i < len(durations) else 1.0
        ep_rps[ep][i] = (ep_counts[ep][i] / dur) if dur > 0 else 0
        ep_pass_rps[ep][i] = (ep_pass_counts[ep][i] / dur) if dur > 0 else 0

overall_rps = [sum(ep_counts[ep][i] for ep in endpoints) / (durations[i] if i < len(durations) else 1.0) for i in range(n_points)]
overall_pass_rps = [sum(ep_pass_counts[ep][i] for ep in endpoints) / (durations[i] if i < len(durations) else 1.0) for i in range(n_points)]

overall_p90 = []
overall_avg = []
for i in range(n_points):
    p90_vals = [ep_pxx[ep]["p90"][i] for ep in endpoints if ep_counts[ep][i] > 0]
    avg_vals = [ep_avg[ep][i] for ep in endpoints if ep_counts[ep][i] > 0]
    overall_p90.append(max(p90_vals) if p90_vals else 0)
    overall_avg.append(max(avg_vals) if avg_vals else 0)


# ---------------- SUMMARY TABLE WITH OPTIONAL SLA ----------------

# NEW: Helper function to extract the core service name (e.g., 'orders', 'users')
def extract_core_service_name(ep_name):
    # Strip protocol if present (though Artillery URLs usually don't have it here)
    name = ep_name.replace("http://", "").replace("https://", "")
    # Find the hostname part
    if '/' in name:
        hostname = name.split('/')[0]
    else:
        hostname = name
    
    # Extract the service name, assuming it's the first segment before a hash/dash
    # e.g., 'orders-994656192388.us-central1.run.app' -> 'orders'
    # e.g., 'orders-nafj7uberq-uc.a.run.app' -> 'orders'
    # Try to find the service name before the first '-' or '.' after the start
    match = re.match(r"([a-zA-Z]+)(?:[-.0-9a-zA-Z]*)\.", hostname)
    if match:
        return match.group(1)
    
    # Fallback for simple names
    return hostname.split('.')[0].split('-')[0]

# NEW: Refined SLA lookup function
def get_sla_for_endpoint(ep_name, sla_dict, default_sla=DEFAULT_SLA, default_tph=DEFAULT_TPH):
    if not sla_dict:
        return default_sla, default_tph
    
    # 1. Try exact match (Artillery's endpoint key usually contains the full URL/path)
    # Canonicalize the Artillery endpoint key (strip path variable like {{ userId }} and trailing slashes)
    canon_ep_name = re.sub(r"/\{\{.*?\}\}", "", ep_name).rstrip("/")
    if canon_ep_name in sla_dict:
        return tuple(sla_dict[canon_ep_name])
    
    # 2. Try match against SLA keys stripped of path variables
    for key in sla_dict:
        canon_sla_key = key.rstrip("/")
        if canon_ep_name == canon_sla_key:
             return tuple(sla_dict[key])

    # 3. Try to match by core service name (more flexible matching for dynamic hosts)
    ep_service_name = extract_core_service_name(ep_name)
    
    for key, val in sla_dict.items():
        sla_service_name = extract_core_service_name(key)
        
        # Check if the extracted service names match AND the Artillery endpoint contains the SLA key's base path
        if ep_service_name and ep_service_name == sla_service_name:
            # We assume a match on the core service name is sufficient for SLA lookup
            return tuple(val)
            
    # Fallback to default
    return default_sla, default_tph


summary_rows = []
for ep in endpoints:
    # Use the full, final aggregate counts from the overall run
    total = sum(ep_counts[ep])
    pass_count = sum(ep_pass_counts[ep]) # This still relies on intermediate 2xx counts being summed
    # However, the calculation below for the summary table is more robust for final total
    
    # Fallback to the accurate aggregated counts from all counters if needed
    # We will trust the intermediate aggregation logic for now, but ensure the final
    # percentage calculation relies on the summary_rows data.
    
    # The true total counts come from the keys like:
    # plugins.metrics-by-endpoint.orders-994656192388.us-central1.run.app/orders.codes.200 (for pass)
    # Which are merged into agg_counters. We use the calculated `total` and `pass_count` here for simplicity
    # and trust they were correctly generated by the aggregation logic above.
    
    # Let's ensure the total count is calculated robustly from the aggregated summaries
    agg_summary = agg_summaries.get(f"plugins.metrics-by-endpoint.response_time.{ep}", {})
    total_tx_count_agg = agg_summary.get("count", total) # Prefer the count from response_time summary
    
    # Total Pass Count is often missing from the summary, so we use the counter aggregation result
    # For now, stick to the `sum(ep_counts[ep])` and `sum(ep_pass_counts[ep])` derived from intermediates.
    
    base_row = {
        "Transaction": ep,
        "Avg (ms)": round(agg_summary.get("mean", 0), 2),
        "Min (ms)": agg_summary.get("min", 0) if agg_summary.get("min", 0) != float("inf") else 0,
        "Max (ms)": agg_summary.get("max", 0),
        "P90": round(agg_summary.get("p90", 0), 2),
        "P95": round(agg_summary.get("p95", 0), 2),
        "P99(ms)": round(agg_summary.get("p99", 0), 2),
        "Count": total_tx_count_agg, # Use the response_time summary count for final total
        "Pass_Count": pass_count, # Use the sum of 2xx from intermediates
        "Fail_Count": total_tx_count_agg - pass_count
    }

    if SLA_ENABLED:
        sla_val, tph_val = get_sla_for_endpoint(ep, sla_data)
        base_row["SLA(ms)"] = sla_val
        base_row["Expected_TPH"] = tph_val

    summary_rows.append(base_row)

columns = ["Transaction"]
if SLA_ENABLED:
    columns += ["SLA(ms)"]
columns += ["Avg (ms)", "Min (ms)", "Max (ms)", "P90", "P95", "P99(ms)"]
if SLA_ENABLED:
    columns += ["Expected_TPH"]
columns += ["Count", "Pass_Count", "Fail_Count"]

summary_df = pd.DataFrame(summary_rows, columns=columns).sort_values(by="Transaction")
print("\n--- Summary DataFrame preview ---")
print(summary_df.head(20).to_string(index=False))

# ---------------- ERROR DETAIL TABLE ----------------
error_rows = []
for ep in endpoints:
    for code, count in ep_total_codes[ep].items():
        # Only include non-2xx codes and error codes
        if not (code.isdigit() and code.startswith("2")) and count > 0:
            error_rows.append({"URL": ep, "Status Code": code, "Count": count})

error_df = pd.DataFrame(error_rows, columns=["URL", "Status Code", "Count"]).sort_values(by=["URL", "Status Code"], ascending=[True, False])
print("\n--- Error Detail DataFrame preview ---")
print(error_df.head(20).to_string(index=False))


# ---------------- DEFINE MISSING HEADER VARIABLES ----------------
if time_points:
    start_dt = time_points[0].strftime("%Y-%m-%d %H:%M:%S")
    end_dt = time_points[-1].strftime("%Y-%m-%d %H:%M:%S")
    test_duration_sec = (time_points[-1] - time_points[0]).total_seconds()
    test_duration = str(timedelta(seconds=round(test_duration_sec)))
else:
    start_dt = "N/A"
    end_dt = "N/A"
    test_duration = "0s"
    test_duration_sec = 0
    
# FIX: Calculate total_tx by summing up the aggregated counts from summary_rows
total_tx = sum(row['Count'] for row in summary_rows)

# ---------------- PREPARE PER-ENDPOINT DATA FOR JS ----------------
def safe_id(s):
    """Return a filesystem/HTML safe ID for endpoint names."""
    # Ensure ID starts with a letter or underscore for JS compatibility
    return "_" + re.sub(r"[^0-9a-zA-Z]+", "_", s)

per_ep_data = {}
donut_codes_total = defaultdict(int)
for ep in endpoints:
    sid = safe_id(ep)
    
    # 💥 CRITICAL FIX: Find the corresponding row in summary_rows 
    # and use the final, accurate Count and Pass_Count from the aggregate data.
    summary_row = next((row for row in summary_rows if row["Transaction"] == ep), None)
    
    if summary_row:
        total = summary_row["Count"]
        pass_count_total_calc = summary_row["Pass_Count"]
    else:
        # Fallback if somehow a transaction wasn't in the summary table (shouldn't happen)
        total = 0
        pass_count_total_calc = 0
    # ---------------- END CRITICAL FIX ----------------
    
    if total > 0:
        pass_pct_value = (pass_count_total_calc / total) * 100
        err_pct_value = 100.0 - pass_pct_value
    else:
        pass_pct_value = 0.0
        err_pct_value = 0.0

    for code, counts in ep_codes[ep].items():
        donut_codes_total[code] += sum(counts)

    per_ep_data[ep] = {
        "spark_x": x_labels,
        "spark_y": ep_avg[ep],
        "rt_x": x_labels,
        "rt_y": ep_avg[ep],
        "p90": ep_pxx[ep]["p90"],
        "p95": ep_pxx[ep]["p95"],
        "p99": ep_pxx[ep]["p99"],
        "rps": ep_rps[ep],
        "pass_rps": ep_pass_rps[ep],
        "codes": {code: ep_codes[ep][code] for code in ep_codes[ep].keys()},
        "total": total,
        "pass_pct": f"{pass_pct_value:.2f}",
        "err_pct": f"{err_pct_value:.2f}",
        "safe_id": sid
    }

donut_labels = sorted(donut_codes_total.keys())
donut_values = [donut_codes_total[k] for k in donut_labels]

# ---------------- PHASES --------------------------------
yaml_phases = config_yaml.get("config", {}).get("phases") if isinstance(config_yaml, dict) else None
phases_data = []
cumulative_duration_sec = 0

if yaml_phases and isinstance(yaml_phases, list) and INPUT_YAML:
    for ph in yaml_phases:
        duration = ph.get("duration", 0)
        phase_start_dt = start_time + timedelta(seconds=cumulative_duration_sec)
        cumulative_duration_sec += duration
        phase_end_dt = start_time + timedelta(seconds=cumulative_duration_sec)
        phases_data.append({
            "name": ph.get("name", "Phase"),
            "duration": duration,
            "arrival": ph.get("arrivalRate", ph.get("arrivalCount", 0)),
            "start_dt": phase_start_dt.isoformat(timespec="seconds"),
            "end_dt": phase_end_dt.isoformat(timespec="seconds"),
            "start_sec": cumulative_duration_sec - duration,
            "end_sec": cumulative_duration_sec
        })
elif time_points:
    total_duration = int(test_duration_sec)
    phases_data = [{
        "name": "Total Run",
        "duration": total_duration,
        "arrival": 0,
        "start_dt": time_points[0].isoformat(timespec="seconds"),
        "end_dt": time_points[-1].isoformat(timespec="seconds"),
        "start_sec": 0,
        "end_sec": total_duration
    }]

# ---------------- JSON strings for embedding ----------------
js_PER_EP = json.dumps(per_ep_data)
js_OVERALL_RPS = json.dumps({"x": x_labels, "y": overall_rps, "ep_data": ep_rps})
js_OVERALL_PASS_RPS = json.dumps({"x": x_labels, "y": overall_pass_rps})
js_OVERALL_P90 = json.dumps({"x": x_labels, "y": overall_p90, "ep_data": {ep: ep_pxx[ep]['p90'] for ep in endpoints}})
js_OVERALL_AVG = json.dumps({"x": x_labels, "y": overall_avg, "ep_data": ep_avg})
js_EP_RPS = json.dumps({"x": x_labels, "ep_data": ep_rps, "ep_pass_data": ep_pass_rps})
js_DONUT = json.dumps({"labels": donut_labels, "values": donut_values})
js_PHASES = json.dumps(phases_data)
js_TIME_AXIS_TYPE = 'date' if TIME_AXIS_MODE=='datetime' else 'linear'
js_TIME_AXIS_MODE_PY = TIME_AXIS_MODE

# ---------------- HTML TEMPLATE START ----------------
html = f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Artillery Report</title>

  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css" rel="stylesheet">

  <script src="https://cdn.plot.ly/plotly-2.33.0.min.js"></script>
  <style>
body{{background:#f7f8fa;font-family:'Inter',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;padding:18px;}}
.card{{margin-bottom:16px;border-radius:12px;box-shadow:0 2px 6px rgba(0,0,0,0.08);}}
.card-body{{padding:20px;}}
.table{{font-size:0.875rem;}}
/* center all table cells except the first column (Transaction) */
.table th, .table td {{ text-align:center; vertical-align: middle; }}
.table th:first-child, .table td:first-child {{ text-align:left !important; }}
/* sparkline styling (not used in final screenshot design, but keeping for reference) */
.sparkline{{width:180px;height:36px;display:inline-block;}}
.chart-controls {{ display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 1rem; padding: 10px 0; border-top: 1px solid #eee; border-bottom: 1px solid #eee; }}

/* Transaction list style (from screenshot) */
.accordion-item {{
    border: none !important; /* Ensure no borders on accordion items to keep list look */
}}
/* Apply shading to alternating transaction items for visual separation */
.transaction-item {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 15px; /* Added horizontal padding for shade */
    margin: 5px 0; /* Add margin for border effect */
    border-radius: 8px; /* Slight rounding for shade */
    /* border: 1px solid #eee; */ /* Optional subtle border */
}}
.transaction-item-even {{
    background-color: #fcfcfc; /* Very subtle lighter shade */
}}
.transaction-item-odd {{
    background-color: #f2f2f2; /* Slightly darker subtle shade */
}}
.accordion-item:first-child .transaction-item {{
    /* No border top removal needed with this new box style */
}}
.transaction-name {{
    font-size: 1rem;
    font-weight: 500;
    flex-grow: 1;
    margin-right: 15px;
}}
.transaction-badges {{
    display: flex;
    gap: 8px;
    align-items: center;
}}
.details-button {{
    min-width: 80px;
    /* Use button as accordion header for accessibility */
    background: none !important;
    border: none !important;
    color: #0d6efd !important;
    text-align: right;
    padding: 0;
    font-weight: 500; /* Match look of a link */
}}
/* For accordion style, ensure collapsed content has some padding/margin */
.collapse-graphs {{
    padding-top: 15px;
    padding-bottom: 5px;
    margin-bottom: 0;
    /* border-bottom: 1px solid #eee; */
}}
.accordion-item:last-child .collapse-graphs {{
    /* border-bottom: none; */
}}
.collapse-graphs .row > div {{
    margin-bottom: 15px; /* Spacing between charts in the same row */
}}
</style>
</head>

<body>
<div class='container-fluid'>
  <div class='card'><div class='card-body'>
    <h3>🚀 Artillery Performance Report</h3>
    <div class='row'>
      <div class='col-md-3'><strong>Start:</strong><div>{start_dt}</div></div>
      <div class='col-md-3'><strong>End:</strong><div>{end_dt}</div></div>
      <div class='col-md-3'><strong>Duration:</strong><div>{test_duration}</div></div>
      <div class='col-md-3'><strong>Total tx:</strong><div>{total_tx}</div></div>
    </div>
  </div></div>

  <div class='card mt-3'><div class='card-body'>
    <h5>🎛️ Chart Visibility</h5>
    <div class='chart-controls'>
      <div class="form-check form-check-inline">
        <input class="form-check-input chart-toggle" type="checkbox" id="show-chart-p90" data-target="chart-p90" {'checked' if SHOW_GRAPH_P90 else ''}>
        <label class="form-check-label" for="show-chart-p90">Overall P90</label>
      </div>
      <div class="form-check form-check-inline">
        <input class="form-check-input chart-toggle" type="checkbox" id="show-chart-avg" data-target="chart-avg" checked>
        <label class="form-check-label" for="show-chart-avg">Overall Avg</label>
      </div>
      <div class="form-check form-check-inline">
        <input class="form-check-input chart-toggle" type="checkbox" id="show-chart-ep-rps" data-target="chart-ep-rps" checked>
        <label class="form-check-label" for="show-chart-ep-rps">RPS Per Endpoint</label>
      </div>
      <div class="form-check form-check-inline">
        <input class="form-check-input chart-toggle" type="checkbox" id="show-chart-rps" data-target="chart-rps" {'checked' if SHOW_GRAPH_RPS else ''}>
        <label class="form-check-label" for="show-chart-rps">Overall Throughput</label>
      </div>
      <div class="form-check form-check-inline">
        <input class="form-check-input chart-toggle" type="checkbox" id="show-chart-combined" data-target="chart-combined" {'checked' if SHOW_GRAPH_COMBINED else ''}>
        <label class="form-check-label" for="show-chart-combined">RPS vs P90 Combined</label>
      </div>
    </div>
  </div></div>
  
  <div class='card mt-3'><div class='card-body'>
    <h5>📄 Transaction Summary</h5>
    <div class="table-responsive" style="max-height: 400px; overflow-y: auto;">
      {summary_df.to_html(classes="table table-striped table-bordered table-sm text-center", index=False, justify="center")}
    </div>
  </div></div>

  <div class='card mt-3'><div class='card-body'>
    <h5>🚨 Error and Status Code Breakdown</h5>
    <div class='row'>
      <div class='col-md-6'>
        <h6>Detailed Non-2xx Status Codes and Errors</h6>
        <div class="table-responsive" style="max-height: 300px; overflow-y: auto;">
          {error_df.to_html(classes="table table-striped table-bordered table-sm text-center", index=False, justify="center")}
        </div>
      </div>
      <div class='col-md-6'>
        <h6>Overall HTTP Status Code Distribution</h6>
        <div id="chart-codes" style="height:300px;">
          <div id="donut" style="width:100%; height:300px; margin: 0 auto;"></div>
        </div>
      </div>
    </div>
  </div></div>

  <div class='card mt-3'><div class='card-body'>
    <h5>📈 Overall Charts</h5>
    <div id="chart-rps" style="height:300px; display: {'block' if SHOW_GRAPH_RPS else 'none'};"></div>
    <div id="chart-p90" style="height:300px; display: {'block' if SHOW_GRAPH_P90 else 'none'};"></div>
    <div id="chart-avg" style="height:300px; display: block;"></div>
    <div id="chart-ep-rps" style="height:300px; display: block;"></div>
    <div id="chart-combined" style="height:300px; display: {'block' if SHOW_GRAPH_COMBINED else 'none'};"></div>
  </div></div>

  <div class='card mt-3'><div class='card-body'>
    <h5>🔎 Per-endpoint breakdown</h5>
    <div class="accordion" id="perEndpointAccordion">
"""

# 🔹 Accordion structure for each endpoint
for i, ep in enumerate(endpoints):
    d = per_ep_data[ep]
    # Determine shading class for the transaction item
    shading_class = "transaction-item-even" if i % 2 == 0 else "transaction-item-odd"
    
    # Each transaction list item is an accordion item header
    html += f"""
        <div class="accordion-item">
            <div class="transaction-item {shading_class}">
                <div class="transaction-name">{ep}</div>
                <div class="transaction-badges">
                    <span class='badge bg-secondary'>Total: {d['total']}</span>
                    <span class='badge {"bg-success" if float(d['err_pct']) < 1.0 else "bg-success"}'>Succ: {d['pass_pct']}%</span>
                    <span class='badge {"bg-danger" if float(d['err_pct']) > 0.0 else "bg-secondary"}'>Err: {d['err_pct']}%</span>
                </div>
                <button class="btn btn-sm btn-link details-button ms-3" type="button" 
                        data-bs-toggle="collapse" data-bs-target="#collapse-{d["safe_id"]}" 
                        aria-expanded="false" aria-controls="collapse-{d["safe_id"]}" 
                        data-ep-id="{ep}">
                    Details
                </button>
            </div>
            
            <div id="collapse-{d["safe_id"]}" class="accordion-collapse collapse collapse-graphs" data-bs-parent="#perEndpointAccordion">
                <div class="accordion-body p-0">
                    <h6 class="mb-3"><b>{ep}</b> - Response Time, RPS, and Status Codes</h6>
                    <div class='row'>
                        <div class='col-md-6'><div id='ep-{d["safe_id"]}-rt' style='height:260px;'></div></div>
                        <div class='col-md-6'><div id='ep-{d["safe_id"]}-rps' style='height:260px;'></div></div>
                    </div>
                    <div class='row'>
                        <div class='col-12'><div id='ep-{d["safe_id"]}-codes' style='height:260px;'></div></div>
                    </div>
                </div>
            </div>
        </div>
    """

# Close the Accordion and Card Body/Container
html += f"""
    </div>
</div></div>
</div> 
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>


<script>
// JSON DATA INJECTION 
const PER_EP = {js_PER_EP};
const OVERALL_RPS_DATA = {js_OVERALL_RPS};
const OVERALL_PASS_RPS_DATA = {js_OVERALL_PASS_RPS};
const OVERALL_P90_DATA = {js_OVERALL_P90};
const OVERALL_AVG_DATA = {js_OVERALL_AVG};
const EP_RPS_DATA = {js_EP_RPS};
const DONUT = {js_DONUT};
const PHASES = {js_PHASES};

// STRING DATA INJECTION
const TIME_AXIS_TYPE = '{js_TIME_AXIS_TYPE}';
const TIME_AXIS_MODE_PY = '{js_TIME_AXIS_MODE_PY}';

// BOOLEAN DATA INJECTION
const SHOW_GRAPH_RPS = {str(SHOW_GRAPH_RPS).lower()};
const SHOW_GRAPH_P90 = {str(SHOW_GRAPH_P90).lower()};
const SHOW_GRAPH_COMBINED = {str(SHOW_GRAPH_COMBINED).lower()};
const SHOW_GRAPH_CODES = {str(SHOW_GRAPH_CODES).lower()};

const SHOW_PHASE_SHAPES = true;

// small helper to add phase rectangles/annotations
function addPhaseShapes(layout, yMax, dataPoints) {{
    if (!SHOW_PHASE_SHAPES) return;
    layout.shapes = layout.shapes || [];
    layout.annotations = layout.annotations || [];
    if (!PHASES || PHASES.length === 0) return;
    let safeYMax = Math.max(yMax || 1, 1);
    if (yMax < 10 && yMax > 0) safeYMax = 10;
    
    if (TIME_AXIS_MODE_PY === 'datetime') {{
        PHASES.forEach(ph => {{ 
            if (!ph.start_dt || !ph.end_dt || ph.start_dt === ph.end_dt) return;
            layout.shapes.push({{type:'rect', x0:ph.start_dt, x1:ph.end_dt, y0:0, y1:1, fillcolor:'rgba(0,123,255,0.06)', line:{{width:0}}, xref:'x', yref:'paper'}});
            let startTS = new Date(ph.start_dt).getTime();
            let endTS = new Date(ph.end_dt).getTime();
            let center = new Date((startTS + endTS)/2).toISOString().substring(0,19);
            layout.annotations.push({{x:center, y:safeYMax*1.02, text:ph.name||'Phase', showarrow:false, font:{{size:10,color:'#007bff'}}, xref:'x', yref:'y'}});
        }});
    }} else {{
        PHASES.forEach(ph => {{
            layout.shapes.push({{type:'rect', x0:ph.start_sec, x1:ph.end_sec, y0:0, y1:1, fillcolor:'rgba(0,123,255,0.06)', line:{{width:0}}, yref:'paper'}});
            layout.annotations.push({{x:(ph.start_sec+ph.end_sec)/2, y:safeYMax*1.02, text:ph.name||'Phase', showarrow:false, font:{{size:10,color:'#007bff'}}}});
        }});
    }}
}}

// Reusable function to draw main graphs
function drawMainGraph(chartId, title, yAxisTitle, mainData, epData=null, passData=null) {{
    let maxVals = mainData.y.filter(y=>!isNaN(y) && y!==null);
    if (epData) Object.values(epData).forEach(arr=>{{ maxVals = maxVals.concat(arr.filter(y=>!isNaN(y) && y!==null)); }});
    let maxY = maxVals.length>0 ? Math.max(...maxVals) : 10;
    let traces = [{{x: mainData.x, y: mainData.y, mode:'lines', name:title, line:{{width:3}}}}];
    if (epData) {{
        Object.keys(epData).forEach(ep => {{ 
            traces.push({{x: mainData.x, y: epData[ep], mode:'lines', name: ep, visible: 'legendonly', line:{{width:1}}}});
        }});
    }}
    if (passData && passData.y.length > 0) {{
        traces.push({{x: passData.x, y: passData.y, mode:'lines', name:'Pass-Only (2xx)', visible:'legendonly', line:{{width:3, dash:'dash'}}}});
    }}
    let layout = {{ title: title, xaxis: {{type: TIME_AXIS_TYPE}}, yaxis: {{title: yAxisTitle}}, hovermode:'x unified' }};
    addPhaseShapes(layout, maxY, mainData);
    Plotly.newPlot(chartId, traces, layout, {{responsive:true}});
}}

// Reusable function for per-endpoint detail graphs (called on collapse show event)
function drawEndpointGraphs(ep) {{
    let id = PER_EP[ep].safe_id;
    let rt_div = document.getElementById('ep-'+id+'-rt');
    let rps_div = document.getElementById('ep-'+id+'-rps');
    let codes_div = document.getElementById('ep-'+id+'-codes');
    let endpointData = PER_EP[ep];

    // Response Time Chart
    if (rt_div && endpointData.rt_y.length>0) {{ 
        let traces = [
            {{x:endpointData.rt_x, y:endpointData.rt_y, mode:'lines', name:'Avg RT', line:{{width:3}}}},
            {{x:endpointData.rt_x, y:endpointData.p90, mode:'lines', name:'P90'}},
            {{x:endpointData.rt_x, y:endpointData.p95, mode:'lines', name:'P95'}},
            {{x:endpointData.rt_x, y:endpointData.p99, mode:'lines', name:'P99'}}
        ];
        let allY = traces.flatMap(t=>t.y).filter(y=>!isNaN(y));
        let maxY = allY.length>0 ? Math.max(...allY) : 100;
        let layout = {{title: ep+' Response Times (Percentiles & Avg)', xaxis: {{type: TIME_AXIS_TYPE}}, yaxis: {{title:'ms'}}}};
        addPhaseShapes(layout, maxY, endpointData);
        Plotly.newPlot(rt_div, traces, layout, {{responsive:true}});
    }}
    
    // RPS Chart
    if (rps_div && endpointData.rps.length>0) {{ 
        let layoutR = {{title: ep+' Requests per Second', xaxis:{{type: TIME_AXIS_TYPE}}, yaxis:{{title:'RPS'}}}};
        addPhaseShapes(layoutR, Math.max(...endpointData.rps), endpointData);
        Plotly.newPlot(rps_div, [{{x:endpointData.rt_x, y:endpointData.rps, mode:'lines', name:'RPS'}}], layoutR, {{responsive:true}});
    }}
    
    // Codes Chart
    if (codes_div && Object.keys(endpointData.codes).length>0) {{
        let traces_codes = [];
        Object.keys(endpointData.codes).forEach(code => {{
            traces_codes.push({{x:endpointData.rt_x, y:endpointData.codes[code], mode:'lines', name:code, stackgroup:'one'}});
        }});
        let allCodeY = traces_codes.flatMap(t=>t.y).filter(y=>!isNaN(y));
        let layoutC = {{title: ep+' HTTP Codes (Stacked)', xaxis:{{type: TIME_AXIS_TYPE}}, yaxis:{{title:'count'}}}};
        addPhaseShapes(layoutC, allCodeY.length>0 ? Math.max(...allCodeY) : 10, endpointData);
        Plotly.newPlot(codes_div, traces_codes, layoutC, {{responsive:true}});
    }}
}}


document.addEventListener('DOMContentLoaded', function(){{
    
    // 1. Hook up visibility toggles for Main Charts
    document.querySelectorAll('.chart-toggle').forEach(toggle => {{
        let targetId = toggle.getAttribute('data-target');
        let targetDiv = document.getElementById(targetId);
        
        toggle.addEventListener('change', function() {{
            if (targetDiv) {{
                targetDiv.style.display = this.checked ? "block" : "none";
                
                if (this.checked) {{
                     if (targetId === 'chart-rps' && OVERALL_RPS_DATA.y.length > 0) {{
                         drawMainGraph('chart-rps', 'Overall Throughput (RPS) vs Time', 'Requests/s', OVERALL_RPS_DATA, null, OVERALL_PASS_RPS_DATA);
                     }} else if (targetId === 'chart-p90' && OVERALL_P90_DATA.y.length > 0) {{
                         drawMainGraph('chart-p90', 'Overall P90 Response Time vs Time', 'ms', OVERALL_P90_DATA, OVERALL_P90_DATA.ep_data);
                     }} else if (targetId === 'chart-combined' && OVERALL_RPS_DATA.y.length>0 && OVERALL_P90_DATA.y.length>0) {{
                         let layout_comb = {{title:'RPS vs Overall P90', xaxis:{{type: TIME_AXIS_TYPE}}, yaxis:{{title:'RPS', side:'left'}}, yaxis2:{{title:'P90 ms', overlaying:'y', side:'right'}}}};
                         addPhaseShapes(layout_comb, Math.max(...OVERALL_RPS_DATA.y), OVERALL_RPS_DATA);
                         Plotly.newPlot('chart-combined', [
                            {{x: OVERALL_RPS_DATA.x, y: OVERALL_RPS_DATA.y, mode:'lines', name:'RPS', yaxis:'y1'}},
                            {{x: OVERALL_P90_DATA.x, y: OVERALL_P90_DATA.y, mode:'lines', name:'P90', yaxis:'y2'}}
                         ], layout_comb, {{responsive:true}});
                     }} else {{
                         // Only relayout if the chart was already drawn
                         Plotly.relayout(targetDiv, {{autosize:true}});
                     }}
                }}
            }}
        }});
    }});

    // 2. Draw all initial Main Charts
    
    if (OVERALL_AVG_DATA.y.length > 0) {{
        drawMainGraph('chart-avg', 'Overall Average Response Time vs Time', 'ms', OVERALL_AVG_DATA, OVERALL_AVG_DATA.ep_data);
    }}
    if (Object.keys(EP_RPS_DATA.ep_data).length > 0) {{
        let overall_y_rps = EP_RPS_DATA.x.map((_, i) => Object.values(EP_RPS_DATA.ep_data).reduce((s, arr) => s + arr[i], 0));
        let mainRpsData = {{x: EP_RPS_DATA.x, y: overall_y_rps}};
        drawMainGraph('chart-ep-rps', 'Requests Per Second (per endpoint)', 'Requests/s', mainRpsData, EP_RPS_DATA.ep_data);
    }}

    if (SHOW_GRAPH_RPS && OVERALL_RPS_DATA.y.length > 0) {{
        drawMainGraph('chart-rps', 'Overall Throughput (RPS) vs Time', 'Requests/s', OVERALL_RPS_DATA, null, OVERALL_PASS_RPS_DATA);
    }}
    if (SHOW_GRAPH_P90 && OVERALL_P90_DATA.y.length > 0) {{
        drawMainGraph('chart-p90', 'Overall P90 Response Time vs Time', 'ms', OVERALL_P90_DATA, OVERALL_P90_DATA.ep_data);
    }}
    if (SHOW_GRAPH_COMBINED && OVERALL_RPS_DATA.y.length>0 && OVERALL_P90_DATA.y.length>0) {{
        let layout_comb = {{title:'RPS vs Overall P90', xaxis:{{type: TIME_AXIS_TYPE}}, yaxis:{{title:'RPS', side:'left'}}, yaxis2:{{title:'P90 ms', overlaying:'y', side:'right'}}}};
        addPhaseShapes(layout_comb, Math.max(...OVERALL_RPS_DATA.y), OVERALL_RPS_DATA);
        Plotly.newPlot('chart-combined', [
            {{x: OVERALL_RPS_DATA.x, y: OVERALL_RPS_DATA.y, mode:'lines', name:'RPS', yaxis:'y1'}},
            {{x: OVERALL_P90_DATA.x, y: OVERALL_P90_DATA.y, mode:'lines', name:'P90', yaxis:'y2'}}
        ], layout_comb, {{responsive:true}});
    }}

    // 3. Donut Chart
    if (DONUT.labels && DONUT.labels.length>0) {{
        Plotly.newPlot('donut', [{{labels: DONUT.labels, values: DONUT.values, type:'pie', hole:0.6, marker: {{colors: ['#28a745', '#ffc107', '#dc3545', '#17a2b8', '#6c757d']}}}}],
            {{title: 'Overall HTTP Status Code Breakdown', margin:{{t:40,b:20,l:20,r:20}}}}, {{responsive:true}});
    }}


    // 4. Per-Endpoint Charts (Accordion)
    document.querySelectorAll('.details-button').forEach(button => {{
        let targetId = button.getAttribute('data-bs-target').replace('#', '');
        let collapseElement = document.getElementById(targetId);
        let epName = button.getAttribute('data-ep-id');

        if (collapseElement) {{
             // Redraw on 'shown' to ensure correct sizing after collapse expands
             collapseElement.addEventListener('shown.bs.collapse', function () {{
                 if (epName) {{
                    drawEndpointGraphs(epName);
                 }}
             }});
        }}
    }});

    // 5. DataTables
    // Apply DataTables to both summary and error tables
    $('table').DataTable({{ paging: true, searching: true, info: true, order: [] }});

}});

</script>
</body></html>
"""


# ---------------- WRITE HTML ----------------
try:
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"🎉 Successfully generated dashboard at {OUTPUT_HTML}")
except Exception as e:
    print(f"❌ Error writing HTML output file: {e}")
