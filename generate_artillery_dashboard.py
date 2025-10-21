#!/usr/bin/env python3
"""
Artillery Multi-Report Dashboard Generator
Merges multiple Artillery JSON reports into a single, interactive HTML dashboard.
Enhanced with:
 - Configurable time granularity
 - Proper timezone (EST) handling
 - Phase visibility control for elapsed vs datetime mode
 - ✅ New: Optional --app-name, --test-name, and --show-overall-metrics CLI arguments.
 - ✅ New: Improved header layout and metrics panel alignment.
 - Time-series graphs use MAX aggregation for RT metrics within each time bucket.
 - SLA Compliance Check (P95 vs SLA) in the summary table.
 - ✅ FIX: Handles 'mean': null in Artillery reports during aggregation.
 - ✅ NEW: Adds success-only (2xx) single-digit metrics to the header for clarity.
 - 🛑 NEW FIX: Uses P50 (50th Percentile) for all 'Avg RT' displays and time-series data.
 - 🛑 NEW FIX: Dynamic granularity calculation based on test duration and user input.
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
import pytz

# ---------------- CONFIG - DEFAULT VALUES ----------------
TIME_AXIS_MODE = "datetime"  # "elapsed" or "datetime"
# 🛑 MODIFIED: Initialize GRANULARITY_SEC to a default, but it will be recalculated later.
GRANULARITY_SEC = 30  # seconds for graph and bucket granularity

# 🛑 NEW DEFAULTS FOR GRANULARITY
# Default granularity used if user does not provide one.
GRANULARITY_DEFAULT_ELAPSED_SEC = 30       # Default for elapsed axis mode (30 seconds)
GRANULARITY_DEFAULT_DATETIME_STR = '2min'  # Default for datetime axis mode

# 🛑 NEW CONSTANTS FOR AUTO-CALCULATION
AUTO_CALC_MAP = {
    # (Duration Threshold in seconds): (Granularity in seconds, Granularity string)
    900: (30, '30S'),     # <= 15 minutes: 30 seconds
    1800: (60, '1min'),   # 15m to 30m: 1 minute
    3600: (120, '2min'),  # 30m to 60m: 2 minutes (Matches datetime default for this range)
    7200: (300, '5min'),  # 1h to 2h: 5 minutes
    float('inf'): (600, '10min'), # > 2 hours: 10 minutes
}

PERCENTILES_AND_MAX = ["p50", "p90", "p95", "p99", "max", "avg"] 
PERCENTILES_ONLY = ["p50", "p90", "p95", "p99"] # Used for standard percentile lists


# Chart Visibility Defaults (Adjusted for consistency with previous solution)
SHOW_GRAPH_RPS = False
SHOW_GRAPH_AVG = False
SHOW_GRAPH_P90 = False
SHOW_GRAPH_P95 = False 
SHOW_GRAPH_COMBINED = True      # RPS vs P90
SHOW_GRAPH_ALL_RT = True        # Combined Avg, P90, P95, P99, Max (Hidden by default in user's initial template, but included here for completeness)
SHOW_GRAPH_RPS_VS_P95 = True    # RPS vs P95 dual axis (Hidden by default in user's initial template, but included here for completeness)
SHOW_GRAPH_CODES = True
SHOW_GRAPH_DONUT = True

# Data Filtering
FILTER_START_SEC = 0
FILTER_END_SEC = None

# EST timezone configuration
TZ_EST = pytz.timezone("US/Eastern")

def to_est(dt_or_ms):
    """
    Convert a timestamp (milliseconds since epoch or datetime) into an EST-aware datetime.
    """
    if isinstance(dt_or_ms, (int, float)):
        return datetime.fromtimestamp(int(dt_or_ms) / 1000.0, tz=TZ_EST).replace(microsecond=0)
    elif isinstance(dt_or_ms, datetime):
        return dt_or_ms.astimezone(TZ_EST) if dt_or_ms.tzinfo else TZ_EST.localize(dt_or_ms.replace(microsecond=0))
    else:
        raise TypeError(f"Unsupported type for to_est(): {type(dt_or_ms)}")

# =========================================================================
# 🛑 NEW: GRANULARITY HELPER FUNCTIONS (Logic moved from thought process)
# =========================================================================

def _convert_pandas_freq_to_seconds(freq_str):
    """Converts a Pandas frequency string (e.g., '5min', '2h') into seconds."""
    try:
        # Check for numeric strings (elapsed mode input)
        if str(freq_str).isdigit():
             return int(freq_str)
             
        # Create a dummy timedelta object from the frequency string (datetime mode input)
        td = pd.to_timedelta(freq_str)
        return td.total_seconds()
    except ValueError:
        print(f"ERROR: Invalid frequency string '{freq_str}'.")
        return 0

def _determine_granularity(total_duration_sec, user_granularity, time_axis_mode):
    """
    Determines the final time-series granularity based on user input,
    axis mode, and total test duration.
    
    Returns: (granularity_sec, resample_period_str)
    """
    
    # 1. USER OVERRIDE: If user_granularity is provided, use it.
    if user_granularity is not None:
        granularity = str(user_granularity).strip()
        
        if time_axis_mode == 'elapsed':
            try:
                gran_sec = int(granularity)
                if gran_sec > 0:
                    return gran_sec, f'{gran_sec}S'
                else:
                    print(f"WARNING: Granularity must be > 0. Using default.")
            except ValueError:
                print(f"WARNING: Invalid elapsed granularity '{granularity}'. Must be an integer (seconds). Using default.")
        
        elif time_axis_mode == 'datetime':
            gran_sec = _convert_pandas_freq_to_seconds(granularity)
            if gran_sec > 0:
                return gran_sec, granularity
            else:
                print(f"WARNING: Invalid datetime granularity '{granularity}'. Using default.")

    # 2. AUTO-CALCULATION/DEFAULTS
    
    # Auto-calculation based on test duration
    for threshold, (gran_sec, gran_str) in AUTO_CALC_MAP.items():
        if total_duration_sec <= threshold:
            if time_axis_mode == 'datetime':
                # Return the frequency string for datetime mode
                return _convert_pandas_freq_to_seconds(gran_str), gran_str
            else:
                # Return the seconds for elapsed mode
                return gran_sec, f'{gran_sec}S'

    # 3. FALLBACK: Use defined defaults 
    if time_axis_mode == 'datetime':
        gran_sec = _convert_pandas_freq_to_seconds(GRANULARITY_DEFAULT_DATETIME_STR)
        return gran_sec, GRANULARITY_DEFAULT_DATETIME_STR
    else:
        return GRANULARITY_DEFAULT_ELAPSED_SEC, f'{GRANULARITY_DEFAULT_ELAPSED_SEC}S'

# =========================================================================

# Default SLA/TPL values
DEFAULT_SLA = 500  # ms
DEFAULT_TPH = 0

# ---------------- CLI ARGUMENTS ----------------
parser = argparse.ArgumentParser(description="Merge up to N Artillery JSON reports into one HTML dashboard.")
parser.add_argument("--json", required=True, help="Comma-separated list of Artillery JSON report files (or single file).")
parser.add_argument("--yaml", required=False, help="Optional Artillery YAML config file (for phases).")
parser.add_argument("--sla", type=str, default=None, help="Optional SLA JSON path.")
parser.add_argument("--output", required=True, help="Output HTML file path.")
# ✅ Added new optional arguments
parser.add_argument("--mode", default=TIME_AXIS_MODE, choices=["datetime", "elapsed"], help="Time axis mode: 'datetime' or 'elapsed' (default: %(default)s).")
# 🛑 MODIFIED: Change default to None and remove type=int to allow '5min' strings.
parser.add_argument("--granularity", default=None, help="Time bucket granularity in seconds (for elapsed, e.g., 60) or Pandas frequency string (for datetime, e.g., 5min). Overrides auto-calculation.")
parser.add_argument("--app-name", default=None, help="Optional application name to display in the report header.")
parser.add_argument("--test-name", default=None, help="Optional test name (e.g., Load Test 1) to display in the report header.")
parser.add_argument("--show-overall-metrics", action='store_true', help="Flag to display the key single-digit overall performance metrics.")
args = parser.parse_args()

# Apply optional arguments to configuration variables
TIME_AXIS_MODE = args.mode
# 🛑 MODIFIED: Capture user granularity string/value. GRANULARITY_SEC will be set later.
USER_GRANULARITY_INPUT = args.granularity 
APP_NAME = args.app_name
TEST_NAME = args.test_name
SHOW_OVERALL_METRICS = args.show_overall_metrics

# MIN_ARTILLERY_INTERVAL logic is now redundant as GRANULARITY_SEC is calculated later,
# but we need to re-assign the value to satisfy the existing code structure before
# the final calculation. We'll move the check after the final calculation.


MIN_ARTILLERY_INTERVAL = 10 
if GRANULARITY_SEC < MIN_ARTILLERY_INTERVAL:
    print(f"⚠️ Warning: Requested granularity ({GRANULARITY_SEC}s) is less than the typical Artillery snapshot interval ({MIN_ARTILLERY_INTERVAL}s).")
    #print(f"    Setting granularity to {MIN_ARTILLERY_INTERVAL}s to prevent distorted RPS spikes.")
    GRANULARITY_SEC = MIN_ARTILLERY_INTERVAL
    
# ---------------- INPUT FILES ----------------
if args.json:
    INPUT_JSONS = [p.strip() for p in args.json.split(",") if p.strip()]
else:
    INPUT_JSONS = []

INPUT_YAML = args.yaml
INPUT_SLA = args.sla
OUTPUT_HTML = args.output

existing_files = []
missing_files = []
for f in INPUT_JSONS:
    p = Path(f).resolve()
    if p.exists():
        existing_files.append(str(p))
    else:
        missing_files.append(str(p))

if not existing_files:
    print("❌ Error: None of the provided JSON files exist. Exiting.")
    sys.exit(1)

if missing_files:
    print("⚠️ Warning: Some JSON files were not found and will be skipped:")
    for f in missing_files:
        print(f"    - File not found: {f}")

INPUT_JSONS = existing_files
print(f"✅ Proceeding with {len(INPUT_JSONS)} existing JSON file(s).")

# ---------------- LOAD SLA JSON ----------------
sla_data = {}
SLA_ENABLED = False
if INPUT_SLA:
    sla_path = Path(INPUT_SLA).resolve()
    if not sla_path.exists():
        print(f"⚠️ Warning: SLA file '{sla_path}' not found.")
    else:
        try:
            with open(sla_path, "r", encoding="utf-8") as f:
                sla_data = json.load(f) or {}
            SLA_ENABLED = bool(sla_data)
            print(f"✅ Loaded SLA definitions ({len(sla_data)} entries).")
        except Exception as e:
            print(f"⚠️ SLA parse error: {e}")
else:
    print("ℹ️ No SLA file provided. SLA columns will be omitted.")

# ---------------- LOAD JSON INTERMEDIATE SNAPSHOTS ----------------
# ---------------- LOAD JSON INTERMEDIATE SNAPSHOTS ----------------
all_intermediate = []
all_agg_counters = defaultdict(int)
# Ensure the aggregate summary can store all required fields
# Initialize 'mean' to 0 for safe calculation
all_agg_summaries = defaultdict(lambda: {"count": 0, "mean": 0, "min": float("inf"), "max": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0}) # Initialize P50
overall_max_instant_rps = 0 # ✅ NEW: Track the max instantaneous rate from the JSON

for json_file in INPUT_JSONS:
    p = Path(json_file)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load {p}: {e}")
        continue

    intermediate_list = data.get("intermediate", [])
    if intermediate_list:
        for snap in intermediate_list:
            snap["_source_file"] = str(p)
            
            # 👇 FIX: Find the absolute MAX instantaneous RPS from all snapshots
            instant_rate = snap.get("rates", {}).get("http.request_rate")
            if instant_rate is not None and instant_rate > overall_max_instant_rps:
                overall_max_instant_rps = instant_rate
                
        all_intermediate.extend(intermediate_list)

    # Logic: Sum counters, and merge summaries using weighted mean for 'mean' and MAX for percentiles/max.
    for k, v in data.get("aggregate", {}).get("counters", {}).items():
        all_agg_counters[k] += v
    for k, s in data.get("aggregate", {}).get("summaries", {}).items():
        existing = all_agg_summaries[k]
        n1, n2 = existing.get("count", 0), s.get("count", 0)
        
        if n1 + n2 > 0:
            # ✅ FIX for TypeError: Check if 'mean' is None (null in JSON) before calculation
            existing_mean_val = existing.get("mean") if existing.get("mean") is not None else 0
            new_mean_val = s.get("mean") if s.get("mean") is not None else 0
            
            # Keep weighted mean calculation in the aggregated summary, even though we won't use it for the table's "Avg"
            existing["mean"] = (existing_mean_val * n1 + new_mean_val * n2) / (n1 + n2)
            existing["min"] = min(existing["min"], s.get("min", existing["min"]))
            existing["max"] = max(existing["max"], s.get("max", existing["max"])) 
            
            # P50 is now included in PERCENTILES_ONLY due to CHANGE 1
            for p_key in PERCENTILES_ONLY:
                existing[p_key] = max(existing.get(p_key, 0), s.get(p_key, 0)) # Max aggregation for percentiles
            existing["count"] = n1 + n2

agg_counters = all_agg_counters
agg_summaries = all_agg_summaries
intermediate = all_intermediate


if not intermediate:
    print("❌ No snapshots found in provided JSON files. Exiting.")
    sys.exit(1)

# ---------------- TIME ALIGNMENT & BUCKET DEFINITION ----------------
valid_intermediate = [snap for snap in intermediate if snap.get("firstMetricAt")]
intermediate = valid_intermediate

if not intermediate:
    print("❌ No valid time-based snapshots found. Exiting.")
    sys.exit(1)

# Find the overall start and end time of the test run
all_metric_times = [snap["firstMetricAt"] for snap in intermediate] + [snap["lastMetricAt"] for snap in intermediate if snap.get("lastMetricAt")]
min_ts_ms = min(all_metric_times)
max_ts_ms = max(all_metric_times)
start_time_unaligned = to_est(min_ts_ms)
end_time = to_est(max_ts_ms)

# Calculate total duration needed for auto-granularity
test_duration_sec_unaligned = (end_time - start_time_unaligned).total_seconds()


# ----------------- 🛑 NEW: DETERMINE FINAL GRANULARITY -----------------
# Determine the final granularity used for data aggregation
GRANULARITY_SEC, RESAMPLE_PERIOD_STR = _determine_granularity(
    test_duration_sec_unaligned, 
    USER_GRANULARITY_INPUT, # User-provided input string/value
    TIME_AXIS_MODE
)

# Enforce minimum interval check (can use the newly calculated GRANULARITY_SEC)
MIN_ARTILLERY_INTERVAL = 10 
if GRANULARITY_SEC < MIN_ARTILLERY_INTERVAL:
    print(f"⚠️ Warning: Calculated granularity ({GRANULARITY_SEC}s) is less than the typical Artillery snapshot interval ({MIN_ARTILLERY_INTERVAL}s). Setting to {MIN_ARTILLERY_INTERVAL}s.")
    GRANULARITY_SEC = MIN_ARTILLERY_INTERVAL
    RESAMPLE_PERIOD_STR = f'{GRANULARITY_SEC}S'

print(f"INFO: Total test duration is {test_duration_sec_unaligned:.0f} seconds.")
print(f"INFO: Time series granularity set to {RESAMPLE_PERIOD_STR} (seconds: {GRANULARITY_SEC}).")
# -----------------------------------------------------------------------


# Align start time to the nearest preceding granularity boundary for cleaner buckets
start_ts_sec = start_time_unaligned.timestamp()
# This crucial line now uses the calculated GRANULARITY_SEC
start_ts_sec_aligned = start_ts_sec - (start_ts_sec % GRANULARITY_SEC) 
start_time_aligned = datetime.fromtimestamp(start_ts_sec_aligned, tz=TZ_EST)

# Define the buckets
# This line also uses the calculated GRANULARITY_SEC
BUCKET_DURATION = timedelta(seconds=GRANULARITY_SEC) 
time_points = []
current_time = start_time_aligned

while current_time < end_time + BUCKET_DURATION:
    time_points.append(current_time)
    current_time += BUCKET_DURATION

n_points = len(time_points)
if n_points <= 1:
     # Fallback for very short tests or single snapshot
    time_points = [start_time_aligned]
    n_points = 1

# Prepare X-axis labels based on mode
x_labels = [
    t.isoformat(timespec="seconds")
    if TIME_AXIS_MODE == "datetime"
    # This calculation still uses start_time_aligned, which is good.
    else round((t - start_time_aligned).total_seconds(), 2) 
    for t in time_points
]

# ---------------- DISCOVER ENDPOINTS ----------------
def discover_endpoints(counters, summaries):
    """Find all unique endpoints from counters and summaries."""
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
# Use PERCENTILES_AND_MAX to include max and avg for time-series data
ep_pxx = {ep: {p: [0] * n_points for p in PERCENTILES_AND_MAX} for ep in endpoints} 
ep_counts = {ep: [0] * n_points for ep in endpoints}
ep_pass_counts = {ep: [0] * n_points for ep in endpoints}
ep_codes = {ep: defaultdict(lambda: [0] * n_points) for ep in endpoints}

ep_total_codes = defaultdict(lambda: defaultdict(int)) # Final summary (not time-series)

# ---------------- FILL SNAPSHOTS (AGGREGATION) ----------------
for snap in intermediate:
    t_ms = snap.get("firstMetricAt")
    if not t_ms:
        continue
    
    snap_dt = to_est(t_ms)

    # Determine which bucket this snapshot falls into
    time_diff_sec = (snap_dt - start_time_aligned).total_seconds()
    if time_diff_sec < 0:
        continue # Skip data before the aligned start time

    bucket_index = int(time_diff_sec // GRANULARITY_SEC)
    
    if bucket_index >= n_points:
        continue

    sums = snap.get("summaries", {})
    counters = snap.get("counters", {})
    
    # Process response time summaries (MAX aggregation)
    for key, summary in sums.items():
        if key.startswith("plugins.metrics-by-endpoint.response_time."):
            ep = key.replace("plugins.metrics-by-endpoint.response_time.", "")
            if ep in endpoints:
                # We use MAX aggregation for all RT metrics to capture the peak from ANY snapshot in that bucket
                
                # Use 0 if the value is None (null in JSON)
                snap_p50 = summary.get("p50") if summary.get("p50") is not None else 0 # Get P50 value
                snap_mean = summary.get("mean") if summary.get("mean") is not None else 0
                snap_max = summary.get("max") if summary.get("max") is not None else 0
                
                current_avg = ep_pxx[ep]["avg"][bucket_index]
                current_max = ep_pxx[ep]["max"][bucket_index]
                
                # 🛑 CHANGE 2: Use MAX of P50 for the 'avg' series data (used by graphs)
                ep_pxx[ep]["avg"][bucket_index] = max(current_avg, snap_p50)
                
                ep_pxx[ep]["max"][bucket_index] = max(current_max, snap_max)
                
                # P50 is now handled here, due to CHANGE 1 (PERCENTILES_ONLY update)
                for p in PERCENTILES_ONLY: 
                    current_p = ep_pxx[ep][p][bucket_index]
                    snap_p = summary.get(p) if summary.get(p) is not None else 0
                    ep_pxx[ep][p][bucket_index] = max(current_p, snap_p)
                
    # Process status code counters (SUM aggregation)
    for k, v in counters.items():
        if k.startswith("plugins.metrics-by-endpoint."):
            part = k.replace("plugins.metrics-by-endpoint.", "")
            if ".codes." in part:
                ep_part, code = part.split(".codes.")
                if ep_part in endpoints:
                    # Sum counts
                    ep_counts[ep_part][bucket_index] += v
                    ep_codes[ep_part][code][bucket_index] += v
                    ep_total_codes[ep_part][code] += v # For final summary table
                    if code.isdigit() and code.startswith("2"):
                        ep_pass_counts[ep_part][bucket_index] += v
            elif ".errors." in part:
                ep_part, code = part.split(".errors.")
                code_key = f"ERR:{code}"
                if ep_part in endpoints:
                    # Sum counts
                    ep_counts[ep_part][bucket_index] += v
                    ep_codes[ep_part][code_key][bucket_index] += v
                    ep_total_codes[ep_part][code_key] += v # For final summary table

# ---------------- CALCULATE RPS & OVERALL METRICS ----------------
BUCKET_DURATION_FLOAT = float(GRANULARITY_SEC)

# Compute per-endpoint and overall throughput values (requests per second)
ep_rps = {ep: [0] * n_points for ep in endpoints}
ep_pass_rps = {ep: [0] * n_points for ep in endpoints}

for ep in endpoints:
    for i in range(n_points):
        ep_rps[ep][i] = ep_counts[ep][i] / BUCKET_DURATION_FLOAT
        ep_pass_rps[ep][i] = ep_pass_counts[ep][i] / BUCKET_DURATION_FLOAT

overall_rps = [
    sum(ep_counts[ep][i] for ep in endpoints) / BUCKET_DURATION_FLOAT
    for i in range(n_points)
]
overall_pass_rps = [
    sum(ep_pass_counts[ep][i] for ep in endpoints) / BUCKET_DURATION_FLOAT
    for i in range(n_points)
]

# Overall RT metrics are simply the MAX of the already MAX-aggregated endpoint values
overall_p50 = [max(ep_pxx[ep]["p50"][i] for ep in endpoints) for i in range(n_points)] # Now correctly populated and available
overall_p90 = [max(ep_pxx[ep]["p90"][i] for ep in endpoints) for i in range(n_points)]
overall_p95 = [max(ep_pxx[ep]["p95"][i] for ep in endpoints) for i in range(n_points)] 
overall_p99 = [max(ep_pxx[ep]["p99"][i] for ep in endpoints) for i in range(n_points)]
overall_max = [max(ep_pxx[ep]["max"][i] for ep in endpoints) for i in range(n_points)] 
overall_avg = [max(ep_pxx[ep]["avg"][i] for ep in endpoints) for i in range(n_points)] # This now holds MAX P50 due to CHANGE 2

# ---------------- HEADER VARIABLES ----------------
if time_points and len(time_points) > 1:
    start_dt = time_points[0].strftime("%Y-%m-%d %H:%M:%S %z")
    end_dt = time_points[-1].strftime("%Y-%m-%d %H:%M:%S %z")
    test_duration_sec = (time_points[-1] - time_points[0]).total_seconds()
    test_duration = str(timedelta(seconds=round(test_duration_sec)))
else:
    start_dt = "N/A"
    end_dt = "N/A"
    test_duration = "0s"
    test_duration_sec = 0

# ---------------- OVERALL SINGLE-DIGIT METRICS ----------------
# Calculate single-digit metrics based on final aggregates (agg_summaries)

# 1. ALL REQUESTS (Max of all endpoint aggregates for percentiles)
total_weighted_sum_rt = 0
overall_total_count = 0
overall_max_p50 = 0 # 🛑 CHANGE 3: Initialize P50 max
overall_max_p90 = 0
overall_max_p95 = 0
overall_max_max_rt = 0

for ep, summary in agg_summaries.items():
    
    if summary["count"] > 0:
        # Use 0 if mean/p50 is None (null)
        mean_val = summary.get("mean") if summary.get("mean") is not None else 0
        p50_val = summary.get("p50") if summary.get("p50") is not None else 0 # Retrieve P50
        p90_val = summary.get("p90") if summary.get("p90") is not None else 0
        p95_val = summary.get("p95") if summary.get("p95") is not None else 0
        max_val = summary.get("max") if summary.get("max") is not None else 0
        
        # Weighted mean calculation (only kept for internal logic, not displayed as 'Avg')
        if ep.startswith("plugins.metrics-by-endpoint.response_time."):
            total_weighted_sum_rt += mean_val * summary["count"]
            overall_total_count += summary["count"]
        
        # For worst-case percentiles and max (max of ALL summary maxes/percentiles, including error groups)
        overall_max_p50 = max(overall_max_p50, p50_val) # 🛑 CHANGE 3: Max aggregation for P50
        overall_max_p90 = max(overall_max_p90, p90_val)
        overall_max_p95 = max(overall_max_p95, p95_val)
        overall_max_max_rt = max(overall_max_max_rt, max_val)

# Fallback count (if endpoint-specific counts were 0 but global counts were present)
if overall_total_count == 0:
     overall_total_count = agg_summaries.get("http.response_time.all", {}).get("count", 0)

overall_avg_rt_ms = (total_weighted_sum_rt / overall_total_count) if overall_total_count > 0 else 0
#overall_max_rps = max(overall_rps) if overall_rps else 0
overall_max_rps = round(overall_max_instant_rps, 1) # Use the instantaneous max rate
overall_avg_rps = (overall_total_count / test_duration_sec) if test_duration_sec > 0 else 0
total_tx = overall_total_count

# 2. SUCCESSFUL REQUESTS (2xx Only - New Logic)
SUCCESS_SUMMARY_KEY = "http.response_time.2xx"
success_summary = agg_summaries.get(SUCCESS_SUMMARY_KEY, {})

# Safely retrieve values, defaulting to 0 if key not found or value is None
success_p50_rt = success_summary.get("p50") if success_summary.get("p50") is not None else 0 # Retrieve P50
success_p90_rt = success_summary.get("p90") if success_summary.get("p90") is not None else 0
success_p95_rt = success_summary.get("p95") if success_summary.get("p95") is not None else 0
success_max_rt = success_summary.get("max") if success_summary.get("max") is not None else 0
success_count = success_summary.get("count", 0)

if success_count == 0:
    # If no 2xx metric found or count is zero, default to "N/A"
    success_avg_rt_label = "N/A"
    success_p90_rt_label = "N/A"
    success_p95_rt_label = "N/A"
    success_max_rt_label = "N/A"
else:
    # 🛑 CHANGE 4: Use P50 for the 'Avg RT' label
    success_avg_rt_label = f"{success_p50_rt:.1f} ms"
    success_p90_rt_label = f"{success_p90_rt:.1f} ms"
    success_p95_rt_label = f"{success_p95_rt:.1f} ms"
    success_max_rt_label = f"{success_max_rt:.1f} ms"

# 3. Compile final metrics dictionary
OVERALL_METRICS = {
    # All Requests (Existing)
    # 🛑 CHANGE 4: Use Max P50 for Overall Avg RT
    "All Req. Avg RT (ms)": round(overall_max_p50, 1),
    "All Req. P90 RT (ms)": round(overall_max_p90, 1),
    "All Req. P95 RT (ms)": round(overall_max_p95, 1),
    "All Req. Max RT (ms)": round(overall_max_max_rt, 1),
    
    # Successful Requests (New)
    # 🛑 CHANGE 4: Use 2xx P50 for 2xx Avg RT
    "2xx Avg RT (ms)": success_avg_rt_label,
    "2xx P90 RT (ms)": success_p90_rt_label,
    "2xx P95 RT (ms)": success_p95_rt_label,
    "2xx Max RT (ms)": success_max_rt_label,
    "2xx Total Requests": success_count,

    # Throughput & Total (Shared)
    "Total Requests": overall_total_count,
    "Max RPS": round(overall_max_rps, 1),
    "Avg RPS (Throughput)": round(overall_avg_rps, 1),
}


# ---------------- SUMMARY TABLE ----------------
def extract_core_service_name(ep_name):
    """Heuristic: extract a simple service name from the endpoint hostname."""
    name = ep_name.replace("http://", "").replace("https://", "")
    hostname = name.split("/")[0] if "/" in name else name
    match = re.match(r"([a-zA-Z]+)(?:[-.0-9a-zA-Z]*)\.", hostname)
    if match:
        return match.group(1)
    return hostname.split('.')[0].split('-')[0]

def get_sla_for_endpoint(ep_name, sla_dict, default_sla=DEFAULT_SLA, default_tph=DEFAULT_TPH):
    """Resolve SLA (ms) and expected TPH for an endpoint using fallback strategies."""
    if not sla_dict:
        return default_sla, default_tph
    canon_ep_name = re.sub(r"/\{\{.*?\}\}", "", ep_name).rstrip("/")
    if canon_ep_name in sla_dict:
        return tuple(sla_dict[canon_ep_name])
    for key in sla_dict:
        if canon_ep_name == key.rstrip("/"):
            return tuple(sla_dict[key])
    ep_service_name = extract_core_service_name(ep_name)
    for key, val in sla_dict.items():
        if extract_core_service_name(key) == ep_service_name:
            return tuple(val)
    return default_sla, default_tph

summary_rows = []
for ep in endpoints:
    agg_summary_key = f"plugins.metrics-by-endpoint.response_time.{ep}"
    agg_summary = agg_summaries.get(agg_summary_key, {})
    total_tx_count_agg = agg_summary.get("count", sum(ep_counts.get(ep, [0]))) 
    
    # Safely retrieve values, treating None as 0
    mean_val = agg_summary.get("mean") if agg_summary.get("mean") is not None else 0
    min_val = agg_summary.get("min") if agg_summary.get("min") is not None else 0
    max_val = agg_summary.get("max") if agg_summary.get("max") is not None else 0
    p50_val = agg_summary.get("p50") if agg_summary.get("p50") is not None else 0 # Retrieve P50
    p90_val = agg_summary.get("p90") if agg_summary.get("p90") is not None else 0
    p95_val = agg_summary.get("p95") if agg_summary.get("p95") is not None else 0
    p99_val = agg_summary.get("p99") if agg_summary.get("p99") is not None else 0

    if total_tx_count_agg > 0:
         # Use the sum of bucketed pass counts if aggregate count is zero (fallback)
         pass_count_agg = sum(v for k, v in agg_counters.items() if k.startswith(f"plugins.metrics-by-endpoint.{ep}.codes.") and k.endswith((".200", ".201", ".202", ".203", ".204", ".205", ".206", ".207", ".208", ".226")))
         fail_count_agg = total_tx_count_agg - pass_count_agg
    else:
        pass_count_agg = sum(ep_pass_counts.get(ep, [0]))
        fail_count_agg = sum(ep_counts.get(ep, [0])) - pass_count_agg
        total_tx_count_agg = sum(ep_counts.get(ep, [0]))
        
    row = {
        "Transaction": ep,
        # 🛑 CHANGE 5: Use P50 for the 'Avg (ms)' column in the summary table
        "Avg (ms)": round(p50_val, 2),
        "Min (ms)": min_val if min_val != float("inf") else 0,
        "Max (ms)": round(max_val, 2),
        "P50": round(p50_val, 2), # Add P50 explicitly if desired (optional)
        "P90": round(p90_val, 2),
        "P95": round(p95_val, 2),
        "P99(ms)": round(p99_val, 2),
        "Count": total_tx_count_agg,
        "Pass_Count": pass_count_agg,
        "Fail_Count": fail_count_agg
    }

    if SLA_ENABLED:
        sla_val, tph_val = get_sla_for_endpoint(ep, sla_data)
        row["SLA(ms)"] = sla_val
        row["Expected_TPH"] = tph_val
        # SLA Compliance Check for P95
        p95_check_val = row.get("P95", 0)
        if total_tx_count_agg == 0:
            row["SLA P95 Status"] = "N/A (No transactions)"
        elif p95_check_val == 0:
            row["SLA P95 Status"] = "N/A (No RT data)"
        elif p95_check_val <= sla_val:
            row["SLA P95 Status"] = "Met"
        else:
            row["SLA P95 Status"] = "Not Met"
            
    summary_rows.append(row)

# Define column order for the summary table
columns = ["Transaction"] 
if SLA_ENABLED: columns += ["SLA(ms)"]
# Columns now reflect P50 being used as the 'Avg (ms)'
columns += ["Avg (ms)", "Min (ms)", "Max (ms)", "P90", "P95", "P99(ms)"]
if SLA_ENABLED: columns += ["Expected_TPH"]
columns += ["Count", "Pass_Count", "Fail_Count"]
if SLA_ENABLED: columns += ["SLA P95 Status"]
summary_df = pd.DataFrame(summary_rows, columns=columns).sort_values(by="Transaction")
#print("\n--- Summary DataFrame preview ---")
#print(summary_df.head(20).to_string(index=False))

# ---------------- ERROR DETAIL TABLE ----------------
error_rows = []
for ep in endpoints:
    for code, count in ep_total_codes[ep].items():
        if not (code.isdigit() and code.startswith("2")) and count > 0:
            error_rows.append({"URL": ep, "Status Code": code, "Count": count})
error_df = pd.DataFrame(error_rows, columns=["URL", "Status Code", "Count"]).sort_values(by=["URL", "Status Code"], ascending=[True, False])
#print("\n--- Error Detail DataFrame preview ---")
#print(error_df.head(20).to_string(index=False))

# ---------------- PREPARE PER-ENDPOINT DATA FOR JS ----------------
def safe_id(s):
    """Return a filesystem/HTML safe ID for endpoint names."""
    return "_" + re.sub(r"[^0-9a-zA-Z]+", "_", s)

per_ep_data = {}
donut_codes_total = defaultdict(int)
for ep in endpoints:
    sid = safe_id(ep)
    summary_row = next((row for row in summary_rows if row["Transaction"] == ep), None)
    total = summary_row["Count"] if summary_row else 0
    pass_count_total_calc = summary_row["Pass_Count"] if summary_row else 0
    pass_pct_value = (pass_count_total_calc / total) * 100 if total > 0 else 0.0
    err_pct_value = 100.0 - pass_pct_value if total > 0 else 0.0

    # Total status code count for donut chart is still based on the final aggregates
    for code, counts in ep_total_codes[ep].items():
        donut_codes_total[code] += counts
        
    per_ep_data[ep] = {
        # 'spark_y' and 'rt_y' (which are the chart lines for 'Avg') now use the P50 values from ep_pxx['avg'] (due to CHANGE 2)
        "spark_x": x_labels,
        "spark_y": ep_pxx[ep]["avg"],
        "rt_x": x_labels,
        "rt_y": ep_pxx[ep]["avg"],
        "p90": ep_pxx[ep]["p90"],
        "p95": ep_pxx[ep]["p95"],
        "p99": ep_pxx[ep]["p99"],
        "max": ep_pxx[ep]["max"], # Include max for per-endpoint detail
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


config_yaml = {}
if INPUT_YAML and Path(INPUT_YAML).exists():
    try:
        with open(INPUT_YAML,"r",encoding="utf-8") as f:
            config_yaml = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️ Could not read YAML config: {e}")
            
# ---------------- PHASES --------------------------------
yaml_phases = config_yaml.get("config", {}).get("phases") if isinstance(config_yaml, dict) else None
phases_data = []
cumulative_duration_sec = 0

if yaml_phases and isinstance(yaml_phases, list) and INPUT_YAML:
    for ph in yaml_phases:
        duration = ph.get("duration", 0)
        phase_start_dt = start_time_aligned + timedelta(seconds=cumulative_duration_sec)
        cumulative_duration_sec += duration
        phase_end_dt = start_time_aligned + timedelta(seconds=cumulative_duration_sec)
        phases_data.append({
            "name": ph.get("name", "Phase"),
            "duration": duration,
            "arrival": ph.get("arrivalRate", ph.get("arrivalCount", 0)),
            "start_dt": phase_start_dt.isoformat(timespec="seconds"),
            "end_dt": phase_end_dt.isoformat(timespec="seconds"),
            "start_sec": cumulative_duration_sec - duration,
            "end_sec": cumulative_duration_sec
        })
elif time_points and len(time_points) > 1:
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


# ---------------- EXTEND LAST PHASE TO END OF TEST DATA ----------------
if phases_data:
    # Get the true end time of the test run from the calculated duration
    last_elapsed_sec = test_duration_sec # Already calculated from time_points
    last_timestamp_dt = time_points[-1] if time_points else None
    
    # Check if the calculated end time is later than the last phase's end time
    if last_timestamp_dt and last_elapsed_sec > phases_data[-1]['end_sec']:
        
        last_phase = phases_data[-1]
        
        # 🛑 FIX: Overwrite the last phase's boundaries to the test's absolute end
        last_phase_start_sec = last_phase['start_sec']
        
        # Update elapsed seconds
        last_phase['end_sec'] = last_elapsed_sec
        # Update datetime string (using the same ISO format as your current code)
        last_phase['end_dt'] = last_timestamp_dt.isoformat(timespec="seconds") 
        
        # Update the duration metric to reflect the extended time
        last_phase['duration'] = round(last_elapsed_sec - last_phase_start_sec)
        
# ---------------- JSON strings for embedding ----------------
js_PER_EP = json.dumps(per_ep_data)
js_OVERALL_RPS = json.dumps({"x": x_labels, "y": overall_rps, "ep_data": ep_rps})
js_OVERALL_PASS_RPS = json.dumps({"x": x_labels, "y": overall_pass_rps})
js_OVERALL_P90 = json.dumps({"x": x_labels, "y": overall_p90, "ep_data": {ep: ep_pxx[ep]['p90'] for ep in endpoints}})
js_OVERALL_P95 = json.dumps({"x": x_labels, "y": overall_p95, "ep_data": {ep: ep_pxx[ep]['p95'] for ep in endpoints}})
js_OVERALL_AVG = json.dumps({"x": x_labels, "y": overall_avg, "ep_data": {ep: ep_pxx[ep]['avg'] for ep in endpoints}})
js_EP_RPS = json.dumps({"x": x_labels, "ep_data": ep_rps, "ep_pass_data": ep_pass_rps})
js_DONUT = json.dumps({"labels": donut_labels, "values": donut_values})
js_PHASES = json.dumps(phases_data)
print(json.dumps(phases_data, indent=4))
js_TIME_AXIS_TYPE = 'date' if TIME_AXIS_MODE == 'datetime' else 'linear'
js_TIME_AXIS_MODE_PY = TIME_AXIS_MODE 

# New JSON data for combined RT graph
js_OVERALL_ALL_RT = json.dumps({ 
    "x": x_labels, 
    "avg": overall_avg, 
    "p90": overall_p90, 
    "p95": overall_p95, 
    "p99": overall_p99, 
    "max": overall_max 
})

# Expose granularity to JS (seconds) for elapsed mode
js_GRANULARITY_SEC = GRANULARITY_SEC

# Define X-axis title based on mode
js_X_AXIS_TITLE = 'Time' if TIME_AXIS_MODE == 'datetime' else 'Elapsed Seconds'


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
    .table th, .table td {{ text-align:center; vertical-align: middle; }}
    .table th:first-child, .table td:first-child {{ text-align:left !important; }}
    .sparkline{{width:180px;height:36px;display:inline-block;}}
    .chart-controls {{ display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 1rem; padding: 10px 0; border-top: 1px solid #eee; border-bottom: 1px solid #eee; }}
    .transaction-item {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 15px; margin: 5px 0; border-radius: 8px; }}
    .transaction-item-even {{ background-color: #fcfcfc; }}
    .transaction-item-odd {{ background-color: #f2f2f2; }}
    .transaction-name {{ font-size: 1rem; font-weight: 500; flex-grow: 1; margin-right: 15px; }}
    .transaction-badges {{ display: flex; gap: 8px; align-items: center; }}
    .details-button {{ min-width: 80px; background: none !important; border: none !important; color: #0d6efd !important; text-align: right; padding: 0; font-weight: 500; }}
    .collapse-graphs {{ padding-top: 15px; padding-bottom: 5px; margin-bottom: 0; }}
    .key-metric-label {{ font-size: 0.8rem; font-weight: 500; color: #6c757d; text-align: left; }}
    .key-metric-value {{ font-size: 1.3rem; font-weight: 700; color: #212529; line-height: 1.1; text-align: left; }}
    .key-metric-wrapper {{ border-left: 1px solid #eee; padding-left: 15px; }}
    .key-metric-wrapper:first-child {{ border-left: none; padding-left: 0; }}
</style>
</head>
<body>

<div class='container-fluid'>

    <div class='card'><div class='card-body'>
        <h3 class='mb-4'>🚀 Artillery Performance Report {' - ' + (TEST_NAME or 'N/A')} </h3>

        <div class='row mb-3'>
            <div class='col-md-3'><strong>App Name:</strong><div>{APP_NAME or 'N/A'}</div></div>
            <div class='col-md-3'><strong>Start:</strong><div>{start_dt}</div></div>
            <div class='col-md-3'><strong>End:</strong><div>{end_dt}</div></div>
            <div class='col-md-3'><strong>Duration:</strong><div>{test_duration}</div></div>
        </div>
        <div class='row'>
            <div class='col-md-3'><strong>Total Transactions:</strong><div>{total_tx}</div></div>
            <div class='col-md-9'></div>
        </div>
    </div></div>
    
    {f'''
    <div class='card mt-3'><div class='card-body'>
        <h5 class='mb-3'>📊 Overall Single-Digit Metrics (Test Summary)</h5>

        <h6 class='mt-4 text-primary'>All Requests (Max P-values Across All Endpoints/Errors)</h6>
        <div class='row g-3'>
            <div class='col-md-3 col-sm-6 key-metric-wrapper'>
                <div class='key-metric-label'>Overall Avg RT (Weighted Mean)</div>
                <div class='key-metric-value'>{OVERALL_METRICS.get("All Req. Avg RT (ms)", 0)} ms</div>
            </div>
            <div class='col-md-3 col-sm-6 key-metric-wrapper'>
                <div class='key-metric-label'>Overall P90 RT (Worst Case)</div>
                <div class='key-metric-value'>{OVERALL_METRICS.get("All Req. P90 RT (ms)", 0)} ms</div>
            </div>
            <div class='col-md-3 col-sm-6 key-metric-wrapper'>
                <div class='key-metric-label'>Overall P95 RT (Worst Case)</div>
                <div class='key-metric-value'>{OVERALL_METRICS.get("All Req. P95 RT (ms)", 0)} ms</div>
            </div>
            <div class='col-md-3 col-sm-6 key-metric-wrapper'>
                <div class='key-metric-label'>Max RT (Test Max)</div>
                <div class='key-metric-value'>{OVERALL_METRICS.get("All Req. Max RT (ms)", 0)} ms</div>
            </div>
        </div>

        <h6 class='mt-4 text-success'>Successful Requests (2xx Only)</h6>
        <div class='row g-3'>
            <div class='col-md-3 col-sm-6 key-metric-wrapper'>
                <div class='key-metric-label'>2xx Avg RT (P50)</div>
                <div class='key-metric-value'>{OVERALL_METRICS.get("2xx Avg RT (ms)", "N/A")}</div>
            </div>
            <div class='col-md-3 col-sm-6 key-metric-wrapper'>
                <div class='key-metric-label'>2xx P90 RT</div>
                <div class='key-metric-value'>{OVERALL_METRICS.get("2xx P90 RT (ms)", "N/A")}</div>
            </div>
            <div class='col-md-3 col-sm-6 key-metric-wrapper'>
                <div class='key-metric-label'>2xx P95 RT</div>
                <div class='key-metric-value'>{OVERALL_METRICS.get("2xx P95 RT (ms)", "N/A")}</div>
            </div>
            <div class='col-md-3 col-sm-6 key-metric-wrapper'>
                <div class='key-metric-label'>2xx Max RT</div>
                <div class='key-metric-value'>{OVERALL_METRICS.get("2xx Max RT (ms)", "N/A")}</div>
            </div>
        </div>

        <h6 class='mt-4 text-secondary'>Throughput</h6>
        <div class='row g-3'>
            <div class='col-md-3 col-sm-6 key-metric-wrapper'>
                <div class='key-metric-label'>Total Requests</div>
                <div class='key-metric-value'>{OVERALL_METRICS.get("Total Requests", 0)}</div>
            </div>
            <div class='col-md-3 col-sm-6 key-metric-wrapper'>
                <div class='key-metric-label'>Max RPS (Peak Rate)</div>
                <div class='key-metric-value'>{OVERALL_METRICS.get("Max RPS", 0)}</div>
            </div>
            <div class='col-md-3 col-sm-6 key-metric-wrapper'>
                <div class='key-metric-label'>Avg RPS (Throughput)</div>
                <div class='key-metric-value'>{OVERALL_METRICS.get("Avg RPS (Throughput)", 0)}</div>
            </div>
        </div>
    </div></div>
    ''' if SHOW_OVERALL_METRICS else ''}
    
        <div class='card mt-3'><div class='card-body'>
        <h5 class='mb-3'>📋 Transaction Summary Table</h5>
        {summary_df.to_html(classes='table table-striped table-bordered', index=False, escape=False)}
    </div></div>
    
    <div class='row mt-3'>
    
    <div class='col-md-6'>
    
    <div class='card mt-3'><div class='card-body'>
        <h5 class='mb-3'>🚨 Error Details Table</h5>
        <p class='text-muted'>Non-2xx Status Codes and Artillery Errors aggregated across all endpoints.</p>
        {error_df.to_html(classes='table table-striped table-bordered', index=False, escape=False)}
    </div></div>
    
    </div>

    <div class='col-md-6'>
    {f'''
    <div class='card mt-3'><div class='card-body'>
        <h5 class='mb-3'>🍩 Overall Status Code Breakdown</h5>
        <div id='donut' style='height: 450px;'></div>
    </div></div>
    ''' if SHOW_GRAPH_DONUT else ''}

    </div>
    

    
    <div class='card mt-3'><div class='card-body'>
        <h5 class='mb-3'>📈 Overall Time-Series Graphs</h5>
        
        <div class="chart-controls">
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="checkbox" id="checkRPS" {'checked' if SHOW_GRAPH_RPS else ''}>
                <label class="form-check-label" for="checkRPS">Requests Per Second (RPS)</label>
            </div>
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="checkbox" id="checkAllRT" {'checked' if SHOW_GRAPH_ALL_RT else ''}>
                <label class="form-check-label" for="checkAllRT">All RT Percentiles (Avg/P90/P95/P99/Max)</label>
            </div>
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="checkbox" id="checkCombined" {'checked' if SHOW_GRAPH_COMBINED else ''}>
                <label class="form-check-label" for="checkCombined">RPS vs P90 (Dual Axis)</label>
            </div>
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="checkbox" id="checkRPSvsP95" {'checked' if SHOW_GRAPH_RPS_VS_P95 else ''}>
                <label class="form-check-label" for="checkRPSvsP95">RPS vs P95 (Dual Axis)</label>
            </div>
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="checkbox" id="checkPhase" checked>
                <label class="form-check-label" for="checkPhase">Show Phases</label>
            </div>
        </div>
        
        <div id='chart_rps' style='height: 350px; display: {'block' if SHOW_GRAPH_RPS else 'none'};'></div>
        <div id='chart_overall_rt_combined' style='height: 350px; display: {'block' if SHOW_GRAPH_ALL_RT else 'none'};'></div>
        <div id='chart_combined' style='height: 350px; display: {'block' if SHOW_GRAPH_COMBINED else 'none'};'></div>
        <div id='chart_rps_vs_p95' style='height: 350px; display: {'block' if SHOW_GRAPH_RPS_VS_P95 else 'none'};'></div>
    </div></div>
    

    
    
    
    <div class='card mt-3'><div class='card-body'>
        <h5 class='mb-3'>🔗 Per-Transaction Details (Accordion)</h5>
        
        <div class="accordion" id="endpointAccordion">
        {''.join(f'''
            <div class="accordion-item transaction-item transaction-item-{'even' if i % 2 == 0 else 'odd'}">
                <h2 class="accordion-header transaction-name" id="heading_{per_ep_data[ep]['safe_id']}">
                    {ep}
                </h2>
                <div class="transaction-badges">
                    <span class="badge bg-primary text-light me-2">Total: {per_ep_data[ep]['total']}</span>
                    <span class="badge bg-success me-2">Pass: {per_ep_data[ep]['pass_pct']}%</span>
                    <span class="badge bg-danger">Fail: {per_ep_data[ep]['err_pct']}%</span>
                    <div id="spark_{per_ep_data[ep]['safe_id']}" class="sparkline me-3"></div>
                    <button class="btn btn-sm details-button" type="button" data-bs-toggle="collapse" 
                            data-bs-target="#collapse_{per_ep_data[ep]['safe_id']}" 
                            aria-expanded="false" 
                            aria-controls="collapse_{per_ep_data[ep]['safe_id']}"
                            data-ep-id="{ep}">
                        Details »
                    </button>
                </div>
            </div>
            <div id="collapse_{per_ep_data[ep]['safe_id']}" class="accordion-collapse collapse collapse-graphs" 
                 aria-labelledby="heading_{per_ep_data[ep]['safe_id']}" data-bs-parent="#endpointAccordion">
                <div class="accordion-body">
                
                    <div class="row">
                        <div class="col-md-6">
                            <h6 class='mb-3'>Requests Per Second (RPS)</h6>
                            <div id='chart_rps_{per_ep_data[ep]['safe_id']}' style='height: 300px;'></div>
                        </div>
                        <div class="col-md-6">
                            <h6 class='mb-3'>Response Time Percentiles (Avg is P50)</h6>
                            <div id='chart_rt_{per_ep_data[ep]['safe_id']}' style='height: 300px;'></div>
                        </div>
                    </div>
                
                    <div class="row mt-4">
                        <div class="col-12">
                            {(
                                 f"""
                                 <h6 class='mb-3'>Status Code Counts</h6>
                                 <div id='chart_codes_{per_ep_data[ep]['safe_id']}' style='height: 300px;'></div>
                                 """
                             ) if SHOW_GRAPH_CODES else ''}
                        </div>
                    </div>
                    
                </div>
            </div>
        ''' for i, ep in enumerate(endpoints))}
        </div>
    </div></div>

</div> <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>

<script>

// ----------------------------------------------------------------------------------
// PYTHON DATA PASSED TO JAVASCRIPT
// ----------------------------------------------------------------------------------
const PER_EP = {js_PER_EP};
const OVERALL_RPS_DATA = {js_OVERALL_RPS};
const OVERALL_PASS_RPS_DATA = {js_OVERALL_PASS_RPS};
const OVERALL_P90_DATA = {js_OVERALL_P90};
const OVERALL_P95_DATA = {js_OVERALL_P95};
const OVERALL_AVG_DATA = {js_OVERALL_AVG}; // Contains MAX P50 for time-series
const OVERALL_ALL_RT_DATA = {js_OVERALL_ALL_RT};
const EP_RPS_DATA = {js_EP_RPS};
const DONUT = {js_DONUT};
const PHASES = {js_PHASES};
const TIME_AXIS_TYPE = '{js_TIME_AXIS_TYPE}';
const TIME_AXIS_MODE_PY = '{js_TIME_AXIS_MODE_PY}';
const GRANULARITY_SEC = {js_GRANULARITY_SEC};
const X_AXIS_TITLE = '{js_X_AXIS_TITLE}';

console.log('TIME_AXIS_MODE_PY:', TIME_AXIS_MODE_PY); // Should be 'elapsed' or 'datetime'
console.log('GRANULARITY_SEC:', GRANULARITY_SEC); // Should be 'elapsed' or 'datetime'
    

// ----------------------------------------------------------------------------------
// UTILITY FUNCTIONS (JS)
// ----------------------------------------------------------------------------------

/**
 * Creates phase shape annotations for Plotly graphs based on time axis mode.
 * @param {{string}} graphId - The ID of the Plotly div to attach phases to.
 * @returns {{Array}} An array of Plotly shape objects.
 */
function getPhaseShapes(graphId) {{
    const plot = document.getElementById(graphId);
    if (!plot) return {{shapes: [], annotations: []}}; 

    const showPhaseCheckbox = document.getElementById('checkPhase');
    if (!showPhaseCheckbox || !showPhaseCheckbox.checked) {{
        return {{shapes: [], annotations: []}};
    }}

    const annotations = [];
    const shapes = [];
    const height = 1.0; 

    PHASES.forEach(phase => {{
        let x0, x1;
        let annotationX; // 🛑 RESTORED: This variable is required for label alignment
        let xAnchor;

        // Determine X-coordinates and label position based on mode
        if (TIME_AXIS_MODE_PY === 'datetime') {{
            x0 = phase.start_dt;
            x1 = phase.end_dt;
            annotationX = x0; 
            xAnchor = 'left';
        }} else {{
            x0 = phase.start_sec;
            x1 = phase.end_sec;
            annotationX = (x0 + x1) / 2; // Calculate center point
            xAnchor = 'center';
        }}

        // 1. Phase vertical strip (shaded background)
        shapes.push({{
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: x0,
            y0: 0,
            x1: x1,
            y1: height,
            fillcolor: 'rgba(173, 216, 230, 0.15)',
            line: {{ width: 0 }},
            layer: 'below'
        }});
        
        // 2. Phase Demarcation Lines (Dotted)
        shapes.push({{
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: x0,
            y0: 0,
            x1: x0,
            y1: height,
            line: {{
                color: 'rgba(0, 0, 0, 0.4)',
                width: 1,
                dash: 'dot' 
            }}
        }});

        // 3. Phase Name Annotation (Simple Name)
        // 🛑 FIX: Use $${{{...}}} to correctly escape the JS template literal variable with dot-notation
        const phaseNameText = `<b>${{phase.name}}</b>`; 

        annotations.push({{
            x: annotationX,              // Use calculated X-coordinate
            y: 0.97, // Lower than 1.05 to prevent clutter with chart title
            xref: 'x',
            yref: 'paper',
            text: phaseNameText,
            showarrow: false,
            xanchor: xAnchor,
            yanchor: 'bottom',
            font: {{ size: 10, color: '#333' }},
            textangle: 0
        }});

        // 4. Phase Details Annotation (Subtle, secondary line)
        annotations.push({{
            x: annotationX,              
            y: 0.92,
            xref: 'x',
            yref: 'paper',
            // 🛑 FIX: Use $${{{...}}} for f-string compatibility
            text: `(Rate: ${{phase.arrival}} / ${{phase.duration}}s)`, 
            showarrow: false,
            xanchor: xAnchor,
            yanchor: 'bottom',
            font: {{ size: 8, color: 'rgba(51, 51, 51, 0.7)' }},
            textangle: 0
        }});
    }});

    return {{shapes: shapes, annotations: annotations}};
}}

// ----------------------------------------------------------------------------------
// CORE DRAWING FUNCTIONS (PLOTLY)
// ----------------------------------------------------------------------------------

/**
 * Generic function to draw a simple line graph.
 */
function drawSingleYAxisGraph(graphId, title, dataSet, yAxisTitle, phaseTitle) {{
    // 🛑 CRASH FIX: Check for undefined/null dataSet and its required properties.
    if (!dataSet || !dataSet.x || dataSet.x.length === 0) {{
        console.warn(`[${{graphId}}] Skipping Single Y-Axis Graph: Data set is missing or empty.`);
        return;
    }}
    
    const traces = [];
    
    // Add main line (Overall)
    traces.push({{
        x: dataSet.x,
        y: dataSet.y,
        mode: 'lines',
        name: yAxisTitle,
        line: {{color: '#007bff'}}
    }});
    
    const isElapsed = TIME_AXIS_MODE_PY === 'elapsed';
    const phaseData = getPhaseShapes(graphId);
    const xAxisDtick = isElapsed ? GRANULARITY_SEC : 1000 * GRANULARITY_SEC;
    
    
    
    const layout = {{
        title: title,
        xaxis: {{
            type: TIME_AXIS_TYPE,
            title: X_AXIS_TITLE,
            tickformat: TIME_AXIS_MODE_PY === 'datetime' ? '%H:%M:%S' : '',
            showgrid: true,
            // 🛑 FIX: Add dtick logic for elapsed/linear mode
            
            dtick: xAxisDtick
            //TIME_AXIS_MODE_PY === 'elapsed' ? GRANULARITY_SEC  : null // Set tick interval to 2x granularity (60s ticks)
        }},
        yaxis: {{
            title: yAxisTitle,
            rangemode: 'tozero'
        }},
        shapes: phaseData.shapes,
        annotations: phaseData.annotations,
        margin: {{t: 40, b: 60, l: 60, r: 20}},
        hovermode: 'x unified',
        legend: {{
            orientation: 'h', // Horizontal orientation
            y: -0.2,          // Position below the chart area (normalized coordinates)
            x: 0.5,           // Center the legend horizontally
            xanchor: 'center' // Anchor the legend to the center point
        }}
    }};

    Plotly.newPlot(graphId, traces, layout, {{responsive: true}});
}}

/**
 * Draws the Overall Combined RT Percentiles graph.
 */
function drawOverallAllRTGraph(graphId, title, dataSet) {{
    // 🛑 CRASH FIX (Final Version): Check for undefined/null dataSet and its required properties.
    // This is the check that fixes the original TypeError on line 752.
    if (!dataSet || !dataSet.x || dataSet.x.length === 0) {{
        console.warn(`[${{graphId}}] Skipping Overall RT Percentiles Graph: Data set is missing or empty.`);
        return;
    }}
    
    const traces = [
        {{ x: dataSet.x, y: dataSet.avg, mode: 'lines', name: 'Avg RT (P50)', line: {{color: '#007bff'}} }},
        {{ x: dataSet.x, y: dataSet.p90, mode: 'lines', name: 'P90 RT', line: {{color: '#ffc107'}} }},
        {{ x: dataSet.x, y: dataSet.p95, mode: 'lines', name: 'P95 RT', line: {{color: '#dc3545'}} }},
        {{ x: dataSet.x, y: dataSet.p99, mode: 'lines', name: 'P99 RT', line: {{color: '#6c757d'}} }},
        {{ x: dataSet.x, y: dataSet.max, mode: 'lines', name: 'Max RT', line: {{color: '#212529', dash: 'dot'}} }}
    ];

    const isElapsed = TIME_AXIS_MODE_PY === 'elapsed';
    
    const phaseData = getPhaseShapes(graphId);
    const xAxisDtick = isElapsed ? GRANULARITY_SEC : 1000 * GRANULARITY_SEC;
    
    const layout = {{
        title: title,
        xaxis: {{
            type: TIME_AXIS_TYPE,
            title: X_AXIS_TITLE,
            tickformat: TIME_AXIS_MODE_PY === 'datetime' ? '%H:%M:%S' : '',
            showgrid: true,
            // 🛑 FIX: Add dtick logic for elapsed/linear mode
            dtick: xAxisDtick
            //TIME_AXIS_MODE_PY === 'elapsed' ? GRANULARITY_SEC  : null // Set tick interval to 2x granularity (60s ticks)
        }},
        yaxis: {{
            title: 'Response Time (ms)',
            rangemode: 'tozero'
        }},
        shapes: phaseData.shapes,
        annotations: phaseData.annotations,
        margin: {{t: 40, b: 60, l: 60, r: 20}},
        hovermode: 'x unified',
        legend: {{
            orientation: 'h', // Horizontal orientation
            y: -0.2,          // Position below the chart area (normalized coordinates)
            x: 0.5,           // Center the legend horizontally
            xanchor: 'center' // Anchor the legend to the center point
        }}
    }};

    Plotly.newPlot(graphId, traces, layout, {{responsive: true}});
}}


/**
 * Generic function to draw a dual Y-axis graph (RPS vs RT).
 * @param {{string}} graphId - The ID of the Plotly div.
 * @param {{string}} title - Chart title.
 * @param {{Object}} rpsData - RPS dataset (x, y, ep_data)
 * @param {{Object}} rtData - RT dataset (x, y, ep_data)
 * @param {{string}} rtMetricName - Name of the RT metric (e.g., 'P90 Response Time (ms)')
 */
function drawDualYAxisGraph(graphId, title, rpsData, rtData, rtMetricName) {{
    // 🛑 CRASH FIX: Check for undefined/null dataSet and its required properties.
    if (!rpsData || !rpsData.x || rpsData.x.length === 0 || !rtData || !rtData.x || rtData.x.length === 0) {{
        console.warn(`[${{graphId}}] Skipping Dual Y-Axis Graph: One or both data sets are missing or empty.`);
        return;
    }}
    
    const traces = [
        // 1. RPS (Primary Y-axis: y1)
        {{
            x: rpsData.x,
            y: rpsData.y,
            mode: 'lines',
            name: 'Overall RPS',
            yaxis: 'y1',
            line: {{color: '#28a745'}} // Green for RPS
        }},
        // 2. RT Metric (Secondary Y-axis: y2)
        {{
            x: rtData.x,
            y: rtData.y,
            mode: 'lines',
            name: `Overall ${{rtMetricName}}`,
            yaxis: 'y2',
            line: {{color: '#dc3545'}} // Red/Danger for RT
        }}
    ];
    
    const isElapsed = TIME_AXIS_MODE_PY === 'elapsed';
    
    const phaseData = getPhaseShapes(graphId);
    const xAxisDtick = isElapsed ? GRANULARITY_SEC : 1000 * GRANULARITY_SEC;
    
    const layout = {{
        title: title,
        xaxis: {{
            type: TIME_AXIS_TYPE,
            title: X_AXIS_TITLE,
            tickformat: TIME_AXIS_MODE_PY === 'datetime' ? '%H:%M:%S' : '',
            showgrid: true,
            // 🛑 FIX: Add dtick logic for elapsed/linear mode
            dtick: xAxisDtick
            //TIME_AXIS_MODE_PY === 'elapsed' ? GRANULARITY_SEC  : null // Set tick interval to 2x granularity (60s ticks)
        }},
        yaxis: {{
            title: 'Requests / sec',
            rangemode: 'tozero',
            color: '#28a745',
            side: 'left'
        }},
        yaxis2: {{
            title: rtMetricName,
            rangemode: 'tozero',
            color: '#dc3545',
            overlaying: 'y',
            side: 'right',
            showgrid: false
        }},
        shapes: phaseData.shapes,
        annotations: phaseData.annotations,
        margin: {{t: 40, b: 60, l: 60, r: 60}},
        hovermode: 'x unified',
        legend: {{
            orientation: 'h', // Horizontal orientation
            y: -0.2,          // Position below the chart area (normalized coordinates)
            x: 0.5,           // Center the legend horizontally
            xanchor: 'center' // Anchor the legend to the center point
        }}
    }};

    Plotly.newPlot(graphId, traces, layout, {{responsive: true}});
}}

/**
 * Draws all charts for a specific endpoint (RPS, RT Percentiles, Status Codes).
 */

/**
 * Draws all charts for a specific endpoint (RPS, RT Percentiles, Status Codes).
 */
function drawEndpointGraphs(ep) {{
    const data = PER_EP[ep];
    const safeId = data.safe_id;

    // CRASH FIX: Check data validity before drawing
    if (!data || !data.rt_x || data.rt_x.length === 0) {{
        console.warn(`[${{ep}}] Skipping Endpoint Graphs: Data is missing or empty.`);
        return;
    }}
    console.log('hello');
    console.log(GRANULARITY_SEC);
    
    const isElapsed = TIME_AXIS_MODE_PY === 'elapsed';

    // 🛑 NEW: Define dtick based on the axis mode
    const xAxisDtick = isElapsed ? GRANULARITY_SEC : 1000 * GRANULARITY_SEC; // 120 seconds for elapsed, Set tick interval to 5 minutes (in milliseconds) # 60 seconds * 1000 milliseconds/second * 5 minutes
    const xAxisTickFormat = isElapsed ? '' : '%H:%M:%S';
    
    // ----------------------------------------------------
    // Layout template for all charts (with new adjustments)
    // ----------------------------------------------------
    const baseLayout = {{
        // 🛑 FIX 1: Smaller title font size
        title: {{ 
            text: '', // Text will be set individually
            font: {{ size: 14 }} // Smaller font size for clutter reduction
        }},
        // 🛑 FIX 2: Increased bottom margin for better separation
        margin: {{t: 40, b: 80, l: 60, r: 20}}, 
        hovermode: 'x unified',
        legend: {{
            orientation: 'h', 
            y: -0.3, // 🛑 FIX 2: Pushed legend further down
            x: 0.5, 
            xanchor: 'center' 
        }},
        
        xaxis: {{
            type: TIME_AXIS_TYPE,
            title: X_AXIS_TITLE,
            tickformat: TIME_AXIS_MODE_PY === 'datetime' ? '%H:%M:%S' : '',
            showgrid: true,
            // 🛑 FIX: Add dtick logic for elapsed/linear mode
            dtick: xAxisDtick
            //TIME_AXIS_MODE_PY === 'elapsed' ? GRANULARITY_SEC  : null // Set tick interval to 2x granularity (60s ticks)
        }},
        yaxis: {{ rangemode: 'tozero' }},
        // 🛑 FIX 3: Chart Border Shape Definition (Applies to all)
        shapes: [{{
            type: 'rect',
            xref: 'paper',
            yref: 'paper',
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
            line: {{
                color: 'rgba(0, 0, 0, 0.1)', // Light gray border
                width: 1
            }},
            layer: 'below',
            fillcolor: 'transparent'
        }}],
        annotations: [] // Start with empty annotations
    }};
    
    
    // ----------------------------------------------------
    // 1. RPS Graph
    // ----------------------------------------------------
    const rpsTraces = [
        {{ x: data.rt_x, y: data.rps, mode: 'lines', name: 'Total RPS', line: {{color: '#007bff'}} }}, 
        {{ x: data.rt_x, y: data.pass_rps, mode: 'lines', name: 'Successful (2xx) RPS', line: {{color: '#28a745', dash: 'dash'}} }}
    ];
    
    const rpsLayout = Object.assign({{}}, baseLayout, {{
        title: {{ text: `${{ep}} - Requests Per Second`, font: baseLayout.title.font }},
        yaxis: {{ title: 'Requests / sec', rangemode: 'tozero' }},
    }});

    Plotly.newPlot(`chart_rps_${{safeId}}`, rpsTraces, rpsLayout, {{responsive: true}});

    // ----------------------------------------------------
    // 2. RT Percentiles Graph
    // ----------------------------------------------------
    const rtTraces = [
        {{ x: data.rt_x, y: data.rt_y, mode: 'lines', name: 'Avg RT (P50)', line: {{color: '#1f77b4', width: 2}} }}, 
        {{ x: data.rt_x, y: data.p90, mode: 'lines', name: 'P90 RT', line: {{color: '#ff7f0e', width: 1.5}} }}, 
        {{ x: data.rt_x, y: data.p95, mode: 'lines', name: 'P95 RT', line: {{color: '#d62728', width: 1.5}} }}, 
        {{ x: data.rt_x, y: data.p99, mode: 'lines', name: 'P99 RT', line: {{color: '#9467bd', width: 1, dash: 'dot'}} }}, 
        {{ x: data.rt_x, y: data.max, mode: 'lines', name: 'Max RT', line: {{color: '#7f7f7f', width: 1, dash: 'dot'}} }}
    ];

    const rtLayout = Object.assign({{}}, baseLayout, {{
        title: {{ text: `${{ep}} - Response Time Percentiles`, font: baseLayout.title.font }},
        yaxis: {{ title: 'Response Time (ms)', rangemode: 'tozero' }},
    }});

    Plotly.newPlot(`chart_rt_${{safeId}}`, rtTraces, rtLayout, {{responsive: true}});

    // ----------------------------------------------------
    // 3. Status Codes Graph
    // ----------------------------------------------------
    if (document.getElementById(`chart_codes_${{safeId}}`)) {{
        const codeTraces = [];
        const sortedCodes = Object.keys(data.codes).sort();
        
        sortedCodes.forEach(code => {{
            if (data.codes[code].some(c => c > 0)) {{ 
                const traceColor = code.startsWith('2') ? '#28a745' : // Success
                                   code.startsWith('3') ? '#ffc107' : // Redirect
                                   code.startsWith('4') ? '#dc3545' : // Client Error
                                   code.startsWith('5') ? '#6c757d' : // Server Error
                                   '#000000';
                                   
                codeTraces.push({{
                    x: data.rt_x,
                    y: data.codes[code],
                    name: code,
                    type: 'scatter', 
                    mode: 'lines',
                    line: {{ width: 2, color: traceColor }}
                }});
            }}
        }});

        const codesLayout = Object.assign({{}}, baseLayout, {{
            title: {{ text: `${{ep}} - Status Code Counts Over Time`, font: baseLayout.title.font }},
            yaxis: {{ title: 'Transaction Count', rangemode: 'tozero' }},
        }});

        Plotly.newPlot(`chart_codes_${{safeId}}`, codeTraces, codesLayout, {{responsive: true}});
    }}
}}

/**
 * Draws the sparkline chart.
 * @param {{string}} graphId - The ID of the Plotly div.
 * @param {{Object}} dataSet - The data object for the sparkline (x, y).
 * @param {{string}} color - Line color.
 */
function drawSparkline(graphId, dataSet, color) {{
    if (!document.getElementById(graphId)) return;
    
    const maxVal = Math.max(...dataSet.y);
    const yAxisRange = [0, maxVal * 1.2 || 100]; // 20% padding, avoid 0 max

    const trace = {{
        x: dataSet.x,
        y: dataSet.y,
        mode: 'lines',
        line: {{ color: color, width: 1 }},
        hoverinfo: 'none' // Don't show hover on sparkline
    }};

    const layout = {{
        autosize: true,
        margin: {{ l: 0, r: 0, b: 0, t: 0, pad: 0 }},
        xaxis: {{ showgrid: false, zeroline: false, showticklabels: false }},
        yaxis: {{ showgrid: false, zeroline: false, showticklabels: false, range: yAxisRange }},
        plot_bgcolor: '#fcfcfc',
        paper_bgcolor: '#fcfcfc'
    }};

    Plotly.newPlot(graphId, [trace], layout, {{displayModeBar: false}});
}}


// ----------------------------------------------------------------------------------
// INIT
// ----------------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', function () {{

    // Chart visibility toggles
    document.getElementById('checkRPS')?.addEventListener('change', function () {{
        const chartDiv = document.getElementById('chart_rps');
        chartDiv.style.display = this.checked ? 'block' : 'none';
        if (this.checked) Plotly.Plots.resize(chartDiv); // 🛑 FIX
    }});
    document.getElementById('checkAllRT')?.addEventListener('change', function () {{
        const chartDiv = document.getElementById('chart_overall_rt_combined');
        chartDiv.style.display = this.checked ? 'block' : 'none';
        if (this.checked) Plotly.Plots.resize(chartDiv); // 🛑 FIX
    }});
    document.getElementById('checkCombined')?.addEventListener('change', function () {{
        const chartDiv = document.getElementById('chart_combined');
        chartDiv.style.display = this.checked ? 'block' : 'none';
        if (this.checked) Plotly.Plots.resize(chartDiv); // 🛑 FIX
    }});
    document.getElementById('checkRPSvsP95')?.addEventListener('change', function () {{
        const chartDiv = document.getElementById('chart_rps_vs_p95');
    chartDiv.style.display = this.checked ? 'block' : 'none';
    if (this.checked) Plotly.Plots.resize(chartDiv); // 🛑 FIX
    }});
    document.getElementById('checkPhase')?.addEventListener('change', function () {{
        // Re-draw all open Plotly charts to update phases
        Object.keys(PER_EP).forEach(ep => {{
            const collapseElement = document.getElementById(`collapse_${{PER_EP[ep].safe_id}}`);
            if (collapseElement?.classList.contains('show')) {{
                drawEndpointGraphs(ep);
            }}
        }});
        // Re-draw all overall charts
        if (document.getElementById('chart_rps')?.style.display === 'block') drawSingleYAxisGraph('chart_rps', 'Requests Per Second', OVERALL_RPS_DATA, 'Requests / sec');
        if (document.getElementById('chart_overall_rt_combined')?.style.display === 'block') drawOverallAllRTGraph('chart_overall_rt_combined', 'Overall RT Percentiles (Avg is P50)', OVERALL_ALL_RT_DATA); 
        if (document.getElementById('chart_combined')?.style.display === 'block') drawDualYAxisGraph('chart_combined', 'RPS vs Overall P90', OVERALL_RPS_DATA, OVERALL_P90_DATA, 'P90 Response Time (ms)');
        if (document.getElementById('chart_rps_vs_p95')?.style.display === 'block') drawDualYAxisGraph('chart_rps_vs_p95', 'RPS vs Overall P95', OVERALL_RPS_DATA, OVERALL_P95_DATA, 'P95 Response Time (ms)');
    }});


    // 1. Sparklines - Draw on initial load
    Object.keys(PER_EP).forEach(ep => {{
        const s = PER_EP[ep];
        drawSparkline("spark_"+s.safe_id, {{x: s.spark_x, y: s.spark_y}}, '#007bff');
    }});

    // 2. Overall Charts - Draw on initial load if visible (based on PYTHON defaults)
    if (document.getElementById('chart_rps')?.style.display === 'block') {{
        drawSingleYAxisGraph('chart_rps', 'Requests Per Second', OVERALL_RPS_DATA, 'Requests / sec');
    }}
    if (document.getElementById('chart_overall_rt_combined')?.style.display === 'block') {{
        // FIX: The defensive check added above prevents the crash
        drawOverallAllRTGraph('chart_overall_rt_combined', 'Overall RT Percentiles (Avg is P50)', OVERALL_ALL_RT_DATA);
    }}
    if (document.getElementById('chart_combined')?.style.display === 'block') {{
        // RPS vs P90
        drawDualYAxisGraph('chart_combined', 'RPS vs Overall P90', OVERALL_RPS_DATA, OVERALL_P90_DATA, 'P90 Response Time (ms)');
    }}
    if (document.getElementById('chart_rps_vs_p95')?.style.display === 'block') {{
        // RPS vs P95
        drawDualYAxisGraph('chart_rps_vs_p95', 'RPS vs Overall P95', OVERALL_RPS_DATA, OVERALL_P95_DATA, 'P95 Response Time (ms)');
    }}
    
    // 3. Donut Chart
    if (DONUT.labels && DONUT.labels.length>0) {{
        Plotly.newPlot('donut', [{{labels: DONUT.labels, values: DONUT.values, type:'pie', hole:0.6}}], {{title: 'Overall HTTP Status Code Breakdown', margin:{{t:40,b:20,l:20,r:20}}}}, {{responsive:true}});
    }}

    // 4. Per-Endpoint Charts (Accordion)
    document.querySelectorAll('.details-button').forEach(button => {{
        let targetId = button.getAttribute('data-bs-target').replace('#', '');
        let collapseElement = document.getElementById(targetId);
        let epName = button.getAttribute('data-ep-id');

        if (collapseElement) {{
             // The collapse 'shown' event is used to draw the chart once it becomes visible
             collapseElement.addEventListener('shown.bs.collapse', function () {{
                 if (epName) {{
                    drawEndpointGraphs(epName);
                 }}
             }});
        }}
    }});

    // 5. DataTables
    $('table').DataTable({{ paging: true, searching: true, info: true, order: [] }});
}});
</script>
</body>
</html>
"""

# ---------------- OUTPUT HTML FILE ----------------
try:
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✅ Report generated successfully at: {OUTPUT_HTML}")
except Exception as e:
    print(f"\n❌ Error writing output HTML file: {e}")
