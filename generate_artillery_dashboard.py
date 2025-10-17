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

# ---------------- CONFIG ----------------
# --- NEW CONFIG FLAGS ---
# Default axis mode
TIME_AXIS_MODE = "elapsed" # Use "datetime" or "elapsed"

# --- Phase Visibility Logic ---
# 1. Base the flag on the TIME_AXIS_MODE (as requested)
if TIME_AXIS_MODE == "datetime":
    SHOW_PHASE_SHAPES = False
else:
    SHOW_PHASE_SHAPES = True 
# SHOW_PHASE_SHAPES is further updated below based on YAML presence.
# ------------------------

PERCENTILES = ["p90","p95","p99"]
SHOW_GRAPH_RPS = True
SHOW_GRAPH_P90 = True
SHOW_GRAPH_COMBINED = True
SHOW_GRAPH_CODES = True
SHOW_GRAPH_DONUT = True
FILTER_START_SEC = 0
FILTER_END_SEC = None

# ---------------- CLI ARGUMENTS ----------------
parser = argparse.ArgumentParser(description="Merge up to 3 Artillery JSON reports into one HTML dashboard.")
parser.add_argument("--json", required=True, help="Comma-separated list of Artillery JSON report files.")
parser.add_argument("--yaml", required=False, help="Path to Artillery YAML config file.")
parser.add_argument("--sla",type=str,default=None,help="Optional SLA JSON file defining transaction SLA (ms) and Expected_TPH per transaction" )
parser.add_argument("--output", required=True, help="Output HTML file path.")

# Handle argument parsing safely in case of interactive testing
try:
    args = parser.parse_args()
except SystemExit:
    # If no arguments are provided and the script exits, provide a helpful message
    print("\nError: Arguments not provided. Run from command line like:")
    print("python gen_art_dash_multi.py --json report1.json,report2.json --output report.html")
    sys.exit(1)
except:
    # Minimal dummy arguments for demonstration if parsing fails mysteriously
    class DummyArgs:
        json = "dummy.json"
        yaml = None
        output = "output.html"
    args = DummyArgs()
    
# Check if dummy data generation is needed only if files were NOT specified
if args.json == "dummy.json":
    INPUT_JSONS = []
else:
    INPUT_JSONS = [p.strip() for p in args.json.split(",") if p.strip()]

# ---------------- VALIDATE JSON FILES EXISTENCE ----------------
existing_files = [f for f in INPUT_JSONS if Path(f).exists()]
missing_files = [f for f in INPUT_JSONS if not Path(f).exists()]

# Fail only if none of the provided JSON files exist
if not existing_files:
    print("❌ Error: None of the provided JSON files exist.")
    for f in missing_files:
        print(f"   - {f}")
    sys.exit(1)

# Warn and continue if some are missing
if missing_files:
    print("⚠️ Warning: Some JSON files were not found and will be skipped:")
    for f in missing_files:
        print(f"   - {f}")

print("✅ Proceeding with existing JSON files:")
for f in existing_files:
    print(f"   - {f}")

# Update the list to only include existing files
INPUT_JSONS = existing_files

INPUT_YAML = args.yaml
OUTPUT_HTML = args.output

# ---------------- LOAD SLA JSON (Optional) ----------------
INPUT_SLA = args.sla
sla_data = {}

if INPUT_SLA:
    if not Path(INPUT_SLA).exists():
        print(f"⚠️ Warning: SLA file '{INPUT_SLA}' not found. Default SLA values will be used.")
    else:
        try:
            with open(INPUT_SLA, "r") as f:
                sla_data = json.load(f)
            print(f"✅ Loaded SLA definitions from {INPUT_SLA}")
        except json.JSONDecodeError:
            print(f"⚠️ Warning: SLA file '{INPUT_SLA}' is not valid JSON. Using defaults.")
            sla_data = {}
            
# --- PHASE SHAPE OVERRIDE based on YAML presence ---
# If a YAML file path was NOT provided, force the phase flag OFF.
if not INPUT_YAML:
    # This correctly addresses the user's explicit request: 
    # If YML is not provided, the phases flag will be off.
    SHOW_PHASE_SHAPES = False
# ---------------------------------------------------


# ---------------- LOAD DATA ----------------
all_intermediate = []
all_agg_counters = defaultdict(int)
all_agg_summaries = defaultdict(lambda: {"count":0,"mean":0,"min":float("inf"),"max":0,"p90":0,"p95":0,"p99":0})

for json_file in INPUT_JSONS:
    path = Path(json_file)
    if not path.exists():
        continue
    with open(path,"r",encoding="utf-8") as f:
        data = json.load(f)
    intermediate = data.get("intermediate",[])
    all_intermediate.extend(intermediate)
    # Aggregate counters
    for k,v in data.get("aggregate",{}).get("counters",{}).items():
        all_agg_counters[k] += v
    # Aggregate summaries
    for k,s in data.get("aggregate",{}).get("summaries",{}).items():
        existing = all_agg_summaries[k]
        n1,n2 = existing.get("count",0), s.get("count",0)
        if n1+n2>0:
            existing["mean"] = (existing["mean"]*n1 + s.get("mean",0)*n2)/(n1+n2)
            existing["min"] = min(existing["min"], s.get("min",0))
            existing["max"] = max(existing["max"], s.get("max",0))
            for p in PERCENTILES:
                existing[p] = max(existing[p], s.get(p,0))
            existing["count"] = n1+n2

agg_counters = all_agg_counters
agg_summaries = all_agg_summaries
intermediate = all_intermediate

# --- Dummy Data Generation if no real data is loaded ---
if not intermediate:
    if INPUT_JSONS and INPUT_JSONS[0]:
        if any(Path(f).exists() for f in INPUT_JSONS):
             raise SystemExit("❌ No valid data found in provided JSON files.")
    
    print("❌ No valid data found. Generating minimal dummy data for structure.")
    now = datetime.now()
    time_points = [now]
    # Use ISO format for x_labels to match phase data format
    x_labels = [time_points[0].isoformat(timespec='seconds')]
    endpoints = ["test_endpoint"]
    overall_rps = [1]
    overall_p90 = [100]
    overall_avg = [50] 
    donut_labels = []
    donut_values = []
    per_ep_data = {}
    total_duration = 60
    # Use ISO format for phases_data
    phases_data=[{"name":"Default Phase","duration":total_duration,"arrival":0, "start_dt": now.isoformat(timespec='seconds'), "end_dt": (now + timedelta(seconds=total_duration)).isoformat(timespec='seconds'), "start_sec": 0, "end_sec": total_duration}]
    total_tx = 0
    if not intermediate:
        summary_df = pd.DataFrame([{
            "Transaction": "test_endpoint",
            "SLA(ms)": DEFAULT_SLA,
            "Avg (ms)": 0,
            "Min (ms)": 0,
            "Max (ms)": 0,
            "P90": 0,
            "P95": 0,
            "P99(ms)": 0,
            "Expected_TPH": DEFAULT_TPH,
            "Count": 0,
            "Pass_Count": 0,
            "Fail_Count": 0
        }])
    error_df = pd.DataFrame()
    
    ep_p90 = {"test_endpoint": [90]}
    ep_avg = {"test_endpoint": [45]}
    ep_rps = {"test_endpoint": [1]}
    overall_pass_rps = [1]
    
    
    js_PER_EP = json.dumps({})
    js_OVERALL_RPS = json.dumps({"x": x_labels, "y": overall_rps, "ep_data": ep_rps}) 
    js_OVERALL_P90 = json.dumps({"x": x_labels, "y": overall_p90, "ep_data": ep_p90}) 
    js_OVERALL_AVG = json.dumps({"x": x_labels, "y": overall_avg, "ep_data": ep_avg}) 
    js_OVERALL_PASS_RPS = json.dumps({"x": x_labels, "y": overall_pass_rps}) 
    js_EP_RPS = json.dumps({"x": x_labels, "ep_data": ep_rps, "ep_pass_data": ep_rps}) 
    js_DONUT = json.dumps({"labels": donut_labels, "values": donut_values})
    js_PHASES = json.dumps(phases_data)
    
    #start_dt = time_points[0].isoformat(timespec='seconds')
    #end_dt = time_points[0].isoformat(timespec='seconds')
    #filtered_elapsed_seconds = [0]
    start_dt_obj = time_points[0] if time_points else datetime.now()
    end_dt_obj = time_points[-1] if time_points else datetime.now()
    
    # Calculate the test duration (timedelta)
    duration_delta = end_dt_obj - start_dt_obj
    
    # Format the duration as H:MM:SS or D days, H:MM:SS
    def format_duration(td):
        total_seconds = int(td.total_seconds())
        # Calculate days, hours, minutes, and seconds
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Format string
        if days > 0:
            return f"{days} days, {hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    test_duration = format_duration(duration_delta)

    # Prepare ISO strings for display
    start_dt = start_dt_obj.isoformat(timespec='seconds')
    end_dt = end_dt_obj.isoformat(timespec='seconds')
    total_tx = sum(sum(ep_counts[ep]) for ep in endpoints)
    
    # Jumps to HTML generation
else:
    # ---------------- LOAD YAML ----------------
    config_yaml = {}
    if INPUT_YAML and Path(INPUT_YAML).exists():
        try:
            with open(INPUT_YAML,"r",encoding="utf-8") as f:
                config_yaml = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"⚠️ Could not read YAML config: {e}")

    # ---------------- DISCOVER ENDPOINTS ----------------
    def discover_endpoints(counters,summaries):
        eps = set()
        for k in counters.keys():
            if k.startswith("plugins.metrics-by-endpoint."):
                part = k.replace("plugins.metrics-by-endpoint.","")
                if ".codes." in part:
                    eps.add(part.split(".codes.")[0])
                elif ".errors." in part:
                    eps.add(part.split(".errors.")[0])
        for k in summaries.keys():
            if k.startswith("plugins.metrics-by-endpoint.response_time."):
                eps.add(k.replace("plugins.metrics-by-endpoint.response_time.",""))
        return sorted(eps)

    endpoints = discover_endpoints(agg_counters,agg_summaries)

    # ---------------- TIME AXIS ----------------
    time_points = [datetime.fromtimestamp(snap["firstMetricAt"]/1000.0) for snap in intermediate]
    start_time = time_points[0]
    
    # --- Fix RPS spike issue ---
    durations = [
        max(1.0, (time_points[i+1]-time_points[i]).total_seconds())
        if i<len(time_points)-1 else 
        10.0
        for i in range(len(time_points))
    ]
    # ---------------- END FIX ----------------
    
    elapsed_seconds = [(t-start_time).total_seconds() for t in time_points]

    # ---------------- FILTER TIME RANGE ----------------
    if FILTER_END_SEC is None:
        FILTER_END_SEC = float("inf")
    filtered_indices = [i for i,t in enumerate(elapsed_seconds) if FILTER_START_SEC <= t <= FILTER_END_SEC]

    if not filtered_indices:
        # Handle case where filter removes all data
        raise SystemExit("❌ No data points fall within the selected filter range.")
    
    time_points = [time_points[i] for i in filtered_indices]
    durations = [durations[i] for i in filtered_indices] 
    filtered_elapsed_seconds = [(t - time_points[0]).total_seconds() for t in time_points]
    
    # CRITICAL: Use ISO 8601 format for x_labels when in datetime mode
    x_labels = [t.isoformat(timespec='seconds') if TIME_AXIS_MODE=="datetime" else round((t-time_points[0]).total_seconds(),2) for t in time_points]
    
    # ---------------- DATA CONTAINERS ----------------
    ep_avg = {ep:[0]*len(time_points) for ep in endpoints}
    ep_pxx = {ep:{p:[0]*len(time_points) for p in PERCENTILES} for ep in endpoints}
    ep_codes = {ep:defaultdict(lambda:[0]*len(time_points)) for ep in endpoints}
    ep_counts = {ep:[0]*len(time_points) for ep in endpoints}
    ep_pass_counts = {ep:[0]*len(time_points) for ep in endpoints}

    # ---------------- FILL SNAPSHOTS ----------------
    for idx_orig,snap in enumerate(intermediate):
        if idx_orig not in filtered_indices:
            continue
        idx = filtered_indices.index(idx_orig)
        sums = snap.get("summaries",{})
        counters = snap.get("counters",{})
        for key,summary in sums.items():
            if key.startswith("plugins.metrics-by-endpoint.response_time."):
                ep = key.replace("plugins.metrics-by-endpoint.response_time.","")
                if ep in endpoints:
                    ep_avg[ep][idx] = summary.get("mean",0)
                    for p in PERCENTILES:
                        ep_pxx[ep][p][idx] = summary.get(p,0)
        for k,v in counters.items():
            if k.startswith("plugins.metrics-by-endpoint."):
                part = k.replace("plugins.metrics-by-endpoint.","")
                if ".codes." in part:
                    ep_part, code = part.split(".codes.")
                    ep_counts[ep_part][idx] += v
                    ep_codes[ep_part][code][idx] = v
                    if code.isdigit() and code.startswith("2"):
                        ep_pass_counts[ep_part][idx] += v
                elif ".errors." in part:
                    ep_part, code = part.split(".errors.")
                    ep_counts[ep_part][idx] += v
                    ep_codes[ep_part][f"ERR:{code}"][idx] = v

    # ---------------- CALCULATE OVERALL ----------------
    ep_rps = {ep:[ep_counts[ep][i]/durations[i] for i in range(len(time_points))] for ep in endpoints}
    ep_pass_rps = {ep:[ep_pass_counts[ep][i]/durations[i] for i in range(len(time_points))] for ep in endpoints}
    
    overall_rps = [sum(ep_counts[ep][i] for ep in endpoints)/durations[i] for i in range(len(time_points))]
    overall_pass_rps = [sum(ep_pass_counts[ep][i] for ep in endpoints)/durations[i] for i in range(len(time_points))]
    overall_p90 = [max(ep_pxx[ep]["p90"][i] for ep in endpoints) for i in range(len(time_points))]
    overall_avg = [max(ep_avg[ep][i] for ep in endpoints) for i in range(len(time_points))]

    # ---------------- SUMMARY TABLE WITH SLA ----------------
    
    def get_sla_for_endpoint(ep_name, sla_dict, default_sla=500, default_tph=0):
        """
        Retrieve SLA and Expected_TPH for a given transaction/endpoint.
        Tries exact match first; if not found, removes trailing slashes and matches base domain.
        Debug prints added to see why matching fails.
        """
        #print(f"\nLooking up SLA for endpoint: '{ep_name}'")

        # Exact match
        if ep_name in sla_dict:
            #print(f"Exact match found: {ep_name} -> {sla_dict[ep_name]}")
            return sla_dict[ep_name]
        
        # Try matching without trailing slashes
        ep_base = ep_name.rstrip("/")
        for k in sla_dict:
            k_base = k.rstrip("/")
            #print(f"Comparing '{ep_base}' with SLA key '{k_base}'")
            if k_base == ep_base:
                #print(f"Match found: {k} -> {sla_dict[k]}")
                return sla_dict[k]
        
        # Fallback to defaults
        #print(f"No match found for '{ep_name}', using defaults: SLA={default_sla}, TPH={default_tph}")
        return [default_sla, default_tph]

    
    DEFAULT_SLA = 500  # ms
    DEFAULT_TPH = 0    # transactions per hour
    
    # Determine if SLA JSON is actually provided and loaded
    SLA_ENABLED = bool(sla_data)
    
    # ---------------- SUMMARY TABLE ----------------
    summary_rows = []
    for ep in endpoints:
        total = sum(ep_counts[ep])
        pass_count = sum(ep_pass_counts[ep])
        fail_count = total - pass_count
        agg_summary = agg_summaries.get(f"plugins.metrics-by-endpoint.response_time.{ep}",{})
    
        row = {
            "Transaction": ep,
            "Avg (ms)": round(agg_summary.get("mean", 0), 2),
            "Min (ms)": agg_summary.get("min", 0),
            "Max (ms)": agg_summary.get("max", 0),
            "P90": agg_summary.get("p90", 0),
            "P95": agg_summary.get("p95", 0),
            "P99(ms)": agg_summary.get("p99", 0),
            "Count": total,
            "Pass_Count": pass_count,
            "Fail_Count": fail_count
        }
        for p in PERCENTILES:
            if p.lower() == "p99":
                continue  # already added as "P99(ms)"
            row[p.upper()] = agg_summary.get(p,0)
            
        # Only add SLA columns if SLA JSON was provided
        if SLA_ENABLED:
            sla_val, tph_val = get_sla_for_endpoint(ep, sla_data, DEFAULT_SLA, DEFAULT_TPH)
            row["SLA(ms)"] = sla_val
            row["Expected_TPH"] = tph_val

        summary_rows.append(row)

    # Decide column order dynamically
    columns = ["Transaction"]
    if SLA_ENABLED:
        columns += ["SLA(ms)"]
    columns += ["Avg (ms)", "Min (ms)", "Max (ms)", "P90", "P95", "P99(ms)"]
    if SLA_ENABLED:
        columns += ["Expected_TPH"]
    columns += ["Count", "Pass_Count", "Fail_Count"]

    summary_df = pd.DataFrame(summary_rows, columns=columns).sort_values(by="Transaction")
    
    #summary_df = pd.DataFrame(summary_rows).sort_values(by="Transaction")

    # ---------------- ERROR TABLE ----------------
    error_rows,error_agg=[],defaultdict(int)
    for ep in endpoints:
        for code,arr in ep_codes[ep].items():
            s = sum(arr)
            if s>0 and not (code.isdigit() and code.startswith("2")):
                error_rows.append({"Endpoint":ep,"Status Code":code,"Count":int(s)})
                error_agg[code] += s
    error_df = pd.DataFrame(error_rows)
    if not error_df.empty:
        error_df["Proportion"] = (error_df["Count"]/error_df["Count"].sum()*100).round(2)
    donut_labels,donut_values=list(error_agg.keys()),list(error_agg.values())

    # ---------------- PER-ENDPOINT DATA ----------------
    def safe_id(s): return re.sub(r"[^0-9a-zA-Z]+","_",s)
    per_ep_data={}
    for ep in endpoints:
        sid = safe_id(ep)
        total = sum(ep_counts[ep])
        pass_pct_value = sum(sum(arr) for code,arr in ep_codes[ep].items() if code.isdigit() and code.startswith("2"))/(total or 1)*100
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
            "codes": {code:ep_codes[ep][code] for code in ep_codes[ep].keys()},
            "total": total,
            "pass_pct": f"{pass_pct_value:.2f}",
            "err_pct": f"{100-pass_pct_value:.2f}",
            "safe_id": sid
        }

    # ---------------- PHASES - Calculate Datetime Boundaries ----------------
    phases_data=[]
    yaml_phases = config_yaml.get("config",{}).get("phases") if isinstance(config_yaml,dict) else None
    
    cumulative_duration_sec = 0
    # Use the phases from YAML only if it was provided and loaded.
    if yaml_phases and isinstance(yaml_phases,list) and INPUT_YAML:
        for ph in yaml_phases:
            duration = ph.get("duration",0)
            phase_start_dt = start_time + timedelta(seconds=cumulative_duration_sec)
            cumulative_duration_sec += duration
            phase_end_dt = start_time + timedelta(seconds=cumulative_duration_sec)
            
            phases_data.append({
                "name":ph.get("name","Phase"),
                "duration":duration,
                "arrival":ph.get("arrivalRate",ph.get("arrivalCount",0)),
                # CRITICAL: Use ISO 8601 format for Plotly shapes
                "start_dt": phase_start_dt.isoformat(timespec='seconds'), 
                "end_dt": phase_end_dt.isoformat(timespec='seconds'), 
                "start_sec": cumulative_duration_sec - duration,
                "end_sec": cumulative_duration_sec
            })
    else:
        # If no YAML or phases, create a single "Default Phase" covering the whole run
        total_duration = int((time_points[-1]-time_points[0]).total_seconds()) if time_points else 60
        # CRITICAL: Use ISO 8601 format for default phase
        phases_data=[{"name":"Default Phase","duration":total_duration,"arrival":0, "start_dt": start_time.isoformat(timespec='seconds'), "end_dt": time_points[-1].isoformat(timespec='seconds') if time_points else (start_time + timedelta(seconds=total_duration)).isoformat(timespec='seconds'), "start_sec": 0, "end_sec": total_duration}]
    
    # ---------------- JS JSON STRINGS ----------------
    ep_p90 = {ep: per_ep_data[ep]['p90'] for ep in endpoints}
    ep_avg = {ep: per_ep_data[ep]['rt_y'] for ep in endpoints}
    ep_rps_data = {ep: per_ep_data[ep]['rps'] for ep in endpoints}
    ep_pass_rps_data = {ep: per_ep_data[ep]['pass_rps'] for ep in endpoints}

    js_PER_EP = json.dumps(per_ep_data)
    js_OVERALL_RPS = json.dumps({"x": x_labels, "y": overall_rps, "ep_data": ep_rps_data}) 
    js_OVERALL_PASS_RPS = json.dumps({"x": x_labels, "y": overall_pass_rps})
    js_OVERALL_P90 = json.dumps({"x": x_labels, "y": overall_p90, "ep_data": ep_p90}) 
    js_OVERALL_AVG = json.dumps({"x": x_labels, "y": overall_avg, "ep_data": ep_avg}) 
    js_EP_RPS = json.dumps({"x": x_labels, "ep_data": ep_rps_data, "ep_pass_data": ep_pass_rps_data})
    js_DONUT = json.dumps({"labels": donut_labels, "values": donut_values})
    js_PHASES = json.dumps(phases_data)

    #start_dt = time_points[0].isoformat(timespec='seconds') if time_points else ""
    #end_dt = time_points[-1].isoformat(timespec='seconds') if time_points else ""
    #total_tx = sum(sum(ep_counts[ep]) for ep in endpoints)
    start_dt_obj = time_points[0] if time_points else datetime.now()
    end_dt_obj = time_points[-1] if time_points else datetime.now()
    
    # Calculate the test duration (timedelta)
    duration_delta = end_dt_obj - start_dt_obj
    
    # Format the duration as H:MM:SS or D days, H:MM:SS
    def format_duration(td):
        total_seconds = int(td.total_seconds())
        # Calculate days, hours, minutes, and seconds
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Format string
        if days > 0:
            return f"{days} days, {hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    test_duration = format_duration(duration_delta)

    # Prepare ISO strings for display
    start_dt = start_dt_obj.isoformat(timespec='seconds')
    end_dt = end_dt_obj.isoformat(timespec='seconds')
    total_tx = sum(sum(ep_counts[ep]) for ep in endpoints)


# ---------------- HTML + CSS ----------------
bootstrap = """
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdn.datatables.net/2.0.7/css/dataTables.bootstrap5.min.css" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<style>
body{background:#f7f8fa;font-family:'Inter',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;padding:18px;}
.card{margin-bottom:16px;border-radius:12px;box-shadow:0 2px 6px rgba(0,0,0,0.08);}
.card-body{padding:20px;}
.table{font-size:0.875rem;text-align:center;}
.table th:first-child,
.table td:first-child {
    text-align: left !important;
}
.endpoint-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.badge-success{background-color:#28a745 !important;color:#fff;}
.badge-danger{background-color:#dc3545 !important;color:#fff;}
.sparkline{width:180px;height:36px;display:inline-block;}
.toggle-group { display: flex; align-items: center; justify-content: flex-end; margin-bottom: 10px; }
.toggle-group label { margin-right: 10px; font-weight: 500; font-size: 0.85rem; }
.chart-controls { display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 1rem; padding: 10px 0; border-top: 1px solid #eee; border-bottom: 1px solid #eee; }
</style>
"""


# ---------------- BUILD HTML ----------------
html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Artillery Report</title>{bootstrap}</head><body>
<div class='container-fluid'>
<div class='card'><div class='card-body'>
<h3>🚀 Artillery Performance Report</h3>
<div class='row'>
<div class='col-md-3'><strong>Start:</strong><div>{start_dt}</div></div>
<div class='col-md-3'><strong>End:</strong><div>{end_dt}</div></div>
<div class='col-md-3'><strong>End:</strong><div>{test_duration}</div></div>
<div class='col-md-3'><strong>Total tx:</strong><div>{total_tx}</div></div>
</div></div></div>

<div class='card'><div class='card-body'>
    <h5>🎛️ Chart Visibility</h5>
    <div class='chart-controls'>
        <div class="form-check form-check-inline">
            <input class="form-check-input chart-toggle" type="checkbox" id="show-chart-p90" data-target="card-chart-p90" checked>
            <label class="form-check-label" for="show-chart-p90">Overall P90</label>
        </div>
        <div class="form-check form-check-inline">
            <input class="form-check-input chart-toggle" type="checkbox" id="show-chart-avg" data-target="card-chart-avg" checked>
            <label class="form-check-label" for="show-chart-avg">Overall Avg</label>
        </div>
        <div class="form-check form-check-inline">
            <input class="form-check-input chart-toggle" type="checkbox" id="show-chart-ep-rps" data-target="card-chart-ep-rps" checked>
            <label class="form-check-label" for="show-chart-ep-rps">RPS Per Endpoint</label>
        </div>
        <div class="form-check form-check-inline">
            <input class="form-check-input chart-toggle" type="checkbox" id="show-chart-rps" data-target="card-chart-rps" checked>
            <label class="form-check-label" for="show-chart-rps">Overall Throughput</label>
        </div>
        <div class="form-check form-check-inline">
            <input class="form-check-input chart-toggle" type="checkbox" id="show-chart-combined" data-target="card-chart-combined" checked>
            <label class="form-check-label" for="show-chart-combined">RPS vs P90 Combined</label>
        </div>
    </div>
    <h5>📄 Summary</h5>
        <div class="table-responsive" style="max-height: 400px; overflow-y: auto;">

{summary_df.to_html(classes="table table-striped table-bordered table-sm text-center", index=False, justify="center") if not summary_df.empty else "<div>No summary data available.</div>"}</div>
</div></div>

<div class='row'>
<div class='col-md-6'><div class='card'><div class='card-body'><h5>⚠️ Errors</h5>
{error_df.to_html(classes="table table-striped table-bordered", index=False) if not error_df.empty else "<div>No endpoint errors.</div>"}
</div></div></div>
<div class='col-md-6'><div class='card'><div class='card-body'><h5>🍩 Error Distribution</h5><div id='donut' style='height:400px;'></div></div></div></div>
</div>

<div class='card' id='card-chart-p90'><div class='card-body'>
    <div class='toggle-group'>
        <label for='toggle-overall-p90'>Show Endpoint Lines</label>
        <div class="form-check form-switch">
            <input class="form-check-input" type="checkbox" id="toggle-overall-p90">
        </div>
    </div>
    <h5>⏱️ Overall P90 Response Time (with per-URL timeline)</h5>
    <div id='chart-p90' style='height:420px;'></div>
</div></div>

<div class='card' id='card-chart-avg'><div class='card-body'>
    <div class='toggle-group'>
        <label for='toggle-overall-avg'>Show Endpoint Lines</label>
        <div class="form-check form-switch">
            <input class="form-check-input" type="checkbox" id="toggle-overall-avg">
        </div>
    </div>
    <h5>⏱️ Overall Average Response Time (with per-URL timeline)</h5>
    <div id='chart-avg' style='height:420px;'></div>
</div></div>

<div class='card' id='card-chart-ep-rps'><div class='card-body'>
    <div class='toggle-group'>
        <label for='toggle-ep-rps'>Show Endpoint Lines</label>
        <div class="form-check form-switch">
            <input class="form-check-input" type="checkbox" id="toggle-ep-rps">
        </div>
    </div>
    <h5>💥 Requests Per Second (per endpoint) timeline</h5>
    <div id='chart-ep-rps' style='height:420px;'></div>
</div></div>

<div class='card' id='card-chart-rps'><div class='card-body'>
    <div class='toggle-group'>
        <label for='toggle-overall-rps'>Show Pass-Only Throughput</label>
        <div class="form-check form-switch">
            <input class="form-check-input" type="checkbox" id="toggle-overall-rps">
        </div>
    </div>
    <h5>📊 Throughput Overall timeline (RPS)</h5>
    <div id='chart-rps' style='height:420px;'></div>
</div></div>

<div class='card' id='card-chart-combined'><div class='card-body'><h5>📈 RPS vs Overall P90</h5><div id='chart-combined' style='height:420px;'></div></div></div>
<div class='card'><div class='card-body'><h5>🔎 Per-endpoint breakdown</h5>
"""

# Per-endpoint sparkline + collapsibles
for ep in endpoints:
    d = per_ep_data[ep]
    # Corrected 'badge badge-success' to 'badge bg-success' for Bootstrap 5
    html += f"""
    <div class='endpoint-row'>
        <div style='flex:1'><b>{ep}</b></div>
        <div><span class='badge bg-secondary'>Total: {d['total']}</span>
        <span class='badge bg-success'>2xx: {d['pass_pct']}%</span>
        <span class='badge bg-danger'>Err: {d['err_pct']}%</span></div>
        <div style='width:180px;margin-left:12px;'><div id='spark-{d["safe_id"]}' class='sparkline'></div></div>
        <a class='btn btn-sm btn-outline-primary' data-bs-toggle='collapse' href='#collapse-{d["safe_id"]}' role='button' aria-expanded='false' aria-controls='collapse-{d["safe_id"]}'>Details</a>
    </div>
    <div class='collapse' id='collapse-{d["safe_id"]}'>
        <div class='card card-body'>
            <div class='row'>
                <div class='col-md-6'><div id='ep-{d["safe_id"]}-rt' style='height:260px;'></div></div> <div class='col-md-6'><div id='ep-{d["safe_id"]}-rps' style='height:260px;'></div></div> </div>
            <div class='row'>
                <div class='col-md-12'><div id='ep-{d["safe_id"]}-codes' style='height:260px;'></div></div>
            </div>
        </div>
    </div>
    """

html += "</div></div>"

# ---------------- JS ----------------
html += f"""
<script src='https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js'></script>
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/2.0.7/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/2.0.7/js/dataTables.bootstrap5.min.js"></script>

<script>
const PER_EP = {js_PER_EP};
const OVERALL_RPS_DATA = {js_OVERALL_RPS};
const OVERALL_PASS_RPS_DATA = {js_OVERALL_PASS_RPS}; 
const OVERALL_P90_DATA = {js_OVERALL_P90};
const OVERALL_AVG_DATA = {js_OVERALL_AVG};
const EP_RPS_DATA = {js_EP_RPS};
const DONUT = {js_DONUT};
const PHASES = {js_PHASES};
const TIME_AXIS_TYPE = '{'date' if TIME_AXIS_MODE=='datetime' else 'linear'}';
const TIME_AXIS_MODE = '{TIME_AXIS_MODE}';
const SHOW_PHASE_SHAPES = {str(SHOW_PHASE_SHAPES).lower()}; // FLAG from Python
const FILTERED_ELAPSED_SECONDS = {json.dumps(filtered_elapsed_seconds)};

// Utility function for phase rectangles
function addPhaseShapes(layout, yMax, dataPoints) {{
    if (!SHOW_PHASE_SHAPES) return; // CHECK FLAG

    layout.shapes = layout.shapes || [];
    layout.annotations = layout.annotations || [];

    // Safety check: only add phases if there are more than just the "Default Phase"
    // OR if the single phase actually has a specified duration that matches the test (not just 60s default)
    if (PHASES.length === 0) return;
    
    const safeYMax = Math.max(yMax, 1); 

    if (TIME_AXIS_MODE === 'datetime') {{
        PHASES.forEach(function(ph) {{
            let x0_val = ph.start_dt;
            let x1_val = ph.end_dt;
            
            if (!x0_val || !x1_val || x0_val === x1_val) return;

            layout.shapes.push({{
                type:'rect',
                x0:x0_val,
                x1:x1_val,
                y0:0,           
                y1:1,           
                fillcolor:'rgba(0,123,255,0.08)',
                line:{{width:0}},
                xref:'x',       
                yref:'paper'    // Relative to the plot area (0 to 1)
            }});

            // Calculate annotation position (center datetime)
            let startTS = new Date(x0_val).getTime();
            let endTS = new Date(x1_val).getTime();
            // Using the precise ISO string as the x coordinate for annotation
            let centerTS = new Date((startTS + endTS) / 2).toISOString().substring(0, 19); 

            // Place annotation slightly above the highest data point (relative to data)
            layout.annotations.push({{
                x:centerTS,
                y:safeYMax * 1.05,
                text:ph.name||'Phase',
                showarrow:false,
                font:{{size:10,color:'#007bff'}},
                xref:'x',
                yref:'y'
            }});
        }});

    }} else {{ // Linear time axis (elapsed seconds)
        PHASES.forEach(function(ph) {{
            let x0=ph.start_sec, x1=ph.end_sec;
            layout.shapes.push({{
                type:'rect',
                x0:x0,
                x1:x1,
                y0:0,
                y1:1,           
                fillcolor:'rgba(0,123,255,0.08)',
                line:{{width:0}},
                yref:'paper'    
            }});
            layout.annotations.push({{
                x:(x0+x1)/2,
                y:safeYMax*1.05,
                text:ph.name||'Phase',
                showarrow:false,
                font:{{size:10,color:'#007bff'}}
            }});
        }});
    }}
}}

// Reusable function to draw a main graph with optional endpoint lines and error toggle
function drawMainGraph(chartId, title, yAxisTitle, mainData, epData = null, toggleId = null, passData = null) {{
    // Calculate max_y from all visible traces (main + endpoints) for phase annotation placement
    let max_y_values = mainData.y.filter(y => !isNaN(y) && y !== null);
    if (epData) {{
        Object.values(epData).forEach(arr => {{
            max_y_values = max_y_values.concat(arr.filter(y => !isNaN(y) && y !== null));
        }});
    }}
    let max_y = max_y_values.length > 0 ? Math.max(...max_y_values) : 10;
    
    // Use double backslashes in regex to prevent Python SyntaxWarning
    let traces = [{{x:mainData.x,y:mainData.y,mode:'lines',name:title.replace(/ *\\([^)]*\\) */g, ""), line:{{width:3}}}}]; // Main line (Overall)

    // Add endpoint lines (now solid)
    if (epData) {{
        Object.keys(epData).forEach(ep => {{
            traces.push({{
                x: mainData.x,
                y: epData[ep],
                mode: 'lines',
                name: ep,
                line: {{width: 1}}, // SOLID LINE
                visible: 'legendonly' // Hidden by default
            }});
        }});
    }}
    
    // Add pass-only data as a hidden line
    if (passData) {{
        traces.push({{
            x: passData.x,
            y: passData.y,
            mode: 'lines',
            name: 'Pass-Only (2xx)',
            line: {{width: 3, dash: 'dash'}},
            visible: 'legendonly' // Hidden by default
        }});
    }}

    let layout = {{
        title: title,
        xaxis: {{type: TIME_AXIS_TYPE}},
        yaxis: {{title: yAxisTitle}},
        hovermode: 'x unified'
    }};
    addPhaseShapes(layout, max_y, mainData);
    
    Plotly.newPlot(chartId, traces, layout, {{responsive:true}});
    
    // Add toggle functionality
    if (toggleId) {{
        document.getElementById(toggleId).addEventListener('change', function(e) {{
            let update = {{visible: []}};
            let chartDiv = document.getElementById(chartId);
            let numEndpoints = epData ? Object.keys(epData).length : 0;
            let totalTraces = chartDiv.data.length;
            
            if (epData) {{
                // Logic for P90/Avg/EP-RPS (show/hide individual endpoint lines)
                let mainLineIndex = 0; 
                let epStartIndex = 1; 
                
                update.visible[mainLineIndex] = true; 

                if (e.target.checked) {{
                    for(let i = 0; i < numEndpoints; i++) {{
                         update.visible[epStartIndex + i] = true; // Show all endpoints
                    }}
                }} else {{
                    for(let i = 0; i < numEndpoints; i++) {{
                         update.visible[epStartIndex + i] = 'legendonly'; // Hide all endpoints
                    }}
                }}
                
                Plotly.restyle(chartDiv, update);

            }} else if (passData) {{
                // Logic for Overall Throughput (toggle between Total and Pass-Only)
                let totalRPSIndex = 0;
                let passRPSIndex = totalTraces - 1; // Last trace
                
                if (e.target.checked) {{
                    // Show Pass-Only, Hide Total
                    update.visible[totalRPSIndex] = 'legendonly';
                    update.visible[passRPSIndex] = true;
                }} else {{
                    // Show Total, Hide Pass-Only
                    update.visible[totalRPSIndex] = true;
                    update.visible[passRPSIndex] = 'legendonly';
                }}
                Plotly.restyle(chartDiv, update);
            }}
        }});
    }}
}}


document.addEventListener('DOMContentLoaded',function(){{
    // --- TOP LEVEL CHART VISIBILITY TOGGLE ---
    document.querySelectorAll('.chart-toggle').forEach(toggle => {{
        let targetId = toggle.getAttribute('data-target');
        let targetCard = document.getElementById(targetId);

        // Initial state
        if (targetCard) {{
            targetCard.style.display = toggle.checked ? 'block' : 'none';
        }}

        toggle.addEventListener('change', function() {{
            if (targetCard) {{
                targetCard.style.display = this.checked ? 'block' : 'none';
                // Trigger Plotly relayout to redraw hidden graphs when they become visible
                if (this.checked) {{
                    let chartDiv = targetCard.querySelector('.card-body > div[id^="chart-"]');
                    if (chartDiv) {{
                        Plotly.relayout(chartDiv, {{autosize: true}});
                    }}
                }}
            }}
        }});
    }});


    // Sparklines
    Object.keys(PER_EP).forEach(ep=>{{
        let div=document.getElementById('spark-'+PER_EP[ep].safe_id);
        if(div){{
            Plotly.newPlot(div,[{{x:PER_EP[ep].spark_x,y:PER_EP[ep].spark_y,mode:'lines',line:{{width:1}},hoverinfo:'none'}}],
            {{margin:{{t:2,b:2,l:2,r:2}},xaxis:{{visible:false}},yaxis:{{visible:false}}}},
            {{displayModeBar:false,staticPlot:true,responsive:true}});
        }}
    }});

    // 1. Throughput Overall (RPS)
    if({str(SHOW_GRAPH_RPS).lower()} && OVERALL_RPS_DATA.y.length > 0){{
        drawMainGraph('chart-rps', 'Throughput Overall timeline (RPS)', 'Requests/s', OVERALL_RPS_DATA, null, 'toggle-overall-rps', OVERALL_PASS_RPS_DATA);
    }}

    // 2. Overall P90 Response Time
    if({str(SHOW_GRAPH_P90).lower()} && OVERALL_P90_DATA.y.length > 0){{
        drawMainGraph('chart-p90', 'Overall P90 Response Time', 'ms', OVERALL_P90_DATA, OVERALL_P90_DATA.ep_data, 'toggle-overall-p90');
    }}
    
    // 3. Overall Average Response Time
    if(OVERALL_AVG_DATA.y.length > 0){{
        drawMainGraph('chart-avg', 'Overall Average Response Time', 'ms', OVERALL_AVG_DATA, OVERALL_AVG_DATA.ep_data, 'toggle-overall-avg');
    }}

    // 4. Requests Per Second (per endpoint) timeline
    if(Object.keys(EP_RPS_DATA.ep_data).length > 0){{
        // Calculate the overall sum for the main line
        let overall_y = EP_RPS_DATA.x.map((_, i) => Object.values(EP_RPS_DATA.ep_data).reduce((sum, arr) => sum + arr[i], 0));
        let overall_data = {{x: EP_RPS_DATA.x, y: overall_y}};
        
        drawMainGraph('chart-ep-rps', 'Requests Per Second (per endpoint)', 'Requests/s', overall_data, EP_RPS_DATA.ep_data, 'toggle-ep-rps');
    }}

    // Combined RPS vs P90
    if({str(SHOW_GRAPH_COMBINED).lower()} && OVERALL_RPS_DATA.y.length > 0 && OVERALL_P90_DATA.y.length > 0){{
        let max_y_comb = Math.max(...OVERALL_RPS_DATA.y);
        let layout_comb={{title:'RPS vs Overall P90',xaxis:{{type:TIME_AXIS_TYPE}},yaxis:{{title:'RPS',side:'left'}},yaxis2:{{title:'P90 ms',overlaying:'y',side:'right'}}}};
        addPhaseShapes(layout_comb, max_y_comb, OVERALL_RPS_DATA);
        Plotly.newPlot('chart-combined',[
            {{x:OVERALL_RPS_DATA.x,y:OVERALL_RPS_DATA.y,mode:'lines',name:'RPS',yaxis:'y1'}},
            {{x:OVERALL_P90_DATA.x,y:OVERALL_P90_DATA.y,mode:'lines',name:'P90',yaxis:'y2'}}
        ],layout_comb,{{responsive:true}});
    }}

    // Donut
    if({str(SHOW_GRAPH_DONUT).lower()} && DONUT.labels.length>0){{
        Plotly.newPlot('donut',[{{labels:DONUT.labels,values:DONUT.values,type:'pie',hole:0.6}}],{{margin:{{t:20,b:20,l:20,r:20}}}},{{responsive:true}});
    }}

    // Per-endpoint collapsibles
    Object.keys(PER_EP).forEach(ep=>{{
        let id=PER_EP[ep].safe_id;
        let collapseDiv=document.getElementById('collapse-'+id);
        if(!collapseDiv) return;
        
        let isGraphDrawn = false;
        
        new bootstrap.Collapse(collapseDiv, {{toggle: false}})._element.addEventListener('shown.bs.collapse',function(){{
            if(isGraphDrawn) return;

            let rt_div=document.getElementById('ep-'+id+'-rt');
            let rps_div=document.getElementById('ep-'+id+'-rps');
            let codes_div=document.getElementById('ep-'+id+'-codes');
            let endpointData = PER_EP[ep];

            // 1. Response Time Graph (now including Avg)
            if(rt_div && endpointData.rt_y.length > 0){{
                let traces=[
                    {{x:endpointData.rt_x,y:endpointData.rt_y,mode:'lines',name:'Avg RT'}},
                    {{x:endpointData.rt_x,y:endpointData.p90,mode:'lines',name:'P90'}},
                    {{x:endpointData.rt_x,y:endpointData.p95,mode:'lines',name:'P95'}},
                    {{x:endpointData.rt_x,y:endpointData.p99,mode:'lines',name:'P99'}}
                ];
                let all_rt_y = traces.flatMap(t=>t.y).filter(y => !isNaN(y));
                let max_rt_y = all_rt_y.length > 0 ? Math.max(...all_rt_y) : 100;

                let layout={{title:ep+' Response Times (Percentiles & Avg)',xaxis:{{type:TIME_AXIS_TYPE}},yaxis:{{title:'ms'}}}};
                addPhaseShapes(layout, max_rt_y, endpointData);
                Plotly.react(rt_div,traces,layout,{{responsive:true}});
            }}
            
            // 2. RPS Graph
            if(rps_div && endpointData.rps.length > 0){{
                let rps_y = endpointData.rps;
                let max_rps_y = Math.max(...rps_y) || 10;

                let traces=[{{x:endpointData.rt_x,y:rps_y,mode:'lines',name:'RPS'}}];
                
                let layout_rps={{title:ep+' Requests per Second',xaxis:{{type:TIME_AXIS_TYPE}},yaxis:{{title:'RPS'}}}};
                addPhaseShapes(layout_rps, max_rps_y, endpointData);
                Plotly.react(rps_div,traces,layout_rps,{{responsive:true}});
            }}
            
            // 3. Codes Graph
            if(codes_div && Object.keys(endpointData.codes).length > 0){{
                let traces_codes=[];
                Object.keys(endpointData.codes).forEach(code=>{{
                    // Use 'stackgroup' to ensure counts are additive and easy to read
                    traces_codes.push({{x:endpointData.rt_x,y:endpointData.codes[code],mode:'lines',name:code,stackgroup:'one'}});
                }});
                
                let all_code_y = traces_codes.flatMap(t=>t.y).filter(y => !isNaN(y));
                let max_code_y = all_code_y.length > 0 ? Math.max(...all_code_y) : 10;
                
                let layout_codes={{
                    title:ep+' HTTP Codes (Stacked)',
                    xaxis:{{type:TIME_AXIS_TYPE}},
                    yaxis:{{title:'count'}},
                    showlegend: true 
                }};
                addPhaseShapes(layout_codes, max_code_y, endpointData);
                Plotly.react(codes_div,traces_codes,layout_codes,{{responsive:true}});
            }}
            
            isGraphDrawn = true;
        }},{{once:true}});
    }});
}});
</script>
</body></html>
"""

# ---------------- SAVE HTML ----------------
try:
    with open(OUTPUT_HTML,"w",encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Full merged report generated → {OUTPUT_HTML}")
except Exception as e:
    print(f"❌ Could not write HTML file to {OUTPUT_HTML}: {e}")
