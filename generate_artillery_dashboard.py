# generate_artillery_dashboard.py
import json
import re
from datetime import datetime
from collections import defaultdict
import pandas as pd
import yaml


# ---------------- CONFIG ----------------
#INPUT_JSON = "artillery_result.json"
#INPUT_YAML = "config.yml"
#OUTPUT_HTML = "artillery_report_full.html"


PERCENTILES = ["p90", "p95", "p99"]
TIME_AXIS_MODE = "elapsed"  # 'elapsed' or 'datetime'
SHOW_GRAPHS = True
FILTER_START_SEC = 0
FILTER_END_SEC = None  # None = till end

# ---------------------------------------------------------------------
# 🧩 Step 1: Read environment variables (with sensible defaults)
# ---------------------------------------------------------------------
# GitHub Actions will pass these automatically.
# When running locally, you can still just put the files next to the script.

INPUT_JSON = os.getenv("INPUT_JSON", "artillery_result.json")   # Default local filename
INPUT_YAML = os.getenv("INPUT_YAML", "config.yml")              # Default config
OUTPUT_HTML = os.getenv("OUTPUT_HTML", "artillery_dashboard.html")  # Default HTML output

print(f"📁 JSON Input: {INPUT_JSON}")
print(f"📁 YAML Config: {INPUT_YAML}")
print(f"📄 Output HTML: {OUTPUT_HTML}")

# ---------------- LOAD DATA ----------------
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
with open(INPUT_YAML, "r", encoding="utf-8") as f:
    config_yaml = yaml.safe_load(f)

aggregate = data.get("aggregate", {})
intermediate = data.get("intermediate", [])
agg_counters = aggregate.get("counters", {})
agg_summaries = aggregate.get("summaries", {})

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

# ---------------- TIME AXIS ----------------
if intermediate:
    time_points = [datetime.fromtimestamp(snap["firstMetricAt"] / 1000.0) for snap in intermediate]
    start_time = time_points[0]

    durations = []
    for i in range(len(time_points)):
        if i < len(time_points) - 1:
            d = max(0.0001, (time_points[i + 1] - time_points[i]).total_seconds())
        else:
            d = (time_points[i] - time_points[i - 1]).total_seconds() if i > 0 else 1.0
            if d <= 0:
                d = 1.0
        durations.append(d)
else:
    time_points = []
    start_time = None
    durations = []

# ---------------- FILTER TIME RANGE ----------------
if FILTER_END_SEC is None:
    FILTER_END_SEC = float("inf")

filtered_indices = [
    i for i, t in enumerate(time_points)
    if FILTER_START_SEC <= (t - start_time).total_seconds() <= FILTER_END_SEC
]

time_points = [time_points[i] for i in filtered_indices]
durations = [durations[i] for i in filtered_indices]

# ---------------- X LABELS ----------------
if TIME_AXIS_MODE == "datetime":
    x_labels = [t.strftime("%Y-%m-%d %H:%M:%S") for t in time_points]  # local time
else:
    x_labels = [(t - start_time).total_seconds() for t in time_points]

# ---------------- DATA CONTAINERS ----------------
ep_avg = {ep: [0]*len(time_points) for ep in endpoints}
ep_pxx = {ep: {p: [0]*len(time_points) for p in PERCENTILES} for ep in endpoints}
ep_codes = {ep: defaultdict(lambda: [0]*len(time_points)) for ep in endpoints}
ep_counts = {ep: [0]*len(time_points) for ep in endpoints}

# ---------------- FILL SNAPSHOTS ----------------
for idx_orig, snap in enumerate(intermediate):
    if idx_orig not in filtered_indices:
        continue
    idx = filtered_indices.index(idx_orig)
    sums = snap.get("summaries", {})
    counters = snap.get("counters", {})

    for key, summary in sums.items():
        if key.startswith("plugins.metrics-by-endpoint.response_time."):
            ep = key.replace("plugins.metrics-by-endpoint.response_time.", "")
            if ep in endpoints:
                ep_avg[ep][idx] = summary.get("mean", 0)
                for p in PERCENTILES:
                    ep_pxx[ep][p][idx] = summary.get(p, 0)

    for k, v in counters.items():
        if k.startswith("plugins.metrics-by-endpoint."):
            part = k.replace("plugins.metrics-by-endpoint.", "")
            if ".codes." in part:
                ep_part, code = part.split(".codes.")
                ep_counts[ep_part][idx] += v
                ep_codes[ep_part][code][idx] = v
            elif ".errors." in part:
                ep_part, code = part.split(".errors.")
                code = f"ERR:{code}"
                ep_counts[ep_part][idx] += v
                ep_codes[ep_part][code][idx] = v

# ---------------- CALCULATE RPS ----------------
ep_rps = {}
for ep in endpoints:
    ep_rps[ep] = [ep_counts[ep][i] / durations[i] if durations[i]>0 else 0 for i in range(len(time_points))]

overall_rps = [
    sum(ep_counts[ep][i] for ep in endpoints)/durations[i] if durations[i]>0 else 0
    for i in range(len(time_points))
]

# ---------------- SUMMARY TABLE ----------------
summary_rows = []
for ep in endpoints:
    total = sum(ep_counts[ep])
    pass_count = sum(sum(arr) for code, arr in ep_codes[ep].items() if code.isdigit() and code.startswith("2"))
    fail_count = total - pass_count
    agg_summary = agg_summaries.get(f"plugins.metrics-by-endpoint.response_time.{ep}", {})
    row = {
        "Transaction": ep,
        "Count": total,
        "Pass Count": pass_count,
        "Fail Count": fail_count,
        "Avg (ms)": round(agg_summary.get("mean",0),2),
        "Min (ms)": agg_summary.get("min",0),
        "Max (ms)": agg_summary.get("max",0)
    }
    for p in PERCENTILES:
        row[p.upper()] = agg_summary.get(p,0)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).sort_values(by="Transaction")

# ---------------- ERROR TABLE ----------------
error_rows = []
error_agg = defaultdict(int)
for ep in endpoints:
    for code, arr in ep_codes[ep].items():
        s = sum(arr)
        if s>0 and not (code.isdigit() and code.startswith("2")):
            error_rows.append({"Endpoint":ep,"Status Code":code,"Count":int(s)})
            error_agg[code] += s

error_df = pd.DataFrame(error_rows)
if not error_df.empty:
    error_df["Proportion"] = (error_df["Count"]/error_df["Count"].sum()*100).round(2)

donut_labels = list(error_agg.keys())
donut_values = list(error_agg.values())

# ---------------- PER-ENDPOINT DATA ----------------
def safe_id(s): return re.sub(r"[^0-9a-zA-Z]+","_",s)

per_ep_data = {}
for ep in endpoints:
    sid = safe_id(ep)
    total = sum(ep_counts[ep])
    pass_pct = round(
        sum(sum(arr) for code, arr in ep_codes[ep].items() if code.isdigit() and code.startswith("2")) / (total or 1) * 100, 2
    )
    err_pct = 100 - pass_pct
    per_ep_data[ep] = {
        "spark_x": x_labels,
        "spark_y": ep_avg[ep],
        "rt_x": x_labels,
        "rt_y": ep_avg[ep],
        "p90": ep_pxx[ep]["p90"],
        "p95": ep_pxx[ep]["p95"],
        "p99": ep_pxx[ep]["p99"],
        "codes": {code: ep_codes[ep][code] for code in ep_codes[ep].keys()},
        "total": total,
        "pass_pct": pass_pct,
        "err_pct": err_pct,
        "safe_id": sid
    }

# ---------------- PHASES ----------------
phases_data = []
for ph in config_yaml.get("config", {}).get("phases", []):
    phases_data.append({
        "name": ph.get("name",""),
        "duration": ph.get("duration",0),
        "arrival": ph.get("arrivalRate", ph.get("arrivalRate",0))
    })

# ---------------- HTML + JS ----------------
bootstrap = """
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body{background:#f7f8fa;font-family:'Inter',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;padding:18px;}
.card{margin-bottom:16px;border-radius:12px;box-shadow:0 2px 6px rgba(0,0,0,0.08);}
.card-body{padding:20px;}
.table{font-size:0.875rem;text-align:center;}
.endpoint-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.badge-stat{font-weight:600;padding:6px 12px;border-radius:8px;margin-left:6px;font-size:0.9rem;}
.badge-success{background-color:#28a745;color:#fff;}
.badge-danger{background-color:#dc3545;color:#fff;}
.sparkline{width:180px;height:36px;display:inline-block;}
</style>
"""

js_PER_EP = json.dumps(per_ep_data)
js_overall_rps = json.dumps({"x": x_labels, "y": overall_rps})
js_donut = json.dumps({"labels": donut_labels, "values": donut_values})
js_phases = json.dumps(phases_data)
xaxis_type = "date" if TIME_AXIS_MODE=="datetime" else "linear"

start_dt = time_points[0] if time_points else ""
end_dt = time_points[-1] if time_points else ""
total_tx = sum(sum(ep_counts[ep]) for ep in endpoints)

# ---------------- BUILD HTML ----------------
html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Artillery Report</title>{bootstrap}</head><body>
<div class='container-fluid'>
<div class='card'><div class='card-body'>
<h3>🚀 Artillery Performance Report</h3>
<div class='row'>
<div class='col-md-4'><strong>Start:</strong><div>{start_dt}</div></div>
<div class='col-md-4'><strong>End:</strong><div>{end_dt}</div></div>
<div class='col-md-4'><strong>Total tx:</strong><div>{total_tx}</div></div>
</div></div></div>

<div class='card'><div class='card-body'><h5>📄 Summary</h5>
{summary_df.to_html(classes="table table-striped table-bordered", index=False)}</div></div>

<div class='row'>
<div class='col-md-6'><div class='card'><div class='card-body'><h5>⚠️ Errors</h5>
{error_df.to_html(classes="table table-striped table-bordered", index=False) if not error_df.empty else "<div>No endpoint errors.</div>"}
</div></div></div>
<div class='col-md-6'><div class='card'><div class='card-body'><h5>🍩 Error Distribution</h5><div id='donut' style='height:260px;'></div></div></div></div>
</div>

<div class='card'><div class='card-body'><h5>📈 Avg Response Time</h5><div id='chart-avg' style='height:420px;'></div></div></div>
<div class='card'><div class='card-body'><h5>💥 Requests Per Second</h5><div id='chart-rps' style='height:420px;'></div></div></div>
<div class='card'><div class='card-body'><h5>🔢 HTTP Status Codes</h5><div id='chart-codes' style='height:420px;'></div></div></div>

<div class='card'><div class='card-body'><h5>🔎 Per-endpoint breakdown</h5>
"""

for ep in endpoints:
    d = per_ep_data[ep]
    html += f"""
    <div class='endpoint-row'>
        <div style='flex:1'><b>{ep}</b></div>
        <div><span class='badge badge bg-secondary'>Total: {d['total']}</span>
        <span class='badge badge-success'>2xx: {d['pass_pct']}%</span>
        <span class='badge badge-danger'>Err: {d['err_pct']}%</span></div>
        <div style='width:180px;margin-left:12px;'><div id='spark-{d["safe_id"]}' class='sparkline'></div></div>
        <a class='btn btn-sm btn-outline-primary' data-bs-toggle='collapse' href='#collapse-{d["safe_id"]}'>Details</a>
    </div>
    <div class='collapse' id='collapse-{d["safe_id"]}'>
        <div class='card card-body'>
            <div class='row'>
                <div class='col-md-6'><div id='ep-{d["safe_id"]}-rt' style='height:260px;'></div></div>
                <div class='col-md-6'><div id='ep-{d["safe_id"]}-codes' style='height:260px;'></div></div>
            </div>
        </div>
    </div>
    """

html += "</div></div>"

html += f"""
<script src='https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js'></script>
<script>
const PER_EP = {js_PER_EP};
const OVERALL_RPS = {js_overall_rps};
const DONUT = {js_donut};
const PHASES = {js_phases};
const TIME_AXIS_TYPE = '{xaxis_type}';

function layout_base(title, ytitle) {{
    return {{
        title:{{text:title}},
        xaxis:{{title:'Time', type: TIME_AXIS_TYPE}},
        yaxis:{{title:ytitle}},
        hovermode:'closest',
        shapes: [],
        annotations: []
    }};
}}

function addPhaseShapesAndMarkers(layout, traces, yMax) {{
    let x = 0;
    PHASES.forEach(function(ph) {{
        // Rectangle for phase
        layout.shapes.push({{
            type: 'rect',
            x0: x, x1: x + ph.duration,
            y0: 0, y1: yMax,
            fillcolor: 'rgba(0,123,255,0.08)',
            line: {{ width: 0 }}
        }});

        // Annotation with hover info
        layout.annotations.push({{
            x: x + ph.duration / 2,
            y: yMax * 1.02,
            text: ph.name,
            showarrow: false,
            font: {{ size: 10, color: '#007bff' }},
            xref: 'x', yref: 'y',
            hovertext: `Phase: ${{ph.name}}<br>Duration: ${{ph.duration}}s<br>Arrival Rate: ${{ph.arrival}}`,
            hoverlabel: {{ bgcolor: 'lightblue', font: {{ size: 10 }} }}
        }});

        x += ph.duration;
    }});
}}

document.addEventListener('DOMContentLoaded', function(){{
    // Sparklines
    Object.keys(PER_EP).forEach(ep => {{
        var div = document.getElementById('spark-' + PER_EP[ep].safe_id);
        if(div){{
            var trace = [{{x:PER_EP[ep].spark_x, y:PER_EP[ep].spark_y, mode:'lines', line:{{width:1}}, hoverinfo:'none'}}];
            var layout_s = {{margin:{{t:2,b:2,l:2,r:2}}, xaxis:{{visible:false}}, yaxis:{{visible:false}}}};
            Plotly.newPlot(div, trace, layout_s, {{displayModeBar:false, staticPlot:true, responsive:true}});
        }}
    }});

    if({str(SHOW_GRAPHS).lower()}){{
        // Main charts
        let traces_avg = [], traces_rps=[], traces_codes=[];
        Object.keys(PER_EP).forEach(ep => {{
            let d = PER_EP[ep];
            traces_avg.push({{x:d.rt_x, y:d.rt_y, mode:'lines+markers', name:ep}});
            traces_rps.push({{x:d.rt_x, y:d.rt_y.map(v=>v/100), mode:'lines', name:ep}});
            Object.keys(d.codes).forEach(code => {{
                traces_codes.push({{x:d.rt_x, y:d.codes[code], mode:'lines+markers', name:ep+'_'+code}});
            }});
        }});

        let layout_avg = layout_base('Avg Response Time','ms');
        addPhaseShapesAndMarkers(layout_avg, traces_avg, Math.max(...traces_avg.flatMap(t=>t.y))*1.1);
        Plotly.newPlot('chart-avg', traces_avg, layout_avg, {{responsive:true}});

        let layout_rps = layout_base('Requests per Second','req/s');
        addPhaseShapesAndMarkers(layout_rps, traces_rps, Math.max(...traces_rps.flatMap(t=>t.y))*1.1);
        Plotly.newPlot('chart-rps', traces_rps, layout_rps, {{responsive:true}});

        let layout_codes = layout_base('HTTP Status Codes','count');
        addPhaseShapesAndMarkers(layout_codes, traces_codes, Math.max(...traces_codes.flatMap(t=>t.y))*1.1);
        Plotly.newPlot('chart-codes', traces_codes, layout_codes, {{responsive:true}});

        // Donut chart
        if(DONUT.labels.length>0){{
            Plotly.newPlot('donut', [{{labels:DONUT.labels, values:DONUT.values, type:'pie', hole:0.6}}], {{margin:{{t:20,b:20,l:20,r:20}}}}, {{responsive:true}});
        }}
    }}

    // Collapsible endpoint charts
    document.querySelectorAll('.collapse').forEach(c => {{
        c.addEventListener('shown.bs.collapse', function(){{
            let id = c.id.replace('collapse-','');
            let epKey = Object.keys(PER_EP).find(k=>PER_EP[k].safe_id===id);
            if(!epKey) return;

            let rt_div = document.getElementById('ep-'+id+'-rt');
            let codes_div = document.getElementById('ep-'+id+'-codes');

            if(rt_div){{
                let traces = [
                    {{x:PER_EP[epKey].rt_x, y:PER_EP[epKey].rt_y, mode:'lines+markers', name:'avg'}},
                    {{x:PER_EP[epKey].rt_x, y:PER_EP[epKey].p90, mode:'lines+markers', name:'p90'}},
                    {{x:PER_EP[epKey].rt_x, y:PER_EP[epKey].p95, mode:'lines+markers', name:'p95'}},
                    {{x:PER_EP[epKey].rt_x, y:PER_EP[epKey].p99, mode:'lines+markers', name:'p99'}}
                ];
                let yMax = Math.max(...traces.flatMap(t=>t.y));
                let layout = layout_base(epKey+' response times','ms');
                addPhaseShapesAndMarkers(layout, traces, yMax||1);
                Plotly.react(rt_div, traces, layout, {{responsive:true}});
            }}

            if(codes_div){{
                let traces_codes = [];
                Object.keys(PER_EP[epKey].codes).forEach(code => {{
                    traces_codes.push({{x:PER_EP[epKey].rt_x, y:PER_EP[epKey].codes[code], mode:'lines+markers', name:code}});
                }});
                let yMaxCodes = Math.max(...traces_codes.flatMap(t=>t.y||[0]));
                let layout_codes = layout_base(epKey+' HTTP Codes','count');
                addPhaseShapesAndMarkers(layout_codes, traces_codes, yMaxCodes||1);
                Plotly.react(codes_div, traces_codes, layout_codes, {{responsive:true}});
            }}
        }}, {{once:true}});
    }});
}});
</script>
</body></html>
"""

with open(OUTPUT_HTML,"w",encoding="utf-8") as f:
    f.write(html)
print(f"✅ Full report generated (LOCAL TIME): {OUTPUT_HTML}")
# Sample Python script placeholder
