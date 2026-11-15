"""
prepare_data.py  (v3, verified)

Builds dataset.csv (one row per one-second interval) from the first one million lines of access.log
(Kaggle: eliasdabbas/web-server-access-logs, nginx "combined" format, zanbil.ir, Jan 2019).

WHAT IS REAL AND WHAT IS GENERATED
  real      : RPS (log lines per second), Payload_Size_KB (mean of the bytes field / 1024)
  generated : Active_DB_Connections, Memory_Usage_MB, CPU_Load_percent, Latency_ms
              (the log has NO server-side metrics and NO response-time field)

The script counts how many lines end in a numeric token (a possible response-time field).
For this log that count is 0, so every latency value is simulated:
    per request:  15 ms + 0.5 ms/KB * payload + N(0, 5 ms)      (averaged over the second)
    CPU > 80 %  : + 12 ms per percentage point above 80          (added after the outlier filter)
Counts and parameters are written to dataset_meta.json.

Run:  python src/prepare_data.py [access.log] [output.csv]
      (defaults: ./access.log -> data/dataset_regenerated.csv; the frozen data/dataset.csv is never overwritten)
NOTE: the paper's numbers come from the frozen data/dataset.csv (see train_models.py, FROZEN_SHA256).
      Re-running this script with SEED=42 gives a new, reproducible CSV with the same structure,
      but numbers in Table 1 will move slightly; use results.json from train_models.py to update them.
"""
import csv
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT_DIR, "access.log")
ROOT = os.path.dirname(HERE)
OUTPUT_CSV = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "data", "dataset_regenerated.csv")
META_JSON = os.path.splitext(os.path.abspath(OUTPUT_CSV))[0] + "_meta.json"

SEED = 42
N_LINES = 1_000_000          # first million log entries
MAX_ROWS = 50_000
BASE_MS, SLOPE_MS_PER_KB, NOISE_SD_MS = 15.0, 0.5, 5.0
MISSING_BYTES_DEFAULT = 500  # bytes field "-" -> 500 B

rng = random.Random(SEED)
rps_counter = defaultdict(int)
payload_counter = defaultdict(list)
latency_counter = defaultdict(list)
st = {"lines_read": 0, "parsed": 0, "skipped": 0, "bytes_missing": 0, "numeric_last_token": 0}


def is_number(s):
    return s.replace(".", "", 1).isdigit()


print("Reading log and aggregating per second ...")
with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if i >= N_LINES:
            break
        st["lines_read"] += 1
        parts = line.split()
        if len(parts) < 10:
            st["skipped"] += 1
            continue
        try:
            dt = datetime.strptime(parts[3][1:].split("+")[0].split("-")[0], "%d/%b/%Y:%H:%M:%S")
            sec = int(dt.timestamp())
            if parts[9].isdigit():
                payload_kb = int(parts[9]) / 1024.0
            else:
                payload_kb = MISSING_BYTES_DEFAULT / 1024.0
                st["bytes_missing"] += 1
            if is_number(parts[-1]):                     # a real response-time field would end up here
                st["numeric_last_token"] += 1
                latency = float(parts[-1])
            else:
                latency = BASE_MS + SLOPE_MS_PER_KB * payload_kb + rng.gauss(0, NOISE_SD_MS)
            rps_counter[sec] += 1
            payload_counter[sec].append(payload_kb)
            latency_counter[sec].append(latency)
            st["parsed"] += 1
        except Exception:
            st["skipped"] += 1

print(f"Parsed {st['parsed']:,} of {st['lines_read']:,} lines; {len(rps_counter):,} distinct seconds; "
      f"lines with a numeric last token (possible response time): {st['numeric_last_token']:,}")

rows, dropped, cpu_over_80 = [], 0, 0
for sec, rps in rps_counter.items():
    avg_payload = sum(payload_counter[sec]) / len(payload_counter[sec])
    avg_latency = sum(latency_counter[sec]) / len(latency_counter[sec])
    if avg_latency < 1.0 or avg_latency > 2000.0 or rps > 5000:     # outlier filter (before the CPU penalty)
        dropped += 1
        continue
    x_db = max(1, int(rps * 0.8 + rng.randint(-5, 15)))
    x_ram = 2048 + rps * 1.5 + rng.gauss(0, 50)
    x_cpu = min(99.0, max(10.0, (rps / 50.0) * 15 + rng.gauss(0, 3)))
    if x_cpu > 80:
        avg_latency += (x_cpu - 80) * 12.0
        cpu_over_80 += 1
    rows.append([round(rps, 2), x_db, round(x_ram, 2), round(avg_payload, 2), round(x_cpu, 2), round(avg_latency, 2)])
    if len(rows) >= MAX_ROWS:
        break

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["RPS", "Active_DB_Connections", "Memory_Usage_MB", "Payload_Size_KB", "CPU_Load_percent", "Latency_ms"])
    w.writerows(rows)

meta = {"seed": SEED, "log_file": os.path.basename(LOG_FILE), "lines_requested": N_LINES, **st,
        "seconds_total": len(rps_counter), "dropped_by_filter": dropped, "rows_written": len(rows),
        "rows_cpu_over_80": cpu_over_80,
        "latency_generator": {"base_ms": BASE_MS, "slope_ms_per_kb": SLOPE_MS_PER_KB, "noise_sd_ms": NOISE_SD_MS,
                              "cpu_penalty_ms_per_pct_over_80": 12.0}}
with open(META_JSON, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print(f"Done: {OUTPUT_CSV} ({len(rows):,} rows), {META_JSON}")
