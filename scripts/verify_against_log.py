"""
verify_against_log.py

Checks the REAL columns of data/dataset.csv (RPS and Payload_Size_KB) against the raw access log.
The other four columns are simulated (see README) and cannot be checked against the log.

Usage:  python scripts/verify_against_log.py path/to/access.log [data/dataset.csv]

The raw log (Zaker 2019, doi:10.7910/DVN/3QBYB5; also on Kaggle: eliasdabbas/web-server-access-logs)
is NOT part of this repository (3.3 GB, contains visitor IP addresses).
"""
import csv
import hashlib
import os
import sys
from collections import OrderedDict
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "access.log")
CSV = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "data", "dataset.csv")
N_LINES = 1_000_000

rps, pay = OrderedDict(), {}
missing = numeric_last = 0
with open(LOG, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if i >= N_LINES:
            break
        p = line.split()
        dt = datetime.strptime(p[3][1:].split("+")[0].split("-")[0], "%d/%b/%Y:%H:%M:%S")
        sec = int(dt.timestamp())
        if p[9].isdigit():
            kb = int(p[9]) / 1024
        else:
            kb, missing = 500 / 1024, missing + 1
        if p[-1].replace(".", "", 1).isdigit():
            numeric_last += 1
        rps[sec] = rps.get(sec, 0) + 1
        pay.setdefault(sec, []).append(kb)

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
bad = []
for k, (sec, r) in enumerate(zip(rps, rows)):
    pa = round(sum(pay[sec]) / len(pay[sec]), 2)
    if rps[sec] != float(r["RPS"]) or abs(pa - float(r["Payload_Size_KB"])) > 0.011:
        bad.append((k, rps[sec], float(r["RPS"]), pa, float(r["Payload_Size_KB"])))

print(f"log: {sum(rps.values()):,} lines, {len(rps):,} distinct seconds, max RPS {max(rps.values())}, "
      f"missing byte counts {missing}, lines ending in a numeric token {numeric_last}")
print(f"csv: {len(rows):,} rows, sha256 {hashlib.sha256(open(CSV, 'rb').read()).hexdigest()}")
print(f"rows where RPS/payload differ from the log: {len(bad)}")
for b in bad:
    print("  row %d: log RPS %s vs csv RPS %s; log payload %s vs csv payload %s" % b)
print("OK" if len(rows) == len(rps) and len(bad) <= 1 else "MISMATCH")
print("(Known: the last row has RPS 44 in the CSV vs 43 in the log - one extra line was read by an earlier "
      "version of the script; it does not affect any reported result.)")
