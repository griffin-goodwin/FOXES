import os, re, glob
from datetime import datetime, timedelta
from collections import defaultdict

BASE = "/mnt/data/SDO-AIA-flaring"   # <- change if the root is different
YEAR = "2017"  # change the year
WLS  = ["94","131","171","193","211","304"]  # folders under BASE

# regex to pull "YYYY-MM-DDTHH:MM:SS" from filename
STAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")

def parse_ts_from_path(path):
    m = STAMP_RE.search(os.path.basename(path))
    if not m:
        return None
    return datetime.fromisoformat(m.group(1))

def find_gaps(timestamps, step=timedelta(minutes=1)):
    """Return list of (prev, next, gap_len_minutes) where diff > step."""
    gaps = []
    for a, b in zip(timestamps, timestamps[1:]):
        diff = b - a
        if diff > step:
            gaps.append((a, b, int(diff.total_seconds()//60)))
    return gaps

def human(ts):
    return ts.isoformat(timespec="seconds")

summary = []

for wl in WLS:
    pattern = os.path.join(BASE, wl, f"{YEAR}*.fits")
    files = glob.glob(pattern)
    ts = [parse_ts_from_path(p) for p in files]
    ts = sorted(t for t in ts if t is not None)

    if not ts:
        print(f"[{wl} Å] No files found for {YEAR}")
        summary.append((wl, 0, None, None, 0, 0))
        continue

    gaps = find_gaps(ts, timedelta(minutes=1))

    print(f"\n[{wl} Å]")
    print(f"  Files: {len(ts)}")
    print(f"  First: {human(ts[0])}")
    print(f"  Last : {human(ts[-1])}")
    print(f"  Gap segments (>1 min): {len(gaps)}")

    if gaps:
        biggest = max(gaps, key=lambda g: g[2])
        print(f"Largest gap: {biggest[2]} minutes  ({human(biggest[0])} -> {human(biggest[1])})")
        print("Sample gaps:")
        for g in gaps[:10]:  # show first 10
            print(f"    {human(g[0])} -> {human(g[1])}  ({g[2]} min)")
    else:
        print("  No gaps detected within sequences.")

    summary.append((wl, len(ts), ts[0], ts[-1], len(gaps), max([g[2] for g in gaps], default=0)))

# Overall quick view
print("\n=== Summary ===")
for wl, n, fst, lst, ngaps, maxgap in summary:
    fst_s = human(fst) if fst else "-"
    lst_s = human(lst) if lst else "-"
    print(f"{wl:>3} Å  files={n:6}  gaps={ngaps:5}  max_gap(min)={maxgap:5}  range={fst_s} .. {lst_s}")
