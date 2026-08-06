#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Table 3 1M row n=6: combine p2d2 (docs 0-2) + ext d3/d4/d5 (1 each) -> per-doc table
# + n6 medians (req_speedup, req_time OFF/ON sec, prefill/decode speedup). Raw values only.
import json, glob, statistics, sys
R = "/home/USER/atflash/shared/results"

def items_of(d):
    s = json.load(open(f"{d}/e2e_summary.json"))
    return s["groups"][0]["items"]

rows = []
p2d2 = f"{R}/20260716_103143_phasec_1m_qual_p2d2"
for it in items_of(p2d2):
    rows.append(("p2d2(0-2)", it))
for s in (3, 4, 5):
    ds = sorted(glob.glob(f"{R}/*phasec_1m_tim_ext_d{s}"))
    if not ds:
        print(f"[warn] ext d{s} dir missing"); continue
    try:
        for it in items_of(ds[-1]):
            rows.append((f"ext_d{s}", it))
    except FileNotFoundError:
        print(f"[warn] ext d{s} e2e_summary not ready")

print(f"== 1M e2e combined  n={len(rows)} ==")
F = ["id", "req_time_s_off", "req_time_s_win", "req_speedup_win_vs_off",
     "prefill_speedup_win_vs_off", "decode_speedup_win_vs_off",
     "prefill_ms_off", "prefill_ms_win"]
for src, it in rows:
    d = {k: (round(it[k], 3) if isinstance(it.get(k), float) else it.get(k)) for k in F}
    print(f"  {src:10} id={d['id']}  OFF={d['req_time_s_off']}s ON={d['req_time_s_win']}s "
          f"req={d['req_speedup_win_vs_off']}x pf={d['prefill_speedup_win_vs_off']}x dec={d['decode_speedup_win_vs_off']}x")

def med(k):
    v = [it[k] for _, it in rows if it.get(k) is not None]
    return statistics.median(v) if v else float("nan")

print("\n-- n%d MEDIAN --" % len(rows))
print(f"  req_speedup(win/off) = {med('req_speedup_win_vs_off'):.4f}x")
print(f"  req_time_off = {med('req_time_s_off'):.1f}s   req_time_win = {med('req_time_s_win'):.1f}s")
print(f"  prefill_speedup = {med('prefill_speedup_win_vs_off'):.4f}x   decode_speedup = {med('decode_speedup_win_vs_off'):.4f}x")
print(f"  (existing n3 p2d2 published: req 1.3014x / OFF 765.8s / ON 588.2s)")
