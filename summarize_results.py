#!/usr/bin/env python3
"""Summarise a pi_pipeline.py results log (results/run_*.jsonl) into tables for the report.

Usage:
  python summarize_results.py                 # newest results/run_*.jsonl
  python summarize_results.py results/run_X.jsonl
Also writes <log>.csv next to the log.
"""
import csv, glob, json, os, statistics, sys

path = sys.argv[1] if len(sys.argv) > 1 else None
if path is None:
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "run_*.jsonl")))
    if not files:
        sys.exit("No results/run_*.jsonl found")
    path = files[-1]

meta, recs = {}, []
with open(path, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if "meta" in d:
            meta = d["meta"]
        else:
            recs.append(d)

def num(x, fmt="{:.1f}"):
    return "-" if x is None else fmt.format(x)

rows = []
for i, r in enumerate(recs, 1):
    o = r.get("observer") or {}
    rows.append({
        "iter": r.get("iteration", 1), "n": i,
        "input": r["input"].replace("\n", " ")[:44],
        "type": r["type"], "verdict": r["verdict"],
        "source": r["explanation_source"], "fallback": r.get("fallback_reason") or "",
        "prompt_tok": o.get("prompt_tokens"), "gen_tok": o.get("generated_tokens"),
        "ttft_s": None if o.get("ttft_ms") is None else o["ttft_ms"] / 1000,
        "tok_per_s": o.get("decode_tok_per_s"),
        "llm_s": None if r.get("llm_wall_ms") is None else r["llm_wall_ms"] / 1000,
        "e2e_s": r["end_to_end_ms"] / 1000,
        "detect_ms": r["analyze_ms"],
        "rss_mb": r["peak_rss_mb"], "temp_c": r["temp_after_c"], "throttled": r["throttled_after"],
    })

print(f"Log: {path}")
print(f"Detectors load {meta.get('detectors_load_s')} s | Llama load {meta.get('llm_load_s')} s | "
      f"max_seq_len {meta.get('max_seq_len')} | no_llm {meta.get('no_llm')}\n")
hdr = f"{'it':>2} {'#':>2} {'input':<44} {'type':<4} {'verdict':<9} {'src':<8} {'ptok':>4} {'gtok':>4} {'ttft':>5} {'tok/s':>5} {'e2e_s':>6} {'det_ms':>6} {'rssMB':>5} {'C':>5}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['iter']:>2} {r['n']:>2} {r['input']:<44} {r['type']:<4} {r['verdict']:<9} {r['source']:<8} "
          f"{num(r['prompt_tok'],'{}'):>4} {num(r['gen_tok'],'{}'):>4} {num(r['ttft_s']):>5} {num(r['tok_per_s']):>5} "
          f"{r['e2e_s']:>6.1f} {r['detect_ms']:>6.1f} {r['rss_mb']:>5.0f} {num(r['temp_c']):>5}")

def stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"min {min(vals):.1f} | median {statistics.median(vals):.1f} | mean {statistics.mean(vals):.1f} | max {max(vals):.1f}"

print("\n=== Per iteration (iteration 1 starts with a cold process; later ones are warm in the same process) ===")
for it in sorted({r["iter"] for r in rows}):
    g = [r for r in rows if r["iter"] == it]
    print(f"Iteration {it}: {len(g)} messages")
    print(f"  end-to-end s : {stat([r['e2e_s'] for r in g])}")
    print(f"  first token s: {stat([r['ttft_s'] for r in g])}")
    print(f"  decode tok/s : {stat([r['tok_per_s'] for r in g])}")
    print(f"  first message e2e / ttft: {g[0]['e2e_s']:.1f} s / {num(g[0]['ttft_s'])} s")

fb = [r for r in rows if r["source"] != "llama"]
print(f"\nTemplate fallbacks: {len(fb)} of {len(rows)}")
for r in fb:
    print(f"  #{r['n']} {r['input']!r}: {r['fallback']}")
print(f"Max peak RAM {max(r['rss_mb'] for r in rows):.0f} MB | "
      f"max temp {num(max([r['temp_c'] for r in rows if r['temp_c'] is not None], default=None))} C | "
      f"throttle states seen: {sorted({r['throttled'] for r in rows if r['throttled']})}")

csv_path = os.path.splitext(path)[0] + ".csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\nCSV written: {csv_path}")
