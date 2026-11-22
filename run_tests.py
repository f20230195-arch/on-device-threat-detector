#!/usr/bin/env python3
"""Reproducible test of the detectors on test_cases.csv (safe/scam SMS, safe/phishing URLs).

  python run_tests.py              # detectors + rules + verdict only (fast, deterministic)
  python run_tests.py --llm        # also run Llama for every case (slow; records latency)

Counting: a case is FLAGGED if the verdict is CAUTION or HIGH RISK; HIGH-only is reported separately.
test_cases.csv is a SYNTHETIC set written independently of the models; add real messages (with personal
details removed) to the CSV to extend it. Do not tune rules against these results.
"""
import argparse, csv, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pi_pipeline as P

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default=os.path.join(HERE, "test_cases.csv"))
ap.add_argument("--llm", action="store_true")
ap.add_argument("--proj-dir", default=str(P.DEFAULT_PROJ_DIR))
ap.add_argument("--model-dir", default=str(P.DEFAULT_MODEL_DIR))
args = ap.parse_args()

P.init_detectors(args.proj_dir)
if args.llm:
    print(f"Llama loaded in {P.init_llm(args.model_dir):.2f} s")

with open(args.cases, newline="", encoding="utf-8") as f:
    cases = list(csv.DictReader(f))

rows = []
for c in cases:
    t0 = time.perf_counter()
    if args.llm:
        rec = P.run(c["text"], verbose=False)
        verdict, e2e = rec["verdict"], rec["end_to_end_ms"]
    else:
        verdict = P.overall_verdict(P.analyze(c["text"]))
        e2e = (time.perf_counter() - t0) * 1000
    bad_class = c["expected"] in ("scam", "phishing")
    flagged = verdict in ("CAUTION", "HIGH RISK")
    rows.append({**c, "verdict": verdict, "flagged": flagged, "high": verdict == "HIGH RISK",
                 "correct": flagged == bad_class, "correct_high_only": (verdict == "HIGH RISK") == bad_class,
                 "ms": round(e2e, 1)})
    mark = "ok  " if rows[-1]["correct"] else "MISS"
    print(f"{mark} {c['category']:<22} {verdict:<9} {c['text'][:70]}")

print("\n=== Per category (flagged = CAUTION or HIGH RISK) ===")
print(f"{'category':<24}{'cases':>6}{'flagged':>9}{'HIGH only':>11}{'correct':>9}{'correct(HIGH only)':>20}")
for cat in dict.fromkeys(r["category"] for r in rows):
    g = [r for r in rows if r["category"] == cat]
    print(f"{cat:<24}{len(g):>6}{sum(r['flagged'] for r in g):>9}{sum(r['high'] for r in g):>11}"
          f"{sum(r['correct'] for r in g):>9}{sum(r['correct_high_only'] for r in g):>20}")
bad = [r for r in rows if r["expected"] in ("scam", "phishing")]
good = [r for r in rows if r["expected"] == "safe"]
print(f"\nScams/phishing caught: {sum(r['flagged'] for r in bad)}/{len(bad)} (HIGH only: {sum(r['high'] for r in bad)}/{len(bad)})")
print(f"Safe items falsely flagged: {sum(r['flagged'] for r in good)}/{len(good)} (HIGH only: {sum(r['high'] for r in good)}/{len(good)})")

os.makedirs(P.RESULTS_DIR, exist_ok=True)
out = P.RESULTS_DIR / f"tests_{time.strftime('%Y%m%d_%H%M%S')}.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"Saved: {out}")
