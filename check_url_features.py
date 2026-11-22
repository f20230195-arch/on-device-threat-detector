#!/usr/bin/env python3
"""Check that pi_pipeline.url_features() is IDENTICAL to features() in url_detector.py (the training script).

url_detector.py runs a training job when imported, so this script does NOT import it: it reads the
file, extracts only SUSPICIOUS_WORDS, FEATURE_NAMES, safe_parse() and features(), and runs them.

Usage:
  python check_url_features.py                          # built-in edge-case URLs
  python check_url_features.py --csv phishing_site_urls.csv --n 5000   # plus real dataset rows
"""
import argparse, ast, os, re, sys
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pi_pipeline as P

ap = argparse.ArgumentParser()
ap.add_argument("--train", default=os.path.join(HERE, "url_detector.py"))
ap.add_argument("--csv"); ap.add_argument("--n", type=int, default=5000)
args = ap.parse_args()

src = open(args.train, encoding="utf-8").read()
keep = []
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("safe_parse", "features"):
        keep.append(node)
    elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in ("SUSPICIOUS_WORDS", "FEATURE_NAMES") for t in node.targets):
        keep.append(node)
ns = {"re": re, "urlparse": urlparse}
exec(compile(ast.Module(body=keep, type_ignores=[]), args.train, "exec"), ns)
train_features = ns["features"]

ok = True
for name in ("SUSPICIOUS_WORDS", "FEATURE_NAMES"):
    same = ns[name] == getattr(P, name)
    ok &= same
    print(f"{name}: {'IDENTICAL' if same else 'DIFFERENT'}")

urls = [
    "https://www.google.com", "http://secure-login-paypal.verify-account.xyz/update",
    "http://paytm-secure-offer.xyz/?id=123", "http://bit.ly/xyz", "192.168.1.1/admin",
    "http://[::1/login", "http://a]b.com/x?y=1", "http://user@evil.com/%20%41", "HTTP://EXAMPLE.COM/A/B/C?x=1&y=2",
    "example.com", "https://xn--80ak6aa92e.com/", "", "   http://padded.com   ", "http://1.2.3.4:8080/path",
]
if args.csv:
    import pandas as pd
    df = pd.read_csv(args.csv, encoding="latin-1")
    col = next((c for c in df.columns if c.lower() == "url"), df.columns[0])
    urls += df[col].dropna().sample(min(args.n, len(df)), random_state=1).astype(str).tolist()

bad = 0
for u in urls:
    a, b = train_features(u), P.url_features(u)
    if a != b:
        bad += 1
        if bad <= 5:
            print("MISMATCH:", repr(u), "\n  training:", a, "\n  pi      :", b)
ok &= bad == 0
print(f"\nChecked {len(urls)} URLs: {len(urls) - bad} identical, {bad} different")
print("RESULT:", "PASS - features are identical" if ok else "FAIL")
sys.exit(0 if ok else 1)
