import re
import pandas as pd
import joblib
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

CSV_FILE = "phishing_site_urls.csv"
MAX_ROWS = 100000          # the file is large, this keeps it fast

# ---------- Load data ----------
df = pd.read_csv(CSV_FILE, encoding="latin-1")
print("Columns found:", df.columns.tolist())

cols = {c.lower(): c for c in df.columns}
url_col = cols.get("url") or df.columns[0]
label_col = (cols.get("label") or cols.get("type")
             or cols.get("result") or df.columns[-1])
df = df[[url_col, label_col]].dropna()
df.columns = ["url", "label"]

# Turn labels into 0 (safe) / 1 (bad)
SAFE = {"good", "benign", "legitimate", "safe", "0"}
df["y"] = (~df["label"].astype(str).str.lower().str.strip().isin(SAFE)).astype(int)
print("\nLabel counts (0 = safe, 1 = bad):")
print(df["y"].value_counts())

if len(df) > MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=42)

# ---------- Feature extraction ----------
SUSPICIOUS_WORDS = ["login", "verify", "update", "secure", "account",
                    "bank", "confirm", "free", "bonus", "paypal", "signin"]

def safe_parse(url):
    """Return (host, path, query). Falls back to manual splitting
    when urlparse fails on malformed URLs (e.g. stray brackets)."""
    try:
        p = urlparse(url if "://" in url else "http://" + url)
        return p.netloc.lower(), p.path, p.query
    except ValueError:
        rest = url.split("://", 1)[-1]
        host, _, remainder = rest.partition("/")
        path, _, query = remainder.partition("?")
        return host.lower(), "/" + path, query

def features(url):
    url = str(url).strip()
    host, path, query = safe_parse(url)
    low = url.lower()
    return [
        len(url),
        len(host),
        len(path),
        len(query),
        url.count("."),
        url.count("-"),
        url.count("/"),
        url.count("@"),
        url.count("?"),
        url.count("="),
        url.count("%"),
        sum(c.isdigit() for c in url),
        sum(c.isdigit() for c in url) / max(len(url), 1),
        host.count("."),                                   # subdomain depth
        int(bool(re.fullmatch(r"[\d.:]+", host))),         # IP instead of domain
        int(low.startswith("https")),
        sum(w in low for w in SUSPICIOUS_WORDS),
    ]

FEATURE_NAMES = ["len", "host_len", "path_len", "query_len", "dots",
                 "hyphens", "slashes", "at", "qmark", "equals", "percent",
                 "digits", "digit_ratio", "subdomains", "is_ip", "https",
                 "suspicious_words"]

print("\nExtracting features (this can take a minute)...")
X = pd.DataFrame([features(u) for u in df["url"]], columns=FEATURE_NAMES)
y = df["y"].values

# ---------- Train and evaluate ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, max_depth=20,
                               class_weight="balanced",
                               random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\n=== Phishing URL detector results ===")
print(classification_report(y_test, model.predict(X_test),
                            target_names=["safe", "bad"]))

joblib.dump(model, "url_model.joblib")

# ---------- Interactive demo ----------
print("Type a URL (or 'q' to quit):")
while True:
    u = input("> ")
    if u.strip().lower() == "q":
        break
    p = model.predict_proba(pd.DataFrame([features(u)],
                                         columns=FEATURE_NAMES))[0][1]
    print(f"{'PHISHING / BAD' if p > 0.5 else 'LIKELY SAFE'} "
          f"(risk score: {p:.2f})")