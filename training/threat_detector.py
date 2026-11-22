import re
import joblib
import pandas as pd
from urllib.parse import urlparse

# ---------- Load the trained models ----------
sms_model = joblib.load("sms_model.joblib")
url_model = joblib.load("url_model.joblib")

# ---------- URL feature extraction (same as url_detector.py) ----------
SUSPICIOUS_WORDS = ["login", "verify", "update", "secure", "account",
                    "bank", "confirm", "free", "bonus", "paypal", "signin"]

FEATURE_NAMES = ["len", "host_len", "path_len", "query_len", "dots",
                 "hyphens", "slashes", "at", "qmark", "equals", "percent",
                 "digits", "digit_ratio", "subdomains", "is_ip", "https",
                 "suspicious_words"]

def safe_parse(url):
    try:
        p = urlparse(url if "://" in url else "http://" + url)
        return p.netloc.lower(), p.path, p.query
    except ValueError:
        rest = url.split("://", 1)[-1]
        host, _, remainder = rest.partition("/")
        path, _, query = remainder.partition("?")
        return host.lower(), "/" + path, query

def url_features(url):
    url = str(url).strip()
    host, path, query = safe_parse(url)
    low = url.lower()
    return [
        len(url), len(host), len(path), len(query),
        url.count("."), url.count("-"), url.count("/"), url.count("@"),
        url.count("?"), url.count("="), url.count("%"),
        sum(c.isdigit() for c in url),
        sum(c.isdigit() for c in url) / max(len(url), 1),
        host.count("."),
        int(bool(re.fullmatch(r"[\d.:]+", host))),
        int(low.startswith("https")),
        sum(w in low for w in SUSPICIOUS_WORDS),
    ]

# ---------- Trusted domains (Tranco list + built-in fallback) ----------
TRANCO_FILE = "tranco_top_1000_latest.csv"   # <-- change if your file name differs

BUILTIN_TRUSTED = {
    "google.com", "youtube.com", "wikipedia.org", "github.com",
    "microsoft.com", "apple.com", "amazon.com", "amazon.in",
    "facebook.com", "instagram.com", "linkedin.com", "twitter.com",
    "x.com", "whatsapp.com", "bits-pilani.ac.in", "gov.in", "nic.in",
}

# Hosting platforms that anyone can publish on. Never auto-trust these.
ABUSABLE_HOSTS = {"blogspot.com", "github.io", "weebly.com", "wixsite.com",
                  "herokuapp.com", "netlify.app", "web.app", "pages.dev",
                  "vercel.app", "000webhostapp.com", "firebaseapp.com"}

DOMAIN_RE = re.compile(r"[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?(?:\.[a-z0-9\-]+)*\.[a-z]{2,}")

def load_trusted(path):
    trusted = set()
    try:
        with open(path, encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                matches = DOMAIN_RE.findall(line.strip().strip('"').lower())
                if matches:
                    d = matches[-1]
                    if d.startswith("www."):
                        d = d[4:]
                    trusted.add(d)
        print(f"Loaded {len(trusted)} trusted domains from {path}")
    except FileNotFoundError:
        print(f"WARNING: {path} not found. Using built-in list only.")
    return (trusted | BUILTIN_TRUSTED) - ABUSABLE_HOSTS

TRUSTED_DOMAINS = load_trusted(TRANCO_FILE)
print("Startup check: google.com trusted?", "google.com" in TRUSTED_DOMAINS)

def parent_domains(host):
    """www.mail.google.com -> [www.mail.google.com, mail.google.com, google.com, com]"""
    parts = host.split(".")
    return [".".join(parts[i:]) for i in range(len(parts))]

def is_trusted(url):
    host, _, _ = safe_parse(url)
    host = host.split(":")[0]
    candidates = parent_domains(host)
    if any(c in ABUSABLE_HOSTS for c in candidates):
        return False
    return any(c in TRUSTED_DOMAINS for c in candidates)

# ---------- Detection functions ----------
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

def check_url(url):
    if is_trusted(url):
        return 0.02, True
    x = pd.DataFrame([url_features(url)], columns=FEATURE_NAMES)
    return url_model.predict_proba(x)[0][1], False

def check_sms(text):
    return sms_model.predict_proba([text])[0][1]

def looks_like_bare_url(text):
    t = text.strip()
    return " " not in t and ("." in t or "://" in t)

def analyze(text):
    text = text.strip()
    result = {"type": None, "sms_risk": None, "url_risks": {}}

    if looks_like_bare_url(text):
        result["type"] = "URL"
        result["url_risks"][text] = check_url(text)
    else:
        result["type"] = "SMS"
        result["sms_risk"] = check_sms(text)
        for link in URL_PATTERN.findall(text):
            result["url_risks"][link] = check_url(link)
    return result

def verdict(score, threshold=0.5):
    return "HIGH RISK" if score > threshold else "LOW RISK"

# ---------- Interactive demo ----------
if __name__ == "__main__":
    print("\n=== On-Device Threat Detector ===")
    print("Paste an SMS or a URL (or 'q' to quit)\n")
    while True:
        text = input("> ")
        if text.strip().lower() == "q":
            break
        if not text.strip():
            continue

        r = analyze(text)
        print(f"\nInput type: {r['type']}")
        if r["sms_risk"] is not None:
            print(f"  Message scam score: {r['sms_risk']:.2f} -> {verdict(r['sms_risk'])}")
        for link, (score, trusted) in r["url_risks"].items():
            tag = " [trusted domain]" if trusted else ""
            print(f"  Link {link[:60]}")
            print(f"    Phishing score: {score:.2f} -> {verdict(score)}{tag}")
        print()