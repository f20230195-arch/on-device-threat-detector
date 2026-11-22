#!/usr/bin/env python3
"""
On-device threat, phishing and scam detector - Raspberry Pi version.

Ported from the Colab "Cell 2" pipeline (pipeline.py). The detectors, rules and
verdict logic are copied unchanged; only these things differ:
  * no Colab / Google Drive / Hugging Face login code, local paths only
  * everything runs offline (HF_HUB_OFFLINE=1)
  * ExecuTorch's PyTorchObserver timing line is CAPTURED and parsed, not hidden
  * per-message timing, RAM and CPU temperature are logged to results/*.jsonl

Expected layout (all next to this script):
  threat_project/        sms_model_v2.joblib  url_model.joblib  tranco_top_1000_latest.csv
  llama32_1b_instruct/   model.pte  tokenizer.json  tokenizer_config.json  chat_template.jinja

Usage:
  python pi_pipeline.py                      # run the 11 demo messages once
  python pi_pipeline.py --repeat 3           # same, 3 times in ONE process (cold vs warm)
  python pi_pipeline.py --text "some SMS"    # one message
  python pi_pipeline.py --no-llm             # detectors only (template explanation)
  python pi_pipeline.py --interactive        # live demo: type or paste a message/URL, q to quit
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import contextlib
import ctypes
import json
import re
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
DEFAULT_PROJ_DIR = BASE / "threat_project"
DEFAULT_MODEL_DIR = BASE / "llama32_1b_instruct"
RESULTS_DIR = BASE / "results"

# ---------- Detector constants (identical to training / Colab) ----------
SUSPICIOUS_WORDS = ["login", "verify", "update", "secure", "account",
                    "bank", "confirm", "free", "bonus", "paypal", "signin"]
FEATURE_NAMES = ["len", "host_len", "path_len", "query_len", "dots",
                 "hyphens", "slashes", "at", "qmark", "equals", "percent",
                 "digits", "digit_ratio", "subdomains", "is_ip", "https",
                 "suspicious_words"]
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly",
              "cutt.ly", "rb.gy", "shorturl.at", "bitly.com", "buff.ly"}
BRANDS = ["paytm", "phonepe", "gpay", "sbi", "hdfc", "icici", "axis", "paypal",
          "amazon", "flipkart", "google", "microsoft", "apple", "netflix",
          "irctc", "aadhaar", "uidai", "incometax"]


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


# ---------- Trusted domains ----------
BUILTIN_TRUSTED = {
    "google.com", "youtube.com", "wikipedia.org", "github.com",
    "microsoft.com", "apple.com", "amazon.com", "amazon.in",
    "facebook.com", "instagram.com", "linkedin.com", "twitter.com",
    "x.com", "whatsapp.com", "bits-pilani.ac.in", "gov.in", "nic.in",
}
# Never auto-trust: shorteners hide the destination, and these hosts let anyone publish.
ABUSABLE_HOSTS = {"blogspot.com", "github.io", "weebly.com", "wixsite.com",
                  "herokuapp.com", "netlify.app", "web.app", "pages.dev",
                  "vercel.app", "000webhostapp.com", "firebaseapp.com"} | SHORTENERS
DOMAIN_RE = re.compile(r"[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?(?:\.[a-z0-9\-]+)*\.[a-z]{2,}")


def load_trusted(path):
    trusted = set()
    try:
        with open(path, encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                m = DOMAIN_RE.findall(line.strip().strip('"').lower())
                if m:
                    d = m[-1]
                    trusted.add(d[4:] if d.startswith("www.") else d)
    except FileNotFoundError:
        print("WARNING: Tranco file not found, using built-in list only")
    return (trusted | BUILTIN_TRUSTED) - ABUSABLE_HOSTS


# Loaded by init_detectors() / init_llm()
sms_model = None
url_model = None
TRUSTED_DOMAINS = set()
tok = None
llm = None


def init_detectors(proj_dir):
    global sms_model, url_model, TRUSTED_DOMAINS
    sms_model = joblib.load(Path(proj_dir) / "sms_model_v2.joblib")
    url_model = joblib.load(Path(proj_dir) / "url_model.joblib")
    TRUSTED_DOMAINS = load_trusted(Path(proj_dir) / "tranco_top_1000_latest.csv")


def parent_domains(host):
    parts = host.split(".")
    return [".".join(parts[i:]) for i in range(len(parts))]


def is_trusted(url):
    host, _, _ = safe_parse(url)
    cands = parent_domains(host.split(":")[0])
    if any(c in ABUSABLE_HOSTS for c in cands):
        return False
    return any(c in TRUSTED_DOMAINS for c in cands)


def brand_lookalike(url):
    """Brand name used as a word in a domain that is not the real, trusted site."""
    if is_trusted(url):
        return None
    host = safe_parse(url)[0].split(":")[0]
    tokens = re.split(r"[.\-_]", host)
    for b in BRANDS:
        if b in tokens:
            return b
    return None


# ---------- SMS scam rules (English + Hinglish) ----------
SMS_RULES = [
    (r"\bkyc\b", "it asks you to update your KYC"),
    (r"(account|a/c|card|sim|number|wallet).{0,40}(block|suspend|deactivat|band ho|expire)",
     "it threatens to block or close your account"),
    (r"(send|share|enter|submit|give|bhej|batao).{0,30}(otp|pin|cvv|password|aadhaar|aadhar|pan card|card number)",
     "it asks you to hand over sensitive details"),
    (r"(lottery|lucky draw|kbc|you have won|you've won|jeet)", "it claims you won a prize"),
    (r"(immediately|urgent|within \d+ ?(hours|hrs)|\d+ ghante|abhi |last chance|today only)",
     "it pressures you to act fast"),
]
NEGATION = re.compile(r"\b(do not|don't|dont|never|mat)\s+(share|send|give|enter|bhej|batao)\b[^.!?]*")
RULE_BOOST, RULE_CAP = 0.12, 0.90


def rule_hits(text):
    low = NEGATION.sub(" ", text.lower())    # ignore "do not share your OTP"
    return [why for pat, why in SMS_RULES if re.search(pat, low)]


# ---------- Detection ----------
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)


def check_url(url):
    if is_trusted(url):
        return 0.02, True
    x = pd.DataFrame([url_features(url)], columns=FEATURE_NAMES)
    score = float(url_model.predict_proba(x)[0][1])
    host = safe_parse(url)[0].split(":")[0]
    if host in SHORTENERS:                 # destination hidden -> never low risk
        score = max(score, 0.55)
    if brand_lookalike(url):               # fake brand domain
        score = min(0.95, score + 0.15)
    return score, False


def check_sms(text):
    return float(sms_model.predict_proba([text])[0][1])


def looks_like_bare_url(text):
    t = text.strip()
    return " " not in t and ("." in t or "://" in t)


def analyze(text):
    text = text.strip()
    r = {"type": None, "sms_model": None, "sms_risk": None,
         "rule_hits": [], "url_risks": {}}
    if looks_like_bare_url(text):
        r["type"] = "URL"
        r["url_risks"][text] = check_url(text)
    else:
        r["type"] = "SMS"
        r["sms_model"] = check_sms(text)
        r["rule_hits"] = rule_hits(text)
        boosted = r["sms_model"] + RULE_BOOST * len(r["rule_hits"])
        r["sms_risk"] = min(RULE_CAP, boosted) if r["rule_hits"] else r["sms_model"]
        for link in URL_PATTERN.findall(text):
            r["url_risks"][link] = check_url(link)
    return r


CAUTION_T = 0.45     # same value as the Colab notebook


def verdict(score):
    if score > 0.5:
        return "HIGH RISK"
    if score > CAUTION_T:
        return "CAUTION"
    return "LOW RISK"


def overall_verdict(r):
    scores = [s for s, _ in r["url_risks"].values()]
    if r["sms_risk"] is not None:
        scores.append(r["sms_risk"])
    return verdict(max(scores)) if scores else "LOW RISK"


# ---------- Plain-language signals (facts for the LLM) ----------
def url_reasons(url):
    host, path, _ = safe_parse(url)
    host = host.split(":")[0]
    low = url.lower()
    reasons = []
    if host in SHORTENERS:
        reasons.append("it uses a link shortener that hides the real destination")
    b = brand_lookalike(url)
    if b:
        reasons.append(f"it pretends to be {b} but is not the real {b} website")
    if re.fullmatch(r"[\d.]+", host):
        reasons.append("it uses a raw IP address instead of a website name")
    if "@" in url:
        reasons.append("it contains an @ symbol, which can disguise the real site")
    words = [w for w in SUSPICIOUS_WORDS if w in low]
    if words:
        reasons.append("its address contains words often used in scams (" + ", ".join(words) + ")")
    if host.count("-") >= 2:
        reasons.append("the site name has several hyphens")
    if host.count(".") >= 3:
        reasons.append("the site name has many sub-parts")
    if not low.startswith("https"):
        reasons.append("it does not use a secure https connection")
    return reasons


def signals(r):
    sig = []
    if r["type"] == "SMS":
        m = r["sms_model"]
        if m > 0.5:
            sig.append("The message wording closely matches known scam messages.")
        elif m > 0.3:
            sig.append("The message wording partly resembles known scam messages.")
        if r["rule_hits"]:
            sig.append("The message has scam warning signs: " + "; ".join(r["rule_hits"]) + ".")
        if m <= 0.3 and not r["rule_hits"]:
            sig.append("The message wording does not match known scam messages.")
    for link, (score, trusted) in r["url_risks"].items():
        if trusted:
            sig.append("The link goes to a well-known website.")
        elif score > 0.3:
            rs = url_reasons(link)
            sig.append("The link looks risky" + (": " + "; ".join(rs) + "." if rs else "."))
        else:
            sig.append("The link shows no clear warning signs.")
    return sig


def action_line(overall, kind="SMS"):
    if overall == "HIGH RISK":
        if kind == "URL":
            return "Do not open this link or enter any details on it."
        return "Do not click any links, reply, or share any details. Delete it and report it as spam."
    if overall == "CAUTION":
        return "Do not click links or share any details yet. Check through an official channel first."
    return "No action needed, but stay careful with unexpected requests for money, OTPs or passwords."


# ---------- Llama via ExecuTorch ----------
SYSTEM_PROMPT = (
    "You help ordinary people understand why a message or link was judged risky or safe. "
    "Write 2 short sentences in plain language using ONLY the facts given. "
    "Call the item a 'message' or a 'link' exactly as stated in Item type. "
    "Do not add anything that is not in the facts. Do not give advice. No introduction."
)
EXAMPLES = [
    ("Item type: message\nOverall verdict: HIGH RISK\nFacts:\n- The message wording closely matches known scam messages.\n"
     "- The link looks risky: it does not use a secure https connection.",
     "This message is very likely a scam because its wording matches known scam messages. "
     "The link in it also looks unsafe because it does not use a secure connection."),
    ("Item type: link\nOverall verdict: CAUTION\nFacts:\n- The link looks risky: its address contains words often used in scams (login).",
     "This link is not clearly dangerous, but its address contains a word often used in scams, so it is worth checking."),
    ("Item type: message\nOverall verdict: LOW RISK\nFacts:\n- The message wording does not match known scam messages.",
     "This message looks safe because its wording does not match known scam messages."),
]
REFUSAL_MARKERS = ["i can't", "i cannot", "can't assist", "i'm not able", "i am not able"]


def init_llm(model_dir):
    """Load tokenizer + ExecuTorch model. Returns load time in seconds."""
    global tok, llm
    from transformers import AutoTokenizer
    from optimum.executorch import ExecuTorchModelForCausalLM
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    llm = ExecuTorchModelForCausalLM.from_pretrained(str(model_dir))
    return time.perf_counter() - t0


def template_explanation(r, overall):
    lead = {"HIGH RISK": "This looks risky.", "CAUTION": "This is not clearly safe.",
            "LOW RISK": "This looks safe."}[overall]
    return lead + " " + " ".join(signals(r))


@contextlib.contextmanager
def capture_output():
    """Capture everything written to stdout/stderr, including C/C++ prints
    (ExecuTorch's disclaimer and PyTorchObserver line), by redirecting the
    file descriptors. result['text'] is filled in when the block exits."""
    result = {"text": ""}
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        yield result
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            ctypes.CDLL(None).fflush(None)
        except Exception:
            pass
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        tmp.seek(0)
        result["text"] = tmp.read().decode("utf-8", errors="replace")
        tmp.close()


OBS_RE = re.compile(r"PyTorchObserver\s*(\{[^}]*\})")


def parse_observer(text):
    """Turn the PyTorchObserver JSON line into readable metrics (ms, tokens/s)."""
    m = OBS_RE.search(text or "")
    if not m:
        return None
    try:
        o = json.loads(m.group(1))
        start = o["inference_start_ms"]
        gen = o["generated_tokens"]
        decode_ms = o["inference_end_ms"] - o["prompt_eval_end_ms"]
        return {
            "prompt_tokens": o["prompt_tokens"],
            "generated_tokens": gen,
            "prefill_ms": o["prompt_eval_end_ms"] - start,
            "ttft_ms": o["first_token_ms"] - start,
            "decode_ms": decode_ms,
            "decode_tok_per_s": round(gen / (decode_ms / 1000), 2) if decode_ms > 0 else None,
            "total_ms": o["inference_end_ms"] - start,
            "raw": m.group(1),
        }
    except (KeyError, ValueError, TypeError):
        return None


def explain(r, overall, max_seq_len=700):
    """Returns (text, source, info). Post-processing is identical to the Colab version."""
    item = "message" if r["type"] == "SMS" else "link"
    facts = (f"Item type: {item}\nOverall verdict: {overall}\nFacts:\n"
             + "\n".join("- " + s for s in signals(r)))
    info = {"llm_wall_ms": None, "observer": None, "fallback_reason": None}

    if llm is None:                       # --no-llm
        info["fallback_reason"] = "llm_disabled"
        return template_explanation(r, overall), "template", info

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in EXAMPLES:
        messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": facts})
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    t0 = time.perf_counter()
    try:
        with capture_output() as cap:
            out = llm.text_generation(tokenizer=tok, prompt=prompt, max_seq_len=max_seq_len)
    except Exception as e:                # do not crash a demo because of one bad generation
        info["llm_wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        info["fallback_reason"] = f"llm_error: {type(e).__name__}: {e}"
        print(f"  [LLM ERROR] {info['fallback_reason']}", file=sys.stderr)
        return template_explanation(r, overall), "template", info
    info["llm_wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    info["observer"] = parse_observer(cap["text"])

    if out.startswith(prompt):
        out = out[len(prompt):]
    elif "assistant" in out:
        out = out.rsplit("assistant", 1)[-1]
    out = out.split("<|eot_id|>")[0].strip()
    out = "\n".join(l for l in out.splitlines()
                    if l.strip() and not l.lower().startswith(("here", "action:"))).strip()

    if not out:
        info["fallback_reason"] = "empty"
    elif len(out) > 400:
        info["fallback_reason"] = "too_long"
    elif any(m in out.lower() for m in REFUSAL_MARKERS):
        info["fallback_reason"] = "refusal"
    if info["fallback_reason"]:
        return template_explanation(r, overall), "template", info
    return out, "llama", info


# ---------- Pi telemetry ----------
def vcgencmd(*args):
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True,
                              text=True, timeout=2).stdout.strip()
    except Exception:
        return None


def cpu_temp_c():
    s = vcgencmd("measure_temp")          # e.g. temp=49.4'C
    m = re.search(r"([\d.]+)", s or "")
    return float(m.group(1)) if m else None


def peak_rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)   # KB -> MB on Linux


# ---------- Full pipeline ----------
def run(text, max_seq_len=700, verbose=True):
    temp_before = cpu_temp_c()
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    r = analyze(text)
    overall = overall_verdict(r)
    analyze_ms = (time.perf_counter() - t0) * 1000

    text_out, source, info = explain(r, overall, max_seq_len)
    action = action_line(overall, r["type"])
    e2e_ms = (time.perf_counter() - t_start) * 1000

    rec = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": text,
        "type": r["type"],
        "verdict": overall,
        "sms_model": r["sms_model"],
        "sms_risk": r["sms_risk"],
        "rule_hits": len(r["rule_hits"]),
        "url_scores": {k: round(v[0], 4) for k, v in r["url_risks"].items()},
        "explanation_source": source,
        "fallback_reason": info["fallback_reason"],
        "explanation": text_out,
        "analyze_ms": round(analyze_ms, 2),
        "llm_wall_ms": info["llm_wall_ms"],
        "observer": info["observer"],
        "end_to_end_ms": round(e2e_ms, 1),
        "peak_rss_mb": peak_rss_mb(),
        "temp_before_c": temp_before,
        "temp_after_c": cpu_temp_c(),
        "throttled_after": vcgencmd("get_throttled"),
    }

    if verbose:
        print(f"VERDICT: {overall}  (input type: {r['type']})")
        if r["sms_model"] is not None:
            print(f"  Message scam score: model {r['sms_model']:.2f} -> final {r['sms_risk']:.2f}")
            if r["rule_hits"]:
                print(f"  Rules matched: {len(r['rule_hits'])}")
        for link, (score, trusted) in r["url_risks"].items():
            print(f"  Link {link[:60]}: {score:.2f}" + (" [trusted domain]" if trusted else ""))
        print(f"\nWhy ({source}): {text_out}")
        print(f"Action: {action}")
        o = info["observer"]
        obs = (f"prompt {o['prompt_tokens']} tok, first token {o['ttft_ms']/1000:.1f} s, "
               f"{o['decode_tok_per_s']} tok/s" if o else "no observer data")
        print(f"  [timing] detectors {analyze_ms:.0f} ms | LLM {((info['llm_wall_ms'] or 0)/1000):.1f} s "
              f"({obs}) | end-to-end {e2e_ms/1000:.1f} s | peak RAM {rec['peak_rss_mb']:.0f} MB "
              f"| {rec['temp_after_c']} C | {rec['throttled_after']}")
        if info["fallback_reason"]:
            print(f"  [note] template used: {info['fallback_reason']}")
        print("-" * 60)
    return rec


TESTS = [
    "Congratulations! You won a free prize. Claim now: http://bit.ly/xyz",
    "http://secure-login-paypal.verify-account.xyz/update",
    "Are we meeting at 5 for the lab session?",
    "https://www.google.com",
    "Dear customer, your SBI account will be blocked today. Update KYC now: http://sbi-kyc-update.xyz/login",
    "Your OTP is 482913. Do not share it with anyone. - HDFC Bank",
    "Hey, are you free for dinner tomorrow?",
    "Dear customer, your SBI account will be blocked today. Update your KYC immediately to avoid suspension.",
    "Aapka bank account band ho jayega. Abhi KYC update karein varna 24 ghante mein block.",
    "You have won Rs 25,00,000 in the KBC lottery. Send your Aadhaar and bank details to claim.",
    "http://paytm-secure-offer.xyz/?id=123",
]


def main():
    ap = argparse.ArgumentParser(description="Offline scam / phishing detector with Llama explanations")
    ap.add_argument("--text", help="analyse one message or URL instead of the demo set")
    ap.add_argument("--repeat", type=int, default=1, help="run the set N times in one process")
    ap.add_argument("--no-llm", action="store_true", help="detectors only, template explanation")
    ap.add_argument("--interactive", action="store_true", help="live demo: read messages/URLs from the keyboard")
    ap.add_argument("--max-seq-len", type=int, default=700)
    ap.add_argument("--proj-dir", default=str(DEFAULT_PROJ_DIR))
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = ap.parse_args()

    t0 = time.perf_counter()
    init_detectors(args.proj_dir)
    detectors_s = time.perf_counter() - t0
    llm_load_s = None
    if not args.no_llm:
        llm_load_s = init_llm(args.model_dir)

    print(f"Detectors loaded in {detectors_s:.2f} s"
          + (f" | Llama loaded in {llm_load_s:.2f} s" if llm_load_s is not None else " | Llama disabled"))
    print(f"Pi temp {cpu_temp_c()} C | {vcgencmd('get_throttled')} | RSS after load {peak_rss_mb():.0f} MB")
    print("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    texts = [args.text] if args.text else TESTS

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(json.dumps({"meta": {"detectors_load_s": round(detectors_s, 3),
                                       "llm_load_s": None if llm_load_s is None else round(llm_load_s, 3),
                                       "no_llm": args.no_llm, "max_seq_len": args.max_seq_len}}) + "\n")
        if args.interactive:
            print("Interactive mode: paste an SMS or a URL and press Enter. Type q to quit.")
            while True:
                try:
                    t = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if t.lower() in ("q", "quit", "exit"):
                    break
                if not t:
                    continue
                rec = run(t, args.max_seq_len)
                rec["iteration"] = 1
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log.flush()
        else:
            for it in range(1, args.repeat + 1):
                if args.repeat > 1:
                    print(f"\n##### Iteration {it} of {args.repeat} #####")
                for t in texts:
                    print(f"\n> {t}")
                    rec = run(t, args.max_seq_len)
                    rec["iteration"] = it
                    log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    log.flush()
    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()
