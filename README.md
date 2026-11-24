# Bharat AI-SoC Student Challenge: On-device Threat, Phishing and Scam Detection (Raspberry Pi 5, fully offline)

An offline assistant that checks a pasted **SMS** or **URL** for scam and phishing signs, decides a verdict with
trained models and rules, and explains it in plain language using **Llama 3.2 1B Instruct** running through
**ExecuTorch (XNNPACK)** on the Arm CPU of a **Raspberry Pi 5**. Nothing leaves the device.

Team: **Anoushka Birje, Aaditya Shrivastava** | Demo video: **(https://drive.google.com/file/d/1F53SoAn8jTEqGpmxzc5pxnkgOcaqYwYF/view?usp=drive_link)** | Technical report: [Technical_Report.pdf](docs/Technical_Report.pdf)

> Phase 1 scope: SMS scam detector + phishing URL detector + Llama explanation, running on the Pi with measured
> on-device numbers. Network intrusion detection is planned for Phase 2 and is **not** implemented.

## How it works

```
Input (pasted SMS or URL)
   |
analyze(): URL or SMS?
   |-- SMS  -> TF-IDF + logistic regression score  (+ regex scam rules, English and Hinglish)
   |-- link inside an SMS, or a bare URL -> trusted-domain check, otherwise RandomForest on 17 URL features
   v
overall_verdict(): LOW RISK / CAUTION / HIGH RISK         <- decided by CODE, never by Llama
   v
signals(): plain-language facts -> prompt -> Llama 3.2 1B (model.pte via ExecuTorch)
   v
2-sentence explanation (Llama, template fallback if it refuses/rambles/errors) + fixed action line (code)
```

Design rule: **the trained models and rules decide; Llama only explains a verdict it is given.**

Verdict bands (from `pi_pipeline.py`): score > 0.5 = HIGH RISK; > 0.45 = CAUTION; otherwise LOW RISK.
Each SMS rule hit adds 0.12 to the model score (capped at 0.90). A trusted domain scores 0.02; link shorteners
get a floor of 0.55; a brand name on a non-trusted domain adds 0.15 (capped at 0.95).

## Tested hardware and software

- Raspberry Pi 5 (4 GB), official 27 W power supply, official Active Cooler, 64 GB microSD
- Raspberry Pi OS Lite 64-bit (Debian 13 "trixie"), Python 3.13.5, aarch64
- Package versions that were used: see `requirements-pi.txt` (executorch 1.6.0, optimum-executorch pinned to commit
  `dc779b7fb1f2185d7c67f5f152e99b2ed8a0e315`, torch 2.15.0+cpu, transformers 5.0.0rc1, scikit-learn 1.9.1)

## Repository layout

| Path | What it is |
|---|---|
| `pi_pipeline.py` | Standalone pipeline for the Pi: detectors, rules, verdict, Llama explanation, timing capture |
| `run_tests.py`, `test_cases.csv` | Reproducible test of safe/scam SMS and safe/phishing URLs (28 synthetic cases) |
| `check_url_features.py` | Checks the Pi's URL features are identical to the training script's |
| `summarize_results.py` | Turns `results/run_*.jsonl` into tables and CSV |
| `scripts/offline_test.sh` | Runs the pipeline inside an empty network namespace (offline proof) |
| `training/` | Original training scripts (`sms_detector.py`, `url_detector.py`) and legacy CLI |
| `colab/pipeline_colab_export.py` | Original Colab export the Pi pipeline was ported from (provenance only, not runnable as is) |
| `requirements-pi.txt` | `pip freeze` from the Pi (note: `torch==2.15.0+cpu` comes from the PyTorch CPU index) |
| `docs/on_device_results.md` | Measurements, method and caveats |
| `results/` | Raw logs from the Pi runs |
| `threat_project/` | Detector files (see its README) |
| `llama32_1b_instruct/` | Llama files (not committed, see below) |

## Setup on the Raspberry Pi

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git tmux time
mkdir -p ~/threat_detector && cd ~/threat_detector      # copy this repository's files here
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip

# 1) CPU-only PyTorch. IMPORTANT: a plain `pip install torch` on aarch64 pulls CUDA/NVIDIA wheels (several GB)
#    that are useless on a Pi and fill /tmp ("No space left on device").
pip install torch --index-url https://download.pytorch.org/whl/cpu
# 2) Detector dependencies (scikit-learn must match the version that trained the .joblib files)
pip install scikit-learn==1.9.1 pandas joblib
# 3) ExecuTorch + Hugging Face wrapper (an unpinned install resolved to this commit on our Pi)
pip install "git+https://github.com/huggingface/optimum-executorch.git@dc779b7fb1f2185d7c67f5f152e99b2ed8a0e315"
```

This is the sequence that worked on our Pi. A clean reinstall directly from `requirements-pi.txt` has **not** been
tested; use the steps above.

### Files that are not in this repository

- **`llama32_1b_instruct/`**: `model.pte` (about 790 MB), `tokenizer.json`, `tokenizer_config.json`,
  `chat_template.jinja`. Too large for git, and the Llama 3.2 Community Licence applies to the exported model.
  We exported it once on x86 Google Colab (gated model: accept Meta's licence on Hugging Face and log in first):
  ```bash
  pip install git+https://github.com/huggingface/optimum-executorch.git
  optimum-cli export executorch --model "meta-llama/Llama-3.2-1B-Instruct" --task "text-generation" \
    --recipe "xnnpack" --use_custom_sdpa --use_custom_kv_cache --qlinear 8da4w --qembedding 4w \
    --output_dir="./llama32_1b_instruct"
  ```
  The exported file ran **unchanged** on the Pi 5 (no re-export). Copy the four files into `llama32_1b_instruct/`.
- **`threat_project/url_model.joblib`**: regenerate with `training/url_detector.py` (needs the Kaggle
  "phishing site URLs" CSV), or take it from the release. Datasets are not redistributed here:
  SMS Spam Collection (Kaggle) and phishing site URLs (Kaggle). The Tranco top-1000 list is included.

## Run

```bash
source venv/bin/activate
python pi_pipeline.py                  # 11 demo messages, explanation by Llama
python pi_pipeline.py --interactive    # live demo: paste an SMS or URL, q to quit
python pi_pipeline.py --text "Your account is blocked. Update KYC now: http://sbi-kyc-update.xyz"
python pi_pipeline.py --repeat 3       # same set 3x in one process (cold vs warm behaviour)
python pi_pipeline.py --no-llm         # detectors only
```

Each run writes `results/run_*.jsonl` (verdict, explanation, PyTorchObserver timing, RAM, CPU temperature,
throttle state per message). Hugging Face offline mode is set, so nothing contacts the internet at run time.

## Test and benchmark procedure

```bash
python run_tests.py                       # 28 cases: safe/scam SMS, safe/phishing URLs (fast, detectors only)
python check_url_features.py --train training/url_detector.py --csv threat_project/phishing_site_urls.csv --n 5000   # training/inference feature parity
python summarize_results.py               # tables from the newest results/run_*.jsonl
/usr/bin/time -v python pi_pipeline.py --repeat 2 2>&1 | tee run_pipeline.log        # peak RAM from "Maximum resident set size"
sudo unshare --net bash scripts/offline_test.sh 2>&1 | tee run_offline.log            # offline proof
```
Thermal sampling and cold/warm procedure are described in `docs/on_device_results.md`.

## Results (measured on the Raspberry Pi 5)

| Metric | Result |
|---|---|
| Decode speed | about 12.3 tokens/s (11.7 to 13.2) |
| Time to first token, prompt of 325 to 404 tokens | about 4.0 s steady state; 7.7 s in a new process with warm file cache; 13.4 s right after boot |
| End-to-end per message | median 6.0 s steady state (4.3 to 12.9 s across all messages in the run) |
| Peak RAM | about 2.73 GiB (2,863,040 KB) including the model, detectors and Python |
| CPU temperature | 37 to 42 °C at start; 61.5 to 63.1 °C maximum under sustained load; `throttled=0x0` in every sample |
| Detector latency | about 1 ms (SMS), about 24 ms when the URL model runs |
| Test set (28 synthetic cases) | 15/15 scam and phishing items caught; 1/13 safe items falsely flagged |

Full method, per-run numbers and caveats: **`docs/on_device_results.md`**.

## Model metrics (held-out data, measured before deployment)
| Model | Evaluation data | Result |
|---|---|---|
| URL classifier (random forest, 17 features) | 20,000 URLs: stratified 20% split of 100,000 sampled rows, random_state=42 | accuracy 0.89; safe precision/recall 0.92/0.92; malicious precision 0.81, recall 0.80, F1 0.80 |
| SMS classifier | 1,115 messages: 20% split of the Kaggle SMS Spam Collection | accuracy 0.98; scam precision 0.95, recall 0.93 |
URL figures come from re-running training/url_detector.py. They use URL text features only (no page content or domain age) and a random split, so they may be optimistic.

## Limitations

- **False alarms on genuine transactional SMS.** The "safe" training class is mostly casual chat. Example from our
  tests: a bank debit alert scored 0.80 purely from the SMS model (words such as "rs", "500", "account"; no rule fired).
- The scam rules were tuned on the 11 demo messages, and our tests are synthetic and small; no real-message test set
  has been collected yet. The 15/15 result describes textbook-style scams and is not an accuracy claim.
- English and Hinglish only.
- Llama paraphrases loosely; a template explanation is used if it refuses, rambles or returns nothing.
- Per-process start-up is slow (the first generation in a process is slower than later ones); keep the process running.
- CPU only, 4 GB RAM: the 3B Llama model was not tried because it would not fit comfortably.
- Network intrusion detection is not implemented (planned for Phase 2).

## Credits and licences

- Llama 3.2 1B Instruct: Meta, under the Llama 3.2 Community Licence (built with Llama). ExecuTorch and
  optimum-executorch: PyTorch / Hugging Face.
- Data: UCI SMS Spam Collection (via Kaggle), Kaggle "phishing site URLs", Tranco top-1000 list.
- Repository licence: MIT (see LICENSE). This covers the code only; the Llama model keeps its own licence.
- See NOTICE for the Llama 3.2 licence attribution.
