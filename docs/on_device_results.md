# On-device results (Raspberry Pi 5, 4 GB)

All numbers on this page were measured on the Raspberry Pi 5, not on Google Colab. Raw logs are in `results/`.
Each number is one measurement session on one board; they are not averages over many sessions.

## Setup

Raspberry Pi 5 (4 GB), official power supply and Active Cooler, 64 GB microSD, Raspberry Pi OS Lite 64-bit
(Debian 13), Python 3.13.5. Llama 3.2 1B Instruct exported to `model.pte` (XNNPACK, 8da4w weights, 4-bit embeddings,
about 790 MB) and run unchanged on the Pi with executorch 1.6.0 through optimum-executorch. Prompt size:
325 to 404 tokens (system message, three few-shot examples, facts).

## How each metric was obtained

* **Load time**: wall time around tokenizer + `ExecuTorchModelForCausalLM.from\_pretrained` (the model file is
memory-mapped, so much of the real weight loading happens during the first generation).
* **Time to first token** = `first\_token\_ms - inference\_start\_ms` from the `PyTorchObserver` line (captured, not hidden).
* **Decode speed (tokens/s)** = `generated\_tokens / ((inference\_end\_ms - prompt\_eval\_end\_ms) / 1000)`.
* **End-to-end** = detectors + rules + verdict + Llama explanation + action line, timed in `pi\_pipeline.py`.
* **Peak RAM** = `Maximum resident set size` from `/usr/bin/time -v` (and `ru\_maxrss` inside the script).
* **Temperature / throttling**: `vcgencmd measure\_temp`, `get\_throttled`, `measure\_clock arm`. The pipeline records them
after every message; a separate 2-second sampler logged them during the sustained run. `throttled=0x0` means no
under-voltage or throttling flag was set.

## Cold versus warm (same SBI scam SMS, 358 prompt tokens)

|Situation|Detectors load|Llama load|Time to first token|LLM time|End-to-end|Process wall time|Peak RAM|
|-|-|-|-|-|-|-|-|
|Right after boot (7 min uptime, empty file cache)|2.95 s|2.40 s|13.4 s|15.3 s|15.3 s|33.4 s|2,863,040 KB|
|New process, file cache warm|2.26 s|2.37 s|7.7 s|9.6 s|9.6 s|24.1 s|2,854,768 KB|
|Later message in the same process|-|-|about 4.1 s|about 6.0 s|about 6.0 s|-|-|

The first generation in every process is slower (disk reads and first-inference set-up). A demo or application should
load once and keep the process alive. Within a process, time to first token grows with prompt length (roughly 3.7 s at
326 tokens to 4.65 s at 404 tokens in our logs).

## Steady state: 11 demo messages, run twice in one process (22 messages)

||min|median|max|
|-|-|-|-|
|End-to-end (second pass)|4.4 s|6.0 s|8.4 s|
|Time to first token (second pass)|3.7 s|4.0 s|4.6 s|
|Decode speed|11.9 tok/s|12.3 tok/s|13.2 tok/s|

First pass (includes first-call overhead): first message 12.9 s end-to-end, 10.7 s to first token.
Template fallbacks: 0 of 22. Peak RAM for the run: 2,848,384 KB. CPU use averaged about 344% (of 400%).
Detector time: about 1 ms for an SMS without links, about 24 ms when the URL RandomForest runs.

## Sustained load and temperature

`--repeat 5` (five passes of the 11 messages, about 6.5 minutes of continuous generation) with a 2-second sampler:
189 samples, **all `throttled=0x0`**, maximum temperature **61.5 °C** (idle at the start about 37 to 42 °C).
In the 22-message run the pipeline's after-message readings peaked at 63.1 °C, also without throttling.
Power draw was not measured.

## Memory

Resident memory was about 0.9 GB right after loading and about 2.8 GB at the end of the first generation, which fits
in the 4 GB board with the OS and the detectors (about 200 MB used at idle).

## Accuracy checks

* **URL feature parity**: `check\_url\_features.py` extracts `features()` from the training script and compares it with
the Pi pipeline on 14 edge cases plus 5,000 dataset rows: 5,014 of 5,014 identical.
* **Reproducible test set** (`test\_cases.csv`, 28 synthetic cases written independently of the models):

|Category|Cases|Flagged (CAUTION or HIGH RISK)|
|-|-|-|
|Scam SMS|8|8 (all HIGH RISK)|
|Phishing URLs|7|7 (all HIGH RISK)|
|Safe URLs|5|0|
|Safe SMS, casual|4|0|
|Safe SMS, transactional|4|1 (a bank debit alert, HIGH RISK, SMS model score 0.80)|

Caveats: small and synthetic; the scam examples are textbook-style, so the high catch rate says little about real-world
recall. The false alarm is a known weakness (see README, Limitations) and was not patched to fit the test.
On the 11 demo messages the verdicts were as intended (scams HIGH RISK, safe messages LOW RISK), but the rules were tuned on those messages.

## Offline operation

The pipeline was run inside an empty network namespace (`sudo unshare --net`, `scripts/offline\_test.sh`). The log
(`results/run\_offline\_1.log`) shows only a loopback interface, a failed connection attempt
(`\[Errno 101] Network is unreachable`), and then the pipeline loading and processing all 11 demo messages (11 verdict lines in the log). Hugging Face offline mode is set in
the code. \[Add: video of the Pi with Wi-Fi switched off.]

