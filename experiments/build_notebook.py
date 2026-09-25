"""Bundle the repo modules + experiments/ into one self-contained Colab notebook.

python3 experiments/build_notebook.py  -> experiments/notebooks/verascope_experiments.ipynb
Rerun after any code change; the notebook carries the code as a base64 zip.
"""

import base64
import io
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_MODULES = ["audio_prep.py", "claim_detect.py", "diarize_demo.py", "retrieve_evidence.py",
                "stitch.py", "transcribe.py", "verify_claims.py", "verify_pipeline.py",
                "wx_transcribe.py"]
OUT = ROOT / "experiments" / "notebooks" / "verascope_experiments.ipynb"


def bundle():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in REPO_MODULES:
            z.write(ROOT / f, f)
        for f in sorted((ROOT / "experiments").glob("*.py")):
            z.write(f, f"experiments/{f.name}")
    return base64.b64encode(buf.getvalue()).decode()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


INTRO = """
# Verascope paper experiments (E1–E9 + figures) — Google Colab, free T4

**Purpose.** Real measurements for the paper: chunked vs whole-file diarization (E1), Stage 1
latency (E2), ASR WER/cpWER (E3), NLI thresholds + calibration (E4), verdict aggregation under
retrieval noise (E5), claim detection vs context (E6), dedup on own videos (E7, optional), hosted
vs local Stage 1 (E8, optional), Gemma 4 E2B audio diarization zero-shot vs QLoRA vs pyannote (E9),
then figures built only from `results/`.

**Datasets.** AMI Meeting Corpus (Mix-Headset; CC BY 4.0), VoxConverse (HF
`diarizers-community/voxconverse`; CC BY 4.0), FEVER gold evidence (HF
`copenlu/fever_gold_evidence`; CC BY-SA 3.0), ClaimBuster (Zenodo 3609356; CC BY 4.0).

**Prerequisites (do these first).**
1. Runtime → Change runtime type → **T4 GPU**.
2. Colab **Secrets** (key icon, left bar), each with *Notebook access* on:
   - `HF_TOKEN` — account that accepted the gated models
     [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and
     [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0). **Required** (E1–E3, E9).
   - `GROQ_API_KEY` — E6 (and E7/E8). Without it E6 is recorded as skipped.
   - `PYANNOTEAI_API_KEY` — optional, hosted Stage 1 in E2/E8.
3. Google Drive with ~5 GB free: results, figures and E9 checkpoints go to `MyDrive/verascope/`,
   so a dead session resumes where it stopped (re-run all cells; finished items are skipped).
4. Optional E7: put your own videos or `*.lines.json` transcripts in `MyDrive/verascope/own_videos/`.

**Run order.** Leave `SMOKE = True`, *Runtime → Run all* (≈45–60 min, all experiments on 1–2 files /
~50 samples). If it finishes without red errors, set `SMOKE = False` and *Run all* again.

**Expected full-scale T4 time (estimates, not measurements).** setup 10 min · E4 10 min · E5 10 min ·
E6 15 min (API) · E1 ≈2 h · E2 ≈45 min · E3 ≈1 h · E9 prep ≈40 min · E9 Gemma ≈3–4 h · figures 1 min.
Free Colab sessions usually end before all of that: re-run everything in a new session; each
experiment resumes from its `partial.jsonl` and E9 training from its last checkpoint.

**VRAM.** Every experiment runs in its own Python process (`python -m experiments.eN`), so all GPU
memory is released when it exits; the cell prints `nvidia-smi` memory after each one.

**When done:** download `MyDrive/verascope/verascope_results.zip` (made by the last cell) and hand
it back together with this executed notebook.
"""

CONFIG = """
SMOKE = True          # True: 1-2 files / ~50 samples per experiment. Flip to False after a clean smoke run.
RUN = {"E4": True, "E5": True, "E6": True, "E1": True, "E2": True, "E3": True,
       "E7": True, "E8": True, "E9": True}
DRIVE_DIR = "/content/drive/MyDrive/verascope"
LLM_SPEND_CAP_USD = "5.0"   # E6 stops (and asks via BLOCKERS.md) before exceeding this
"""

SETUP = """
import base64, io, json, os, subprocess, sys, time, zipfile
from google.colab import drive, userdata
drive.mount("/content/drive")
ROOT = "/content/verascope"
os.makedirs(ROOT, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(ROOT)
tag = "results_smoke" if SMOKE else "results"
os.environ.update({
    "SMOKE": "1" if SMOKE else "0", "VS_ROOT": ROOT, "VS_DATA": "/content/data",
    "VS_RESULTS": f"{DRIVE_DIR}/{tag}", "VS_FIGURES": f"{DRIVE_DIR}/figures{'_smoke' if SMOKE else ''}",
    "OWN_VIDEOS_DIR": f"{DRIVE_DIR}/own_videos", "VS_GIT_COMMIT": GIT_COMMIT,
    "LLM_SPEND_CAP_USD": LLM_SPEND_CAP_USD, "PYTHONUNBUFFERED": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
# Secrets: read here (subprocesses cannot call userdata) and passed on via env. Never printed.
for k in ("HF_TOKEN", "GROQ_API_KEY", "PYANNOTEAI_API_KEY"):
    try:
        v = userdata.get(k)
    except Exception:
        v = None
    if v:
        os.environ[k] = v.strip().strip('"').strip("'")
    print(f"{k}: {'set' if v else 'MISSING'}")
os.chdir(ROOT)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout)
"""

INSTALL = """
# One resolver pass, no forced pins that fight whisperx (it requires huggingface-hub<1.0, torch~=2.8).
# huggingface_hub>=1.5 was only for the app's hosted NLI call; experiments run NLI locally.
# Resolves (checked for py3.12 and py3.13, linux) to torch 2.8.0, transformers 4.57.x, huggingface-hub 0.36.x.
# "incompatible" warnings about other preinstalled Colab packages are harmless; ERROR lines are not.
!pip install -q "whisperx==3.8.6" "pyannote.audio==4.0.7" groq meeteval jiwer whisper-normalizer datasets librosa
!python -c "import importlib.metadata as m; print({p: m.version(p) for p in ('torch','whisperx','pyannote.audio','transformers','huggingface_hub','ctranslate2','datasets')})"
!python -m experiments.common && python -m experiments.speech && python -m experiments.e1 --selfcheck \\
    && python -m experiments.e9 --selfcheck
# Gated pyannote access check: E1/E2/E3/E8/E9 need it and are skipped (not crashed) without it.
_hf = subprocess.run([sys.executable, "-m", "experiments.common", "hf"], capture_output=True, text=True)
print(_hf.stdout, _hf.stderr[-2000:])
HF_OK = _hf.returncode == 0
print("pyannote access:", "OK" if HF_OK else "FAILED -> fix the token (see above), rerun this cell")
"""

RUNNER = """
STATUS = {}

def run(mod, *args, py=sys.executable):
    \"\"\"One experiment = one process (VRAM freed on exit). Output streamed; failures recorded, not raised.\"\"\"
    t = time.time()
    p = subprocess.Popen([py, "-m", f"experiments.{mod}", *args], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ)
    for line in p.stdout:
        print(line, end="")
    code = p.wait()
    STATUS[" ".join([mod, *args])] = {"exit": code, "minutes": round((time.time() - t) / 60, 1)}
    os.makedirs(os.environ["VS_RESULTS"], exist_ok=True)
    with open(f"{os.environ['VS_RESULTS']}/run_log.jsonl", "a") as f:
        f.write(json.dumps({"step": " ".join([mod, *args]), "exit": code, "smoke": SMOKE,
                            "minutes": STATUS[" ".join([mod, *args])]["minutes"],
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\\n")
    mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print(f"\\n==> {mod} {' '.join(args)}: exit {code}, GPU memory now in use: {mem}")

from IPython.display import Image, display
import glob

def show(prefix):
    \"\"\"Rebuild figures from results/ (seconds) and display this experiment's graphs inline.\"\"\"
    subprocess.run([sys.executable, "-m", "experiments.figures"], cwd=ROOT, env=os.environ,
                   capture_output=True)
    figs = sorted(glob.glob(f"{os.environ['VS_FIGURES']}/{prefix}_*.png"))
    for f in figs:
        print(os.path.basename(f))
        display(Image(filename=f, width=520))
    if not figs:
        print(f"no {prefix} figures (see figures/captions_draft.md for why)")
"""

GEMMA_ENV = """
# Gemma 4 / Unsloth get their OWN virtualenv: its pins (torch 2.10, transformers 5.5, trl 0.24,
# huggingface-hub 1.x) conflict with whisperx's (torch 2.8, huggingface-hub<1), so the two stacks
# never share site-packages. Versions below were resolved together for py3.12 and py3.13 with no
# conflicts. The base environment (E1-E8, E9 prep) is untouched.
GEMMA_PY = "/content/gemma_env/bin/python"
!pip install -q uv
!uv venv -q --allow-existing --python {sys.executable} /content/gemma_env
!uv pip install -q --python {GEMMA_PY} "unsloth==2026.9.11" "unsloth_zoo==2026.9.7" "transformers==5.5.0" \\
    "torch==2.10.0" "torchcodec==0.10.*" "trl==0.24.0" timm pyannote.metrics
!{GEMMA_PY} -c "import importlib.metadata as m; print({p: m.version(p) for p in ('torch','unsloth','transformers','trl','xformers','bitsandbytes')})"
"""

HANDOFF = """
run("figures")
for _p in ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"):
    show(_p)
import shutil
tag = "results_smoke" if SMOKE else "results"
stage = "/content/handoff_zip"
shutil.rmtree(stage, ignore_errors=True)
for d in (tag, f"figures{'_smoke' if SMOKE else ''}", "handoff"):
    if os.path.exists(f"{DRIVE_DIR}/{d}"):
        shutil.copytree(f"{DRIVE_DIR}/{d}", f"{stage}/{d}",
                        ignore=shutil.ignore_patterns("ckpt", "adapter", "*.wav"))
shutil.make_archive(f"{DRIVE_DIR}/verascope_{tag}", "zip", stage)
print(json.dumps(STATUS, indent=1))
print(f"zip: {DRIVE_DIR}/verascope_{tag}.zip")
"""


def build():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip() or "unknown"
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "*.py"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    commit += "+uncommitted" if dirty else ""
    cells = [md(INTRO), code(CONFIG),
             code(f'BUNDLE = "{bundle()}"\nGIT_COMMIT = "{commit}"'),
             code(SETUP), md("## Install (≈5–10 min)"), code(INSTALL), code(RUNNER)]
    # quickest first; everything that needs the gated pyannote models waits for HF_OK
    steps = [("E4", "NLI threshold sweep + calibration (FEVER) · ~10 min", "e4", False),
             ("E5", "Verdict aggregation vs retrieval noise · ~10 min", "e5", False),
             ("E7", "Dedup on own videos (optional) · skips in seconds without inputs", "e7", False),
             ("E8", "Hosted vs local Stage 1 (optional) · skips without hosted keys", "e8", True),
             ("E6", "Claim detection vs context window (ClaimBuster, Groq) · ~40 min, rate-limited", "e6", False),
             ("E2", "Stage 1 latency scaling · ~45 min", "e2", True),
             ("E3", "ASR WER / cpWER (AMI test) · ~1 h", "e3", True),
             ("E1", "Chunked vs whole-file diarization (AMI test, VoxConverse test) · ~2 h", "e1", True)]
    for key, title, mod, needs_hf in steps:
        cond = f'RUN["{key}"] and HF_OK' if needs_hf else f'RUN["{key}"]'
        body = f'if {cond}:\n    run("{mod}")\n'
        if needs_hf:
            body += f'elif RUN["{key}"]:\n    print("skipped: pyannote access check failed (install cell)")\n'
        body += f'show("{mod}")'
        cells += [md(f"## {key} — {title}"), code(body)]
    cells += [md("## E9 — Gemma 4 E2B audio diarization\nPrep builds the windows and runs pyannote "
                 "on each test window (base environment)."),
              code('if RUN["E9"] and HF_OK:\n    run("e9", "prep")'),
              md("Separate environment for Gemma 4 + Unsloth (≈3–5 min)."),
              code('if RUN["E9"]:\n' + "\n".join("    " + l if l.strip() else l
                                                  for l in GEMMA_ENV.strip("\n").splitlines())),
              code('if RUN["E9"] and HF_OK:\n    run("e9", "gemma", py=GEMMA_PY)\n    run("e9", "report")\nshow("e9")'),
              md("## Figures, status, zip for hand-off"), code(HANDOFF)]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB), commit {commit}")


if __name__ == "__main__":
    build()
