"""Shared code for the paper experiments (E1-E9): config, results writer,
resumable per-item records, bootstrap CIs, RTTM/DER/WER helpers, dataset
registry, and the Gemma diarization text format.

Every experiment is `python -m experiments.eN` run from the repo root, one
process per experiment so VRAM is released on exit. SMOKE=1 (default) writes
to results_smoke/, SMOKE=0 to results/, so a smoke run never pollutes the
resume state of a full run.

Self-check (CPU, no downloads): python -m experiments.common
"""

import datetime
import hashlib
import importlib.metadata as md
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------- config
SEED = 42
SMOKE = os.environ.get("SMOKE", "1") == "1"
ROOT = Path(os.environ.get("VS_ROOT", Path(__file__).resolve().parent.parent))
RESULTS = Path(os.environ.get("VS_RESULTS", ROOT / ("results_smoke" if SMOKE else "results")))
DATA = Path(os.environ.get("VS_DATA", ROOT / "data"))
FIGURES = Path(os.environ.get("VS_FIGURES", ROOT / "figures"))
BOOT_N = 1000

# Stop-and-ask threshold from the brief: E6 halts before exceeding this.
LLM_SPEND_CAP_USD = float(os.environ.get("LLM_SPEND_CAP_USD", "5.0"))

# Forgiveness collar, +/- seconds around reference boundaries (NIST md-eval
# convention). pyannote.metrics' `collar` is the TOTAL width, so it gets 2x.
COLLARS_PM = (0.25, 0.0)

PACKAGES = ("torch", "torchaudio", "transformers", "huggingface_hub", "datasets",
            "pyannote.audio", "pyannote.metrics", "whisperx", "faster-whisper",
            "ctranslate2", "jiwer", "meeteval", "groq", "unsloth", "peft",
            "trl", "bitsandbytes", "numpy", "matplotlib")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- env
def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        name, mem = out.splitlines()[0].rsplit(",", 1)
        return name.strip(), round(int(mem) / 1024, 1)
    except Exception:
        return "none", 0


def env_info():
    gpu, vram = gpu_info()
    pk = {}
    for p in PACKAGES:
        try:
            pk[p] = md.version(p)
        except Exception:
            pass
    commit = os.environ.get("VS_GIT_COMMIT")
    if not commit:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                    text=True).stdout.strip() or None
        except Exception:
            commit = None
    return {"gpu": gpu, "vram_gb": vram, "python": platform.python_version(),
            "platform": platform.platform(), "packages": pk, "git_commit": commit}


class GpuPeak:
    """Peak device memory (MiB) by polling nvidia-smi, like time23.py. Catches
    CTranslate2 allocations that torch.cuda.max_memory_allocated cannot see.
    ponytail: 0.5 s poll, a sub-500 ms spike can be missed."""

    def __enter__(self):
        import threading
        self.peak, self._stop = 0, threading.Event()

        def poll():
            while not self._stop.is_set():
                try:
                    m = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=5).stdout
                    self.peak = max(self.peak, int(m.split()[0]))
                except Exception:
                    pass
                self._stop.wait(0.5)
        self._t = threading.Thread(target=poll, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


# ---------------------------------------------------------------- results
class Exp:
    """Resumable results for one experiment.

    add(key, rec) appends to partial.jsonl immediately; done() reloads it so a
    rerun after a dead session skips finished items. Failures are kept per
    run (a failed item is retried next run) and land in summary.json."""

    def __init__(self, name, config):
        self.name = name
        self.dir = RESULTS / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.partial = self.dir / "partial.jsonl"
        self.config = {"seed": SEED, "smoke": SMOKE, **config}
        self.failures = []
        self.datasets = []
        random.seed(SEED)
        log(f"{name}: results -> {self.dir} (smoke={SMOKE})")

    def done(self):
        recs = {}
        if self.partial.exists():
            for line in self.partial.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    recs[r["key"]] = r
        return recs

    def add(self, key, rec):
        rec = {"key": key, **rec}
        with self.partial.open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        return rec

    def fail(self, item, reason):
        reason = str(reason)[:2000]
        log(f"{self.name}: FAIL {item}: {reason}")
        self.failures.append({"item": str(item), "reason": reason})

    def dataset(self, name, version, split, n_units, license=None, source=None, note=None):
        d = {"name": name, "version": version, "split": split, "n_units": n_units,
             "license": license, "source": source, "note": note}
        self.datasets.append(d)
        register_dataset(d)

    def finish(self, summary, error_cases=(), records=None):
        records = list(self.done().values()) if records is None else records
        head = {"experiment": self.name, "config": self.config, "env": env_info(),
                "datasets": self.datasets, "finished_utc": now()}
        write_json(self.dir / "summary.json",
                   {**head, "n_records": len(records), "summary": summary,
                    "failures": self.failures})
        write_json(self.dir / "raw_records.json",
                   {**head, "records": records, "summary": summary, "failures": self.failures})
        write_json(self.dir / "error_cases.json", list(error_cases)[:10])
        write_json(RESULTS / "env.json", env_info())
        log(f"{self.name}: done, {len(records)} records, {len(self.failures)} failures")

    def skip(self, reason):
        """Whole experiment skipped (missing inputs/keys): recorded, not faked."""
        self.fail("*", reason)
        blocker(self.name, reason)
        self.finish({"skipped": True, "reason": reason}, records=[])


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def register_dataset(d):
    p = RESULTS / "datasets.json"
    cur = json.loads(p.read_text()) if p.exists() else []
    cur = [x for x in cur if (x["name"], x["split"]) != (d["name"], d["split"])] + [d]
    write_json(p, cur)


def blocker(exp, reason):
    p = RESULTS / "BLOCKERS.md"
    text = p.read_text() if p.exists() else "# Blockers\n\n"
    line = f"- **{exp}** ({now()}): {reason}\n"
    if f"**{exp}**" not in text or reason not in text:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + line)


def secret(name):
    """Env var, else Kaggle secret, else Colab secret. Never printed."""
    v = os.environ.get(name)
    if v:
        return v
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        pass
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


# ---------------------------------------------------------------- stats
def bootstrap(units, stat, n=BOOT_N, seed=SEED):
    """Percentile 95% CI by resampling units (files/recordings/claims)."""
    units = list(units)
    if not units:
        return {"value": None, "ci95": [None, None], "n": 0}
    rng = random.Random(seed)
    k = len(units)
    vals = sorted(stat([units[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return {"value": stat(units), "ci95": [vals[int(0.025 * n)], vals[int(0.975 * n) - 1]],
            "n": k}


def prf(gold, pred, labels):
    """-> {label: {p, r, f1, support}}, macro_f1, accuracy, confusion[gold][pred]."""
    per, conf = {}, {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        conf[g][p] += 1
    for c in labels:
        tp = conf[c][c]
        fp = sum(conf[g][c] for g in labels) - tp
        fn = sum(conf[c].values()) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                  "support": tp + fn}
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0
    return {"per_class": per, "macro_f1": sum(v["f1"] for v in per.values()) / len(labels),
            "accuracy": acc, "confusion": conf}


def macro_f1(pairs, labels):
    return prf([g for g, _ in pairs], [p for _, p in pairs], labels)["macro_f1"]


def ece(scores, outcomes, bins=10):
    """Expected calibration error + reliability table, equal-width bins."""
    tab, n, e = [], len(scores), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores) if lo <= s < hi or (b == bins - 1 and s == 1.0)]
        if not idx:
            tab.append({"bin": [lo, hi], "n": 0, "conf": None, "acc": None})
            continue
        conf = sum(scores[i] for i in idx) / len(idx)
        acc = sum(outcomes[i] for i in idx) / len(idx)
        e += len(idx) / n * abs(conf - acc)
        tab.append({"bin": [lo, hi], "n": len(idx), "conf": conf, "acc": acc})
    return {"ece": e, "bins": tab, "n": n}


# ---------------------------------------------------------------- downloads
def fetch(url, dest, tries=3):
    """Download once to dest (atomic rename); skip if already present."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "verascope-experiments"})
            with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as e:
            if i == tries - 1:
                raise RuntimeError(f"download failed {url}: {e!r}") from e
            time.sleep(5 * (i + 1))


def stable_hash(*parts):
    return hashlib.sha1("\x1f".join(map(str, parts)).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- RTTM / DER
def read_rttm(path):
    """-> {uri: [(start, end, speaker)]} sorted by start."""
    out = {}
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            s, d = float(f[3]), float(f[4])
            out.setdefault(f[1], []).append((s, s + d, f[7]))
    return {k: sorted(v) for k, v in out.items()}


def write_rttm(path, uri, turns):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(
        f"SPEAKER {uri} 1 {s:.3f} {e - s:.3f} <NA> <NA> {k} <NA> <NA>\n"
        for s, e, k in turns if e > s))


def crop(turns, lo, hi, shift=True):
    """Clip turns to [lo, hi); shift to window-local time when shift=True."""
    off = lo if shift else 0.0
    return [(max(s, lo) - off, min(e, hi) - off, k) for s, e, k in turns if e > lo and s < hi]


def annotation(turns):
    from pyannote.core import Annotation, Segment
    a = Annotation()
    for i, (s, e, k) in enumerate(turns):
        if e > s:
            a[Segment(s, e), i] = str(k)
    return a


def der(ref, hyp, collar_pm, uem=None):
    """Global-optimal-mapping DER (pyannote.metrics) -> component dict in
    seconds: total, fa, miss, conf, der. uem=(start, end) or None."""
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    m = DiarizationErrorRate(collar=2 * collar_pm, skip_overlap=False)
    u = Timeline([Segment(*uem)]) if uem else None
    d = m(annotation(ref), annotation(hyp), uem=u, detailed=True)
    return {"total": d["total"], "fa": d["false alarm"], "miss": d["missed detection"],
            "conf": d["confusion"], "der": d["diarization error rate"]}


def pooled_der(units):
    tot = sum(u["total"] for u in units)
    return (sum(u["fa"] + u["miss"] + u["conf"] for u in units) / tot) if tot else 0.0


def der_summary(units):
    """units: per-file der() dicts -> pooled DER + components with file-level CI."""
    tot = sum(u["total"] for u in units) or 1.0
    return {"der": bootstrap(units, pooled_der),
            "fa": sum(u["fa"] for u in units) / tot,
            "miss": sum(u["miss"] for u in units) / tot,
            "conf": sum(u["conf"] for u in units) / tot,
            "total_ref_speech_s": tot, "n_files": len(units)}


# ---------------------------------------------------------------- WER
_NORM = None


def normalize_text(text):
    """Whisper English normalizer (same rules for ref and hyp)."""
    global _NORM
    if _NORM is None:
        from whisper_normalizer.english import EnglishTextNormalizer
        _NORM = EnglishTextNormalizer()
    return _NORM(text)


def wer_counts(ref, hyp):
    """Normalized word error counts. -> {S, D, I, N, wer}."""
    import jiwer
    r, h = normalize_text(ref), normalize_text(hyp)
    if not r.split():
        return {"S": 0, "D": 0, "I": len(h.split()), "N": 0, "wer": None}
    o = jiwer.process_words(r, h if h.split() else "<empty>")
    ins = o.insertions if h.split() else 0
    subs = o.substitutions if h.split() else 0
    dels = o.deletions if h.split() else len(r.split())
    n = o.hits + o.substitutions + o.deletions
    return {"S": subs, "D": dels, "I": ins, "N": n, "wer": (subs + dels + ins) / n}


def pooled_wer(units):
    n = sum(u["N"] for u in units)
    return sum(u["S"] + u["D"] + u["I"] for u in units) / n if n else 0.0


# ---------------------------------------------------------------- E9 format
# One line of text per window: "S1 0.0-3.4; S2 3.4-7.9; S1 7.9-9.2".
# Local labels S1.. numbered by first appearance, times window-relative,
# rounded to 0.1 s, turns sorted by start. Overlap is expressible: two turns
# of different speakers may cover the same time.
TURN_RE = re.compile(r"S(\d+)\s+(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
STRICT_RE = re.compile(r"^S\d+ \d+\.\d-\d+\.\d(; S\d+ \d+\.\d-\d+\.\d)*$")


def merge_turns(turns, gap=0.3):
    """Join same-speaker turns separated by <= gap s (shortens targets)."""
    out = []
    for s, e, k in sorted(turns):
        j = next((i for i in range(len(out) - 1, -1, -1) if out[i][2] == k), None)
        if j is not None and s - out[j][1] <= gap:
            out[j] = (out[j][0], max(out[j][1], e), k)
        else:
            out.append((s, e, k))
    return sorted(out)


def serialize(turns):
    """turns (window-relative) -> target string with first-appearance labels."""
    names, parts = {}, []
    for s, e, k in sorted(turns):
        names.setdefault(k, f"S{len(names) + 1}")
        s, e = round(s, 1), round(e, 1)
        if e > s:
            parts.append(f"{names[k]} {s:.1f}-{e:.1f}")
    return "; ".join(parts)


def parse(text, window_s):
    """Lenient parse -> (turns, info). info flags strict-format validity and
    hallucination indicators; out-of-window turns are clipped, inverted
    dropped, and each is counted."""
    text = (text or "").strip()
    turns, info = [], {"strict": bool(STRICT_RE.match(text)), "n_raw": 0,
                       "out_of_window": 0, "inverted": 0, "self_overlap": 0,
                       "max_concurrent": 0}
    for m in TURN_RE.finditer(text):
        info["n_raw"] += 1
        k, s, e = f"S{m.group(1)}", float(m.group(2)), float(m.group(3))
        if e <= s:
            info["inverted"] += 1
            continue
        if s >= window_s or e > window_s + 0.05:
            info["out_of_window"] += 1
            if s >= window_s:
                continue
            e = window_s
        turns.append((s, e, k))
    turns.sort()
    for i, (s, e, k) in enumerate(turns):
        for s2, e2, k2 in turns[i + 1:]:
            if s2 >= e:
                break
            if k2 == k:
                info["self_overlap"] += 1
    pts = sorted([(s, 1) for s, _, _ in turns] + [(e, -1) for _, e, _ in turns],
                 key=lambda p: (p[0], p[1]))
    c = 0
    for _, d in pts:
        c += d
        info["max_concurrent"] = max(info["max_concurrent"], c)
    info["parse_failure"] = not turns
    return turns, info


# ---------------------------------------------------------------- self-check
def _selfcheck():
    # bootstrap: constant data -> zero-width CI
    b = bootstrap([1.0] * 20, lambda u: sum(u) / len(u))
    assert b["value"] == 1.0 and b["ci95"] == [1.0, 1.0] and b["n"] == 20
    b = bootstrap(list(range(100)), lambda u: sum(u) / len(u))
    assert b["ci95"][0] < 49.5 < b["ci95"][1], b

    r = prf(["a", "a", "b", "c"], ["a", "b", "b", "c"], ["a", "b", "c"])
    assert r["accuracy"] == 0.75 and abs(r["per_class"]["a"]["r"] - 0.5) < 1e-9
    assert r["confusion"]["a"]["b"] == 1

    e = ece([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0])
    assert abs(e["ece"] - 0.1) < 1e-9, e

    # E9 format round trip + parser flags
    turns = [(0.0, 3.42, "spkB"), (3.4, 7.9, "spkA"), (7.9, 9.2, "spkB"), (5.0, 6.0, "spkB")]
    s = serialize(turns)
    assert s == "S1 0.0-3.4; S2 3.4-7.9; S1 5.0-6.0; S1 7.9-9.2", s
    got, info = parse(s, 30.0)
    assert info["strict"] and not info["parse_failure"] and len(got) == 4
    assert info["max_concurrent"] == 2 and info["self_overlap"] == 0
    got, info = parse("S1 0.0-3.0; S1 2.0-4.0; S2 5.0-4.0; S3 29.0-45.0; S4 31-32 junk", 30.0)
    assert not info["strict"] and info["self_overlap"] == 1 and info["inverted"] == 1
    assert info["out_of_window"] == 2 and got[-1] == (29.0, 30.0, "S3"), got
    assert parse("I think there are two speakers", 30.0)[1]["parse_failure"]
    assert merge_turns([(0, 1, "A"), (1.2, 2, "A"), (2.1, 3, "B"), (5, 6, "A")]) == \
        [(0, 2, "A"), (2.1, 3, "B"), (5, 6, "A")]
    assert crop([(0, 10, "A"), (12, 20, "B")], 5, 15) == [(0, 5, "A"), (7, 10, "B")]

    # RTTM round trip
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        write_rttm(Path(d) / "x.rttm", "m1", [(0.0, 1.5, "A"), (2.0, 3.0, "B")])
        assert read_rttm(Path(d) / "x.rttm") == {"m1": [(0.0, 1.5, "A"), (2.0, 3.0, "B")]}

    # DER (needs pyannote.metrics): label permutation is free under global
    # mapping, a mid-file swap (chunk relabel) is not.
    try:
        import pyannote.metrics  # noqa: F401
    except ImportError:
        print("selfcheck: pyannote.metrics missing, DER checks skipped")
    else:
        ref = [(0, 10, "A"), (10, 20, "B"), (20, 30, "A"), (30, 40, "B")]
        perm = [(s, e, {"A": "x", "B": "y"}[k]) for s, e, k in ref]
        assert der(ref, perm, 0.0)["der"] == 0.0
        swapped = perm[:2] + [(20, 30, "y"), (30, 40, "x")]  # chunk 2 relabelled
        d = der(ref, swapped, 0.0, uem=(0, 40))
        assert abs(d["der"] - 0.5) < 1e-9 and d["conf"] == 20.0, d
        # collar: +/-0.25 s forgives a 0.2 s boundary shift entirely
        shifted = [(0, 10.2, "x"), (10.2, 20, "y"), (20, 30, "x"), (30, 40, "y")]
        assert der(ref, shifted, 0.25)["der"] == 0.0
        assert der(ref, shifted, 0.0)["der"] > 0.0
        s = der_summary([d, der(ref, perm, 0.0)])
        assert abs(s["der"]["value"] - 0.25) < 1e-9, s
    try:
        import jiwer  # noqa: F401
        import whisper_normalizer  # noqa: F401
    except ImportError:
        print("selfcheck: jiwer/whisper_normalizer missing, WER checks skipped")
    else:
        w = wer_counts("Hello, world. It's 5 o'clock!", "hello world it is five o'clock")
        assert w["N"] >= 4 and w["wer"] is not None, w
        w = wer_counts("red green blue gold", "red green gold")  # numbers get merged by the normalizer
        assert (w["D"], w["N"], w["wer"]) == (1, 4, 0.25), w
        assert wer_counts("red green", "")["D"] == 2
    print("common selfcheck ok")


def hf_check():
    """Can HF_TOKEN read the gated pyannote repos? Prints the reason, never the token."""
    from huggingface_hub import hf_hub_download, whoami
    tok = (os.environ.get("HF_TOKEN") or "").strip()
    if not tok:
        print("HF_TOKEN: MISSING (Colab secret not set or notebook access off)")
        return False
    try:
        me = whoami(token=tok)
        auth = me.get("auth", {}).get("accessToken", {})
        print(f"HF token OK: user={me.get('name')} type={auth.get('role', '?')}")
    except Exception as e:
        print(f"HF token INVALID ({type(e).__name__}): create a new Read token at "
              "https://huggingface.co/settings/tokens and paste it into the HF_TOKEN secret "
              "(no quotes/spaces).")
        return False
    ok = True
    for repo in ("pyannote/speaker-diarization-3.1", "pyannote/segmentation-3.0"):
        try:
            hf_hub_download(repo, "config.yaml", token=tok)
            print(f"  {repo}: access OK")
        except Exception as e:
            ok = False
            print(f"  {repo}: NO ACCESS ({type(e).__name__}). Open https://huggingface.co/{repo} "
                  "logged in as that user and accept the conditions. If already accepted and the "
                  "token is fine-grained, enable 'Read access to contents of all public gated repos "
                  "you can access' on it (or use a classic Read token).")
    return ok


if __name__ == "__main__":
    if sys.argv[1:] == ["hf"]:
        sys.exit(0 if hf_check() else 1)
    _selfcheck()
