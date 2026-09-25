"""E2 — Stage 1 latency vs duration, per-stage timing.

Recordings of each duration are built by concatenating DISTINCT AMI dev
meetings (Mix-Headset) and cutting to the exact length. Each is run through
the repo's diarize_demo.run_pipeline_wx (local Stage 1, as deployed) REPS
times after one warm-up; stage times are the pipeline's own "STAGE <name>
<s>s" stderr lines. align and diarize run concurrently, so the additive
stages are prep + transcribe + align||diarize + fusion.

If GROQ_API_KEY and PYANNOTEAI_API_KEY are set, the hosted Stage 1
(diarize_demo.run_pipeline) is timed once per duration and marked hosted.
"""

import asyncio
import contextlib
import io
import os
import re
import sys
import time

from experiments.common import SMOKE, Exp, GpuPeak, log, secret
from experiments import speech as sp

DURS_MIN = [1, 2] if SMOKE else [10, 20, 30, 45, 60]
REPS = 1 if SMOKE else 3
STAGE_RE = re.compile(r"STAGE (.+?) ([\d.]+)s")


class Tee(io.TextIOBase):
    def __init__(self, a):
        self.a, self.buf = a, io.StringIO()

    def write(self, s):
        self.a.write(s)
        return self.buf.write(s)


def build(minutes):
    """Distinct AMI dev meetings, concatenated, cut to `minutes`."""
    path = sp.DATA / "e2" / f"concat_{minutes}min.wav"
    if path.exists():
        return path, meetings_of(path)
    need, parts, used = minutes * 60 * sp.SR, [], []
    import numpy as np
    for m in sp.ami_meetings("dev"):
        a = sp.load16k(sp.ami_audio(m))
        parts.append(a[:need - sum(map(len, parts))])
        used.append(m)
        if sum(map(len, parts)) >= need:
            break
    sp.write_wav(path, np.concatenate(parts))
    path.with_suffix(".txt").write_text(" ".join(used))
    return path, used


def meetings_of(path):
    return path.with_suffix(".txt").read_text().split()


def timed(fn, *a):
    tee = Tee(sys.stderr)
    with GpuPeak() as g, contextlib.redirect_stderr(tee):
        t = time.perf_counter()
        lines, fails = asyncio.run(fn(*a))
        wall = time.perf_counter() - t
    stages = {k: float(v) for k, v in STAGE_RE.findall(tee.buf.getvalue())}
    return {"wall_s": wall, "stages_s": stages, "peak_vram_mib": g.peak, "n_lines": len(lines),
            "n_speakers": len({l[2] for l in lines}), "n_failures": len(fails)}, lines


def main():
    import diarize_demo as dd
    import wx_transcribe as wx
    exp = Exp("E2", {"durations_min": DURS_MIN, "reps": REPS, "warmup": "one 60 s clip first",
                     "local": "diarize_demo.run_pipeline_wx defaults (min_spk 1, max_spk 5)",
                     "language": "en fixed (autodetect picked 'nn' on an AMI clip in a trial run)",
                     "whisperx_arch_cuda_default": wx._DEVICE_DEFAULTS["cuda"][0],
                     "source": "distinct AMI dev meetings concatenated"})
    keys = {k: secret(k) for k in ("GROQ_API_KEY", "PYANNOTEAI_API_KEY")}
    hosted = all(keys.values())
    exp.config["hosted_timed"] = hosted
    if not hosted:
        exp.fail("hosted", "GROQ_API_KEY and/or PYANNOTEAI_API_KEY missing: hosted Stage 1 not timed")
    for k, v in keys.items():
        if v:
            os.environ[k] = v
    recs = [build(m) for m in DURS_MIN]
    exp.dataset("AMI Mix-Headset concatenations", "Edinburgh mirror", "dev",
                len({m for _, used in recs for m in used}), sp.AMI_LICENSE, sp.AMI_AUDIO,
                "meetings used per duration recorded in each record")

    done = exp.done()
    warm = sp.DATA / "e2" / "warmup60.wav"
    if not warm.exists():
        sp.write_wav(warm, sp.load16k(recs[0][0], dur=60))
    log("E2 warm-up (not recorded)")
    asyncio.run(dd.run_pipeline_wx(str(warm), "en"))
    for (path, used), minutes in zip(recs, DURS_MIN):
        for rep in range(REPS):
            key = f"local|{minutes}|{rep}"
            if key in done:
                continue
            try:
                r, _ = timed(dd.run_pipeline_wx, str(path), "en")
                r.update(system="local", minutes=minutes, rep=rep, meetings=used,
                         rtf=r["wall_s"] / (minutes * 60))
                exp.add(key, r)
                log(f"E2 {key}: {r['wall_s']:.1f}s RTF {r['rtf']:.3f} peak {r['peak_vram_mib']} MiB")
            except Exception as e:
                exp.fail(key, repr(e))
        if hosted and f"hosted|{minutes}|0" not in done:
            try:
                r, _ = timed(dd.run_pipeline, str(path), keys["PYANNOTEAI_API_KEY"], "en")
                r.update(system="hosted", minutes=minutes, rep=0, meetings=used,
                         rtf=r["wall_s"] / (minutes * 60))
                exp.add(f"hosted|{minutes}|0", r)
            except Exception as e:
                exp.fail(f"hosted|{minutes}", repr(e))

    rows = list(exp.done().values())
    summary = {}
    for sysname in ("local", "hosted"):
        for m in DURS_MIN:
            rs = [r for r in rows if r["system"] == sysname and r["minutes"] == m]
            if not rs:
                continue
            walls = sorted(r["wall_s"] for r in rs)
            stage_names = sorted({k for r in rs for k in r["stages_s"]})
            summary[f"{sysname}|{m}"] = {
                "n_reps": len(rs), "wall_s_mean": sum(walls) / len(walls),
                "wall_s_min": walls[0], "wall_s_max": walls[-1],
                "rtf_mean": sum(r["rtf"] for r in rs) / len(rs),
                "peak_vram_mib_max": max(r["peak_vram_mib"] for r in rs),
                "stages_s_mean": {k: sum(r["stages_s"].get(k, 0) for r in rs) / len(rs)
                                  for k in stage_names}}
    slow = sorted(rows, key=lambda r: -r["rtf"])[:10]
    exp.finish(summary, [{**r, "why": "highest real-time factor"} for r in slow])


if __name__ == "__main__":
    main()
