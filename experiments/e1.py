"""E1 — chunked vs whole-file diarization (pyannote/speaker-diarization-3.1).

Pipeline object: diarize_demo._diarization_pipeline (the repo's loader, incl.
its fp16 embedding patch), called on in-memory audio. Chunked conditions
diarize each chunk independently, NO cross-chunk speaker matching, and stitch
by time offset with chunk-scoped labels. DER uses pyannote.metrics' global
optimal mapping over the whole file, so inconsistent labels across chunks are
penalised as confusion.
"""

import time

from experiments.common import COLLARS_PM, RESULTS, SMOKE, Exp, bootstrap, der, der_summary, \
    log, write_rttm
from experiments import speech as sp

CHUNKS = [30, 60, 120, 300, 600]
CONDS = ["whole"] + [f"chunk{c}" for c in CHUNKS]
N_AMI = 1 if SMOKE else 16                  # AMI test (full-corpus partition), all 16
N_VOX = 1 if SMOKE else 10                  # VoxConverse test, first 10 >= 15 min
VOX_MIN_S = 60 if SMOKE else 900
SMOKE_CROP_S = 300
MIN_TAIL_S = 5.0


def chunk_plan(dur, size):
    """[0, size, 2*size, ...]; a tail shorter than MIN_TAIL_S joins the previous chunk."""
    b = list(range(0, int(dur), size)) + [dur]
    b = [float(x) for x in b]
    if len(b) > 2 and b[-1] - b[-2] < MIN_TAIL_S:
        b.pop(-2)
    return list(zip(b, b[1:]))


def diarize(pipe, audio):
    df = pipe(audio, min_speakers=None, max_speakers=None)
    rows = df.to_dict("records") if hasattr(df, "to_dict") else df
    return sorted((float(r["start"]), float(r["end"]), str(r["speaker"])) for r in rows)


def run_cond(pipe, audio, dur, cond):
    if cond == "whole":
        return diarize(pipe, audio)
    out = []
    for i, (lo, hi) in enumerate(chunk_plan(dur, int(cond[5:]))):
        seg = audio[int(lo * sp.SR):int(hi * sp.SR)]
        out += [(s + lo, e + lo, f"c{i}_{k}") for s, e, k in diarize(pipe, seg)]
    return out


def files():
    out = []
    for m in sp.ami_meetings("test")[:N_AMI]:
        out.append(("ami", m, sp.ami_audio(m), sp.ami_rttm(m, "test"), sp.ami_uem(m, "test")))
    for uri, path, turns, d in sp.vox_files("test", N_VOX, VOX_MIN_S):
        out.append(("vox", uri, path, turns, (0.0, d)))
    return out


def main():
    exp = Exp("E1", {"model": "pyannote/speaker-diarization-3.1", "conditions": CONDS,
                     "speaker_bounds": "unconstrained (min/max None); the app defaults to 1..5",
                     "collars_pm_s": COLLARS_PM, "chunk_tail_merge_s": MIN_TAIL_S,
                     "cross_chunk_matching": False, "smoke_crop_s": SMOKE_CROP_S if SMOKE else None,
                     "der": "pyannote.metrics DiarizationErrorRate, global optimal mapping, "
                            "overlap scored, collar arg = 2 x collar_pm"})
    import torch
    import diarize_demo as dd
    pipe = dd._diarization_pipeline("cuda" if torch.cuda.is_available() else "cpu")
    fl = files()
    exp.dataset("AMI Mix-Headset (only_words RTTM)", "BUTSpeechFIT AMI-diarization-setup main",
                "test", sum(f[0] == "ami" for f in fl), sp.AMI_LICENSE, sp.AMI_SETUP)
    exp.dataset("VoxConverse", "HF " + sp.VOX_ID, f"test (>= {VOX_MIN_S} s)",
                sum(f[0] == "vox" for f in fl), sp.VOX_LICENSE, "https://huggingface.co/datasets/" + sp.VOX_ID)
    done = exp.done()
    for corpus, uri, path, ref, uem in fl:
        audio = None
        for cond in CONDS:
            key = f"{uri}|{cond}"
            if key in done:
                continue
            try:
                if audio is None:
                    audio = sp.load16k(path, dur=SMOKE_CROP_S if SMOKE else None)
                dur = len(audio) / sp.SR
                ref_c = [t for t in ref if t[0] < dur]
                ref_c = [(s, min(e, dur), k) for s, e, k in ref_c]
                u = (uem[0], min(uem[1], dur)) if uem else (0.0, dur)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                hyp = run_cond(pipe, audio, dur, cond)
                wall = time.perf_counter() - t
                write_rttm(exp.dir / "rttm" / cond / f"{uri}.rttm", uri, hyp)
                rec = {"corpus": corpus, "uri": uri, "cond": cond, "dur_s": dur, "wall_s": wall,
                       "rtf": wall / dur, "uem": u,
                       "peak_torch_mem_mb": (torch.cuda.max_memory_allocated() / 2**20
                                             if torch.cuda.is_available() else None),
                       "n_ref_spk": len({k for _, _, k in ref_c}),
                       "n_hyp_spk": len({k for _, _, k in hyp}), "n_hyp_turns": len(hyp)}
                for c in COLLARS_PM:
                    rec[f"der_c{c}"] = der(ref_c, hyp, c, uem=u)
                exp.add(key, rec)
                log(f"E1 {key}: DER(0.25)={rec['der_c0.25']['der']:.3f} "
                    f"spk {rec['n_hyp_spk']}/{rec['n_ref_spk']} {wall:.0f}s")
            except Exception as e:
                exp.fail(key, repr(e))

    recs = list(exp.done().values())
    summary = {}
    for corpus in ("ami", "vox", "all"):
        for cond in CONDS:
            rs = [r for r in recs if r["cond"] == cond and corpus in (r["corpus"], "all")]
            if not rs:
                continue
            summary[f"{corpus}|{cond}"] = {
                **{f"collar_{c}": der_summary([r[f"der_c{c}"] for r in rs]) for c in COLLARS_PM},
                "spk_count_abs_err": bootstrap([abs(r["n_hyp_spk"] - r["n_ref_spk"]) for r in rs],
                                               lambda u: sum(u) / len(u)),
                "spk_count_exact": sum(r["n_hyp_spk"] == r["n_ref_spk"] for r in rs) / len(rs),
                "wall_s_total": sum(r["wall_s"] for r in rs), "n_files": len(rs)}
    by = {(r["uri"], r["cond"]): r for r in recs}
    errs = []
    for r in recs:
        w = by.get((r["uri"], "whole"))
        if r["cond"] == "whole" or not w:
            continue
        errs.append({"uri": r["uri"], "corpus": r["corpus"], "cond": r["cond"],
                     "der_whole": w["der_c0.25"], "der_chunked": r["der_c0.25"],
                     "delta_der": r["der_c0.25"]["der"] - w["der_c0.25"]["der"],
                     "n_ref_spk": r["n_ref_spk"], "n_hyp_spk_whole": w["n_hyp_spk"],
                     "n_hyp_spk_chunked": r["n_hyp_spk"],
                     "rttm": f"{RESULTS.name}/E1/rttm/{r['cond']}/{r['uri']}.rttm"})
    errs.sort(key=lambda e: -e["delta_der"])
    exp.finish(summary, errs)


def _selfcheck():
    assert chunk_plan(100, 30) == [(0, 30), (30, 60), (60, 90), (90, 100.0)]  # 10 s tail kept
    assert chunk_plan(92, 30) == [(0, 30), (30, 60), (60, 92.0)]      # 2 s tail merged
    assert chunk_plan(20, 30) == [(0, 20.0)]

    class Fake:
        def __call__(self, audio, **k):
            d = len(audio) / sp.SR
            return [{"start": 0.0, "end": d / 2, "speaker": "SPEAKER_00"},
                    {"start": d / 2, "end": d, "speaker": "SPEAKER_01"}]
    import numpy as np
    hyp = run_cond(Fake(), np.zeros(sp.SR * 100, np.float32), 100.0, "chunk30")
    assert hyp[2] == (30.0, 45.0, "c1_SPEAKER_00") and hyp[-1][1] == 100.0, hyp
    assert len({k for *_, k in hyp}) == 8  # 4 chunks x 2 local labels, never merged
    print("e1 selfcheck ok")


if __name__ == "__main__":
    import sys
    _selfcheck() if "--selfcheck" in sys.argv else main()
