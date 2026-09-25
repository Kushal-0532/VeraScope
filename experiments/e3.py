"""E3 — ASR word error rate on AMI test (Mix-Headset), WhisperX.

Hypothesis words: the repo's wx_transcribe.transcribe_aligned (VAD + batched
faster-whisper + wav2vec2 alignment, language autodetected as deployed), for
ARCHS. Reference: AMI manual words (ami_public_manual_1.6.2). Both sides go
through the Whisper English normalizer before jiwer.

Overlap split: a word (ref or hyp) is "overlap" if its midpoint lies where the
only_words RTTM has >= 2 active speakers; WER per region = errors of the
region-restricted word sequences (an approximation: alignment runs per region).

cpWER (meeteval): hypothesis speakers from diarize_demo.group_lines_wx(words,
turns) with turns = E1's whole-file pyannote RTTM for the same meeting (or a
fresh unconstrained diarization when E1 has not produced it).
"""

import gc
import json
import os
import time

from experiments.common import RESULTS, SMOKE, Exp, bootstrap, log, normalize_text, pooled_wer, \
    read_rttm, wer_counts, write_json, write_rttm
from experiments import speech as sp

ARCHS = ["large-v3", "medium"]           # large-v3 per brief; medium = deployed default on cuda
N_MEETINGS = 2 if SMOKE else 16
SMOKE_CROP_S = 300


def text(words):
    return " ".join(w for _, _, w, *_ in sorted(words))


def split_wer(ref, hyp, regions):
    mid = lambda w: (w[0] + w[1]) / 2
    out = {}
    for name, keep in (("overlap", True), ("non_overlap", False)):
        r = [w for w in ref if sp.in_regions(mid(w), regions) == keep]
        h = [w for w in hyp if sp.in_regions(mid(w), regions) == keep]
        out[name] = wer_counts(text(r), text(h))
    return out


def cpwer(ref_words, lines):
    import meeteval
    ref, hyp = {}, {}
    for _, _, w, spk in ref_words:
        ref.setdefault(spk, []).append(w)
    for _, _, spk, t, _ in lines:
        hyp.setdefault(str(spk), []).append(t)
    ref = {k: normalize_text(" ".join(v)) for k, v in ref.items()}
    hyp = {k: normalize_text(" ".join(v)) for k, v in hyp.items()}
    r = meeteval.wer.wer.cp.cp_word_error_rate(reference=ref, hypothesis=hyp)
    return {"errors": r.errors, "length": r.length, "cpwer": r.error_rate,
            "missed_speaker": r.missed_speaker, "falarm_speaker": r.falarm_speaker,
            "n_ref_spk": len(ref), "n_hyp_spk": len(hyp)}


def main():
    import torch
    import diarize_demo as dd
    import wx_transcribe as wx
    exp = Exp("E3", {"archs": ARCHS,
                     "language": "en fixed: autodetect picked 'nn' on an AMI clip in a trial run, "
                                 "which would make WER meaningless",
                     "compute_type": "float16 on cuda (wx_transcribe default)",
                     "normalizer": "whisper_normalizer.english.EnglishTextNormalizer",
                     "reference": "AMI manual words, punctuation tokens dropped",
                     "overlap_rule": "word midpoint in >=2-speaker region of only_words RTTM",
                     "cpwer_turns": "E1 whole-file RTTM, else unconstrained pyannote 3.1",
                     "smoke_crop_s": SMOKE_CROP_S if SMOKE else None})
    meetings = sp.ami_meetings("test")[:N_MEETINGS]
    exp.dataset("AMI Mix-Headset + manual words 1.6.2", "Edinburgh mirror", "test",
                len(meetings), sp.AMI_LICENSE, sp.AMI_WORDS)
    done = exp.done()
    for arch in ARCHS:
        os.environ["WHISPERX_ARCH"] = arch
        for m in meetings:
            key = f"{arch}|{m}"
            if key in done:
                continue
            try:
                path = sp.ami_audio(m)
                crop = SMOKE_CROP_S if SMOKE else None
                if crop:
                    path = sp.write_wav(sp.DATA / "e3" / f"{m}_{crop}.wav", sp.load16k(path, dur=crop))
                ref_w = [w for w in sp.ami_words(m) if not crop or w[1] <= crop]
                turns_ref = [t for t in sp.ami_rttm(m, "test") if not crop or t[0] < crop]
                t = time.perf_counter()
                words = wx.transcribe_aligned(str(path), language="en")
                wall = time.perf_counter() - t
                write_json(exp.dir / "words" / arch / f"{m}.json", words)
                wc = wer_counts(text(ref_w), text(words))
                rec = {"arch": arch, "meeting": m, "wall_s": wall, "n_hyp_words": len(words),
                       "n_ref_words": len(ref_w), "wer": wc,
                       "split": split_wer(ref_w, words, sp.overlap_timeline(turns_ref))}
                rttm = RESULTS / "E1" / "rttm" / "whole" / f"{m}.rttm"
                if rttm.exists() and not crop:
                    turns, src = next(iter(read_rttm(rttm).values())), "E1"
                else:  # diarize once per meeting, reuse for the second arch
                    own = exp.dir / "rttm" / f"{m}{'_crop' if crop else ''}.rttm"
                    if not own.exists():
                        write_rttm(own, m, dd.wx_diarize(str(path), 1, None))
                    turns, src = next(iter(read_rttm(own).values()), []), "fresh"
                rec["cpwer"] = {**cpwer(ref_w, dd.group_lines_wx(words, turns)), "turns_from": src}
                exp.add(key, rec)
                log(f"E3 {key}: WER {wc['wer']:.3f} cpWER {rec['cpwer']['cpwer']:.3f} {wall:.0f}s")
            except Exception as e:
                exp.fail(key, repr(e))
        wx._ASR.clear()
        gc.collect()
        torch.cuda.empty_cache()

    recs = list(exp.done().values())
    summary = {}
    for arch in ARCHS:
        rs = [r for r in recs if r["arch"] == arch]
        if not rs:
            continue
        cp_units = [r["cpwer"] for r in rs]
        summary[arch] = {
            "wer": bootstrap([r["wer"] for r in rs], pooled_wer),
            "overlap": bootstrap([r["split"]["overlap"] for r in rs], pooled_wer),
            "non_overlap": bootstrap([r["split"]["non_overlap"] for r in rs], pooled_wer),
            "cpwer": bootstrap(cp_units, lambda u: sum(x["errors"] for x in u) /
                               max(1, sum(x["length"] for x in u))),
            "per_meeting": {r["meeting"]: r["wer"]["wer"] for r in rs},
            "S_D_I": {k: sum(r["wer"][k] for r in rs) for k in "SDIN"},
            "wall_s_total": sum(r["wall_s"] for r in rs)}
    worst = sorted(recs, key=lambda r: -(r["wer"]["wer"] or 0))[:10]
    errs = []
    for r in worst:
        words = json.loads((exp.dir / "words" / r["arch"] / f"{r['meeting']}.json").read_text())
        ref_w = sp.ami_words(r["meeting"])
        lo = words[len(words) // 2][0] if words else 0.0
        errs.append({"arch": r["arch"], "meeting": r["meeting"], "wer": r["wer"],
                     "split": r["split"], "cpwer": r["cpwer"],
                     "excerpt_window_s": [lo, lo + 30],
                     "ref_excerpt": text([w for w in ref_w if lo <= w[0] < lo + 30]),
                     "hyp_excerpt": text([w for w in words if lo <= w[0] < lo + 30])})
    exp.finish(summary, errs)


if __name__ == "__main__":
    main()
