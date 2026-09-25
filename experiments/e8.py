"""E8 — hosted vs local Stage 1 on the same AMI test meetings (optional).

hosted: diarize_demo.run_pipeline (Groq Whisper + pyannoteAI + wav2vec2 conf)
local:  diarize_demo.run_pipeline_wx (WhisperX + pyannote 3.1)
Both emit the frozen line contract (start, end, speaker, text, conf). Scored
against AMI references: WER (normalised) and line-level DER (each line is a
speaker turn). Agreement: WER of hosted text vs local text, and line-DER with
local as reference. Line confidences are reported per system, not compared:
the scales come from different models. Cost is not computed (provider
invoices); audio seconds processed are recorded.
"""

import os

from experiments.common import SMOKE, Exp, bootstrap, der, log, pooled_der, pooled_wer, secret, \
    wer_counts
from experiments import speech as sp
from experiments.e2 import timed

N = 1 if SMOKE else 3
CROP_S = 120 if SMOKE else None


def main():
    import diarize_demo as dd
    exp = Exp("E8", {"n_meetings": N, "crop_s": CROP_S, "collar_pm_s": 0.25,
                     "der_unit": "fused lines as turns (both systems alike)", "language": "en fixed"})
    keys = {k: secret(k) for k in ("GROQ_API_KEY", "PYANNOTEAI_API_KEY")}
    if not all(keys.values()):
        return exp.skip("hosted keys missing (GROQ_API_KEY and/or PYANNOTEAI_API_KEY)")
    os.environ.update(keys)
    meetings = sp.ami_meetings("test")[:N]
    exp.dataset("AMI Mix-Headset", "Edinburgh mirror", "test", len(meetings), sp.AMI_LICENSE, sp.AMI_AUDIO)
    done = exp.done()
    for m in meetings:
        if m in done:
            continue
        try:
            path = sp.ami_audio(m)
            if CROP_S:
                path = sp.write_wav(sp.DATA / "e8" / f"{m}_{CROP_S}.wav", sp.load16k(path, dur=CROP_S))
            dur = sp.duration(path)
            ref_text = " ".join(w[2] for w in sp.ami_words(m) if w[1] <= dur)
            ref_t = [(s, min(e, dur), k) for s, e, k in sp.ami_rttm(m, "test") if s < dur]
            rec, lines = {"dur_s": dur}, {}
            for name, fn, args in (("local", dd.run_pipeline_wx, ("en",)),
                                   ("hosted", dd.run_pipeline, (keys["PYANNOTEAI_API_KEY"], "en"))):
                r, lines[name] = timed(fn, str(path), *args)
                confs = [l[4] for l in lines[name] if l[4] is not None]
                rec[name] = {**r, "wer": wer_counts(ref_text, " ".join(l[3] for l in lines[name])),
                             "line_der": der(ref_t, [(l[0], l[1], l[2]) for l in lines[name]], 0.25,
                                             uem=(0.0, dur)),
                             "conf_mean": sum(confs) / len(confs) if confs else None,
                             "conf_none_share": 1 - len(confs) / max(1, len(lines[name]))}
            rec["agreement"] = {
                "wer_hosted_vs_local": wer_counts(" ".join(l[3] for l in lines["local"]),
                                                  " ".join(l[3] for l in lines["hosted"])),
                "line_der_hosted_vs_local": der([(l[0], l[1], l[2]) for l in lines["local"]],
                                                [(l[0], l[1], l[2]) for l in lines["hosted"]],
                                                0.25, uem=(0.0, dur))}
            rec["sample_lines"] = {k: [list(l) for l in v[:15]] for k, v in lines.items()}
            exp.add(m, rec)
            log(f"E8 {m}: WER local {rec['local']['wer']['wer']:.3f} hosted {rec['hosted']['wer']['wer']:.3f}")
        except Exception as e:
            exp.fail(m, repr(e))

    recs = list(exp.done().values())
    summary = {"audio_s_processed_each": sum(r["dur_s"] for r in recs)}
    for name in ("local", "hosted"):
        if recs:
            summary[name] = {"wer": bootstrap([r[name]["wer"] for r in recs], pooled_wer),
                             "line_der": bootstrap([r[name]["line_der"] for r in recs], pooled_der),
                             "rtf_mean": sum(r[name]["wall_s"] / r["dur_s"] for r in recs) / len(recs),
                             "conf_mean": [r[name]["conf_mean"] for r in recs]}
    if recs:
        summary["agreement"] = {
            "wer": pooled_wer([r["agreement"]["wer_hosted_vs_local"] for r in recs]),
            "line_der": pooled_der([r["agreement"]["line_der_hosted_vs_local"] for r in recs])}
    exp.finish(summary, [{"meeting": r["key"], "sample_lines": r["sample_lines"],
                          "local_wer": r["local"]["wer"], "hosted_wer": r["hosted"]["wer"]} for r in recs])


if __name__ == "__main__":
    main()
