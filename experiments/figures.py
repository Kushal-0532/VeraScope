"""Paper figures from results/ only (no models). python -m experiments.figures

IEEE two-column: single column 3.45 in, full width 7.16 in, serif 8 pt, no
titles, series told apart by marker/linestyle/hatch (grayscale-safe),
Okabe-Ito palette. Each figure -> figures/<name>.pdf + .png (300 dpi);
figures/captions_draft.md gets one factual caption per figure. A figure whose
results are missing is listed as not generated, never drawn from placeholders.
"""

import json

from experiments.common import FIGURES, RESULTS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL, FULL = 3.45, 7.16
C = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]
LS = ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]
HATCH = ["", "///", "...", "xxx", "\\\\\\", "++"]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
CAPTIONS = []


def load(exp):
    p = RESULTS / exp / "raw_records.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name, caption):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)
    CAPTIONS.append(f"- **{name}**: {caption}")


def missing(name, why):
    CAPTIONS.append(f"- **{name}**: NOT GENERATED ({why})")


# ---------------------------------------------------------------- E1
def e1():
    r = load("E1")
    if not r or not r["records"]:
        return missing("e1_der_vs_chunk", "no E1 results")
    s, conds = r["summary"], r["config"]["conditions"]
    corpora = [c for c in ("ami", "vox") if f"{c}|whole" in s]
    fig, axes = plt.subplots(1, len(corpora), figsize=(COL, 2.0), sharey=True, squeeze=False)
    for ax, corpus in zip(axes[0], corpora):
        xs = [c for c in conds if f"{corpus}|{c}" in s]
        comp = [s[f"{corpus}|{c}"]["collar_0.25"] for c in xs]
        bottom = [0.0] * len(xs)
        for i, (k, lab) in enumerate((("miss", "Missed"), ("fa", "False alarm"), ("conf", "Confusion"))):
            vals = [100 * c[k] for c in comp]
            ax.bar(range(len(xs)), vals, bottom=bottom, color=C[i], hatch=HATCH[i],
                   edgecolor="black", linewidth=0.4, label=lab)
            bottom = [b + v for b, v in zip(bottom, vals)]
        for j, c in enumerate(comp):
            d = c["der"]
            if None not in d["ci95"]:
                ax.errorbar(j, 100 * d["value"], yerr=[[100 * (d["value"] - d["ci95"][0])],
                                                       [100 * (d["ci95"][1] - d["value"])]],
                            color="black", capsize=2, linewidth=0.8)
        ax.set_xticks(range(len(xs)), ["whole" if x == "whole" else x[5:] for x in xs], rotation=0)
        ax.set_xlabel(f"{'AMI' if corpus == 'ami' else 'VoxConverse'}: chunk length (s)")
    axes[0][0].set_ylabel("DER (%), collar ±0.25 s")
    axes[0][-1].legend(frameon=False, loc="upper right")
    n = {c: s[f"{c}|whole"]["n_files"] for c in corpora}
    save(fig, "e1_der_vs_chunk",
         f"DER of pyannote 3.1 for whole-file vs independently diarized chunks stitched without "
         f"cross-chunk speaker matching, split into missed speech, false alarm and speaker "
         f"confusion; error bars: 95% bootstrap CI over files; files per corpus {n}.")

    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for i, cond in enumerate(conds):
        rs = [x for x in r["records"] if x["cond"] == cond]
        if rs:
            ax.scatter([x["n_ref_spk"] for x in rs], [x["n_hyp_spk"] for x in rs], marker=MK[i],
                       s=14, facecolors="none", edgecolors=C[i % len(C)], linewidths=0.8,
                       label="whole" if cond == "whole" else f"{cond[5:]} s")
    lim = max(max(x["n_ref_spk"], x["n_hyp_spk"]) for x in r["records"]) + 1
    ax.plot([0, lim], [0, lim], color="gray", linewidth=0.6, linestyle=":")
    ax.set_yscale("log")
    ax.set_xlabel("Reference speakers per file")
    ax.set_ylabel("Predicted speaker labels per file")
    ax.legend(frameon=False, ncol=2, fontsize=6, loc="upper left")
    save(fig, "e1_speaker_count",
         f"Predicted vs reference number of speakers per file for each condition; chunked "
         f"conditions count chunk-local labels (no cross-chunk matching); dotted line y = x; "
         f"n = {len({x['uri'] for x in r['records']})} files.")


# ---------------------------------------------------------------- E2
def e2():
    r = load("E2")
    if not r or not r["records"]:
        return missing("e2_stage_times", "no E2 results")
    s = r["summary"]
    mins = sorted({x["minutes"] for x in r["records"] if x["system"] == "local"})
    stages = ["prep", "transcribe", "align||diarize", "fusion"]
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    bottom = [0.0] * len(mins)
    for i, st in enumerate(stages):
        vals = [s[f"local|{m}"]["stages_s_mean"].get(st, 0.0) for m in mins]
        ax.bar(range(len(mins)), vals, bottom=bottom, color=C[i], hatch=HATCH[i],
               edgecolor="black", linewidth=0.4, label=st.replace("||", " ‖ "))
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(mins)), [str(m) for m in mins])
    ax.set_xlabel("Recording duration (min)")
    ax.set_ylabel("Time (s)")
    ax.legend(frameon=False, loc="upper left")
    reps = {m: s[f"local|{m}"]["n_reps"] for m in mins}
    save(fig, "e2_stage_times",
         f"Mean per-stage wall time of the local Stage 1 pipeline (align and diarization run "
         f"concurrently) on concatenated distinct AMI dev meetings, GPU {r['env']['gpu']}; "
         f"repetitions per duration {reps}.")

    fig, ax = plt.subplots(figsize=(COL, 1.9))
    for i, sysname in enumerate(("local", "hosted")):
        ms = [m for m in mins if f"{sysname}|{m}" in s]
        if not ms:
            continue
        rs = [[x["rtf"] for x in r["records"] if x["system"] == sysname and x["minutes"] == m] for m in ms]
        mean = [sum(v) / len(v) for v in rs]
        ax.errorbar(ms, mean, yerr=[[a - min(v) for a, v in zip(mean, rs)],
                                    [max(v) - a for a, v in zip(mean, rs)]],
                    marker=MK[i], linestyle=LS[i], color=C[i], capsize=2, label=sysname)
    ax.set_xlabel("Recording duration (min)")
    ax.set_ylabel("Real-time factor")
    ax.legend(frameon=False)
    save(fig, "e2_rtf", "Real-time factor (wall time / audio duration) vs recording duration; "
                        "markers: mean, bars: min-max over repetitions (hosted: 1 run).")


# ---------------------------------------------------------------- E3
def e3():
    r = load("E3")
    if not r or not r["records"]:
        return missing("e3_wer", "no E3 results")
    s = r["summary"]
    archs = [a for a in r["config"]["archs"] if a in s]
    meetings = sorted({x["meeting"] for x in r["records"]})
    fig, ax = plt.subplots(figsize=(FULL, 1.9))
    w = 0.8 / len(archs)
    for i, a in enumerate(archs):
        vals = [100 * (s[a]["per_meeting"].get(m) or 0) for m in meetings]
        ax.bar([j + i * w for j in range(len(meetings))], vals, w, color=C[i], hatch=HATCH[i],
               edgecolor="black", linewidth=0.4, label=a)
    ax.set_xticks([j + w * (len(archs) - 1) / 2 for j in range(len(meetings))], meetings, rotation=45)
    ax.set_ylabel("WER (%)")
    ax.legend(frameon=False)
    save(fig, "e3_wer_per_meeting", f"WhisperX word error rate per AMI test meeting (Mix-Headset, "
                                    f"Whisper English normalizer); n = {len(meetings)} meetings.")

    fig, ax = plt.subplots(figsize=(COL, 1.9))
    for i, a in enumerate(archs):
        for j, reg in enumerate(("non_overlap", "overlap", "wer")):
            b = s[a][reg]
            ax.bar(j + i * 0.38, 100 * b["value"], 0.38, color=C[i], hatch=HATCH[i], edgecolor="black",
                   linewidth=0.4, label=a if j == 0 else None,
                   yerr=[[100 * (b["value"] - b["ci95"][0])], [100 * (b["ci95"][1] - b["value"])]],
                   capsize=2)
    ax.set_xticks([0.19, 1.19, 2.19], ["non-overlap", "overlap", "all"])
    ax.set_ylabel("WER (%)")
    ax.legend(frameon=False)
    save(fig, "e3_wer_overlap", "Pooled WER in overlapped (>= 2 reference speakers) vs "
                                "non-overlapped regions and overall; bars: 95% bootstrap CI over "
                                f"meetings; n = {len(meetings)}.")


# ---------------------------------------------------------------- E4
def e4():
    r = load("E4")
    if not r or not r["records"]:
        return missing("e4_heatmap", "no E4 results")
    s = r["summary"]
    g = s["grid_tune"]
    st = sorted({x["support_t"] for x in g})
    ct = sorted({x["contra_t"] for x in g})
    M = [[next(x["macro_f1"] for x in g if x["support_t"] == a and x["contra_t"] == b) for a in st]
         for b in ct]
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(st)), [f"{v:.2f}" for v in st], rotation=90)
    ax.set_yticks(range(len(ct)), [f"{v:.2f}" for v in ct])
    ax.set_xlabel("Support threshold")
    ax.set_ylabel("Contradiction threshold")
    pa = s["paper_thresholds"]
    bt = s["best_tune_thresholds"]
    for (a, b), mk, lab in (((pa["support_t"], pa["contra_t"]), "o", "paper"),
                            ((bt["support_t"], bt["contra_t"]), "s", "best (tune)")):
        if a in st and b in ct:
            ax.plot(st.index(a), ct.index(b), mk, markerfacecolor="none", markeredgecolor="white",
                    markersize=7, label=lab)
    ax.legend(frameon=False, loc="upper right", fontsize=6, labelcolor="white")
    fig.colorbar(im, ax=ax, label="Macro-F1 (tuning split)")
    save(fig, "e4_heatmap", f"Macro-F1 of the NLI verdict on the FEVER tuning split over the "
                            f"threshold grid; circle: paper setting (0.85/0.15), square: best tuning "
                            f"setting; n = {s['n_tune']} claims.")

    fig, ax = plt.subplots(figsize=(COL, 1.9))
    cv = [x for x in s["selective_heldout"] if x["selective_accuracy"] is not None]
    ax.plot([x["coverage"] for x in cv], [x["selective_accuracy"] for x in cv], marker="o",
            markersize=2, color=C[0])
    ax.set_xlabel("Coverage (share of claims with a decisive verdict)")
    ax.set_ylabel("Accuracy on decided claims")
    save(fig, "e4_selective", f"Selective prediction on the held-out split: symmetric bands "
                              f"0.5 ± w, w in [0, 0.49]; n = {s['n_heldout']} claims.")

    cls = ["supported", "disputed", "unclear"]
    cm = pa["heldout"]["confusion"]
    fig, ax = plt.subplots(figsize=(COL * 0.8, 2.0))
    Mx = [[cm[a][b] for b in cls] for a in cls]
    ax.imshow(Mx, cmap="Greys")
    top = max(max(row) for row in Mx) or 1
    for i in range(3):
        for j in range(3):
            ax.text(j, i, Mx[i][j], ha="center", va="center",
                    color="white" if Mx[i][j] > top / 2 else "black")
    ax.set_xticks(range(3), cls)
    ax.set_yticks(range(3), cls)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold (FEVER mapped)")
    save(fig, "e4_confusion", f"Confusion matrix at the paper thresholds (0.85/0.15) on the "
                              f"held-out split; n = {s['n_heldout']}.")

    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for i, (k, lab) in enumerate((("supports_vs_refutes", "SUPPORTS vs REFUTES"),
                                  ("supports_vs_rest", "SUPPORTS vs rest"))):
        b = [x for x in s["calibration"][k]["bins"] if x["n"]]
        ax.plot([x["conf"] for x in b], [x["acc"] for x in b], marker=MK[i], linestyle=LS[i],
                color=C[i], label=f"{lab} (ECE {s['calibration'][k]['ece']:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.6)
    ax.set_xlabel("Mean entailment score in bin")
    ax.set_ylabel("Empirical frequency of SUPPORTS")
    ax.legend(frameon=False, fontsize=6)
    save(fig, "e4_reliability", "Reliability diagram of the entailment score (10 equal-width "
                                f"bins); n = {s['calibration']['supports_vs_rest']['n']} claims.")


# ---------------------------------------------------------------- E5
def e5():
    r = load("E5")
    if not r or not r["records"]:
        return missing("e5_noise", "no E5 results")
    t = r["summary"]["by_rule_and_noise"]
    ks = r["config"]["noise_levels"]
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for i, rule in enumerate(r["config"]["rules"]):
        v = [t[f"{rule}@k={k}"] for k in ks]
        ax.errorbar([k + (i - 1.5) * 0.06 for k in ks], [x["macro_f1"] for x in v],
                    yerr=[[x["macro_f1"] - x["macro_f1_ci"]["ci95"][0] for x in v],
                          [x["macro_f1_ci"]["ci95"][1] - x["macro_f1"] for x in v]],
                    marker=MK[i], linestyle=LS[i], color=C[i], capsize=2,
                    label=rule.replace("_", " "))
    ax.set_xticks(ks)
    ax.set_xlabel("Injected distractor passages per claim")
    ax.set_ylabel("Macro-F1")
    ax.legend(frameon=False, fontsize=6)
    save(fig, "e5_noise", f"Macro-F1 of four verdict-aggregation rules vs number of distractor "
                          f"passages (FEVER validation, thresholds 0.85/0.15); bars: 95% bootstrap "
                          f"CI over claims; n = {r['summary']['n_claims']} claims.")


# ---------------------------------------------------------------- E6
def e6():
    r = load("E6")
    if not r or not r["summary"].get("by_context"):
        return missing("e6_context", "no E6 results")
    b = r["summary"]["by_context"]
    ks = sorted(b, key=int)
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    for i, (key, lab) in enumerate((("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"))):
        y = [b[k][key]["value"] if key == "f1" else b[k][key] for k in ks]
        ax.plot([int(k) for k in ks], y, marker=MK[i], linestyle=LS[i], color=C[i], label=lab)
    ax.set_xscale("log")
    ax.set_xticks([int(k) for k in ks], ks)
    ax.set_xlabel("Context lines on each side")
    ax.set_ylabel("Score (check-worthy class)")
    ax.legend(frameon=False)
    n = b[ks[0]]["n_lines"]
    save(fig, "e6_context", f"Claim-detection precision, recall and F1 on ClaimBuster debate "
                            f"sentences vs context window (model {r['config']['model']}); "
                            f"n = {n} lines per setting.")
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    for i, k in enumerate(ks):
        ax.scatter(b[k]["tokens_per_100_lines"], b[k]["f1"]["value"], marker=MK[i], color=C[i],
                   label=f"±{k}")
    ax.set_xlabel("Tokens per 100 lines (input + output)")
    ax.set_ylabel("F1")
    ax.legend(frameon=False, ncol=2)
    save(fig, "e6_cost", "Measured token cost per 100 transcript lines vs F1 for each context "
                         "window.")


# ---------------------------------------------------------------- E7 / E8
def e7():
    r = load("E7")
    if not r or not r["records"]:
        return missing("e7_dedup", "E7 skipped or no results")
    rs = r["records"]
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    x = range(len(rs))
    ax.bar([i - 0.2 for i in x], [v["n_claims"] for v in rs], 0.4, color=C[0], label="claims",
           edgecolor="black", linewidth=0.4)
    ax.bar([i + 0.2 for i in x], [v["n_unique"] for v in rs], 0.4, color=C[1], hatch="///",
           label="after dedup", edgecolor="black", linewidth=0.4)
    ax.set_xticks(list(x), [f"V{i + 1}" for i in x])
    ax.set_ylabel("Claims")
    ax.legend(frameon=False)
    save(fig, "e7_dedup", f"Claims detected before and after normalisation-based dedup per video; "
                          f"n = {len(rs)} videos.")


def e8():
    r = load("E8")
    if not r or not r["records"]:
        return missing("e8_hosted_local", "E8 skipped or no results")
    s = r["summary"]
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    for i, name in enumerate(("local", "hosted")):
        for j, k in enumerate(("wer", "line_der")):
            ax.bar(j + i * 0.38, 100 * s[name][k]["value"], 0.38, color=C[i], hatch=HATCH[i],
                   edgecolor="black", linewidth=0.4, label=name if j == 0 else None)
    ax.set_xticks([0.19, 1.19], ["WER", "Line-level DER"])
    ax.set_ylabel("%")
    ax.legend(frameon=False)
    save(fig, "e8_hosted_local", f"Hosted vs local Stage 1 on AMI test meetings; n = "
                                 f"{len(r['records'])} meetings.")


# ---------------------------------------------------------------- E9
def e9():
    r = load("E9")
    s = r["summary"] if r else {}
    systems = [x for x in ("pyannote", "gemma_zeroshot", "gemma_finetuned") if x in s]
    if not systems:
        return missing("e9_der", "no E9 results")
    labels = {"pyannote": "pyannote 3.1", "gemma_zeroshot": "Gemma zero-shot",
              "gemma_finetuned": "Gemma fine-tuned"}
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    for i, sname in enumerate(systems):
        d = s[sname]["der_c0.25"]
        ax.bar(i, 100 * d["value"], color=C[i], hatch=HATCH[i], edgecolor="black", linewidth=0.4,
               yerr=[[100 * (d["value"] - d["ci95"][0])], [100 * (d["ci95"][1] - d["value"])]],
               capsize=2)
    ax.set_xticks(range(len(systems)), [labels[x] for x in systems])
    ax.set_ylabel("DER (%), collar ±0.25 s")
    save(fig, "e9_der", f"Per-window DER (local optimal mapping, {r['config']['window_s']:.0f} s "
                        f"windows) on identical test windows; bars: 95% bootstrap CI resampling "
                        f"recordings; n = {s['n_common_windows']} windows.")
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    w = 0.8 / len(systems)
    for i, sname in enumerate(systems):
        b = s[sname]["by_n_spk"]
        ks = sorted(b, key=int)
        ax.bar([int(k) + (i - (len(systems) - 1) / 2) * w for k in ks],
               [100 * b[k]["der_c0.25"] for k in ks], w, color=C[i], hatch=HATCH[i],
               edgecolor="black", linewidth=0.4, label=labels[sname])
    ax.set_xlabel("Reference speakers in window")
    ax.set_ylabel("DER (%)")
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(frameon=False, fontsize=6)
    save(fig, "e9_der_by_spk", "Pooled per-window DER by number of reference speakers in the "
                               "window (collar ±0.25 s).")
    rows = [r"\begin{tabular}{lrrrrr}", r"\hline",
            r"System & DER (\%) & Spk.\ acc. & Parse fail & s/window & Peak MB \\", r"\hline"]
    for sname in systems:
        m = s[sname]
        rows.append(f"{labels[sname]} & {100 * m['der_c0.25']['value']:.1f} & "
                    f"{m['spk_count_acc']:.2f} & {m['parse_failure_rate']:.2f} & "
                    f"{m['seconds_per_window_mean']:.2f} & {m['peak_torch_mem_mb']:.0f} \\\\")
    rows += [r"\hline", r"\end{tabular}"]
    (FIGURES / "e9_table.tex").write_text("\n".join(rows) + "\n")
    CAPTIONS.append("- **e9_table.tex**: DER, speaker-count accuracy, parse-failure rate, "
                    "generation/inference seconds per window and peak torch memory per system.")


def main():
    for f in (e1, e2, e3, e4, e5, e6, e7, e8, e9):
        try:
            f()
        except Exception as e:
            missing(f.__name__, f"figure code failed: {e!r}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "captions_draft.md").write_text(
        "# Figure captions (draft, factual)\n\n" + "\n".join(CAPTIONS) + "\n")
    print("\n".join(CAPTIONS))




def status():
    """handoff/STATUS.md: per experiment ran / smoke / skipped / failures, from results only."""
    rows = ["| Exp | State | Smoke | Records | Failures | Finished (UTC) |", "|---|---|---|---|---|---|"]
    for e in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"):
        p = RESULTS / e / "summary.json"
        if not p.exists():
            part = RESULTS / e / "partial.jsonl"
            state = "partial (not finished)" if part.exists() else "not run"
            rows.append(f"| {e} | {state} | | | | |")
            continue
        s = json.loads(p.read_text())
        state = "skipped: " + s["summary"]["reason"] if s["summary"].get("skipped") else "finished"
        fails = "; ".join(f"{f['item']}: {f['reason'][:80]}" for f in s["failures"][:3])
        rows.append(f"| {e} | {state} | {s['config']['smoke']} | {s['n_records']} | "
                    f"{len(s['failures'])} {fails} | {s['finished_utc']} |")
    out = FIGURES.parent / "handoff" / "STATUS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Run status ({RESULTS})\n\n" + "\n".join(rows) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
    status()
