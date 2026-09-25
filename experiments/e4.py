"""E4 — NLI threshold sweep + calibration on FEVER (validation split).

One premise per claim (gold evidence sentences joined, truncated by the repo's
SNIPPET_CAP). Thresholds tuned on the tuning half, reported on the held-out
half together with the paper's 0.85 / 0.15.
"""

import random

from experiments.common import SEED, SMOKE, Exp, bootstrap, ece, log, prf
from experiments.nli import (CLASSES, FEVER_ID, FEVER_LICENSE, FEVER_SPLIT, PAPER_T, Scorer,
                             fever_claims, verdict)

N_PER_CLASS = 17 if SMOKE else 1000          # ~3000 claims full, ~50 smoke
SUPPORT_GRID = [round(0.50 + 0.05 * i, 2) for i in range(10)] + [0.99]
CONTRA_GRID = [0.01] + [round(0.05 * i, 2) for i in range(1, 11)]
BANDS = [round(0.01 * i, 2) for i in range(0, 50)]  # selective prediction: 0.5 +/- w


def evaluate(recs, st, ct):
    gold = [r["gold"] for r in recs]
    pred = [verdict([r["score"]], st, ct) for r in recs]
    m = prf(gold, pred, CLASSES)
    m["unclear_share"] = pred.count("unclear") / len(pred) if pred else 0.0
    return m, pred


def main():
    exp = Exp("E4", {"n_per_class": N_PER_CLASS, "support_grid": SUPPORT_GRID,
                     "contra_grid": CONTRA_GRID, "paper_thresholds": PAPER_T,
                     "tune_heldout_split": "50/50 per class, by claim, seed 42",
                     "premise": "gold evidence sentences joined with ' ', cut at SNIPPET_CAP"})
    claims, pools = fever_claims(N_PER_CLASS)
    rng = random.Random(SEED)
    for c in CLASSES:
        group = [x for x in claims if x["gold"] == c]
        rng.shuffle(group)
        for i, x in enumerate(group):
            x["split"] = "tune" if i < len(group) // 2 else "heldout"

    scorer = Scorer()
    exp.config["nli_model"] = __import__("verify_claims").NLI_MODEL
    exp.config["nli_revision"] = scorer.revision
    done = exp.done()
    for i, x in enumerate(claims):
        if x["id"] in done:
            continue
        try:
            premise = " ".join(x["passages"])
            s = scorer(x["claim"], premise)
            exp.add(x["id"], {"claim": x["claim"], "gold": x["gold"], "split": x["split"],
                              "score": s, "premise": premise[:1500],
                              "n_passages": len(x["passages"])})
        except Exception as e:
            exp.fail(x["id"], repr(e))
        if i % 200 == 0:
            log(f"E4 scored {i}/{len(claims)}")

    recs = list(exp.done().values())
    tune = [r for r in recs if r["split"] == "tune"]
    held = [r for r in recs if r["split"] == "heldout"]
    for split in ("tune", "heldout"):
        n = len([r for r in recs if r["split"] == split])
        exp.dataset(f"{FEVER_ID}", "HF main @ run time", f"{FEVER_SPLIT}/{split}", n,
                    FEVER_LICENSE, "https://huggingface.co/datasets/" + FEVER_ID,
                    f"stratified sample, pool sizes {pools}. NLI model's training data "
                    "(multilingual-NLI-26lang-2mil7) includes machine-translated FEVER-NLI "
                    "train: in-domain, not the same items.")

    grid = []
    for st in SUPPORT_GRID:
        for ct in CONTRA_GRID:
            m, _ = evaluate(tune, st, ct)
            grid.append({"support_t": st, "contra_t": ct, "macro_f1": m["macro_f1"],
                         "accuracy": m["accuracy"], "unclear_share": m["unclear_share"]})
    best = max(grid, key=lambda g: (g["macro_f1"], g["accuracy"]))

    def report(st, ct):
        m, pred = evaluate(held, st, ct)
        units = list(zip([r["gold"] for r in held], pred))
        m["macro_f1_ci"] = bootstrap(units, lambda u: prf([a for a, _ in u], [b for _, b in u],
                                                          CLASSES)["macro_f1"])
        m["accuracy_ci"] = bootstrap(units, lambda u: sum(a == b for a, b in u) / len(u))
        return m, pred

    paper_m, paper_pred = report(*PAPER_T)
    best_m, _ = report(best["support_t"], best["contra_t"])

    # selective prediction on held-out: symmetric band 0.5 +/- w
    curve = []
    for w in BANDS:
        pred = [verdict([r["score"]], 0.5 + w, 0.5 - w) for r in held]
        dec = [(r["gold"], p) for r, p in zip(held, pred) if p != "unclear"]
        curve.append({"band": w, "support_t": 0.5 + w, "contra_t": 0.5 - w,
                      "coverage": len(dec) / len(held) if held else 0.0,
                      "selective_accuracy": (sum(g == p for g, p in dec) / len(dec)) if dec else None})

    binary = [r for r in recs if r["gold"] != "unclear"]
    calib = {
        "supports_vs_refutes": ece([r["score"] for r in binary],
                                   [int(r["gold"] == "supported") for r in binary]),
        "supports_vs_rest": ece([r["score"] for r in recs],
                                [int(r["gold"] == "supported") for r in recs]),
        "note": "score = P(entail) normalised against contradiction (single-label zero-shot); "
                "ECE over all scored claims (tune+heldout), 10 equal-width bins",
    }

    wrong = [(r, p) for r, p in zip(held, paper_pred) if p != r["gold"]]
    wrong.sort(key=lambda rp: -abs(rp[0]["score"] - 0.5))
    errors = [{"claim": r["claim"], "premise": r["premise"], "gold": r["gold"], "pred": p,
               "score": r["score"], "thresholds": PAPER_T,
               "why": "most confident wrong verdicts at paper thresholds (held-out)"}
              for r, p in wrong[:10]]

    exp.finish({"n_tune": len(tune), "n_heldout": len(held),
                "paper_thresholds": {"support_t": PAPER_T[0], "contra_t": PAPER_T[1],
                                     "heldout": paper_m},
                "best_tune_thresholds": {**best, "heldout": best_m},
                "grid_tune": grid, "selective_heldout": curve, "calibration": calib}, errors)


if __name__ == "__main__":
    main()
