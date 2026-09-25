"""E5 — verdict aggregation ablation under injected retrieval noise.

FEVER claims with >= 2 distinct gold evidence sentences, each sentence a
separate passage. k in {0,1,2,4} distractor passages from unrelated claims
(nested: the k=1 distractor is in the k=2 set, etc.). Rules compared at the
paper's thresholds (0.85 / 0.15):
  contradiction_priority  verify_claims._aggregate (paper Eq. 6)
  majority                majority of per-passage labels, tie/neutral -> unclear
  mean                    band the mean score
  max_support             support wins if any passage supports (mirror of Eq. 6)
"""

import random
from collections import Counter

from experiments.common import SEED, SMOKE, Exp, bootstrap, log, prf
from experiments.nli import (CLASSES, FEVER_ID, FEVER_LICENSE, FEVER_SPLIT, PAPER_T, Scorer,
                             fever_claims, label, verdict)

N_PER_CLASS = 17 if SMOKE else 500
NOISE = [0, 1, 2, 4]
TO_VERDICT = {"supports": "supported", "contradicts": "disputed", "neutral": "unclear"}


def rule_majority(scores):
    if not scores:
        return "unclear"
    c = Counter(label(s) for s in scores).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return "unclear"
    return TO_VERDICT[c[0][0]]


def rule_mean(scores):
    return verdict([sum(scores) / len(scores)]) if scores else "unclear"


def rule_max_support(scores):
    labs = [label(s) for s in scores]
    if "supports" in labs:
        return "supported"
    return "disputed" if "contradicts" in labs else "unclear"


RULES = {"contradiction_priority": verdict, "majority": rule_majority, "mean": rule_mean,
         "max_support": rule_max_support}


def main():
    exp = Exp("E5", {"n_per_class": N_PER_CLASS, "noise_levels": NOISE,
                     "thresholds": PAPER_T, "rules": list(RULES),
                     "passages": "each distinct gold evidence sentence is one passage",
                     "distractors": "evidence sentences of other sampled claims, nested per k"})
    claims, pools = fever_claims(N_PER_CLASS, min_passages=2)
    exp.dataset(FEVER_ID, "HF main @ run time", f"{FEVER_SPLIT}/multi-evidence", len(claims),
                FEVER_LICENSE, "https://huggingface.co/datasets/" + FEVER_ID,
                f">=2 distinct evidence sentences for SUPPORTS/REFUTES; NEI claims carry the "
                f"single retrieved sentence the dataset provides; pool sizes {pools}. Not tuned on: "
                "rules use the paper's fixed thresholds.")
    scorer = Scorer()
    exp.config["nli_revision"] = scorer.revision
    done = exp.done()
    for i, x in enumerate(claims):
        if x["id"] in done:
            continue
        rng = random.Random(f"{SEED}-{x['id']}")
        others = [y for y in claims if y["claim"] != x["claim"]]
        distract = [rng.choice(rng.choice(others)["passages"]) for _ in range(max(NOISE))]
        try:
            gold_s = [scorer(x["claim"], p) for p in x["passages"]]
            dis_s = [scorer(x["claim"], p) for p in distract]
            exp.add(x["id"], {"claim": x["claim"], "gold": x["gold"], "passages": x["passages"],
                              "gold_scores": gold_s, "distractors": distract,
                              "distractor_scores": dis_s})
        except Exception as e:
            exp.fail(x["id"], repr(e))
        if i % 100 == 0:
            log(f"E5 scored {i}/{len(claims)}")

    recs = list(exp.done().values())
    table, fd = {}, {}
    for k in NOISE:
        for name, rule in RULES.items():
            units = [(r["gold"], rule(r["gold_scores"] + r["distractor_scores"][:k])) for r in recs]
            m = prf([g for g, _ in units], [p for _, p in units], CLASSES)
            m["macro_f1_ci"] = bootstrap(units, lambda u: prf([a for a, _ in u], [b for _, b in u],
                                                              CLASSES)["macro_f1"])
            m["recall_ci"] = {c: bootstrap([u for u in units if u[0] == c],
                                           lambda u: sum(a == b for a, b in u) / len(u))
                              for c in CLASSES}
            table[f"{name}@k={k}"] = {"rule": name, "k": k, **m}
        cp = [(r, verdict(r["gold_scores"] + r["distractor_scores"][:k])) for r in recs]
        fd[k] = {g: {"n": sum(r["gold"] == g for r, _ in cp),
                     "false_disputed_rate": (sum(r["gold"] == g and p == "disputed" for r, p in cp)
                                             / max(1, sum(r["gold"] == g for r, _ in cp)))}
                 for g in ("supported", "unclear")}

    errs = []
    for r in recs:
        k = max(NOISE)
        s = r["gold_scores"] + r["distractor_scores"][:k]
        if r["gold"] == "supported" and verdict(s) == "disputed":
            errs.append({"claim": r["claim"], "gold": r["gold"], "pred": "disputed", "k": k,
                         "passages": [{"text": p, "score": sc, "distractor": i >= len(r["passages"])}
                                      for i, (p, sc) in enumerate(zip(r["passages"] + r["distractors"][:k], s))],
                         "why": "contradiction-priority false 'disputed' on a SUPPORTS claim"})
    errs.sort(key=lambda e: min(p["score"] for p in e["passages"]))
    exp.finish({"n_claims": len(recs), "by_rule_and_noise": table,
                "contradiction_priority_false_disputed": fd}, errs)


if __name__ == "__main__":
    main()
