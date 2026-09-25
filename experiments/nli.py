"""FEVER data + cached NLI scoring through the repo's verify_claims module.

verify_claims._call_model talks to HF serverless zero-shot-classification with
one candidate label. Locally the same model under the transformers
zero-shot pipeline with one label gives the identical quantity:
softmax([contradiction, entailment])[entailment]. score_pair (with its
SNIPPET_CAP truncation) is the repo's own function, unmodified.
"""

import json
import random
import re

from experiments.common import DATA, SEED, SMOKE, log, stable_hash

import verify_claims as vc

LABEL_MAP = {"SUPPORTS": "supported", "REFUTES": "disputed", "NOT ENOUGH INFO": "unclear"}
CLASSES = ["supported", "disputed", "unclear"]
FEVER_ID = "copenlu/fever_gold_evidence"
FEVER_SPLIT = "validation"
FEVER_LICENSE = "CC BY-SA 3.0 (FEVER/Wikipedia); dataset card also lists GPL-3.0"
PAPER_T = (vc.SUPPORT_T, vc.CONTRADICT_T)  # 0.85 / 0.15, as stated in the paper
_PTB = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}",
        "``": '"', "''": '"'}


def clean(s):
    for k, v in _PTB.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+([,.;:!?)\]'])", r"\1", s)
    s = re.sub(r"([(\[])\s+", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def fever_claims(n_per_class, min_passages=1, seed=SEED):
    """Stratified sample from the validation split. -> [{id, claim, gold,
    passages}] where passages are distinct cleaned evidence sentences."""
    from datasets import load_dataset
    ds = load_dataset(FEVER_ID, split=FEVER_SPLIT)
    pools = {c: [] for c in CLASSES}
    seen = set()
    for row in ds:
        gold = LABEL_MAP.get(row["label"])
        if gold is None or row["claim"] in seen:
            continue
        passages = list(dict.fromkeys(clean(e[2]) for e in row["evidence"] if len(e) >= 3 and e[2]))
        # this dataset gives every NEI claim exactly one system-retrieved
        # sentence, so min_passages only binds SUPPORTS/REFUTES
        if len(passages) < (1 if gold == "unclear" else min_passages):
            continue
        seen.add(row["claim"])
        pools[gold].append({"id": str(row["id"]), "claim": row["claim"], "gold": gold,
                            "passages": passages})
    rng = random.Random(seed)
    out = []
    for c in CLASSES:
        rng.shuffle(pools[c])
        out += pools[c][:n_per_class]
    log(f"fever: pools {{{', '.join(f'{c}: {len(p)}' for c, p in pools.items())}}}, "
        f"sampled {len(out)} (min_passages={min_passages})")
    return out, {c: len(p) for c, p in pools.items()}


class Scorer:
    """(claim, snippet) -> P(entail) via verify_claims.score_pair, cached in
    DATA/nli_cache.jsonl so E4 and E5 share scores and reruns are free."""

    def __init__(self):
        import torch
        from transformers import pipeline
        dev = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("zero-shot-classification", model=vc.NLI_MODEL, device=dev)
        vc._call_model = lambda claim, snip: self.pipe(
            snip, candidate_labels=[claim], hypothesis_template="{}")["scores"][0]
        self.path = DATA / ("nli_cache_smoke.jsonl" if SMOKE else "nli_cache.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                k, v = json.loads(line)
                self.cache[k] = v
        self.revision = self._revision()

    @staticmethod
    def _revision():
        try:
            from huggingface_hub import model_info
            return model_info(vc.NLI_MODEL).sha
        except Exception as e:
            return f"unknown ({e!r})"

    def __call__(self, claim, snippet):
        k = stable_hash(claim, snippet[:vc.SNIPPET_CAP])
        if k not in self.cache:
            self.cache[k] = float(vc.score_pair(claim, snippet))
            with self.path.open("a") as f:
                f.write(json.dumps([k, self.cache[k]]) + "\n")
        return self.cache[k]


def verdict(scores, support_t=PAPER_T[0], contra_t=PAPER_T[1]):
    """The repo's own aggregation (Eq. 6) at the given thresholds."""
    vc.SUPPORT_T, vc.CONTRADICT_T = support_t, contra_t
    try:
        return vc._aggregate(scores)["verdict"]
    finally:
        vc.SUPPORT_T, vc.CONTRADICT_T = PAPER_T


def label(score, support_t=PAPER_T[0], contra_t=PAPER_T[1]):
    vc.SUPPORT_T, vc.CONTRADICT_T = support_t, contra_t
    try:
        return vc._label(score)
    finally:
        vc.SUPPORT_T, vc.CONTRADICT_T = PAPER_T
