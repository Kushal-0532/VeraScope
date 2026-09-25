"""E6 — claim detection vs context window on ClaimBuster.

System: the repo's claim_detect (_render_batch + _detect_batch: same prompt,
model, schema, BATCH_SIZE=20), with claim_detect.CONTEXT set to each value in
CONTEXTS. Lines are debate sentences in (File_id, Line_number) order from
groundtruth.csv + crowdsourced.csv; label = check-worthy factual (Verdict 1).
A line is predicted positive when >= 1 claim is emitted for it.

Evaluated lines come from contiguous blocks; context lines are the real
neighbours in the debate, including ones outside the block.
"""

import asyncio
import csv
import os
import random
import re
import time

from experiments.common import DATA, LLM_SPEND_CAP_USD, SEED, SMOKE, Exp, blocker, bootstrap, \
    fetch, log, prf, secret

import claim_detect as cd

CONTEXTS = [1, 2, 5, 10, 20]
BLOCK = 100
N_BLOCKS = 1 if SMOKE else int(os.environ.get("E6_BLOCKS", "4"))  # 50 lines smoke, 400 full (Groq free tier)
# Groq tokens-per-minute budget; free tier is 8000. Calls are paced under it so
# 429s do not eat the retries. Dev tier: raise GROQ_TPM.
GROQ_TPM = int(os.environ.get("GROQ_TPM", "8000"))
SMOKE_LINES = 50
# Price used for the spend cap and the $ column. Tokens are the measurement;
# $ = tokens x this assumed list price. VERIFY against console.groq.com before
# quoting dollars in the paper.
PRICE_IN, PRICE_OUT = 0.10, 0.50           # USD per 1M tokens, conservative
ZENODO = "https://zenodo.org/records/3609356/files/{}?download=1"


def load_debates():
    rows = {}
    for name, prio in (("crowdsourced.csv", 0), ("groundtruth.csv", 1)):
        p = fetch(ZENODO.format(name), DATA / "claimbuster" / name)
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                sid = r["Sentence_id"]
                if sid not in rows or prio:   # groundtruth label wins
                    rows[sid] = {"sid": sid, "text": r["Text"], "speaker": r["Speaker"],
                                 "file": r["File_id"], "line": int(float(r["Line_number"])),
                                 "gold": int(float(r["Verdict"])) == 1, "gt": bool(prio)}
    debates = {}
    for r in rows.values():
        debates.setdefault(r["file"], []).append(r)
    for d in debates.values():
        d.sort(key=lambda r: r["line"])
    return debates


def contiguity(debates):
    pairs = [(a["line"], b["line"]) for d in debates.values() for a, b in zip(d, d[1:])]
    return sum(b - a == 1 for a, b in pairs) / max(1, len(pairs))


class Usage:
    """Wraps groq AsyncCompletions.create to count tokens per call."""

    def __init__(self):
        from groq.resources.chat.completions import AsyncCompletions
        self.calls = []
        orig = AsyncCompletions.create
        usage = self

        async def create(self_, *a, **k):
            t = time.perf_counter()
            resp = await orig(self_, *a, **k)
            u = getattr(resp, "usage", None)
            usage.calls.append({"in": getattr(u, "prompt_tokens", 0) or 0,
                                "out": getattr(u, "completion_tokens", 0) or 0,
                                "s": time.perf_counter() - t, "t": time.time()})
            return resp
        AsyncCompletions.create = create


def cost(tok_in, tok_out):
    return tok_in / 1e6 * PRICE_IN + tok_out / 1e6 * PRICE_OUT


async def run_setting(exp, client, usage, lines, blocks, ctx, done, spent):
    """Sequential, TPM-paced. Recorded seconds are the API call only (pacing
    sleeps and 429 waits excluded), so latency is not inflated by throttling."""
    cd.CONTEXT = ctx
    todo = [(b, s, min(s + cd.BATCH_SIZE, hi)) for b, (lo, hi) in enumerate(blocks)
            for s in range(lo, hi, cd.BATCH_SIZE) if f"c{ctx}-{s}" not in done]
    est = 2500                                    # tokens/request until measured
    for b, lo, hi in todo:
        if spent[0] >= LLM_SPEND_CAP_USD:
            return
        for attempt in range(8):
            window = [c for c in usage.calls if time.time() - c["t"] < 60]
            used = sum(c["in"] + c["out"] for c in window)
            if window and used + est > 0.9 * GROQ_TPM:
                await asyncio.sleep(60 - (time.time() - window[0]["t"]) + 0.5)
                continue
            n0 = len(usage.calls)
            t = time.perf_counter()
            try:
                claims = await cd._detect_batch(client, lines, lo, hi, cd.MODEL)
                break
            except Exception as e:
                wait = re.search(r"try again in ([\d.]+)s", str(e))
                if "429" in str(e) and attempt < 7:
                    await asyncio.sleep(float(wait.group(1)) + 1 if wait else 20)
                    continue
                exp.fail(f"c{ctx}-{lo}", repr(e))
                claims = None
                break
        if claims is None:
            continue
        calls = usage.calls[n0:] or [{"in": 0, "out": 0}]
        tin, tout = sum(c["in"] for c in calls), sum(c["out"] for c in calls)
        est = max(est, tin + tout)
        spent[0] += cost(tin, tout)
        exp.add(f"c{ctx}-{lo}", {"context": ctx, "block": b, "lo": lo, "hi": hi,
                                 "claims": claims, "tokens_in": tin, "tokens_out": tout,
                                 "seconds": time.perf_counter() - t})


def main():
    exp = Exp("E6", {"contexts": CONTEXTS, "batch_size": cd.BATCH_SIZE, "model": cd.MODEL,
                     "paper_context": cd.CONTEXT, "block": BLOCK, "n_blocks": N_BLOCKS,
                     "price_usd_per_1m": {"in": PRICE_IN, "out": PRICE_OUT,
                                          "note": "assumed, verify before quoting $"},
                     "spend_cap_usd": LLM_SPEND_CAP_USD, "groq_tpm": GROQ_TPM,
                     "latency_note": "seconds = API call time; runs sequential + TPM-paced",
                     "positive_class": "Verdict == 1 (check-worthy factual sentence)"})
    key = secret("GROQ_API_KEY")
    if not key:
        return exp.skip("GROQ_API_KEY not set: claim detection needs the Groq API")
    debates = load_debates()
    rng = random.Random(SEED)
    # one flat line list so _render_batch's context never crosses a debate:
    # blocks are drawn inside a debate with >= 20 lines of margin both sides
    files = sorted(debates)
    rng.shuffle(files)
    lines, blocks, meta = [], [], []
    for f in files:
        d = debates[f]
        if len(blocks) == N_BLOCKS:
            break
        if len(d) < BLOCK + 2 * max(CONTEXTS):
            continue
        start = rng.randrange(max(CONTEXTS), len(d) - BLOCK - max(CONTEXTS) + 1)
        base = len(lines)
        for r in d:
            lines.append((float(r["line"]), float(r["line"]) + 1, r["speaker"], r["text"], None))
            meta.append(r)
        n = SMOKE_LINES if SMOKE else BLOCK
        blocks.append((base + start, base + start + n))
    n_eval = sum(hi - lo for lo, hi in blocks)
    exp.dataset("ClaimBuster (groundtruth.csv + crowdsourced.csv)", "Zenodo 3609356 (2020-01-15)",
                "contiguous debate blocks", n_eval, "CC BY 4.0", "https://zenodo.org/records/3609356",
                f"label from groundtruth.csv where present else crowdsourced.csv; adjacent-sentence "
                f"line-number contiguity {contiguity(debates):.3f}; debates {len(debates)}")

    import os
    os.environ["GROQ_API_KEY"] = key
    from groq import AsyncGroq
    usage = Usage()
    client = AsyncGroq(max_retries=0, api_key=key)  # 429s handled by the pacer
    done = exp.done()
    spent = [sum(cost(r["tokens_in"], r["tokens_out"]) for r in done.values())]
    wall = {}

    async def run_all():  # one event loop: the httpx client is bound to it
        for ctx in CONTEXTS:
            t = time.perf_counter()
            await run_setting(exp, client, usage, lines, blocks, ctx, exp.done(), spent)
            wall[ctx] = time.perf_counter() - t
            log(f"E6 context={ctx}: spent ${spent[0]:.3f} (assumed prices)")
            if spent[0] >= LLM_SPEND_CAP_USD:
                blocker("E6", f"LLM spend cap ${LLM_SPEND_CAP_USD} reached at context={ctx}; "
                              "remaining settings not run. Ask the user before raising "
                              "LLM_SPEND_CAP_USD.")
                return
    asyncio.run(run_all())

    recs = list(exp.done().values())
    idx = [i for lo, hi in blocks for i in range(lo, hi)]
    out, errs = {}, []
    for ctx in CONTEXTS:
        rs = [r for r in recs if r["context"] == ctx and r["block"] < len(blocks)]  # ignore older, larger runs
        covered = {i for r in rs for i in range(r["lo"], r["hi"])}
        if not covered:
            continue
        pos = {c["line_id"] for r in rs for c in r["claims"]}
        units = [(meta[i]["gold"], i in pos, meta[i]["gt"]) for i in idx if i in covered]

        def f1(u):
            return prf(["y" if a else "n" for a, _, _ in u], ["y" if b else "n" for _, b, _ in u],
                       ["y", "n"])["per_class"]["y"]["f1"]
        m = prf(["y" if a else "n" for a, _, _ in units], ["y" if b else "n" for _, b, _ in units],
                ["y", "n"])["per_class"]["y"]
        gt_units = [u for u in units if u[2]]
        tin, tout = sum(r["tokens_in"] for r in rs), sum(r["tokens_out"] for r in rs)
        out[str(ctx)] = {"precision": m["p"], "recall": m["r"], "f1": bootstrap(units, f1),
                         "n_lines": len(units), "n_positive": sum(u[0] for u in units),
                         "f1_groundtruth_only": bootstrap(gt_units, f1) if gt_units else None,
                         "tokens_in": tin, "tokens_out": tout,
                         "tokens_per_100_lines": 100 * (tin + tout) / len(units),
                         "usd_per_100_lines_assumed": 100 * cost(tin, tout) / len(units),
                         "wall_s": wall.get(ctx), "batch_seconds_sum": sum(r["seconds"] for r in rs),
                         "s_per_100_lines_api": 100 * sum(r["seconds"] for r in rs) / len(units)}
        if ctx == 2:
            by_line = {c["line_id"]: c["claim_text"] for r in rs for c in r["claims"]}
            for i in idx:
                if i in covered and meta[i]["gold"] != (i in pos):
                    errs.append({"text": meta[i]["text"], "speaker": meta[i]["speaker"],
                                 "debate": meta[i]["file"], "gold_checkworthy": meta[i]["gold"],
                                 "predicted": i in pos, "claim_text": by_line.get(i),
                                 "context_before": [lines[j][3] for j in range(max(0, i - 2), i)],
                                 "gt_label": meta[i]["gt"], "context_window": 2})
    errs.sort(key=lambda e: (not e["gt_label"], e["predicted"]))
    exp.finish({"by_context": out, "spent_usd_assumed": spent[0]}, errs)


if __name__ == "__main__":
    main()
