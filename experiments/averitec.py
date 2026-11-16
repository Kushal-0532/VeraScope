"""E10 data (Phase 37): AVeriTeC dev claims, gold label map, offline evidence pool,
leakage guard -> experiments/e10_items.jsonl.

Evidence pool = the AVeriTeC baseline's `dev_top_k_sentences.json` (BM25 top-100
sentences per claim, with URL), grouped one source per registered domain in rank
order, <= 3 sentences each, <= 10 sources. The 11.5 GB knowledge store is never used.
AVeriTeC is CC BY-NC 4.0 (non-commercial).

    python -m experiments.averitec              # download (~13 MB), write the items
    python -m experiments.averitec --selfcheck  # offline
"""

import datetime
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import credibility  # noqa: E402

REPO = "chenxwh/AVeriTeC"  # a model-type repo (repo_type="model" works)
DEV, POOL = "data/dev.json", "data_store/dev_top_k_sentences.json"
OUT = ROOT / "experiments" / "e10_items.jsonl"

LABEL_MAP = {"Supported": "supported", "Refuted": "disputed",
             "Conflicting Evidence/Cherrypicking": "mixed", "Not Enough Evidence": "nei"}
# Phase 38: VeraScope label key -> gold class.
PRED_MAP = {"strongly_supported": "supported", "mostly_supported": "supported",
            "strongly_disputed": "disputed", "mostly_disputed": "disputed", "mixed": "mixed",
            "not_enough_evidence": "nei", "no_sources": "nei", "couldnt_check": "nei"}
QUOTAS = {"supported": 40, "disputed": 50, "mixed": 30, "nei": 30}  # sums to 150
MAX_SOURCES, PER_SOURCE = 10, 3

_ARCHIVE = re.compile(r"^https?://web\.archive\.org/web/[^/]+/(.+)$")


def download():
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, DEV)), Path(hf_hub_download(REPO, POOL))


def unwrap(url):
    """web.archive.org/web/<ts>/<url> -> <url>; every dev fact-check URL is archived."""
    m = _ARCHIVE.match(url or "")
    return m.group(1) if m else (url or "")


def load_dev(dev_path, pool_path=None):
    """Every dev claim with id (= index = pool claim_id), gold label, and its ranked
    top-k sentences under "top_k" when pool_path is given. Unknown label -> ValueError."""
    claims = json.loads(Path(dev_path).read_text())
    unknown = sorted({c["label"] for c in claims} - LABEL_MAP.keys())
    if unknown:
        raise ValueError(f"unknown AVeriTeC label(s): {unknown}")
    pool = {}
    if pool_path:
        for line in Path(pool_path).read_text().split("\n"):  # not splitlines: sentences hold U+2028
            if line.strip():
                row = json.loads(line)
                pool[row["claim_id"]] = row["top_100"]
    return [dict(c, id=i, gold=LABEL_MAP[c["label"]], top_k=pool.get(i, []))
            for i, c in enumerate(claims)]


def sample(items, n=150, seed=42):
    """Stratified by gold with QUOTAS scaled to n; a short class gives all it has.
    Returns items in id order."""
    rng = random.Random(seed)
    picked = []
    for gold, q in QUOTAS.items():
        ids = sorted(it["id"] for it in items if it["gold"] == gold)
        picked += rng.sample(ids, min(round(q * n / 150), len(ids)))
    keep = set(picked)
    return [it for it in items if it["id"] in keep]


def parse_claim_date(s):
    """Observed format D-M-YYYY, day/month not zero-padded ('9-9-2020', '31-10-2020')."""
    try:
        return datetime.datetime.strptime(s.strip(), "%d-%m-%Y").date()
    except (AttributeError, TypeError, ValueError):
        return None


def _norm(url):
    return unwrap(url).rstrip("/")


def leak_filter(claim, sources):
    """Drop entries (anything with "url") that are the claim's own fact-check article,
    original claim URL or cached claim URL, or on the fact-check article's domain."""
    bad_urls = {_norm(claim.get(k)) for k in
                ("fact_checking_article", "original_claim_url", "cached_original_claim_url")
                if claim.get(k)}
    fca = claim.get("fact_checking_article")
    bad_domain = credibility.registered_domain(unwrap(fca)) if fca else None
    return [s for s in sources
            if _norm(s.get("url")) not in bad_urls
            and not (bad_domain and credibility.registered_domain(unwrap(s.get("url"))) == bad_domain)]


def evidence_pool(claim):
    """claim["top_k"] (ranked sentences) -> Phase 30-shaped sources: one per registered
    domain in first-appearance rank order, <= PER_SOURCE chunks, <= MAX_SOURCES.
    Leak-filtered per sentence before grouping, so a leaked domain never takes a slot."""
    groups = {}
    for s in leak_filter(claim, claim.get("top_k", [])):
        url = unwrap(s["url"])
        d = credibility.registered_domain(url) or url
        g = groups.setdefault(d, {"url": url, "domain": d, "title": "", "chunks": []})
        text = (s.get("sentence") or "").strip()
        if text and text not in g["chunks"] and len(g["chunks"]) < PER_SOURCE:
            g["chunks"].append(text)
    sources = [g for g in groups.values() if g["chunks"]][:MAX_SOURCES]
    for i, s in enumerate(sources, 1):
        s["source_id"] = i
    return sources


def build_item(claim):
    d = parse_claim_date(claim.get("claim_date"))
    return {"id": claim["id"], "claim": claim["claim"], "gold": claim["gold"],
            "claim_date": d.isoformat() if d else None, "sources": evidence_pool(claim)}


def main():
    dev_path, pool_path = download()
    claims = load_dev(dev_path, pool_path)
    print(f"dev claims: {len(claims)}  labels: {dict(Counter(c['label'] for c in claims))}")
    print(f"claim_date samples: {[c['claim_date'] for c in claims[:1] + claims[200:201] + claims[-1:]]}"
          f"  unparseable: {sum(parse_claim_date(c['claim_date']) is None for c in claims)}")
    items = [build_item(c) for c in sample(claims)]
    with OUT.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    leaked = sum(len(leak_filter(c, c["top_k"])) < len(c["top_k"]) for c in claims)
    n_src = sorted(len(it["sources"]) for it in items)
    print(f"wrote {len(items)} items -> {OUT.relative_to(ROOT)}  gold: {dict(Counter(it['gold'] for it in items))}")
    print(f"sources/item min {n_src[0]} median {n_src[len(n_src) // 2]} max {n_src[-1]}; "
          f"claims with leaked pool sentences removed: {leaked}/{len(claims)}")


def _selfcheck():
    assert unwrap("https://web.archive.org/web/20201103001419/https://a.com/x") == "https://a.com/x"
    assert unwrap("https://b.org/y") == "https://b.org/y" and unwrap(None) == ""
    assert parse_claim_date("9-9-2020") == datetime.date(2020, 9, 9)
    assert parse_claim_date("31-10-2020") == datetime.date(2020, 10, 31)
    assert parse_claim_date("None") is None and parse_claim_date(None) is None
    claim = {"fact_checking_article": "https://web.archive.org/web/1/https://www.fc.com/a",
             "original_claim_url": None, "id": 0, "claim": "c", "gold": "nei", "claim_date": "1-1-2020",
             "top_k": [{"sentence": "s1", "url": "https://fc.com/other"},
                       {"sentence": "s2", "url": "https://web.archive.org/web/2/https://news.org/p"},
                       {"sentence": "s3", "url": "https://news.org/q"}]}
    item = build_item(claim)
    assert [s["domain"] for s in item["sources"]] == ["news.org"], item
    assert item["sources"][0]["chunks"] == ["s2", "s3"] and item["claim_date"] == "2020-01-01"
    assert set(PRED_MAP.values()) == set(LABEL_MAP.values())
    print("averitec selfcheck OK")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
