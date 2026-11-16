"""Trust points: URL -> High (3) / Medium (2) / Low or Unrated (1) + a reason.

Precedence (Prior Decision 8), first match wins, host then each parent:
LOW_LIST > HIGH_LIST > SUFFIX_RULES > Lin et al. 2023 pc1 rating > Unrated.
Ratings: ratings/domain_pc1.csv (see ratings/README.md).
"""

import csv
import functools
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

PC1_HIGH = 0.70    # bbc.co.uk is 0.718; 0.75 would push it to Medium (Phase 29 Notes)
PC1_MEDIUM = 0.50
RATINGS = Path(__file__).resolve().parent / "ratings" / "domain_pc1.csv"

# ponytail: tiny suffix list; full Public Suffix List via tldextract if wrong domains show up.
TWO_LEVEL_SUFFIXES = set("co.uk ac.uk gov.uk org.uk com.au gov.au edu.au co.in gov.in "
                         "ac.in nic.in co.nz co.jp co.za com.br com.sg".split())

LOW_LIST = set("""facebook.com instagram.com tiktok.com x.com twitter.com reddit.com
youtube.com quora.com pinterest.com linkedin.com medium.com substack.com blogspot.com
wordpress.com tumblr.com threads.net""".split())

HIGH_LIST = """wikipedia.org britannica.com who.int nature.com science.org sciencedirect.com
thelancet.com nejm.org bmj.com cell.com springer.com wiley.com frontiersin.org plos.org
arxiv.org nih.gov cochranelibrary.com nhs.uk un.org worldbank.org imf.org oecd.org
reuters.com apnews.com afp.com snopes.com politifact.com factcheck.org fullfact.org
altnews.in boomlive.in""".split()
_HIGH_SET = set(HIGH_LIST)

SUFFIX_RULES = [  # (regex on host, reason)
    (re.compile(r"(^|\.)gov(\.[a-z]{2})?$"), "rule: .gov"),
    (re.compile(r"(^|\.)edu(\.[a-z]{2})?$"), "rule: .edu"),
    (re.compile(r"(^|\.)ac\.[a-z]{2}$"), "rule: .ac"),
    (re.compile(r"(^|\.)int$"), "rule: .int"),
    (re.compile(r"(^|\.)mil$"), "rule: .mil"),
    (re.compile(r"(^|\.)nic\.in$"), "rule: .nic.in"),
]

# Tavily include_domains accepts plain hostnames only; the CSV also has paths
# ("rfi.fr/en") and one en-dash typo ("wvec–13newsnow.com") that 400 the request.
_HOSTNAME = re.compile(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+")

TIERS = {"High": 3, "Medium": 2, "Low": 1, "Unrated": 1}


def _host(url):
    try:
        host = (urlsplit(url if "//" in url else "//" + url).hostname or "").lower().strip(".")
    except ValueError:
        return ""
    for p in ("www.", "m.", "amp."):
        if host.startswith(p):
            host = host[len(p):]
    return host if "." in host or host == "localhost" else ""


def registered_domain(url):
    host = _host(url)
    labels = host.split(".")
    n = 3 if ".".join(labels[-2:]) in TWO_LEVEL_SUFFIXES else 2
    return ".".join(labels[-n:])


def _candidates(host):
    """full host, then each parent down to the registered domain."""
    reg = registered_domain(host)
    labels = host.split(".")
    out = []
    for i in range(len(labels)):
        h = ".".join(labels[i:])
        out.append(h)
        if h == reg:
            break
    return out


@functools.lru_cache(maxsize=1)
def _ratings():
    with open(RATINGS, newline="") as f:
        return {r["domain"].lower(): float(r["pc1"]) for r in csv.DictReader(f)}


def _out(tier, reason):
    return {"tier": tier, "points": TIERS[tier], "trust_reason": reason}


def trust(url):
    host = _host(url or "")
    if not host:
        return _out("Unrated", "unrated")
    cands = _candidates(host)
    if any(h in LOW_LIST for h in cands):
        return _out("Low", "list: social/UGC")
    if any(h in _HIGH_SET for h in cands):
        return _out("High", "list: reference/science")
    for rx, reason in SUFFIX_RULES:
        if rx.search(host):
            return _out("High", reason)
    ratings = _ratings()
    for h in cands:
        if h in ratings:
            pc1 = ratings[h]
            tier = "High" if pc1 >= PC1_HIGH else "Medium" if pc1 >= PC1_MEDIUM else "Low"
            return _out(tier, f"rating: {pc1:.2f} (Lin et al. 2023)")
    return _out("Unrated", "unrated")


def high_domains(limit=300):
    """HIGH_LIST + top-rated domains, deterministic, unique, <= limit (Tavily max 300)."""
    out = list(dict.fromkeys(HIGH_LIST))
    seen = set(out)
    for d, pc1 in sorted(_ratings().items(), key=lambda x: (-x[1], x[0])):
        if len(out) >= limit or pc1 < PC1_HIGH:
            break
        if d not in seen and d not in LOW_LIST and _HOSTNAME.fullmatch(d):
            out.append(d)
            seen.add(d)
    return out[:limit]


def _selfcheck():
    assert trust("https://www.reuters.com/x")["tier"] == "High"
    assert trust("https://www.bbc.co.uk/news")["tier"] == "High"
    assert trust("https://www.nytimes.com/a")["points"] >= 2
    assert trust("https://www.thehindu.com/a")["points"] >= 2
    assert trust("https://www.infowars.com/x")["tier"] == "Low"
    assert trust("https://www.naturalnews.com/x")["tier"] == "Low"
    assert trust("https://m.facebook.com/x")["trust_reason"] == "list: social/UGC"
    assert trust("not a url")["tier"] == "Unrated"
    hd = high_domains()
    assert len(hd) <= 300 and len(set(hd)) == len(hd) and set(HIGH_LIST) <= set(hd)
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        for u in sys.argv[1:]:
            print(u, trust(u))
