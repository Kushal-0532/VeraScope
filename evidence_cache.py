"""Record / replay cache for every external call (Groq, Tavily, NLI).

Modes via VERASCOPE_CACHE: off (default) -> just call; record -> hit returns
cached value, miss calls and saves; replay -> hit returns, miss raises CacheMiss
and never calls out. VERASCOPE_CACHE_REFRESH=1 in record mode overwrites hits.
One JSON file per key: <VERASCOPE_CACHE_DIR or ~/.cache/verascope>/<kind>/<key[:2]>/<key>.json
"""

import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


class CacheMiss(RuntimeError):
    def __init__(self, kind, key, parts):
        self.kind, self.key = kind, key
        summary = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)[:200]
        super().__init__(f"not in cache: {kind}/{key[:12]} {summary} — run once with --record")


_stats = defaultdict(lambda: {"hits": 0, "misses": 0, "writes": 0})


def stats():
    return {k: dict(v) for k, v in _stats.items()}


def reset_stats():
    _stats.clear()


def mode():
    m = os.environ.get("VERASCOPE_CACHE", "off")
    if m not in ("off", "record", "replay"):
        raise ValueError(f"VERASCOPE_CACHE must be off|record|replay, got {m!r}")
    return m


def cache_dir():
    return Path(os.environ.get("VERASCOPE_CACHE_DIR") or Path.home() / ".cache" / "verascope")


def key(kind, **parts):
    blob = json.dumps({"kind": kind, **parts}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _path(kind, k):
    return cache_dir() / kind / k[:2] / f"{k}.json"


def _lookup(kind, parts):
    """-> (key, path, hit, value). Raises CacheMiss in replay."""
    m = mode()
    k = key(kind, **parts)
    p = _path(kind, k)
    if m == "off":
        return k, p, False, None
    refresh = m == "record" and os.environ.get("VERASCOPE_CACHE_REFRESH") == "1"
    if not refresh and p.exists():
        _stats[kind]["hits"] += 1
        return k, p, True, json.loads(p.read_text())["value"]
    _stats[kind]["misses"] += 1
    if m == "replay":
        raise CacheMiss(kind, k, parts)
    return k, p, False, None


def _save(kind, parts, p, value):
    if mode() != "record":
        return
    # serialise first: a TypeError leaves no file behind
    blob = json.dumps({"kind": kind, "parts": parts, "value": value,
                       "saved_utc": datetime.now(timezone.utc).isoformat()},
                      ensure_ascii=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(blob)
        os.replace(tmp, p)  # atomic: concurrent writers never tear a file
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _stats[kind]["writes"] += 1


def get_or_call(kind, parts, fn):
    _, p, hit, value = _lookup(kind, parts)
    if hit:
        return value
    value = fn()  # raising -> nothing written
    _save(kind, parts, p, value)
    return value


async def aget_or_call(kind, parts, afn):
    _, p, hit, value = _lookup(kind, parts)
    if hit:
        return value
    value = await afn()
    _save(kind, parts, p, value)
    return value


def _selfcheck():
    with tempfile.TemporaryDirectory() as d:
        os.environ["VERASCOPE_CACHE_DIR"] = d
        calls = []
        fn = lambda: calls.append(1) or {"v": len(calls)}
        os.environ["VERASCOPE_CACHE"] = "record"
        assert get_or_call("t", {"x": 1}, fn) == {"v": 1}
        assert get_or_call("t", {"x": 1}, fn) == {"v": 1} and len(calls) == 1
        os.environ["VERASCOPE_CACHE"] = "replay"
        assert get_or_call("t", {"x": 1}, fn) == {"v": 1} and len(calls) == 1
        try:
            get_or_call("t", {"x": 2}, fn)
            raise AssertionError("expected CacheMiss")
        except CacheMiss:
            pass
        assert len(calls) == 1
        assert key("t", a=1, b=2) == key("t", b=2, a=1) != key("t", a=1, b=3)
        os.environ["VERASCOPE_CACHE"] = "off"
        get_or_call("t", {"x": 9}, fn)
        assert not (Path(d) / "t").joinpath(key("t", x=9)[:2]).exists()
    for k in ("VERASCOPE_CACHE_DIR", "VERASCOPE_CACHE"):
        os.environ.pop(k)
    reset_stats()
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
