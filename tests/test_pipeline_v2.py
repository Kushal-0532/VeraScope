"""Phase 34: v2 pipeline wiring, data contract, golden replay, artifact loader."""
import asyncio
import hashlib
import io
import json
import sys
import wave

import pytest
import tavily
from streamlit.testing.v1 import AppTest

import evidence_cache as ec
import verify_claims
import verify_pipeline as vp
from conftest import ROOT

GOLDEN = ROOT / "tests" / "golden" / "audio2_v2.json"
FIXTURES = ROOT / "tests" / "fixtures" / "cache_audio2"


def project(verdicts_by_line):
    """Stable projection compared against the golden file."""
    out = {}
    for records in verdicts_by_line.values():
        for r in records:
            out[r["claim_id"]] = {
                "label": r["label"], "warnings": sorted(w["key"] for w in r["warnings"]),
                "counts": r["counts"],
                "sources": sorted([s["domain"], "failed" if s.get("failed") else s.get("stance"),
                                   s["tier"]] for s in r["sources"]),
                "share": None if r["share"] is None else round(r["share"], 3)}
    return dict(sorted(out.items()))


def claim(text, line, queries=None):
    return {"claim_text": text, "line_id": line, "speaker": "A", "start": float(line),
            "queries": queries or [text, text + " fact check"]}


class FakeTavily:
    calls = []
    responder = staticmethod(lambda q: [{"url": "https://blog-a.example/x", "score": 0.9,
                                         "content": "Paris is in France.", "title": "t"}])

    def __init__(self, **kw):
        pass

    async def search(self, **kw):
        type(self).calls.append(kw)
        return {"results": type(self).responder(kw["query"])}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    F = type("F", (FakeTavily,), {"calls": []})
    monkeypatch.setattr(tavily, "AsyncTavilyClient", F)
    monkeypatch.setattr(verify_claims, "score_pairs", lambda c, chunks: [0.95] * len(chunks))
    monkeypatch.setenv("VERASCOPE_CACHE_DIR", str(tmp_path / "cache"))
    return F


def run(claims, n_lines=10):
    return asyncio.run(vp.verify_claims_list_v2(claims, n_lines))


def test_golden_replay(monkeypatch):
    g = json.loads(GOLDEN.read_text())
    monkeypatch.setenv("VERASCOPE_CACHE", "replay")
    monkeypatch.setenv("VERASCOPE_CACHE_DIR", str(FIXTURES))
    v, stats = asyncio.run(vp.verify_lines_v2([tuple(l) for l in g["lines"]]))
    assert project(v) == g["projection"]
    assert stats["tavily_credits"] == 0 and stats["cache"]["misses"] == 0
    assert stats["cache"]["hits"] > 0


def test_total_failure_is_couldnt_check(fake):
    def boom(q):
        raise RuntimeError("tavily down")
    fake.responder = staticmethod(boom)
    v, stats = run([claim("Paris is in France.", 1), claim("The sky is green.", 2)])
    assert {r["label"] for rs in v.values() for r in rs} == {"couldnt_check"}
    assert stats["label_counts"]["couldnt_check"] == 2
    assert not any(n for k, n in stats["label_counts"].items() if "disputed" in k)


def test_nli_failure_is_couldnt_check(fake, monkeypatch):
    def boom(c, chunks):
        raise RuntimeError("nli down")
    monkeypatch.setattr(verify_claims, "score_pairs", boom)
    v, stats = run([claim("Paris is in France.", 1)])
    r = v[1][0]
    assert r["label"] == "couldnt_check" and r["counts"]["failed"] == 1
    assert r["counts"]["disagree"] == 0


def test_empty_claims():
    seen = []
    v, stats = asyncio.run(vp.verify_claims_list_v2([], 7, on_stage=seen.append))
    assert v == {} and stats["n_unique_claims"] == 0 and stats["tavily_credits"] == 0
    assert set(stats["elapsed"]) == {"retrieve", "passages", "tier2", "verdict"}
    assert sum(stats["label_counts"].values()) == 0
    assert seen == vp.STAGES_V2


def test_fan_out_same_record(fake):
    v, stats = run([claim("Paris is in France.", 3), claim("paris is in france", 9)])
    assert v[3][0] is v[9][0]
    r = v[3][0]
    assert r["schema_version"] == 2 and r["summary"] is None
    assert [o["line_id"] for o in r["occurrences"]] == [3, 9]
    assert r["label"] == "mostly_supported"  # single Unrated agree -> single_source cap
    s = r["sources"][0]
    for k in ("source_id", "url", "domain", "chunks", "best_chunk", "nli_score", "tier1",
              "stance", "decided_by", "tier", "points", "trust_reason", "found_by"):
        assert k in s, k


def test_stats_arithmetic_and_credits(fake, monkeypatch):
    monkeypatch.setenv("VERASCOPE_CACHE", "record")
    claims = [claim("Paris is in France.", 1), claim("The sky is blue.", 2),
              claim("Water is wet.", 4)]
    v, stats = run(claims)
    assert sum(stats["label_counts"].values()) == stats["n_unique_claims"] == 3
    assert stats["tavily_credits"] == len(fake.calls) == 6
    assert stats["nli_pairs"] == 3
    assert stats["groq_calls"] == {"detect": 1, "tier2": 0, "summary": 0}
    monkeypatch.setenv("VERASCOPE_CACHE", "replay")
    v2, stats2 = run(claims)
    assert stats2["tavily_credits"] == 0 and len(fake.calls) == 6
    assert project(v2) == project(v)


def test_sources_sorted(fake, monkeypatch):
    fake.responder = staticmethod(lambda q: [
        {"url": f"https://{d}/x", "score": 0.9, "content": d, "title": d}
        for d in ("one.example", "wikipedia.org", "two.example")])
    monkeypatch.setattr(verify_claims, "score_pairs",
                        lambda c, chunks: [0.02 if "two" in x else 0.97 for x in chunks])
    v, _ = run([claim("Paris is in France.", 1)])
    st = [(s["stance"], s["points"]) for s in v[1][0]["sources"]]
    assert st == [("agree", 3), ("agree", 1), ("disagree", 1)], st


# ---- artifact loader (AppTest) ----

def _wav():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(b"\0\0" * 8000)
    return buf.getvalue()


def _app(monkeypatch, tmp_path, artifact):
    wav = _wav()
    if artifact.get("digest") == "MATCH":
        artifact["digest"] = hashlib.sha256(wav).hexdigest()
    p = tmp_path / "art.json"
    p.write_text(json.dumps(artifact))
    monkeypatch.setattr(sys, "argv", ["app.py"])
    for k in ["VERASCOPE_STAGE3", "VERASCOPE_LOCAL_NLI", "VERASCOPE_NO_DIARIZE",
              "WHISPERX_ARCH", "WHISPERX_DEVICE", "VERASCOPE_NLI_DEVICE"]:
        monkeypatch.delenv(k, raising=False)   # flags.py sets these directly in other tests
    monkeypatch.setenv("VERASCOPE_ARTIFACT", str(p))
    monkeypatch.setenv("GROQ_API_KEY", "dummy")   # passes the key gate; network stays blocked
    monkeypatch.setenv("PYANNOTEAI_API_KEY", "dummy")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.file_uploader[0].set_value(("a.wav", wav, "audio/wav")).run()
    assert not at.exception, at.exception
    return at


def test_artifact_v2_roundtrip(fake, monkeypatch, tmp_path):
    claims = [claim("Paris is in France.", 0)]
    v, stats = run(claims, n_lines=1)
    art = json.loads(json.dumps({  # through JSON like make_artifact.py
        "schema_version": 2, "source": "a.wav", "digest": "MATCH",
        "lines": [[0.0, 1.0, "A", "Paris is in France.", 0.9]], "failures": [],
        "claims": claims, "n_lines": 1, "verdicts": v, "stats": stats}))
    at = _app(monkeypatch, tmp_path, art)
    md = " ".join(m.value for m in at.markdown)
    assert "Mostly supported" in md, md
    assert any("1 unique claim(s) checked" in c.value for c in at.caption)


def test_artifact_v1_message(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path, {"source": "a.wav", "digest": "MATCH", "lines": [],
                                      "claims": [], "n_lines": 0, "verdicts": {}})
    assert any("Old artifact format" in i.value for i in at.info), [i.value for i in at.info]
