import asyncio
import json
from types import SimpleNamespace as NS

import groq
import pytest

import claim_detect as cd


def fake_groq(monkeypatch, make_claims):
    """Patch groq.AsyncGroq; make_claims(lo, hi) -> list of claim dicts for one batch."""
    calls = []

    class Fake:
        def __init__(self, **kw):
            self.chat = NS(completions=NS(create=self.create))

        async def create(self, **kw):
            rows = [l for l in kw["messages"][1]["content"].splitlines() if "[context]" not in l]
            ids = [int(r[1:r.index("]")]) for r in rows]
            calls.append(kw)
            content = {"claims": make_claims(min(ids), max(ids) + 1)}
            return NS(choices=[NS(message=NS(content=json.dumps(content)))],
                      usage=NS(model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 5}))

    monkeypatch.setattr(groq, "AsyncGroq", Fake)
    return calls


LINES = [(float(i), i + 1.0, "S", f"line {i}", 0.9) for i in range(45)]


def claim(i, qn="paris capital france", qf="paris capital france fact check", text=None):
    return {"line_id": i, "claim_text": text or f"claim {i}",
            "query_neutral": qn, "query_fact_check": qf}


def run(monkeypatch, make_claims, lines=LINES):
    calls = fake_groq(monkeypatch, make_claims)
    return asyncio.run(cd.detect_claims(lines)), calls


def test_happy_path(monkeypatch):
    out, calls = run(monkeypatch, lambda lo, hi: [claim(lo), claim(lo + 1)] if lo == 0 else [])
    assert len(out) == 2
    for c in out:
        assert set(c) == {"claim_text", "line_id", "speaker", "start", "queries"}
        assert c["queries"] == ["paris capital france", "paris capital france fact check"]


def test_call_count_unchanged(monkeypatch):
    _, calls = run(monkeypatch, lambda lo, hi: [])
    assert len(calls) == 3  # ceil(45 / 20)


def test_context_line_rejected(monkeypatch):
    # batch 20:40 renders lines 18,19 as context; a claim on 19 must be dropped there
    out, _ = run(monkeypatch, lambda lo, hi: [claim(19)] if lo == 20 else [])
    assert out == []


@pytest.mark.parametrize("qn,qf,want", [
    ("a b", "", ["a b", "the claim"]),
    ("x" * 500, "a fact check", ["the claim", "a fact check"]),
    ("Same Query", "same  query", ["Same Query"]),
    ("a\nb\t c   d", "a b fact check", ["a b c d", "a b fact check"]),
    (None, None, ["the claim"]),
])
def test_clean_queries(qn, qf, want):
    assert cd._clean_queries("the  claim", qn, qf) == want


def test_long_claim_text_truncated():
    assert cd._clean_queries("y" * 900, "", "z") == ["y" * 400, "z"]


def walk(node):
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert node["required"] == list(node["properties"]), node
            assert node["additionalProperties"] is False, node
        for v in node.values():
            walk(v)
    elif isinstance(node, list):
        for v in node:
            walk(v)


def test_schema_strict():
    walk(cd._SCHEMA["json_schema"]["schema"])
    assert cd._SCHEMA["json_schema"]["strict"] is True
