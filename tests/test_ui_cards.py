"""Phase 35: Claims-tab cards render every v2 label/warning state (AppTest, no pipeline)."""
import json

import pytest

from conftest import ROOT
from test_pipeline_v2 import _app

STATES = json.loads((ROOT / "tests/fixtures/synthetic_ui_states.json").read_text())
LABEL_KEYS = ["strongly_supported", "mostly_supported", "mixed", "mostly_disputed",
              "strongly_disputed", "not_enough_evidence", "no_sources", "couldnt_check"]


def _artifact(verdicts=None, stats=True):
    verdicts = STATES["verdicts"] if verdicts is None else verdicts
    counts = dict.fromkeys(LABEL_KEYS, 0)
    for recs in verdicts.values():
        counts[recs[0]["label"]] += 1
    return {"schema_version": 2, "source": "a.wav", "digest": "MATCH",
            "lines": STATES["lines"], "failures": [], "claims": STATES["claims"],
            "n_lines": len(STATES["lines"]), "verdicts": verdicts,
            "stats": {"n_unique_claims": len(verdicts), "label_counts": counts,
                      "tavily_credits": 0, "groq_calls": {"detect": 1, "tier2": 0, "summary": 0},
                      "tier2_skipped": 0, "quote_failures": 0,
                      "cache": {"hits": 0, "misses": 0, "writes": 0}} if stats else None}


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    at = _app(mp, tmp_path_factory.mktemp("ui"), _artifact())
    yield at, [m.value for m in at.markdown], [c.value for c in at.caption]
    mp.undo()


def _record(state):
    return next(r[0] for r in STATES["verdicts"].values() if r[0]["state"] == state)


def _card(md, state):
    """Index of the card header markdown for a fixture state."""
    text = _record(state)["claim_text"]
    return next(i for i, m in enumerate(md) if f'"{text}" — ' in m)


def test_all_states_present():
    labels = {r[0]["label"] for r in STATES["verdicts"].values()}
    warns = {w["key"] for r in STATES["verdicts"].values() for w in r[0]["warnings"]}
    assert labels == set(LABEL_KEYS)
    assert warns == {"trusted_minority", "single_source", "low_trust_only", "partial_failures"}
    assert len(STATES["verdicts"]) == 12


@pytest.mark.parametrize("state", [r[0]["state"] for r in STATES["verdicts"].values()])
def test_label_header(page, state):
    _, md, _ = page
    rec, head = _record(state), md[_card(md, state)]
    assert f"{rec['emoji']} **{rec['label_text']}**" in head, head
    if rec["share"] is None:
        assert "%" not in head and "points" not in head, head
    else:
        assert f"{rec['share']:.0%} · {rec['agree_points']} vs {rec['disagree_points']} points" in head


def test_pittsburgh_share_string(page):
    _, md, _ = page
    assert "✅ **Strongly supported** (89% · 8 vs 1 points)" in md[_card(md, "pittsburgh")]


def test_warnings(page):
    _, _, caps = page
    for text in ["⚠️ A highly trusted source disagrees",
                 "⚠️ Only one source took a side", "⚠️ Only low-trust sources took a side",
                 "⚠️ 2 of 5 couldn't be read"]:
        assert text in caps, (text, caps)
    assert any("says nothing about whether the claim is true" in c for c in caps)


def test_berlin_grouping(page):
    _, md, _ = page
    i = _card(md, "berlin")
    assert md[i + 1] == "**✅ Agree (3)**"
    j = md.index("**❌ Disagree (1)**", i)
    agree = [m for m in md[i + 2:j] if "checked quote" not in m]
    assert len(agree) == 3 and all(("· Low" in m) or ("· Unrated" in m) for m in agree), agree
    assert "berlin4.example" in md[j + 1] and "· High" in md[j + 1], md[j + 1]


def test_collapsed_groups(page):
    at, _, _ = page
    labels = [e.label for e in at.expander]
    for want in ["➖ No side (3)", "🚫 Off-topic (1)", "⚠️ Couldn't read (2)",
                 "What do the labels mean?", "Run stats"]:
        assert want in labels, (want, labels)


def test_llm_quote_only(page):
    _, md, _ = page
    llm = [s for r in STATES["verdicts"].values() for s in r[0]["sources"]
           if s.get("decided_by") == "llm"]
    shown = [m for m in md if m.endswith("— checked quote")]
    assert len(shown) == len(llm) > 0
    assert not any("nli-only quote" in m for m in md)


def test_transcript_tag(page):
    _, md, _ = page
    assert any(m.endswith("· ❌ Strongly disputed") for m in md), md


def test_not_checked(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path, _artifact(verdicts={}, stats=False))
    md = [m.value for m in at.markdown]
    heads = [m for m in md if '" — ' in m]
    assert len(heads) == 12 and all(h.endswith("⏳ **Not checked yet**") for h in heads), heads


def test_rank_disputed_first():
    import ast
    src = (ROOT / "app.py").read_text()
    rank = ast.literal_eval(src.split("_RANK = ", 1)[1].split("\n\n", 1)[0])
    order = ["couldnt_check", "no_sources", "strongly_supported", "mixed", "strongly_disputed"]
    assert [rank[k] for k in order] == sorted(rank[k] for k in order)
    assert rank["not_enough_evidence"] == rank["no_sources"] < rank["mostly_supported"]


def test_play_from_here_seeks_and_autoplays(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path, _artifact())
    assert not at.get("audio")[0].proto.autoplay
    rec = _record("berlin")
    at.button(key=f"claimseek{rec['claim_text'].lower().rstrip('.')}").click().run()
    audio = at.get("audio")[0].proto
    assert audio.start_time == int(rec["occurrences"][0]["start"]) and audio.autoplay


def test_same_seek_twice_is_a_new_player(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path, _artifact())
    at.button(key="seek2").click().run()            # transcript line 2 (20 s)
    first = at.get("audio")[0].proto
    rec = _record("berlin")                          # its claim card, same 20 s
    at.button(key=f"claimseek{rec['claim_text'].lower().rstrip('.')}").click().run()
    second = at.get("audio")[0].proto
    assert first.start_time == second.start_time == 20
    assert first.end_time != second.end_time and second.autoplay
