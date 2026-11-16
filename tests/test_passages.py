import pytest

import passages
import verify_claims as vc


class Scorer:
    def __init__(self, scores=None, exc=None):
        self.calls, self.scores, self.exc = [], scores, exc

    def __call__(self, claim, chunks):
        self.calls.append(list(chunks))
        if self.exc:
            raise self.exc
        return self.scores if self.scores is not None else [0.5] * len(chunks)


def src(i, chunks):
    return {"source_id": i, "url": f"https://s{i}.org", "chunks": chunks}


def test_picker():
    out = passages.select("c", [src(1, ["a", "b", "c"])], Scorer([0.71, 0.97, 0.40]))
    assert out[0]["best_chunk"] == "b" and out[0]["nli_score"] == 0.97 and out[0]["tier1"] == "agree"


def test_tie_prefers_higher():
    out = passages.select("c", [src(1, ["lo", "hi"])], Scorer([0.1, 0.9]))
    assert out[0]["best_chunk"] == "hi"


@pytest.mark.parametrize("s,label", [(0.85, "agree"), (0.8499, "unsure"),
                                     (0.15, "disagree"), (0.1501, "unsure")])
def test_edges(s, label):
    assert passages.tier1(s) == label


def test_one_call_per_claim():
    sc = Scorer()
    out = passages.select("c", [src(i, ["x", "y", "z"]) for i in (1, 2, 3)], sc)
    assert len(sc.calls) == 1 and len(sc.calls[0]) == 9
    assert [s["source_id"] for s in out] == [1, 2, 3]


def test_no_text_and_errors():
    sc = Scorer()
    out = passages.select("c", [src(1, []), src(2, ["", "  "]), src(3, ["t"])], sc)
    assert out[0]["failed"] == out[1]["failed"] == "no text" and out[2]["tier1"] == "unsure"
    out = passages.select("c", [src(1, ["a"]), src(2, ["b"])], Scorer(exc=RuntimeError("x")))
    assert all(s["failed"] == "nli error: RuntimeError" for s in out)
    assert not any(s.get("tier1") == "disagree" for s in out)


def test_input_not_mutated():
    s = [src(1, ["a"])]
    passages.select("c", s, Scorer([0.9]))
    assert "tier1" not in s[0]


def test_cache_second_select_not_scored(monkeypatch, tmp_path):
    monkeypatch.setenv("VERASCOPE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VERASCOPE_CACHE", "record")
    calls = []
    monkeypatch.setattr(vc, "_score_batch_raw",
                        lambda claim, sn, local: calls.append(list(sn)) or [0.9] * len(sn))
    sources = [src(1, ["a", "b"]), src(2, ["c"])]
    a = passages.select("claim", sources)
    b = passages.select("claim", sources)
    assert a == b and calls == [["a", "b", "c"]]
    # partial hit: only the new chunk is scored, still one batch
    passages.select("claim", [src(1, ["a", "new1", "new2"])])
    assert calls[-1] == ["new1", "new2"]
