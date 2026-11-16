import pytest

import verify_claims as vc

T = {"H": "High", "M": "Medium", "L": "Low", "U": "Unrated"}


def S(spec):
    """'agree:H' / 'no_side:M' / 'failed' -> source dict."""
    if spec == "failed":
        return {"failed": "nli error: X"}
    st, t = spec.split(":")
    return {"stance": st, "tier": T[t]}


CARDS = {
    1: (["agree:H", "agree:H", "agree:M", "no_side:M"], "strongly_supported", 1.0, 8, 0, []),
    2: (["agree:H", "agree:H", "agree:M", "disagree:H"], "mostly_supported", 8 / 11, 8, 3,
        [("trusted_minority", "A highly trusted source disagrees")]),
    3: (["agree:H", "agree:M", "disagree:M", "disagree:H"], "mixed", 0.5, 5, 5, []),
    4: (["agree:L", "agree:L", "disagree:M", "disagree:H", "disagree:M"], "mostly_disputed", 2 / 9, 2, 7, []),
    5: (["disagree:H", "disagree:H", "disagree:M"], "strongly_disputed", 0.0, 0, 8, []),
    6: (["agree:L", "agree:L", "agree:U", "disagree:H"], "mixed", 0.5, 3, 3,
        [("trusted_minority", "A highly trusted source disagrees")]),
    7: (["agree:L", "agree:L", "agree:U"], "mostly_supported", 1.0, 3, 0, [("low_trust_only", None)]),
    8: (["agree:M"], "mostly_supported", 1.0, 2, 0, [("single_source", None)]),
    9: (["agree:H", "agree:H", "no_side:M", "failed", "failed"], "strongly_supported", 1.0, 6, 0,
        [("partial_failures", "2 of 5 couldn't be read")]),
    10: (["no_side:H", "no_side:M", "no_side:L", "off_topic:H"], "not_enough_evidence", None, None, None, []),
    11: ([], "no_sources", None, None, None, []),
    13: (["agree:H"] + ["disagree:H"] * 4, "mostly_disputed", 0.2, 3, 12,
         [("trusted_minority", "A highly trusted source agrees")]),
}


@pytest.mark.parametrize("n", sorted(CARDS))
def test_cards(n):
    specs, label, share, a, d, warns = CARDS[n]
    v = vc.verdict([S(x) for x in specs])
    assert v["label"] == label, v
    assert v["emoji"], v["label_text"] == vc.LABELS[label]
    if share is None:
        assert v["share"] is None and v["agree_points"] is None
    else:
        assert v["share"] == pytest.approx(share) and (v["agree_points"], v["disagree_points"]) == (a, d)
    assert [w["key"] for w in v["warnings"]] == [k for k, _ in warns]
    for (k, text), w in zip(warns, v["warnings"]):
        if text:
            assert w["text"] == text


def test_couldnt_check():
    assert vc.verdict([], retrieval_failed=True)["label"] == "couldnt_check"
    assert vc.verdict([S("failed")] * 4)["label"] == "couldnt_check"


@pytest.mark.parametrize("share,label", [
    (0.80, "strongly_supported"), (0.7999, "mostly_supported"), (0.60, "mostly_supported"),
    (0.5999, "mixed"), (0.50, "mixed"), (0.4001, "mixed"), (0.40, "mostly_disputed"),
    (0.2001, "mostly_disputed"), (0.20, "strongly_disputed"),
])
def test_band_edges(share, label):
    assert vc._band(share) == label


def test_band_float_tolerance():
    assert vc._band(0.8 - 1e-12) == "strongly_supported"
    assert vc._band(0.4 + 1e-12) == "mostly_disputed"


def mirror(spec):
    swap = {"agree": "disagree", "disagree": "agree"}
    st, _, t = spec.partition(":")
    return f"{swap[st]}:{t}" if st in swap else spec


MIRROR = {"strongly_supported": "strongly_disputed", "mostly_supported": "mostly_disputed",
          "mixed": "mixed"}
MIRROR.update({v: k for k, v in MIRROR.items()})


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9, 13])
def test_mirror(n):
    specs = CARDS[n][0]
    a = vc.verdict([S(x) for x in specs])["label"]
    b = vc.verdict([S(mirror(x)) for x in specs])["label"]
    assert b == MIRROR[a], (a, b)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 13])
@pytest.mark.parametrize("extra", ["no_side:H", "off_topic:H", "failed"])
def test_non_voting_sources_dont_change_label(n, extra):
    specs = CARDS[n][0]
    a = vc.verdict([S(x) for x in specs])
    b = vc.verdict([S(x) for x in specs + [extra]])
    assert a["label"] == b["label"]
    if extra == "failed":
        assert b["warnings"][-1]["key"] == "partial_failures"


@pytest.mark.parametrize("spec", ["agree:H", "disagree:H", "agree:L", "disagree:M"])
def test_single_source_never_strong(spec):
    assert not vc.verdict([S(spec), S("no_side:H")])["label"].startswith("strongly")


def test_labels_complete():
    for k in ["strongly_supported", "mostly_supported", "mixed", "mostly_disputed",
              "strongly_disputed", "not_enough_evidence", "no_sources", "couldnt_check",
              "not_checked"]:
        emoji, text = vc.LABELS[k]
        assert emoji and text


def test_points_from_url_when_missing():
    v = vc.verdict([{"stance": "agree", "url": "https://www.reuters.com/x"},
                    {"stance": "disagree", "url": "https://www.infowars.com/x"}])
    assert (v["agree_points"], v["disagree_points"]) == (3, 1)
