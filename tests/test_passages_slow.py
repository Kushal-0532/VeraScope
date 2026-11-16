"""Needs the cached mDeBERTa model. Run by hand:
VERASCOPE_LOCAL_NLI=1 ~/.venvs/ai/bin/python -m pytest -m slow tests/test_passages_slow.py -q
"""
import os

import pytest

import passages

pytestmark = pytest.mark.slow

CLAIM = "The Eiffel Tower is in Berlin."
WIKI = ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, "
        "France. It is named after the engineer Gustave Eiffel, whose company designed "
        "and built the tower from 1887 to 1889.")
INSTA = "Berlin has its own Eiffel Tower! Spotted this replica on our trip."


def test_berlin(monkeypatch):
    # conftest forces HF offline: the model must already be in the HF cache
    monkeypatch.setenv("HOME", os.path.expanduser("~" + os.environ.get("USER", "")))
    out = passages.select(CLAIM, [{"source_id": 1, "chunks": [WIKI]},
                                  {"source_id": 2, "chunks": [INSTA]}])
    assert out[0]["tier1"] == "disagree", out[0]
    assert out[1]["tier1"] == "agree", out[1]  # known tier-1 failure; Phase 32 fixes
