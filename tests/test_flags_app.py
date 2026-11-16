"""Run-mode flags reach the UI (Phase 25's 3-case AppTest)."""
import importlib.util
import os
import sys

import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT

APP = str(ROOT / "app.py")


def run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["app.py"] + argv)
    for k in ["VERASCOPE_STAGE3", "VERASCOPE_LOCAL_NLI", "VERASCOPE_NO_DIARIZE",
              "WHISPERX_ARCH", "WHISPERX_DEVICE", "VERASCOPE_NLI_DEVICE"]:
        monkeypatch.delenv(k, raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def test_nli_offline_no_diarize(monkeypatch):
    at = run(monkeypatch, ["--nli-offline", "--no-diarize"])
    caps = [c.value for c in at.sidebar.caption]
    assert any("local NLI" in c and "diarization off" in c for c in caps), caps


@pytest.mark.skipif(importlib.util.find_spec("whisperx") is not None,
                    reason="guard only shows when whisperx is missing")
def test_whisperx_missing_guard(monkeypatch):
    at = run(monkeypatch, ["--whisperx-offline", "--whisperx-model", "tiny"])
    errs = [e.value for e in at.error]
    assert any("needs `whisperx`" in e for e in errs), errs


def test_default_mode(monkeypatch):
    at = run(monkeypatch, [])
    assert not [c for c in at.sidebar.caption if "Test modes" in c.value]
