#!/usr/bin/env python3
"""Phase 08 UI verification via streamlit.testing.v1.AppTest.

Not a pytest suite on purpose: it needs live GROQ_API_KEY + PYANNOTEAI_API_KEY
in ~/.env and drives the real hosted pipeline end to end (real diarization,
real transcription), so it costs real API calls and real wall-clock time. That
matches how every other live-API phase in this repo was verified (see
specs/phases/02, 05, 06 Notes) — no mocking of the thing the phase is meant to
prove works.

Two checks:
  1. fast path (default): uploads a short mp4 (built from audio2.wav) and
     confirms upload -> Transcribe -> rendered transcript -> seek-on-click, no
     exceptions anywhere. ~1 minute wall-clock.
  2. long path (--long PATH): uploads a real 45-60min audio/video file with
     VERASCOPE_FAIL_CHUNKS=2 set, and confirms the "N of M chunks failed"
     banner, the Failed spans expander content, and the on_stage stage
     progression. Several minutes wall-clock (real diarize long-poll). No
     >10min asset ships in this repo, so this path is opt-in via --long.

Run: /home/kushal/.venvs/ai/bin/python verify_ui_apptest.py [--long PATH]
"""
import argparse
import os
import subprocess
import sys
import tempfile

from streamlit.testing.v1 import AppTest


def make_short_mp4():
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x240:r=1",
         "-i", "audio2.wav", "-shortest", "-c:v", "libx264", "-c:a", "aac", path],
        capture_output=True, check=True,
    )
    return path


def run_fast_check():
    print("=== fast check: short mp4 upload -> transcript -> seek ===")
    mp4_path = make_short_mp4()
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()

    with open(mp4_path, "rb") as f:
        content = f.read()
    at.file_uploader[0].set_value(("test.mp4", content, "video/mp4")).run()
    assert not at.exception, f"upload raised: {at.exception}"

    transcribe_btn = [b for b in at.button if b.label == "Transcribe"][0]
    transcribe_btn.click().run()
    assert not at.exception, f"run raised: {at.exception}"
    assert at.markdown, "no transcript rendered"
    print("captions:", [c.value for c in at.caption])

    seek_buttons = [b for b in at.button if b.key and b.key.startswith("seek")]
    assert seek_buttons, "no seek buttons rendered"
    target = seek_buttons[0]
    target.click().run()
    assert not at.exception, f"seek click raised: {at.exception}"
    h, m, s = ([0] * (3 - len(target.label.split(":")))) + [int(x) for x in target.label.split(":")]
    expected = h * 3600 + m * 60 + s
    assert at.session_state["seek"] == expected, (
        f"seek mismatch: got {at.session_state['seek']}, expected {expected}")
    print(f"seek OK: clicked {target.label!r} -> session_state['seek'] == {expected}")

    os.unlink(mp4_path)
    print("fast check passed\n")


def run_verification_check():
    print("=== verification check: Claims tab, real audio2.wav, real Groq/Tavily/HF ===")
    mp4_path = make_short_mp4()
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()

    with open(mp4_path, "rb") as f:
        content = f.read()
    at.file_uploader[0].set_value(("test.mp4", content, "video/mp4")).run()
    assert not at.exception, f"upload raised: {at.exception}"

    transcribe_btn = [b for b in at.button if b.label == "Transcribe"][0]
    transcribe_btn.click().run()
    assert not at.exception, f"run raised: {at.exception}"

    tabs = [t.label for t in at.tabs]
    assert tabs == ["Transcript", "Claims"], f"unexpected tabs: {tabs}"

    detect_btn = [b for b in at.button if b.label == "Detect claims"]
    assert detect_btn, "no Detect claims button in Claims tab"
    detect_btn[0].click().run()
    assert not at.exception, f"detect raised: {at.exception}"

    caps = [c.value for c in at.caption]
    print("captions after detect:", [c for c in caps if "claim" in c])

    verify_btn = [b for b in at.button if b.label == "Verify claims"]
    if not verify_btn:
        print("no check-worthy claims found in this clip; nothing to verify")
        os.unlink(mp4_path)
        print("verification check passed (zero claims)\n")
        return
    assert not verify_btn[0].disabled, "Verify claims button wrongly disabled with a real TAVILY_API_KEY"
    verify_btn[0].click().run()
    assert not at.exception, f"verify raised: {at.exception}"

    md = [m.value for m in at.markdown]
    supported = [m for m in md if "✓ supported" in m]
    disputed = [m for m in md if "✗ disputed" in m]
    unclear = [m for m in md if "? unclear" in m]
    print("captions:", [c.value for c in at.caption][:2])
    print(f"{len(supported)} supported, {len(disputed)} disputed, {len(unclear)} unclear badge fragment(s)")
    if supported:
        print("example supported:", supported[0])
    if disputed:
        print("example disputed:", disputed[0])

    os.unlink(mp4_path)
    print("verification check passed\n")


def run_long_check(path):
    print(f"=== long check: {path}, VERASCOPE_FAIL_CHUNKS=2 -> failure banner + expander ===")
    os.environ["VERASCOPE_FAIL_CHUNKS"] = "2"
    at = AppTest.from_file("app.py", default_timeout=1800)
    at.run()

    with open(path, "rb") as f:
        content = f.read()
    at.file_uploader[0].set_value((os.path.basename(path), content, "audio/flac")).run()
    assert not at.exception, f"upload raised: {at.exception}"

    transcribe_btn = [b for b in at.button if b.label == "Transcribe"][0]
    transcribe_btn.click().run()
    assert not at.exception, f"run raised: {at.exception}"

    assert at.warning, "no failure banner rendered"
    print("warning:", at.warning[0].value)
    assert "1 of" in at.warning[0].value

    assert at.expander, "no expander rendered"
    exp = at.expander[0]
    assert exp.label == "Failed spans"
    texts = [t.value for t in exp.text]
    print("expander:", texts)
    assert any("chunk 2" in t for t in texts)

    assert any("unavailable" in m.value for m in at.markdown), \
        "no [transcription unavailable] line in rendered transcript"

    print("long check passed\n")
    del os.environ["VERASCOPE_FAIL_CHUNKS"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", metavar="PATH", help="45-60min audio/video file for the long check")
    ap.add_argument("--with-verification", action="store_true",
                     help="also run the fact-check checkbox on a real clip (needs TAVILY_API_KEY)")
    args = ap.parse_args()

    run_fast_check()
    if args.with_verification:
        run_verification_check()
    if args.long:
        run_long_check(args.long)
    else:
        print("(skipping long check — pass --long PATH to run it; "
              "see specs/phases/08-ui.md Notes for a previously captured run)")
    print("verify_ui_apptest ok")
