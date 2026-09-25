"""Streamlit UI for diarize_demo: audio on the left, Transcript/Claims tabs on the right.

Run: streamlit run app.py
"""

import asyncio
import hashlib
import json
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

import audio_prep
import diarize_demo as dd
import verify_pipeline as vp
from retrieve_evidence import normalize as normalize_claim

load_dotenv(os.path.expanduser("~/.env"))

st.set_page_config(page_title="Speaker Diarization", layout="wide")
st.title("Speaker Diarization")

VIDEO_AUDIO_TYPES = ["mp4", "mkv", "mov", "webm", "wav", "mp3", "flac", "m4a", "ogg"]


@st.cache_data(show_spinner=False)
def extract_flac(_raw_bytes, suffix, digest):
    """Write the upload to disk and ffmpeg-extract 16kHz mono FLAC once per
    upload (keyed on digest, not the bytes themselves — hashing hundreds of MB
    on every widget rerun is slow). Handles video containers; librosa can't."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(_raw_bytes)
        src_path = f.name
    flac_path = tempfile.NamedTemporaryFile(suffix=".flac", delete=False).name
    audio_prep.to_flac(src_path, flac_path)
    os.unlink(src_path)
    return flac_path


# ponytail: one env flag, not a config system. STAGE3=1 selects the local
# WhisperX path; unset keeps the hosted path that is the demo fallback. Phase 19
# deletes the hosted branch and this flag goes with it.
STAGE3 = os.environ.get("VERASCOPE_STAGE3") == "1"


@st.cache_data(show_spinner=False)
def run(digest, _flac_path, language, min_spk, max_spk, max_gap, arch, _on_stage):
    """Full pipeline. Cached on (digest, settings) so tweaking nothing re-runs
    nothing; the underscore-prefixed args are excluded from hashing. arch is in
    the key so changing model size is a cache miss."""
    if STAGE3:
        if arch:
            os.environ["WHISPERX_ARCH"] = arch   # read at call time by wx_transcribe
        return asyncio.run(dd.run_pipeline_wx(
            _flac_path, language or None, min_spk, max_spk, max_gap,
            on_stage=_on_stage))
    return asyncio.run(dd.run_pipeline(
        _flac_path, os.environ.get("PYANNOTEAI_API_KEY"), language or None,
        min_spk, max_spk, max_gap, on_stage=_on_stage))


@st.cache_data(show_spinner=False)
def detect(digest, _lines):
    """Cheap first Stage 2 step: claim detection only. Cached on digest alone
    so it survives unrelated widget reruns without re-running Stage 1."""
    return asyncio.run(vp.detect_only(_lines))


@st.cache_data(show_spinner=False)
def verify(digest, _claims, n_lines, _on_stage):
    """Expensive second Stage 2 step: retrieve + verify. Separate cache key
    from detect() so re-detecting (if ever needed) doesn't force re-spending
    Tavily/HF credits, and vice versa."""
    return asyncio.run(vp.verify_claims_list(_claims, n_lines, on_stage=_on_stage))


_VERDICT_BADGE = {
    "supported": ":green[✓ supported]",
    "disputed": ":red[✗ disputed]",
    "unclear": ":gray[? unclear]",
}
_VERDICT_RANK = {"disputed": 2, "unclear": 1, "supported": 0}


def _group_claims(claims):
    """[{"claim_text","line_id","speaker","start"}, ...] (pre-dedup, one entry
    per source line) -> {claim_id: {"claim_text", "occurrences": [...]}}, one
    row per unique claim, so a claim repeated across lines isn't shown twice."""
    grouped = {}
    for c in claims:
        cid = normalize_claim(c["claim_text"])
        g = grouped.setdefault(cid, {"claim_text": c["claim_text"], "occurrences": []})
        g["occurrences"].append({"line_id": c["line_id"], "speaker": c["speaker"], "start": c["start"]})
    return grouped


def _verdict_for(claim_id, occurrences, verdicts_by_line):
    """A claim's verdict record lives under any of its occurrences' line_ids
    in verdicts_by_line; find it by matching normalized claim_text."""
    for occ in occurrences:
        for record in verdicts_by_line.get(occ["line_id"], []):
            if normalize_claim(record["claim_text"]) == claim_id:
                return record
    return None


with st.sidebar:
    st.header("Settings")
    language = st.text_input("Language hint", value="",
                             help="e.g. en, hi. Blank = autodetect (collapses on code-switched audio)")
    min_spk, max_spk = st.slider("Speakers", 1, 10, (1, 5))
    max_gap = st.slider("New line after silence (s)", 0.5, 5.0, 1.5, 0.5)
    low_conf = st.slider("Low-confidence threshold", 0.0, 1.0, 0.75, 0.05)

    arch = None
    if STAGE3:
        import torch
        if torch.cuda.is_available():
            st.caption(f"Device: cuda — {torch.cuda.get_device_name(0)}")
        else:
            st.warning("Device: cpu — a long file will take many minutes.")
        arch = st.selectbox("Model size", ["tiny", "base", "small", "medium",
                                           "large-v3"], index=3)  # Phase 23: medium default

    tavily_ready = bool(os.environ.get("TAVILY_API_KEY"))
    if not tavily_ready:
        st.caption("Set TAVILY_API_KEY in ~/.env to enable claim verification "
                   "on the Claims tab. Transcription works without it.")

# Stage 1 blocks only on what it actually needs; Stage 2 keys gate the Claims
# tab (Phase 14 behaviour), they never stop the whole app.
if STAGE3:
    if not os.environ.get("HF_TOKEN"):
        st.error("Missing HF_TOKEN. Set it in ~/.env or the environment, and "
                 "accept the gated models at "
                 "huggingface.co/pyannote/speaker-diarization-3.1 and "
                 "huggingface.co/pyannote/segmentation-3.0 with the same account.")
        st.stop()
elif missing := [k for k in ("PYANNOTEAI_API_KEY", "GROQ_API_KEY")
                 if not os.environ.get(k)]:
    st.error(f"Missing required key(s): {', '.join(missing)}. "
             "Set them in ~/.env or the environment.")
    st.stop()

uploaded = st.file_uploader("Audio or video file", type=VIDEO_AUDIO_TYPES)
if not uploaded:
    st.info("Upload an audio or video file to start.")
    st.stop()

raw_bytes = uploaded.getvalue()
digest = hashlib.sha256(raw_bytes).hexdigest()
suffix = os.path.splitext(uploaded.name)[1] or ".wav"

if st.session_state.get("_last_digest") != digest:
    # a fresh upload invalidates any prior run's results, so the transcript/
    # claims panels don't keep showing the previous file's stale content.
    for key in ("lines", "failures", "claims", "claims_n_lines",
                "claim_verdicts", "claim_stats"):
        st.session_state.pop(key, None)
    st.session_state._last_digest = digest

# ponytail: demo fallback (Phase 21). If an artifact from make_artifact.py
# matches this upload's digest, render from it and never touch a model or API.
if (art := os.environ.get("VERASCOPE_ARTIFACT")) and "lines" not in st.session_state:
    a = json.load(open(art))
    if a["digest"] == digest:
        st.session_state.update(
            lines=[tuple(l) for l in a["lines"]], failures=a["failures"],
            claims=a["claims"], claims_n_lines=a["n_lines"],
            claim_verdicts={int(k): v for k, v in a["verdicts"].items()},
            claim_stats=a["stats"])
        st.caption(f"Rendered from artifact {os.path.basename(art)}.")

flac_path = extract_flac(raw_bytes, suffix, digest)
duration = audio_prep.probe_duration(flac_path)
n_chunks = len(audio_prep.plan_chunks(duration))
st.caption(f"{dd.ts(duration)} total, {n_chunks} chunk(s) planned.")

left, right = st.columns([1, 2])

with left:
    st.subheader("Audio")
    # ponytail: seek by re-rendering the player at a new start_time. Streamlit has no
    # playhead API, so clicking a line restarts playback there. Good enough for a demo.
    st.audio(flac_path, start_time=st.session_state.get("seek", 0))
    go = st.button("Transcribe", type="primary", use_container_width=True)

with right:
    if not (go or "lines" in st.session_state):
        st.caption("Press Transcribe.")
        st.stop()

    if go:
        with st.status("Preparing audio...", expanded=True) as status:
            try:
                st.session_state.lines, st.session_state.failures = run(
                    digest, flac_path, language, min_spk, max_spk, max_gap,
                    arch, lambda label: status.update(label=label))
            except Exception as e:
                st.exception(e)
                st.stop()
            status.update(label="Done", state="complete", expanded=False)
            # a fresh transcript invalidates any claims detected against the old one
            for key in ("claims", "claims_n_lines", "claim_verdicts", "claim_stats"):
                st.session_state.pop(key, None)

    lines = st.session_state.lines
    failures = st.session_state.failures
    if not lines:
        st.warning("No speech found.")
        st.stop()

    transcript_tab, claims_tab = st.tabs(["Transcript", "Claims"])

    with transcript_tab:
        if failures:
            st.warning(
                f"{len(failures)} of {n_chunks} chunks failed to transcribe. "
                "Those spans appear as [transcription unavailable] in the transcript."
            )
            with st.expander("Failed spans"):
                for idx, start, end, err in failures:
                    st.text(f"chunk {idx}: {dd.ts(start)}–{dd.ts(end)}  {err}")

        names = {}
        for i, (start, end, spk, text, conf) in enumerate(lines):
            label = "Unknown" if spk is None else names.setdefault(spk, f"Speaker {len(names) + 1}")
            c1, c2 = st.columns([1, 6])
            if c1.button(f"{dd.ts(start)}", key=f"seek{i}", use_container_width=True):
                st.session_state.seek = int(start)
                st.rerun()
            flag = " :orange[(?)]" if conf is not None and conf < low_conf else ""
            c2.markdown(f"**{label}:** {text}{flag}")

        st.caption(f"{len(names)} speaker(s), {len(lines)} lines. (?) = low acoustic confidence.")

        verdicts_by_line = st.session_state.get("claim_verdicts", {})

        def _line_text(i, s, e, spk, t):
            records = verdicts_by_line.get(i, [])
            tag = ""
            if records:
                strongest = max(records, key=lambda r: _VERDICT_RANK[r["verdict"]])
                tag = f" [{strongest['verdict']}]"
            return f'[{dd.ts(s)}-{dd.ts(e)}] {names.get(spk, "Unknown")}: "{t}"{tag}'

        st.download_button(
            "Download .txt",
            "\n".join(_line_text(i, s, e, spk, t) for i, (s, e, spk, t, _) in enumerate(lines)),
            file_name=f"{os.path.splitext(uploaded.name)[0]}_transcript.txt",
        )

    with claims_tab:
        if "claims" not in st.session_state:
            st.caption("Find the factual claims in this transcript, then check them "
                       "against web evidence. Detection is a quick Groq pass; "
                       "verification (below, once claims are found) spends Tavily/HF credits.")
            if st.button("Detect claims", type="primary"):
                with st.spinner("Detecting claims..."):
                    try:
                        claims, n_lines = detect(digest, lines)
                    except Exception as e:
                        st.exception(e)
                        st.stop()
                st.session_state.claims = claims
                st.session_state.claims_n_lines = n_lines
                st.rerun()
            st.stop()

        claims = st.session_state.claims
        n_lines = st.session_state.claims_n_lines
        if not claims:
            st.info("No check-worthy factual claims found in this transcript.")
            st.stop()

        grouped = _group_claims(claims)
        verdicts_by_line = st.session_state.get("claim_verdicts", {})
        verify_stats = st.session_state.get("claim_stats")

        if not verdicts_by_line and not verify_stats:
            st.caption(f"{len(grouped)} unique claim(s) found, "
                       f"{len(claims)} occurrence(s) across the transcript.")
            if st.button("Verify claims", type="primary", disabled=not tavily_ready):
                with st.status("Retrieving evidence...", expanded=True) as status:
                    try:
                        vbl, vstats = verify(
                            digest, claims, n_lines, lambda label: status.update(label=label))
                    except Exception as e:
                        st.exception(e)
                        st.stop()
                    status.update(label="Done", state="complete", expanded=False)
                st.session_state.claim_verdicts = vbl
                st.session_state.claim_stats = vstats
                st.rerun()
            if not tavily_ready:
                st.caption("Set TAVILY_API_KEY in ~/.env to enable verification.")

        if verify_stats:
            n_failed = verify_stats["n_retrieve_failures"] + verify_stats["n_verify_failures"]
            vc = verify_stats["verdict_counts"]
            st.caption(
                f"{verify_stats['n_unique_claims']} unique claim(s) checked — "
                f"{vc['supported']} supported, {vc['disputed']} disputed, {vc['unclear']} unclear."
            )
            if n_failed:
                st.warning(f"{n_failed} evidence lookup(s) failed during verification. "
                           "Affected claims may show as unclear with fewer sources.")
                with st.expander("Verification stage details"):
                    st.text(f"retrieval failures: {verify_stats['n_retrieve_failures']}")
                    st.text(f"verification failures: {verify_stats['n_verify_failures']}")

        for claim_id, g in grouped.items():
            record = _verdict_for(claim_id, g["occurrences"], verdicts_by_line)
            badge = f" {_VERDICT_BADGE[record['verdict']]}" if record else " :gray[not yet verified]"
            with st.container(border=True):
                first = g["occurrences"][0]
                c1, c2 = st.columns([1, 6])
                if c1.button(dd.ts(first["start"]), key=f"claimseek{claim_id}", use_container_width=True):
                    st.session_state.seek = int(first["start"])
                    st.rerun()
                sources = ", ".join(sorted({o["speaker"] or "Unknown" for o in g["occurrences"]}))
                lines_seen = ", ".join(dd.ts(o["start"]) for o in g["occurrences"])
                c2.markdown(f'**{sources}:** "{g["claim_text"]}"{badge}')
                if len(g["occurrences"]) > 1:
                    c2.caption(f"said at {lines_seen}")
                if record:
                    with c2.expander("Evidence"):
                        st.markdown(
                            f"Agreement strength {record['confidence']:.2f} across "
                            f"{len(record['evidence_used'])} source(s) — not a truth probability.")
                        for ev in record["evidence_used"]:
                            st.markdown(f"- [{ev['url']}]({ev['url']}) — {ev['label']} ({ev['score']:.2f})")
                            st.caption(ev["snippet"][:300])
