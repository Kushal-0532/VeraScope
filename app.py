"""Streamlit UI for diarize_demo: audio on the left, Transcript/Claims tabs on the right.

Run: streamlit run app.py
Local/offline testing flags go after `--` (see flags.py):
    streamlit run app.py -- --whisperx-offline --whisperx-model tiny --nli-offline --no-diarize
"""

import asyncio
import hashlib
import importlib.util
import json
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

import evidence_cache
import flags

FLAGS = flags.parse_script_args()   # before anything below reads the env switches

import audio_prep  # noqa: E402
import diarize_demo as dd  # noqa: E402
import verify_pipeline as vp  # noqa: E402
from retrieve_evidence import normalize as normalize_claim  # noqa: E402
from verify_claims import LABELS  # noqa: E402

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
    return asyncio.run(vp.verify_claims_list_v2(_claims, n_lines, on_stage=_on_stage))


# "Strongest" claim on a line wins its transcript tag; disputed first (v1 intent).
_RANK = {"couldnt_check": 0, "not_enough_evidence": 1, "no_sources": 1,
         "mostly_supported": 2, "strongly_supported": 2, "mixed": 3,
         "mostly_disputed": 4, "strongly_disputed": 4}


def _seek(t):
    """on_click callback: runs before the script, so the player renders at t this run
    (no st.rerun, which also threw the view back to the Transcript tab)."""
    st.session_state.seek = int(t)


def _rank(record):
    return _RANK.get(record["label"], 0)


def _md(text):
    """Streamlit markdown renders $...$ as LaTeX; claims and passages quote prices."""
    return str(text).replace("$", "\\$")


def _label_md(record):
    """`✅ **Strongly supported** (89% · 8 vs 1 points)`; no share → no numbers."""
    if record is None:
        emoji, text = LABELS["not_checked"]
        return f"{emoji} **{text}**"
    out = f"{record['emoji']} **{record['label_text']}**"
    if record["share"] is not None:
        out += (f" ({record['share']:.0%} · {record['agree_points']} vs "
                f"{record['disagree_points']} points)")
    return out


def _source_line(src):
    st.markdown(f"\\[{src['source_id']}\\] [{_md(src['domain'])}]({src['url']}) · "
                f"{src['tier']} ({_md(src['trust_reason'])})")
    if src.get("best_chunk"):
        # one line, no leading #/>/+ so scraped page text can't become headings or quotes
        st.caption(_md(" ".join(src["best_chunk"].split())[:300].lstrip("#>+-*\\ ")))
    if src.get("decided_by") == "llm" and src.get("quote"):
        st.markdown(f"*“{_md(src['quote'])}”* — checked quote")


_GROUPS = [("agree", "✅ Agree"), ("disagree", "❌ Disagree"), ("no_side", "➖ No side"),
           ("off_topic", "🚫 Off-topic"), ("failed", "⚠️ Couldn't read")]


def _sources(record):
    groups = {k: [] for k, _ in _GROUPS}
    for src in record["sources"]:
        groups["failed" if src.get("failed") else src["stance"]].append(src)
    for key, title in _GROUPS:
        if not groups[key]:
            continue
        if key in ("agree", "disagree"):   # the sides that vote: always visible
            st.markdown(f"**{title} ({len(groups[key])})**")
            for src in groups[key]:
                _source_line(src)
        else:
            with st.expander(f"{title} ({len(groups[key])})"):
                for src in groups[key]:
                    _source_line(src)


_LEGEND = "\n".join(
    ["| Label | Meaning |", "|---|---|",
     f"| {LABELS['strongly_supported'][0]} {LABELS['strongly_supported'][1]} | ≥ 80% of the trust points that took a side agree |",
     f"| {LABELS['mostly_supported'][0]} {LABELS['mostly_supported'][1]} | 60–80% agree |",
     f"| {LABELS['mixed'][0]} {LABELS['mixed'][1]} | 40–60% agree |",
     f"| {LABELS['mostly_disputed'][0]} {LABELS['mostly_disputed'][1]} | 20–40% agree |",
     f"| {LABELS['strongly_disputed'][0]} {LABELS['strongly_disputed'][1]} | ≤ 20% agree |",
     f"| {LABELS['not_enough_evidence'][0]} {LABELS['not_enough_evidence'][1]} | sources found, none took a side |",
     f"| {LABELS['no_sources'][0]} {LABELS['no_sources'][1]} | the search returned nothing usable |",
     f"| {LABELS['couldnt_check'][0]} {LABELS['couldnt_check'][1]} | every search or scoring call failed — says nothing about truth |",
     f"| {LABELS['not_checked'][0]} {LABELS['not_checked'][1]} | verification hasn't run |",
     "", "Trust points per website: High 3, Medium 2, Low or Unrated 1. One vote per website.",
     "", "| Warning | Meaning |", "|---|---|",
     "| ⚠️ A highly trusted source disagrees / agrees | a High source is on the losing side; Strongly → Mostly |",
     "| ⚠️ Only one source took a side | Strongly → Mostly |",
     "| ⚠️ Only low-trust sources took a side | Strongly → Mostly |",
     "| ⚠️ k of n couldn't be read | some pages failed to load or score; label unchanged |"])


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
        if importlib.util.find_spec("whisperx") is None:
            st.error("--whisperx-offline needs `whisperx` in this environment. Install it in "
                     "a separate venv (it pins its own torch): `python3 -m venv ~/.venvs/wx && "
                     "~/.venvs/wx/bin/pip install whisperx streamlit python-dotenv groq "
                     "tavily-python huggingface_hub`.")
            st.stop()
        import torch
        if os.environ.get("WHISPERX_DEVICE", "cuda" if torch.cuda.is_available() else "cpu") == "cuda":
            st.caption(f"Device: cuda — {torch.cuda.get_device_name(0)}")
        else:
            st.warning("Device: cpu — a long file will take many minutes. Use model `tiny`.")
        default_arch = os.environ.get("WHISPERX_ARCH", "medium")  # Phase 23: medium default
        arch = st.selectbox("Model size", flags.MODELS, index=flags.MODELS.index(default_arch))

    modes = [m for m, on in [("local WhisperX", STAGE3),
                             ("local NLI", os.environ.get("VERASCOPE_LOCAL_NLI") == "1"),
                             ("diarization off", dd._no_diarize())] if on]
    if modes:
        st.caption("Test modes: " + ", ".join(modes))

    tavily_ready = bool(os.environ.get("TAVILY_API_KEY"))
    if not tavily_ready:
        st.caption("Set TAVILY_API_KEY in ~/.env to enable claim verification "
                   "on the Claims tab. Transcription works without it.")

# Stage 1 blocks only on what it actually needs; Stage 2 keys gate the Claims
# tab (Phase 14 behaviour), they never stop the whole app.
if STAGE3:
    if not os.environ.get("HF_TOKEN") and not dd._no_diarize():  # gated pyannote models only
        st.error("Missing HF_TOKEN. Set it in ~/.env or the environment, and "
                 "accept the gated models at "
                 "huggingface.co/pyannote/speaker-diarization-3.1 and "
                 "huggingface.co/pyannote/segmentation-3.0 with the same account.")
        st.stop()
elif missing := [k for k in ("PYANNOTEAI_API_KEY", "GROQ_API_KEY")
                 if not os.environ.get(k)
                 and not (k == "PYANNOTEAI_API_KEY" and dd._no_diarize())]:
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
    if a.get("schema_version") != 2:
        st.info("Old artifact format — regenerate with make_artifact.py")
    elif a["digest"] == digest:
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
    # autoplay after a seek click: a paused player that moved its start point looks
    # like nothing happened, especially when it is scrolled out of view.
    st.audio(flac_path, start_time=st.session_state.get("seek", 0),
             autoplay="seek" in st.session_state)
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

    # tracked tab state, so a rerun (seek click, verify) keeps the tab the user is on
    transcript_tab, claims_tab = st.tabs(["Transcript", "Claims"], key="view", on_change="rerun")

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
            c1.button(f"{dd.ts(start)}", key=f"seek{i}", use_container_width=True,
                      on_click=_seek, args=(start,))
            flag = " :orange[(?)]" if conf is not None and conf < low_conf else ""
            tag = ""
            if records := st.session_state.get("claim_verdicts", {}).get(i):
                top = max(records, key=_rank)
                tag = f" · {top['emoji']} {top['label_text']}"
            c2.markdown(f"**{label}:** {_md(text)}{flag}{tag}")

        st.caption(f"{len(names)} speaker(s), {len(lines)} lines. (?) = low acoustic confidence.")

        verdicts_by_line = st.session_state.get("claim_verdicts", {})

        def _line_text(i, s, e, spk, t):
            records = verdicts_by_line.get(i, [])
            tag = ""
            if records:
                tag = f" [{max(records, key=_rank)['label']}]"
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
                    except evidence_cache.CacheMiss:
                        st.warning("Not in cache: run once with --record")
                        st.stop()
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
                with st.status(vp.STAGES_V2[0], expanded=True) as status:
                    try:
                        vbl, vstats = verify(
                            digest, claims, n_lines, lambda label: status.update(label=label))
                    except evidence_cache.CacheMiss:
                        status.update(label="Not in cache", state="error")
                        st.warning("Not in cache: run once with --record")
                        st.stop()
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
            lc = verify_stats["label_counts"]
            st.caption(f"{verify_stats['n_unique_claims']} unique claim(s) checked — " + ", ".join(
                f"{n} {k.replace('_', ' ')}" for k, n in lc.items() if n))
            if lc["couldnt_check"]:
                st.warning(f"{lc['couldnt_check']} claim(s) couldn't be checked "
                           "(search or scoring failed). That is not a dispute.")
            with st.expander("Run stats"):
                g = verify_stats.get("groq_calls", {})
                c = verify_stats.get("cache", {})
                st.markdown(
                    f"- Tavily credits: {verify_stats.get('tavily_credits', 0)}\n"
                    f"- Groq calls: detect {g.get('detect', 0)}, tier-2 {g.get('tier2', 0)}, "
                    f"summary {g.get('summary', 0)}\n"
                    f"- Tier-2 skipped (over budget): {verify_stats.get('tier2_skipped', 0)}, "
                    f"LLM answers rejected (quote not found in the source, NLI label kept): "
                    f"{verify_stats.get('quote_failures', 0)}\n"
                    f"- Cache: {c.get('hits', 0)} hits, {c.get('misses', 0)} misses")
        with st.expander("What do the labels mean?"):
            st.markdown(_LEGEND)

        for claim_id, g in grouped.items():
            record = _verdict_for(claim_id, g["occurrences"], verdicts_by_line)
            with st.container(border=True):
                first = g["occurrences"][0]
                c1, c2 = st.columns([1, 12])
                c1.button("▶", key=f"claimseek{claim_id}", help="Play from here",
                          on_click=_seek, args=(first["start"],))
                speakers = ", ".join(sorted({names.get(o["speaker"], o["speaker"] or "Unknown")
                                             for o in g["occurrences"]}))
                c2.markdown(f'**{_md(speakers)}** at {dd.ts(first["start"])}: '
                            f'"{_md(g["claim_text"])}" — {_label_md(record)}')
                if len(g["occurrences"]) > 1:
                    c2.caption("said at " + ", ".join(dd.ts(o["start"]) for o in g["occurrences"]))
                if record is None:
                    continue
                for w in record["warnings"]:
                    c2.caption(f"⚠️ {_md(w['text'])}")
                if record["label"] == "couldnt_check":
                    c2.caption("Search or scoring failed — this says nothing about "
                               "whether the claim is true.")
                with c2:
                    _sources(record)
                # Phase 36: "Summarize sources" button goes here.
