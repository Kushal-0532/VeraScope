#!/usr/bin/env python3
"""Speaker diarization + transcription demo.

wav -> librosa (16k mono) -> pyannoteAI hosted (who spoke when)
                          -> whisper (what was said)
                          -> wav2vec2 (per-segment acoustic confidence)

Usage: python3 diarize_demo.py path/to/audio.wav
Needs PYANNOTEAI_API_KEY, read from the environment or ~/.env, for diarization.
"""

import argparse
import asyncio
import math
import os
import sys
import tempfile
import time

import audio_prep
import stitch

WAV2VEC2 = "facebook/wav2vec2-base-960h"
SR = 16000
CONFIDENCE_MAX_CLIP_S = 10.0  # cap scored clip length; see Phase 07 notes
CONFIDENCE_WINDOW_S = 4.0  # fixed window for whole-file concurrent scoring


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def load_audio(path):
    """Anything libsndfile reads -> float32 mono at 16kHz."""
    import librosa
    if not os.path.isfile(path):
        die(f"no such file: {path}")
    try:
        audio, _ = librosa.load(path, sr=SR, mono=True)
    except Exception as e:
        die(f"could not read {path}: {e}")
    if len(audio) < SR // 10:
        die(f"{path} has less than 100ms of audio")
    return audio


def diarize(path, token, min_speakers, max_speakers):
    """pyannoteAI hosted diarization on the full file. Always one job, never
    chunked (Prior Decision 2: speaker labels are per-job cluster IDs and
    don't align across separate jobs)."""
    from pyannoteai.sdk import Client
    c = Client(token)
    url = c.upload(path)
    job = c.diarize(url, min_speakers=min_speakers, max_speakers=max_speakers)
    out = c.retrieve(job)
    turns = [(d["start"], d["end"], d["speaker"]) for d in out["output"]["diarization"]]
    return sorted(turns, key=lambda t: t[0])


async def run_pipeline(src, pyannoteai_key, language=None, min_spk=1, max_spk=5,
                        max_gap=1.5, on_stage=None, score_confidence=True):
    """Full pipeline: prep once, then diarize (long-poll), transcribe
    (concurrent chunks), and score acoustic confidence all concurrently.
    -> ([(start, end, speaker, text, confidence)], failures).

    Confidence scoring runs on fixed whole-file windows (see
    _score_windows/_aggregate_line_score) rather than per final line, because
    the final line spans aren't known until diarize's turns arrive — scoring
    windows needs only the raw audio, so it overlaps with diarize's long
    network wait instead of running serially after everything else finishes.
    Diarize dominates wall clock on real recordings (measured: 8+ min of a
    ~11 min total on a real 48-min file), so hiding confidence scoring's CPU
    time under that wait is the single biggest lever available without
    chunking diarization itself (Prior Decision 2 forbids that).

    on_stage(label), if given, is called at stage-level transitions (prep done,
    gather done) — from the main event-loop thread only, never from inside the
    diarize to_thread call, so it's safe for callers (e.g. Streamlit) that
    require single-thread UI updates."""
    import transcribe  # lazy: transcribe.py imports this module at top level

    with tempfile.TemporaryDirectory() as workdir:
        t0 = time.perf_counter()
        flac = audio_prep.to_flac(src, os.path.join(workdir, "full.flac"))
        dur = audio_prep.probe_duration(flac)
        plan = audio_prep.plan_chunks(dur)
        chunks = audio_prep.split(flac, plan, os.path.join(workdir, "chunks"))
        audio = load_audio(flac) if score_confidence else None
        print(f"STAGE prep {time.perf_counter() - t0:.1f}s, {len(chunks)} chunk(s)", file=sys.stderr)
        if on_stage:
            on_stage(f"Prepared {ts(dur)}, {len(chunks)} chunk(s). "
                     "Diarizing + transcribing + scoring...")

        async def timed_diarize():
            t = time.perf_counter()
            turns = await asyncio.to_thread(diarize, flac, pyannoteai_key, min_spk, max_spk)
            print(f"STAGE diarize {time.perf_counter() - t:.1f}s", file=sys.stderr)
            return turns

        async def timed_transcribe():
            t = time.perf_counter()
            result = await transcribe.transcribe_chunks(chunks, language)
            print(f"STAGE transcribe {time.perf_counter() - t:.1f}s", file=sys.stderr)
            return result

        async def timed_confidence():
            if not score_confidence:
                return []
            t = time.perf_counter()
            windows = await asyncio.to_thread(_score_windows, audio)
            print(f"STAGE confidence (overlapped with diarize) {time.perf_counter() - t:.1f}s",
                  file=sys.stderr)
            return windows

        t_gather = time.perf_counter()
        turns, (chunk_words, failures), windows = await asyncio.gather(
            timed_diarize(), timed_transcribe(), timed_confidence())
        print(f"STAGE gather (wall clock) {time.perf_counter() - t_gather:.1f}s", file=sys.stderr)
        if on_stage:
            on_stage("Stitching transcript...")

        t = time.perf_counter()
        words = stitch.stitch(chunk_words)
        lines = group_lines(words, turns, max_gap)
        scored_lines = [
            (s, e, spk, text, _aggregate_line_score(s, e, windows) if score_confidence else None)
            for s, e, spk, text in lines
        ]
        print(f"STAGE stitch+fusion {time.perf_counter() - t:.1f}s", file=sys.stderr)

        return scored_lines, failures


def confidence_scorer(audio):
    """wav2vec2 CTC mean top-token probability per segment.

    Cheap acoustic-clarity signal: high = clean speech Whisper likely got right,
    low = music/noise/crosstalk, so you can flag suspect lines instead of
    trusting Whisper's hallucinations on silence.
    """
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    proc = Wav2Vec2Processor.from_pretrained(WAV2VEC2)
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2).eval()

    def score(start, end):
        end = min(end, start + CONFIDENCE_MAX_CLIP_S)  # cap compute on long lines
        clip = audio[int(start * SR):int(end * SR)]
        if len(clip) < SR // 10:  # <100ms: nothing to score
            return None
        inputs = proc(clip, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        return float(logits.softmax(-1).max(-1).values.mean())

    return score


def _score_windows(audio, window_s=CONFIDENCE_WINDOW_S):
    """wav2vec2 confidence over fixed windows covering the whole file,
    independent of transcript line boundaries. Needs only the raw audio --
    not transcription, not diarization -- so the caller can run it
    concurrently with diarize()'s long network wait instead of serially
    after it. -> [(start, end, score_or_None), ...] covering [0, duration).
    """
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    proc = Wav2Vec2Processor.from_pretrained(WAV2VEC2)
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2).eval()

    duration = len(audio) / SR
    n = max(1, math.ceil(duration / window_s))
    out = []
    for i in range(n):
        start = i * window_s
        end = min(start + window_s, duration)
        clip = audio[int(start * SR):int(end * SR)]
        if len(clip) < SR // 10:
            out.append((start, end, None))
            continue
        inputs = proc(clip, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        out.append((start, end, float(logits.softmax(-1).max(-1).values.mean())))
    return out


def _aggregate_line_score(line_start, line_end, windows):
    """Overlap-duration-weighted average of the windows covering this line.
    None if no scored window overlaps it (e.g. line falls in a <100ms gap)."""
    total_w, total = 0.0, 0.0
    for w_start, w_end, s in windows:
        if s is None:
            continue
        ov = overlap(line_start, line_end, w_start, w_end)
        if ov <= 0:
            continue
        total_w += ov
        total += ov * s
    return total / total_w if total_w > 0 else None


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(seg_start, seg_end, turns):
    """Speaker whose turns overlap this segment the most. None if no overlap."""
    best, best_ov = None, 0.0
    for t_start, t_end, spk in turns:
        ov = overlap(seg_start, seg_end, t_start, t_end)
        if ov > best_ov:
            best, best_ov = spk, ov
    return best


def group_lines(words, turns, max_gap=1.5):
    """Per-word speakers -> [(start, end, speaker, text)] merged into turns.

    A word landing in a diarization gap inherits the previous word's speaker,
    so short pauses don't shatter a line into Unknowns. New line whenever the
    speaker changes or silence exceeds max_gap.
    """
    lines, prev = [], None
    for start, end, word in words:
        spk = assign_speaker(start, end, turns) or prev
        gap = start - lines[-1][1] if lines else 0.0
        if lines and spk == lines[-1][2] and gap <= max_gap:
            s, _, k, text = lines[-1]
            lines[-1] = (s, end, k, f"{text} {word}")
        else:
            lines.append((start, end, spk, word))
        prev = spk
    return lines


def ts(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main():
    # ~/.env for PYANNOTEAI_API_KEY; a real env var still wins (load_dotenv doesn't override)
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.env"))

    ap = argparse.ArgumentParser(description="speaker diarization demo")
    ap.add_argument("audio", help="path to a .wav file")
    ap.add_argument("--pyannoteai-key", default=os.environ.get("PYANNOTEAI_API_KEY"))
    ap.add_argument("--min-speakers", type=int, default=1)
    ap.add_argument("--max-speakers", type=int, default=5)
    ap.add_argument("--language", default=None,
                    help="whisper language hint, e.g. en (default: autodetect)")
    ap.add_argument("--max-gap", type=float, default=1.5,
                    help="silence (s) that starts a new line for the same speaker")
    ap.add_argument("--low-conf", type=float, default=0.75,
                    help="mark lines below this alignment confidence with (?)")
    ap.add_argument("--whisper-arch", default=None,
                    help="whisperx model size (tiny..large-v3); Stage 3 only")
    ap.add_argument("--device", default=None,
                    help="cuda or cpu; default auto. Stage 3 only")
    args = ap.parse_args()

    # ponytail: one env flag selects the local WhisperX path, same switch app.py
    # uses. Phase 19 deletes the hosted branch and both flags go with it.
    stage3 = os.environ.get("VERASCOPE_STAGE3") == "1" or bool(
        args.whisper_arch or args.device)
    if args.whisper_arch:
        os.environ["WHISPERX_ARCH"] = args.whisper_arch
    if args.device:
        os.environ["WHISPERX_DEVICE"] = args.device

    if args.min_speakers > args.max_speakers or args.min_speakers < 1:
        die("need 1 <= --min-speakers <= --max-speakers")

    if not stage3 and not args.pyannoteai_key:
        die("no PYANNOTEAI_API_KEY: set it in ~/.env or the environment, "
            "or pass --pyannoteai-key (https://dashboard.pyannote.ai/).")

    if stage3:
        lines, failures = asyncio.run(run_pipeline_wx(
            args.audio, args.language, args.min_speakers, args.max_speakers,
            args.max_gap))
    else:
        lines, failures = asyncio.run(run_pipeline(
            args.audio, args.pyannoteai_key, args.language,
            args.min_speakers, args.max_speakers, args.max_gap))
    if not lines:
        die("pipeline found no speech")
    for i, start, end, msg in failures:
        print(f"warning: chunk {i} ({ts(start)}-{ts(end)}) failed: {msg}", file=sys.stderr)

    # stable speaker numbering by first appearance; confidence already scored
    # inside run_pipeline, concurrently with diarize (see its docstring)
    names, out = {}, []
    for start, end, spk, text, conf in lines:
        label = "Unknown" if spk is None else names.setdefault(
            spk, f"Speaker {len(names) + 1}")
        flag = " (?)" if conf is not None and conf < args.low_conf else ""
        out.append(f'[{ts(start)}-{ts(end)}] {label}: "{text}"{flag}')

    print()
    print("\n".join(out))
    print(f"\n{len(names)} speaker(s), {len(out)} lines. (?) = low acoustic confidence.",
          file=sys.stderr)


def _selfcheck():
    """Timeline math only — models need real audio and a token."""
    assert ts(0) == "00:00" and ts(75) == "01:15" and ts(3675) == "01:01:15"
    assert overlap(0, 10, 5, 20) == 5 and overlap(0, 5, 10, 20) == 0
    turns = [(0, 10, "A"), (9, 30, "B")]
    assert assign_speaker(0, 8, turns) == "A"
    assert assign_speaker(12, 20, turns) == "B"
    assert assign_speaker(9, 9.5, turns) == "A"   # exact tie: first turn wins
    assert assign_speaker(50, 60, turns) is None  # gap in diarization

    turns = [(0, 5, "A"), (5, 10, "B")]
    words = [(0.0, 1.0, "hi"), (1.0, 2.0, "there"), (6.0, 7.0, "hello")]
    lines = group_lines(words, turns)
    assert [(l[2], l[3]) for l in lines] == [("A", "hi there"), ("B", "hello")]
    assert lines[0][0] == 0.0 and lines[0][1] == 2.0  # line spans its words

    # same speaker, long silence -> two lines
    long_pause = [(0.0, 1.0, "one"), (4.0, 5.0, "two")]
    assert len(group_lines(long_pause, [(0, 10, "A")], max_gap=1.5)) == 2

    # word in a diarization gap inherits the previous speaker
    gapped = [(0.0, 1.0, "a"), (20.0, 21.0, "b")]
    assert group_lines(gapped, [(0, 5, "A")])[-1][2] == "A"

    # _aggregate_line_score: pure math, no model needed
    windows = [(0.0, 8.0, 0.9), (8.0, 16.0, 0.5), (16.0, 24.0, None)]
    assert _aggregate_line_score(0.0, 8.0, windows) == 0.9
    assert abs(_aggregate_line_score(4.0, 12.0, windows) - (4 * 0.9 + 4 * 0.5) / 8) < 1e-9
    assert _aggregate_line_score(16.0, 24.0, windows) is None  # only a None window overlaps
    assert _aggregate_line_score(100.0, 101.0, windows) is None  # no window overlaps at all
    print("selfcheck ok")




# ---------------------------------------------------------------------------
# Stage 3 (WhisperX, local GPU). Added alongside the hosted path, not replacing
# it, so the working app survives until Phase 19 does the deletions on the box.
# ---------------------------------------------------------------------------

_DIARIZER = {}  # (device,) -> DiarizationPipeline


def _diarization_pipeline(device):
    """Lazy singleton. HF_TOKEN read here, not passed in (Phase 18 drops the
    token positional)."""
    if device not in _DIARIZER:
        token = os.environ.get("HF_TOKEN")
        if not token:
            die("no HF_TOKEN: set it in ~/.env or the environment "
                "(https://huggingface.co/settings/tokens), and accept the gated "
                "models at https://huggingface.co/pyannote/speaker-diarization-3.1 "
                "and https://huggingface.co/pyannote/segmentation-3.0")
        try:
            from whisperx import DiarizationPipeline
        except ImportError:
            # moved between whisperx releases; both spellings seen in the wild
            from whisperx.diarize import DiarizationPipeline
        try:
            try:
                # whisperx >= 3.8: kwarg is `token`, and the default model moved
                # to speaker-diarization-community-1; pin 3.1 (Phase 18).
                # PYANNOTE_MODEL is a Phase 23 measurement knob only; the
                # default stays 3.1 until Phase 18's spec says otherwise.
                _DIARIZER[device] = DiarizationPipeline(
                    model_name=os.environ.get("PYANNOTE_MODEL",
                                              "pyannote/speaker-diarization-3.1"),
                    token=token, device=device)
            except TypeError:
                _DIARIZER[device] = DiarizationPipeline(use_auth_token=token, device=device)
        except Exception as e:
            die(f"could not load pyannote diarization ({e}). If this is a 401/403, "
                "accept the gated models at "
                "https://huggingface.co/pyannote/speaker-diarization-3.1 and "
                "https://huggingface.co/pyannote/segmentation-3.0 with the same "
                "account as HF_TOKEN.")
        pipe = getattr(_DIARIZER[device], "model", None)
        # Phase 23: batch knobs, env-tunable. Measured flat at fp32 on the T4
        # (32/64/128 all ~120 s); left in for the fp16 path.
        for attr, env in (("segmentation_batch_size", "PYANNOTE_SEG_BATCH"),
                          ("embedding_batch_size", "PYANNOTE_EMB_BATCH")):
            v = os.environ.get(env)
            if v and pipe is not None and hasattr(pipe, attr):
                setattr(pipe, attr, int(v))
        if str(device).startswith("cuda") and os.environ.get("PYANNOTE_FP16", "1") != "0":
            _fp16_embedding_resnet(pipe)
    return _DIARIZER[device]


def _fp16_embedding_resnet(pipe):
    """Phase 23: 95% of diarize wall is the WeSpeaker embedding ResNet in fp32.
    Autocast the ResNet body only. Autocasting the whole pipeline NaNs the
    fbank front-end (measured: 2048 NaN embeddings, clusters collapse 5 -> 2);
    ResNet-only gives cosine 1.0000 vs fp32 on 8 probe chunks, byte-identical
    turns on the 48-min file, 113 s -> 68 s. Skipped silently on a pipeline
    whose embedding model is shaped differently (fp32 path still correct)."""
    import torch
    resnet = getattr(getattr(getattr(pipe, "_embedding", None), "model_", None), "resnet", None)
    if resnet is None or getattr(resnet, "_verascope_fp16", False):
        return
    orig = resnet.forward

    def forward(*a, **k):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = orig(*a, **k)
        return (tuple(o.float() if torch.is_tensor(o) else o for o in out)
                if isinstance(out, tuple) else out.float())
    resnet.forward = forward
    resnet._verascope_fp16 = True


def wx_diarize(path, min_speakers=1, max_speakers=5):
    """Local pyannote 3.1 via whisperx -> [(start, end, speaker)] sorted by start.

    Same return shape as the hosted diarize(), so assign_speaker/group_lines
    take it unmodified. One unchunked pass over the whole file (Decision 2)."""
    import wx_transcribe
    df = _diarization_pipeline(wx_transcribe._device())(
        path, min_speakers=min_speakers, max_speakers=max_speakers)
    # whisperx returns a DataFrame; iterate generically so a plain list of dicts
    # (older/newer releases) works too.
    rows = df.to_dict("records") if hasattr(df, "to_dict") else df
    turns = [(float(r["start"]), float(r["end"]), str(r["speaker"])) for r in rows]
    return sorted(turns, key=lambda t: t[0])


def group_lines_wx(words, turns, max_gap=1.5):
    """[(start, end, word, score)] + turns -> [(start, end, speaker, text, conf)].

    Same line-breaking rules as group_lines (new line on speaker change or
    silence > max_gap; a word in a diarization gap inherits the previous
    speaker). conf is the duration-weighted mean of the word alignment scores
    (Decision 6); a line whose words all scored None yields None, not 0.0.
    """
    lines, acc, prev = [], [], None
    for start, end, word, score in words:
        spk = assign_speaker(start, end, turns) or prev
        gap = start - lines[-1][1] if lines else 0.0
        if lines and spk == lines[-1][2] and gap <= max_gap:
            s, _, k, text = lines[-1]
            lines[-1] = (s, end, k, f"{text} {word}")
            acc[-1].append((start, end, score))
        else:
            lines.append((start, end, spk, word))
            acc.append([(start, end, score)])
        prev = spk
    return [(s, e, k, text, _weighted_score(ws))
            for (s, e, k, text), ws in zip(lines, acc)]


def _weighted_score(word_scores):
    """[(start, end, score|None)] -> duration-weighted mean, or None if every
    score is None. Zero-length words fall back to weight 1 so a degenerate
    timestamp can't silently drop a real score."""
    total = total_w = 0.0
    for start, end, score in word_scores:
        if score is None:
            continue
        w = max(end - start, 0.0) or 1.0
        total += w * score
        total_w += w
    return total / total_w if total_w > 0 else None


async def run_pipeline_wx(src, language=None, min_spk=1, max_spk=5,
                          max_gap=1.5, on_stage=None):
    """Local Stage 1: prep -> transcribe+align -> diarize -> fuse.
    -> ([(start, end, speaker, text, confidence)], []).

    Phase 23 ordering: ASR alone (GPU-bound), then alignment (mostly CPU
    backtracking) overlapped with diarization (GPU). Running ASR and diarize
    together was measured to be no faster than sequential on the T4 (both
    GPU-bound, they just time-slice), while align||diarize does overlap.
    The empty failures list is kept so app.py / main() unpacking is untouched
    (Decision 8)."""
    import wx_transcribe

    with tempfile.TemporaryDirectory() as workdir:
        t0 = time.perf_counter()
        flac = audio_prep.to_flac(src, os.path.join(workdir, "full.flac"))
        dur = audio_prep.probe_duration(flac)
        print(f"STAGE prep {time.perf_counter() - t0:.1f}s", file=sys.stderr)
        if on_stage:
            on_stage(f"Prepared {ts(dur)}. Transcribing...")

        def timed(name, fn, *a):
            t = time.perf_counter()
            out = fn(*a)
            print(f"STAGE {name} {time.perf_counter() - t:.1f}s", file=sys.stderr)
            return out

        asr = await asyncio.to_thread(timed, "transcribe", wx_transcribe.transcribe,
                                      flac, language)
        if on_stage:
            on_stage("Aligning words and diarizing...")
        t = time.perf_counter()
        words, turns = await asyncio.gather(
            asyncio.to_thread(timed, "align", wx_transcribe.align_words, asr),
            asyncio.to_thread(timed, "diarize", wx_diarize, flac, min_spk, max_spk))
        print(f"STAGE align||diarize {time.perf_counter() - t:.1f}s, "
              f"{len(words)} words, {len(turns)} turns", file=sys.stderr)
        if on_stage:
            on_stage("Assigning speakers...")

        t = time.perf_counter()
        lines = group_lines_wx(words, turns, max_gap)
        print(f"STAGE fusion {time.perf_counter() - t:.1f}s, {len(lines)} lines",
              file=sys.stderr)
        return lines, []


def _selfcheck_wx():
    """group_lines_wx / _weighted_score: pure arithmetic, no model, no network."""
    turns = [(0, 5, "A"), (5, 10, "B")]
    words = [(0.0, 1.0, "hi", 0.9), (1.0, 2.0, "there", 0.7), (6.0, 7.0, "hello", 0.5)]
    lines = group_lines_wx(words, turns)
    assert [(l[2], l[3]) for l in lines] == [("A", "hi there"), ("B", "hello")], lines
    assert lines[0][0] == 0.0 and lines[0][1] == 2.0
    assert abs(lines[0][4] - 0.8) < 1e-9, lines[0]          # equal durations -> mean
    assert lines[1][4] == 0.5

    # duration weighting actually weights: 3s@0.9 + 1s@0.5 -> 0.8
    w = [(0.0, 3.0, "long", 0.9), (3.0, 4.0, "x", 0.5)]
    assert abs(group_lines_wx(w, [(0, 10, "A")])[0][4] - 0.8) < 1e-9

    # same speaker, long silence -> two lines
    assert len(group_lines_wx([(0.0, 1.0, "one", 0.9), (4.0, 5.0, "two", 0.9)],
                              [(0, 10, "A")], max_gap=1.5)) == 2
    # word in a diarization gap inherits the previous speaker
    assert group_lines_wx([(0.0, 1.0, "a", 0.9), (20.0, 21.0, "b", 0.9)],
                          [(0, 5, "A")])[-1][2] == "A"
    # all-None scores -> None, not 0.0, not a crash
    none_line = group_lines_wx([(0.0, 1.0, "a", None), (1.0, 2.0, "b", None)],
                               [(0, 5, "A")])[0]
    assert none_line[4] is None, none_line
    # mixed None/real -> the real ones decide
    assert group_lines_wx([(0.0, 1.0, "a", None), (1.0, 2.0, "b", 0.6)],
                          [(0, 5, "A")])[0][4] == 0.6
    # zero-length word keeps its score instead of vanishing
    assert _weighted_score([(1.0, 1.0, 0.4)]) == 0.4
    # frozen 5-tuple schema
    for s, e, spk, text, conf in group_lines_wx(words, turns):
        assert isinstance(s, float) and isinstance(e, float) and isinstance(text, str)
        assert spk is None or isinstance(spk, str)
        assert conf is None or 0.0 <= conf <= 1.0
    print("selfcheck_wx ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        _selfcheck_wx()
    else:
        main()
