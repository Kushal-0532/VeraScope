#!/usr/bin/env python3
"""WhisperX transcription + word alignment, local, whole file.

flac -> whisperx VAD -> faster-whisper (batched) -> wav2vec2 CTC alignment
     -> [(start, end, word, score)] in time order.

Replaces the Groq transcription path, the manual chunk planner/splitter and
the seam de-duplicator: VAD chunks internally on silence, so there are no
seams to repair and no chunk plan to compute. score is the alignment confidence in [0, 1]
(Prior Decision 6) -- None when alignment was skipped.

Usage: python3 wx_transcribe.py --selfcheck   (offline, no model download)
"""

import logging
import os
import sys

# Defaults; each is overridden by the env var of the same name so a device
# change needs no code edit (Prior Decision 4). None = pick from the device.
WHISPERX_ARCH = None          # None -> large-v3 on cuda, small on cpu
WHISPERX_COMPUTE_TYPE = None  # None -> float16 on cuda, int8 on cpu
WHISPERX_BATCH_SIZE = None    # None -> 16 on cuda, 4 on cpu
WHISPERX_THREADS = 4
WHISPERX_DEVICE = None        # None -> cuda if available else cpu

_DEVICE_DEFAULTS = {
    # arch, compute_type, batch_size
    # Phase 23: medium, not large-v3, on cuda (user OK 2026-09-09). Measured on
    # the T4 for the 48-min file: large-v3 113 s vs medium 68 s ASR; int8 and
    # batch 24/32 were within 7 s of each other, so the plain fp16 row stays.
    "cuda": ("medium", "float16", 16),
    "cpu": ("small", "int8", 4),
}

_ASR = {}    # (arch, device, compute_type, threads) -> FasterWhisperPipeline
_ALIGN = {}  # (language, device) -> (model, metadata) | None if unsupported

log = logging.getLogger("wx_transcribe")


def _env(name, default, cast=str):
    """Env wins over the module default, read at call time so a test (or the
    Phase 20 model-size widget) can set it after import."""
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


def _device():
    d = _env("WHISPERX_DEVICE", WHISPERX_DEVICE)
    if d:
        return d
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _defaults(device):
    return _DEVICE_DEFAULTS.get(device, _DEVICE_DEFAULTS["cpu"])


def _batch_size(device):
    return _env("WHISPERX_BATCH_SIZE", WHISPERX_BATCH_SIZE or _defaults(device)[2], int)


def _asr_model(device):
    """Lazy singleton per (arch, device, compute_type, threads)."""
    arch_d, compute_d, _ = _defaults(device)
    key = (
        _env("WHISPERX_ARCH", WHISPERX_ARCH or arch_d),
        device,
        _env("WHISPERX_COMPUTE_TYPE", WHISPERX_COMPUTE_TYPE or compute_d),
        _env("WHISPERX_THREADS", WHISPERX_THREADS, int),
    )
    if key not in _ASR:
        import whisperx
        arch, dev, compute_type, threads = key
        log.info("loading whisperx %s on %s (%s, %d threads)", arch, dev, compute_type, threads)
        _ASR[key] = whisperx.load_model(
            arch, dev, compute_type=compute_type, threads=threads)
    return _ASR[key]


def _align_model(language, device):
    """Lazy singleton per (language, device). None -> no alignment model for
    this language; cached so the warning is logged once, not per call."""
    key = (language, device)
    if key not in _ALIGN:
        import whisperx
        try:
            _ALIGN[key] = whisperx.load_align_model(language_code=language, device=device)
        except Exception as e:
            log.warning("no alignment model for language %r (%s); falling back to "
                        "segment timings, confidence will be None", language, e)
            _ALIGN[key] = None
    return _ALIGN[key]


def transcribe_aligned(flac_path, language=None, on_progress=None):
    """-> [(start, end, word, score)], time-ordered, whole file, no chunking.

    language=None autodetects and uses the detected code for alignment.
    on_progress(pct: float 0-100), if given, is threaded into both the ASR and
    the alignment pass for Phase 20's UI.
    """
    return align_words(transcribe(flac_path, language, on_progress), on_progress)


def transcribe(flac_path, language=None, on_progress=None):
    """ASR only -> opaque state for align_words(). Split out in Phase 23 so
    run_pipeline_wx can hold the GPU for ASR alone, then overlap the
    CPU-heavy alignment with diarization."""
    import whisperx

    device = _device()
    audio = whisperx.load_audio(flac_path)
    model = _asr_model(device)

    kw = {"batch_size": _batch_size(device), "language": language}
    if on_progress:
        kw["progress_callback"] = on_progress
    result = model.transcribe(audio, **kw)
    return device, audio, result, language or result.get("language")


def align_words(state, on_progress=None):
    """transcribe() state -> [(start, end, word, score)], time-ordered."""
    import whisperx

    device, audio, result, lang = state
    aligned = _align_model(lang, device) if lang else None

    if aligned is None:
        # No align model (or no language at all): Whisper's own segment timings,
        # one "word" per segment, no confidence.
        words = [(float(s["start"]), float(s["end"]), s["text"].strip(), None)
                 for s in result["segments"] if s.get("text", "").strip()]
    else:
        model_a, meta = aligned
        akw = {"progress_callback": on_progress} if on_progress else {}
        # Phase 23: fp16 autocast on the wav2vec2 pass; same 8614 words and
        # scores on the 48-min file, 50 s -> 43 s (the rest is CPU backtracking).
        import torch
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=str(device).startswith("cuda")):
            ra = whisperx.align(result["segments"], model_a, meta, audio, device, **akw)
        words = []
        for w in ra["word_segments"]:
            text = w.get("word", "").strip()
            # whisperx drops start/end on words the CTC pass couldn't place
            # (numerals, chars outside the alignment alphabet). Untimed words
            # can't be assigned a speaker or grouped, so skip them.
            if not text or w.get("start") is None or w.get("end") is None:
                continue
            score = w.get("score")
            words.append((float(w["start"]), float(w["end"]), text,
                          None if score is None else float(score)))

    # Stable sort: crosstalk made word order non-monotonic for real in Phase 02,
    # and group_lines_wx requires time-ordered input (fusion contract).
    words.sort(key=lambda w: w[0])
    return words


def _fake_whisperx(arch_seen, unsupported=()):
    """Stand-in module: no download, no network, no torch."""
    import types
    m = types.ModuleType("whisperx")

    m.load_audio = lambda path: [0.0]

    class _Pipe:
        def transcribe(self, audio, batch_size=None, language=None, progress_callback=None):
            if progress_callback:
                progress_callback(100.0)
            return {"language": language or "en",
                    "segments": [{"start": 0.0, "end": 2.0, "text": "hi there"},
                                 {"start": 3.0, "end": 4.0, "text": "  "}]}

    def load_model(arch, device, compute_type=None, threads=None):
        arch_seen.append(arch)
        return _Pipe()

    def load_align_model(language_code=None, device=None):
        if language_code in unsupported:
            raise ValueError(f"no default align-model for language: {language_code}")
        return ("model_a", {"meta": True})

    def align(segments, model_a, meta, audio, device, progress_callback=None):
        if progress_callback:
            progress_callback(100.0)
        return {"word_segments": [
            {"word": "there", "start": 1.0, "end": 2.0, "score": 0.8},
            {"word": " hi ", "start": 0.0, "end": 1.0, "score": 0.9},  # out of order
            {"word": "  ", "start": 2.0, "end": 2.5, "score": 0.5},    # empty -> dropped
            {"word": "1999", "start": None, "end": None, "score": None},  # untimed -> dropped
        ]}

    m.load_model, m.load_align_model, m.align = load_model, load_align_model, align
    return m


def _selfcheck():
    """Pure logic against a fake whisperx: no model, no network, no GPU."""
    os.environ["WHISPERX_DEVICE"] = "cpu"  # never touch torch here
    warnings = []
    log.addHandler(type("H", (logging.Handler,), {"emit": lambda s, r: warnings.append(r)})())
    log.setLevel(logging.WARNING)

    arch_seen = []
    sys.modules["whisperx"] = _fake_whisperx(arch_seen)
    _ASR.clear(); _ALIGN.clear()

    words = transcribe_aligned("fake.flac")
    assert words == [(0.0, 1.0, "hi", 0.9), (1.0, 2.0, "there", 0.8)], words
    assert all(isinstance(s, float) and isinstance(e, float) and isinstance(t, str)
               for s, e, t, _ in words)
    assert words == sorted(words, key=lambda w: w[0])          # ordering restored
    assert all(c is None or 0.0 <= c <= 1.0 for *_, c in words)
    assert arch_seen == ["small"]                              # cpu default
    assert not warnings

    # progress callback reaches both passes
    pct = []
    transcribe_aligned("fake.flac", on_progress=pct.append)
    assert pct == [100.0, 100.0], pct

    # env override of the arch is picked up (and re-keys the singleton)
    os.environ["WHISPERX_ARCH"] = "large-v3"
    transcribe_aligned("fake.flac")
    assert arch_seen == ["small", "large-v3"], arch_seen
    del os.environ["WHISPERX_ARCH"]

    # unsupported language -> segment timings, score None, exactly one warning,
    # no raise; the second call reuses the cached miss and does not re-warn.
    sys.modules["whisperx"] = _fake_whisperx(arch_seen, unsupported={"yo"})
    _ASR.clear(); _ALIGN.clear()
    fb = transcribe_aligned("fake.flac", language="yo")
    assert fb == [(0.0, 2.0, "hi there", None)], fb           # blank segment dropped
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    transcribe_aligned("fake.flac", language="yo")
    assert len(warnings) == 1, [r.getMessage() for r in warnings]

    del sys.modules["whisperx"]
    _ASR.clear(); _ALIGN.clear()
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        print("error: only --selfcheck is implemented as a CLI entry point "
              "(real runs need whisperx installed on the GPU box)", file=sys.stderr)
        sys.exit(1)
