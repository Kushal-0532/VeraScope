"""Speech data + audio helpers shared by E1, E2, E3, E8, E9.

AMI:  audio  Edinburgh mirror, Mix-Headset (the pyannote benchmark condition)
      RTTM/UEM/lists  BUTSpeechFIT/AMI-diarization-setup, only_words
      words  ami_public_manual_1.6.2.zip (for WER references)
VoxConverse: HF diarizers-community/voxconverse (parquet, streamed; audio
      written to disk undecoded, so no torchcodec dependency).
"""

import io
import json
import re
import subprocess
import wave
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

from experiments.common import DATA, fetch, log, read_rttm

SR = 16000
AMI_SETUP = "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main/"
AMI_AUDIO = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/{m}/audio/{m}.Mix-Headset.wav"
AMI_WORDS = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AMI_LICENSE = "CC BY 4.0"
VOX_ID = "diarizers-community/voxconverse"
VOX_LICENSE = "CC BY 4.0"


# ---------------------------------------------------------------- audio
def load16k(path, start=None, dur=None):
    """Any file -> float32 mono 16 kHz via ffmpeg (what whisperx.load_audio does)."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-f", "s16le", "-ac", "1", "-ar", str(SR), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def write_wav(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    return float(out.stdout.strip())


def overlap_timeline(turns):
    """Regions where >= 2 reference speakers talk -> [(start, end)]."""
    pts = sorted([(s, 1) for s, _, _ in turns] + [(e, -1) for _, e, _ in turns],
                 key=lambda p: (p[0], p[1]))
    out, c, lo = [], 0, None
    for t, d in pts:
        c += d
        if c >= 2 and lo is None:
            lo = t
        elif c < 2 and lo is not None:
            if t > lo:
                out.append((lo, t))
            lo = None
    return out


def in_regions(t, regions):
    import bisect
    i = bisect.bisect_right([a for a, _ in regions], t) - 1
    return i >= 0 and regions[i][0] <= t < regions[i][1]


# ---------------------------------------------------------------- AMI
def ami_meetings(split):
    p = fetch(AMI_SETUP + f"lists/{split}.meetings.txt", DATA / "ami" / f"{split}.meetings.txt")
    return [m.strip() for m in p.read_text().split() if m.strip()]


def ami_rttm(m, split):
    p = fetch(AMI_SETUP + f"only_words/rttms/{split}/{m}.rttm", DATA / "ami" / "rttm" / f"{m}.rttm")
    turns = read_rttm(p)
    return turns.get(m) or next(iter(turns.values()))


def ami_uem(m, split):
    """(start, end) from the setup's UEM (dev/test only), else None."""
    try:
        p = fetch(AMI_SETUP + f"uems/{split}/{m}.uem", DATA / "ami" / "uem" / f"{m}.uem")
        f = p.read_text().split()
        return float(f[2]), float(f[3])
    except Exception:
        return None


def ami_audio(m):
    return fetch(AMI_AUDIO.format(m=m), DATA / "ami" / "audio" / f"{m}.Mix-Headset.wav")


_ZIP = None


def ami_words(m):
    """Reference words -> [(start, end, word, speaker_letter)] sorted by start.
    Only <w> elements with times; punctuation tokens dropped."""
    global _ZIP
    if _ZIP is None:
        _ZIP = zipfile.ZipFile(fetch(AMI_WORDS, DATA / "ami" / "ami_public_manual_1.6.2.zip"))
    out, skipped = [], 0
    for name in _ZIP.namelist():
        g = re.match(rf"(?:.*/)?words/{re.escape(m)}\.([A-Z])\.words\.xml$", name)
        if not g:
            continue
        for el in ET.parse(io.BytesIO(_ZIP.read(name))).getroot().iter():
            if el.tag.split("}")[-1] != "w" or not (el.text or "").strip():
                continue
            if el.get("punc") == "true":
                continue
            s, e = el.get("starttime"), el.get("endtime")
            if s is None or e is None:
                skipped += 1
                continue
            out.append((float(s), float(e), el.text.strip(), g.group(1)))
    if skipped:
        log(f"ami_words {m}: {skipped} untimed words skipped")
    return sorted(out)


# ---------------------------------------------------------------- VoxConverse
def vox_files(split, n, min_dur=0.0, max_scan=None):
    """First n files of the split with duration >= min_dur, written to
    DATA/vox/<split>/. -> [(uri, path, turns, dur)]. Manifest-cached."""
    root = DATA / "vox" / split
    man = root / f"manifest_{n}_{int(min_dur)}.json"
    if man.exists():
        rows = json.loads(man.read_text())
        return [(u, root / f"{u}.wav", [tuple(t) for t in turns], d) for u, turns, d in rows]
    from datasets import Audio, load_dataset
    ds = load_dataset(VOX_ID, split=split, streaming=True).cast_column("audio", Audio(decode=False))
    rows = []
    for i, ex in enumerate(ds):
        if len(rows) >= n or (max_scan and i >= max_scan):
            break
        a = ex["audio"]
        uri = re.sub(r"\W", "", (a.get("path") or "").rsplit("/", 1)[-1].rsplit(".", 1)[0]) or f"{split}{i:03d}"
        raw = root / f"{uri}.raw"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(a["bytes"])
        wav = write_wav(root / f"{uri}.wav", load16k(raw))
        raw.unlink()
        d = duration(wav)
        if d < min_dur:
            wav.unlink()
            continue
        turns = sorted(zip(map(float, ex["timestamps_start"]), map(float, ex["timestamps_end"]),
                           map(str, ex["speakers"])))
        rows.append((uri, turns, d))
        log(f"vox {split}: {uri} {d / 60:.1f} min, {len({t[2] for t in turns})} spk")
    man.write_text(json.dumps(rows))
    return [(u, root / f"{u}.wav", turns, d) for u, turns, d in rows]


def _selfcheck():
    turns = [(0, 5, "A"), (4, 6, "B"), (5.5, 7, "C"), (10, 12, "A"), (11, 11.5, "B")]
    assert overlap_timeline(turns) == [(4, 5), (5.5, 6), (11, 11.5)], overlap_timeline(turns)
    reg = overlap_timeline(turns)
    assert in_regions(4.5, reg) and not in_regions(5.2, reg) and in_regions(11.2, reg)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        x = np.sin(np.arange(SR * 2) / 10).astype(np.float32) * 0.5
        p = write_wav(Path(d) / "a.wav", x)
        y = load16k(p, start=0.5, dur=1.0)
        assert len(y) == SR and abs(duration(p) - 2.0) < 0.01
    print("speech selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
