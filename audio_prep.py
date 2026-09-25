#!/usr/bin/env python3
"""ffmpeg audio prep: any input -> full-length 16kHz mono FLAC + overlapping chunks.

librosa decodes into an in-memory float32 array (~230MB for 60min) and can't
open video containers. ffmpeg works on files instead, handles video, and is
the only route to chunk-splitting.

Usage: python3 audio_prep.py --selfcheck
"""

import os
import subprocess
import sys

CHUNK_S = 600
OVERLAP_S = 10
MAX_CHUNK_BYTES = 25 * 1024 * 1024


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def to_flac(src, dst):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", src,
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", dst],
        capture_output=True, text=True, check=True,
    )
    return dst


def plan_chunks(duration, chunk_s=CHUNK_S, overlap_s=OVERLAP_S):
    """-> [(start, end)]. Short input = one chunk. No tail shorter than overlap_s."""
    if duration <= chunk_s:
        return [(0.0, duration)]

    step = chunk_s - overlap_s
    plan = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_s, duration)
        if plan and duration - start < overlap_s:
            # tail too short to stand alone: extend the previous chunk instead
            prev_start, _ = plan[-1]
            plan[-1] = (prev_start, duration)
            break
        plan.append((start, end))
        start += step
    return plan


def split(flac_path, plan, outdir):
    os.makedirs(outdir, exist_ok=True)
    out = []
    for i, (start, end) in enumerate(plan):
        chunk_path = os.path.join(outdir, f"chunk_{i:03d}.flac")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-ss", str(start), "-t", str(end - start),
             "-i", flac_path, "-ac", "1", "-ar", "16000", "-c:a", "flac", chunk_path],
            capture_output=True, text=True, check=True,
        )
        size = os.path.getsize(chunk_path)
        print(f"chunk {i}: {start:.1f}-{end:.1f}s  {size/1e6:.2f}MB", file=sys.stderr)
        if size >= MAX_CHUNK_BYTES:
            die(f"{chunk_path} is {size/1e6:.2f}MB, over the 25MB limit "
                "(Prior Decision 7: fall back to Opus if this fires for real)")
        out.append((chunk_path, start))
    return out


def _selfcheck():
    assert plan_chunks(500) == [(0.0, 500)]
    assert plan_chunks(600) == [(0.0, 600)]

    plan = plan_chunks(3600)
    assert plan[-1][1] == 3600.0
    for i in range(1, len(plan)):
        prev_start, prev_end = plan[i - 1]
        start, end = plan[i]
        assert start < prev_end, "chunks must overlap"
        assert abs(prev_end - start - OVERLAP_S) < 1e-9, (prev_end, start)
    for start, end in plan:
        assert end - start >= OVERLAP_S

    # duration that would otherwise leave a short tail chunk
    plan2 = plan_chunks(605)
    assert plan2[-1][1] == 605.0
    for start, end in plan2:
        assert end - start >= OVERLAP_S

    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        die("only --selfcheck is implemented as a CLI entry point; "
            "import probe_duration/to_flac/plan_chunks/split for real use")
