# Phase 23 timing harness: warm models on audio.wav, then time run_pipeline_wx(test.mp3) with peak VRAM.
import asyncio, os, subprocess, sys, time, threading
import diarize_demo as dd
peak = [0]
def mon(stop):
    while not stop.is_set():
        m = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip()
        peak[0] = max(peak[0], int(m or 0)); time.sleep(2)
asyncio.run(dd.run_pipeline_wx("audio.wav"))
stop = threading.Event(); th = threading.Thread(target=mon, args=(stop,), daemon=True); th.start()
t = time.perf_counter()
lines, _ = asyncio.run(dd.run_pipeline_wx(sys.argv[1] if len(sys.argv) > 1 else "test.mp3"))
wall = time.perf_counter() - t
stop.set(); th.join()
knobs = {k: v for k, v in os.environ.items() if k.startswith(("WHISPERX_", "PYANNOTE_"))}
print(f"RESULT pipeline {wall:.1f}s | {len(lines)} lines | {len({l[2] for l in lines})} speakers | peak VRAM {peak[0]} MiB | {knobs}")
