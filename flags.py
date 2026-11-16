"""Run-mode flags for local/offline testing.

Each flag only sets an env switch the pipeline already reads at call time, so
the CLI (diarize_demo.py) and the UI (app.py) share one definition:

    streamlit run app.py -- --whisperx-offline --whisperx-model tiny --nli-offline
    python diarize_demo.py audio.wav --whisperx-offline --no-diarize

Streamlit passes everything after `--` to the script (sys.argv).
"""

import argparse
import os
import sys

MODELS = ["tiny", "base", "small", "medium", "large-v3"]


def add_args(ap):
    ap.add_argument("--whisperx-offline", action="store_true",
                    help="transcribe with local WhisperX (+ local pyannote) on this machine "
                         "instead of Groq + pyannoteAI")
    ap.add_argument("--whisperx-model", "--whisper-arch", dest="whisperx_model",
                    choices=MODELS, default=None,
                    help="WhisperX model size; implies --whisperx-offline. Use tiny on a laptop")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None,
                    help="device for WhisperX and local NLI; default auto (cuda if available)")
    ap.add_argument("--nli-offline", action="store_true",
                    help="run the NLI verifier locally (transformers) instead of HF Inference")
    ap.add_argument("--no-diarize", action="store_true",
                    help="skip diarization: every line is SPEAKER_00 (no pyannote, no API)")
    return ap


def apply(args):
    """Translate parsed flags into the env switches. Explicit flags win over
    whatever ~/.env or the shell already set."""
    if args.whisperx_offline or args.whisperx_model or args.device:
        os.environ["VERASCOPE_STAGE3"] = "1"
    if args.whisperx_model:
        os.environ["WHISPERX_ARCH"] = args.whisperx_model
    if args.device:
        os.environ["WHISPERX_DEVICE"] = args.device
        os.environ["VERASCOPE_NLI_DEVICE"] = args.device
    if args.nli_offline:
        os.environ["VERASCOPE_LOCAL_NLI"] = "1"
    if args.no_diarize:
        os.environ["VERASCOPE_NO_DIARIZE"] = "1"
    return args


def parse_script_args(argv=None):
    """For app.py: parse only the flags above, ignore anything else."""
    ap = add_args(argparse.ArgumentParser(prog="app.py", add_help=False))
    args, _ = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    return apply(args)


def _selfcheck():
    keys = ["VERASCOPE_STAGE3", "WHISPERX_ARCH", "WHISPERX_DEVICE",
            "VERASCOPE_NLI_DEVICE", "VERASCOPE_LOCAL_NLI", "VERASCOPE_NO_DIARIZE"]
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        parse_script_args([])
        assert not any(k in os.environ for k in keys), "no flags must set nothing"
        parse_script_args(["--nli-offline", "--unrelated", "x"])
        assert os.environ["VERASCOPE_LOCAL_NLI"] == "1"
        assert "VERASCOPE_STAGE3" not in os.environ, "NLI flag must not switch ASR"
        parse_script_args(["--whisperx-model", "tiny", "--device", "cpu", "--no-diarize"])
        assert os.environ["VERASCOPE_STAGE3"] == "1"
        assert os.environ["WHISPERX_ARCH"] == "tiny"
        assert os.environ["WHISPERX_DEVICE"] == os.environ["VERASCOPE_NLI_DEVICE"] == "cpu"
        assert os.environ["VERASCOPE_NO_DIARIZE"] == "1"
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        add_args(argparse.ArgumentParser(description=__doc__.splitlines()[0])).print_help()
