# VeraScope

Capstone project: automatic fact-checking of spoken audio and video.

Pipeline:

1. Transcribe audio and align words to timestamps (hosted Groq whisper-large-v3-turbo,
   or local WhisperX on a GPU).
2. Diarize speakers (hosted pyannoteAI, or local pyannote 3.1) and merge words and
   turns into a speaker-labelled, timestamped transcript with a confidence per line.
3. Detect check-worthy claims in the transcript with an LLM (Groq gpt-oss-20b,
   strict JSON output).
4. Retrieve web evidence for each claim (Tavily) and score it with NLI (mDeBERTa
   XNLI via the HF Inference API): supported / disputed / unclear.
5. Show the transcript and claims in Transcript / Claims tabs in a Streamlit UI.

Run the UI:

    streamlit run app.py                      # hosted Stage 1 (Groq + pyannoteAI)
    VERASCOPE_STAGE3=1 streamlit run app.py   # local WhisperX Stage 1 (GPU)

Needs `ffmpeg` on the PATH. API keys are read from `~/.env` (a real environment
variable wins):

- hosted Stage 1: `GROQ_API_KEY`, `PYANNOTEAI_API_KEY`
- local Stage 1: `HF_TOKEN`, from an account that accepted the gated
  `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` models
- claims: `GROQ_API_KEY` to detect them; `TAVILY_API_KEY` and `HF_TOKEN` to
  verify them. Without `TAVILY_API_KEY` the Verify button is disabled;
  transcription still works.

There is no requirements file. `onbox.sh` is the install used on the GPU box
(whisperx, streamlit, groq, tavily-python, python-dotenv, `huggingface_hub>=1.5`,
`transformers>=5`). The hosted path also needs `pyannoteai-sdk`.

Command line, without the UI:

    python diarize_demo.py audio.wav                        # hosted
    python diarize_demo.py audio.wav --whisper-arch medium  # local WhisperX

Status: Stages 1 and 2 work on the hosted stack. The local WhisperX Stage 1 runs on
an AWS g5.xlarge (A10G): the 48:46 demo file goes through Stage 1 in 96.4 s, down
from 289.9 s on a T4 before the speed work. The hosted code is still in the repo
alongside the local code; removing it is the last open step of the Stage 3 plan.

Experiments: `experiments/` holds a benchmark suite (E1 to E9) that calls the repo's
own pipeline code on public data (AMI, VoxConverse, FEVER). `build_notebook.py`
packs it into a notebook; `aws/aws_run.sh` runs that notebook headless on a
g5.xlarge. The full run (25 Sep 2026) gave, among others: whole-file pyannote 3.1
DER 9.3% over 26 files vs 52.4% when 600 s chunks are diarized separately (E1);
WhisperX WER 24.2% (medium) / 23.8% (large-v3) on 16 AMI test meetings (E3); NLI
verdict macro-F1 0.694 at the app's 0.85 / 0.15 thresholds on 1,500 held-out FEVER
claims (E4). E6, E7 and E8 produced no results. Outputs are in
`experiments/notebooks/` (not committed).

Layout: `app.py` (UI), `audio_prep.py` / `transcribe.py` / `stitch.py` (hosted
speech), `wx_transcribe.py` (WhisperX), `diarize_demo.py` (diarization, fusion,
pipeline entry points), `claim_detect.py` / `retrieve_evidence.py` /
`verify_claims.py` / `verify_pipeline.py` (fact-checking), `experiments/`
(evaluation suite), `onbox.sh` (GPU box setup), `onbox_check.py` (on-box
acceptance checks), `time23.py` (timing harness), `make_artifact.py` (saves a run
so the UI can replay it with `VERASCOPE_ARTIFACT=<json>`, no GPU or API needed),
`verify_ui_apptest.py` / `test_transcribe.py` (tests). Most modules take
`--selfcheck`.

Specs, phase notes, the hand-off doc, notebooks and audio are kept locally and are
git-ignored. This repo was started on 25 Sep 2026 from a copy of
`capstone_project/`, whose own git history (15 commits, 22 Aug to 9 Sep 2026) holds
the development log.
