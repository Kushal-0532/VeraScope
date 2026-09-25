# VeraScope

Capstone project: automatic fact-checking of spoken audio.

Pipeline:

1. Transcribe audio (Whisper / WhisperX) and align words to timestamps.
2. Diarize speakers and stitch them into a speaker-labelled transcript.
3. Detect check-worthy claims in the transcript with an LLM (Groq).
4. Retrieve evidence for each claim and verify it.
5. Show transcript and claims side by side in a Streamlit UI.

Run the UI:

    streamlit run app.py

API keys (Groq, etc.) are read from `~/.env`.

Layout: `app.py` (UI), `transcribe.py` / `wx_transcribe.py` / `diarize_demo.py` /
`stitch.py` (speech), `claim_detect.py` / `retrieve_evidence.py` /
`verify_claims.py` / `verify_pipeline.py` (fact-checking), `experiments/`
(evaluation scripts), `onbox.sh` (GPU box run script).
