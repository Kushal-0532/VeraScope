# VeraScope
### AI-Powered Audio Transcription and Fact-Checking

VeraScope is an AI-powered fact-checking application that analyzes audio and video recordings to identify and verify factual claims. It combines speech recognition, speaker diarization, large language models, and external evidence retrieval to provide speaker-labelled transcripts and claim verification results.

## Features


- **Speech Transcription:** Converts audio and video recordings into text using Whisper-based models.
- **Speaker Diarization:** Identifies individual speakers and associates them with transcript segments.
- **Claim Detection:** Uses LLMs to identify potentially fact-checkable statements.
- **Evidence Retrieval:** Searches external sources for information relevant to detected claims.
- **Claim Verification:** Classifies claims as supported, disputed, or unclear based on retrieved evidence.
- **Interactive Interface:** Provides a Streamlit dashboard to explore transcripts, claims, and verification results.

## Workflow

```text
Audio / Video Input
        |
        v
Audio Preprocessing
        |
        v
Speech Transcription
        |
        v
Speaker Diarization
        |
        v
Claim Detection
        |
        v
Evidence Retrieval
        |
        v
Claim Verification
        |
        v
Interactive Dashboard
```

## Technology Stack

| Component | Technologies |
|---|---|
| Language | Python |
| Interface | Streamlit |
| Speech Recognition | Whisper, WhisperX |
| Speaker Diarization | pyannote |
| Language Models | Groq API |
| Evidence Retrieval | Tavily API |
| Audio Processing | FFmpeg |

## Getting Started

### Prerequisites

- Python
- Git
- FFmpeg
- Required API credentials

### Installation

Clone the repository:

```bash
git clone https://github.com/Kushal-0532/VeraScope.git
cd VeraScope
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Install the required dependencies and configure the necessary API credentials in your environment.

### Run the Application

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit to access VeraScope.

## Project Structure

The repository includes modules for audio preprocessing, transcription, speaker diarization, claim detection, evidence retrieval, and verification, along with experimental scripts and the main Streamlit application.

## Limitations

VeraScope is an experimental fact-checking tool. Its results depend on transcription accuracy, speaker identification, language model outputs, and the relevance of retrieved evidence. Verification results should be reviewed alongside their supporting sources rather than treated as definitive conclusions.

## Future Scope

- Improve transcription and diarization accuracy.
- Expand multilingual support.
- Enhance claim detection and evidence retrieval.
- Improve verification reliability and processing efficiency.
