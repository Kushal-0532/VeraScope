"""Mock-based test for transcribe.py. No network, no API key."""

import asyncio
from unittest.mock import AsyncMock

import pytest

import transcribe as T


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


def test_transcribe_file_extracts_word_tuples(tmp_path):
    fake_data = {
        "words": [
            {"word": "Hello,", "start": 0.02, "end": 0.4},
            {"word": " ", "start": 0.4, "end": 0.41},  # whitespace-only, dropped
            {"word": "world", "start": 0.5, "end": 0.9},
        ],
        "segments": [],
    }
    client = AsyncMock()
    client.audio.transcriptions.create = AsyncMock(return_value=_FakeResp(fake_data))

    path = tmp_path / "fake.wav"
    path.write_bytes(b"not real audio")

    words = asyncio.run(T.transcribe_file(client, str(path)))

    assert words == [(0.02, 0.4, "Hello,"), (0.5, 0.9, "world")]
    assert all(isinstance(w, tuple) and len(w) == 3 for w in words)


def test_transcribe_file_falls_back_to_segments(tmp_path):
    fake_data = {
        "words": None,
        "segments": [
            {"words": [{"word": "from", "start": 0.0, "end": 0.2}]},
            {"words": [{"word": "segments", "start": 0.2, "end": 0.6}]},
        ],
    }
    client = AsyncMock()
    client.audio.transcriptions.create = AsyncMock(return_value=_FakeResp(fake_data))

    path = tmp_path / "fake.wav"
    path.write_bytes(b"not real audio")

    words = asyncio.run(T.transcribe_file(client, str(path)))

    assert words == [(0.0, 0.2, "from"), (0.2, 0.6, "segments")]


def test_transcribe_one_dies_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(T, "load_dotenv", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        asyncio.run(T.transcribe_one("audio.wav"))


if __name__ == "__main__":
    import pathlib
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_transcribe_file_extracts_word_tuples(pathlib.Path(d))
        test_transcribe_file_falls_back_to_segments(pathlib.Path(d))
    print("selfcheck ok")
