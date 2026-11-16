"""Session-wide test isolation: no keys, no network, no model downloads."""
import os
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KEYS = ["GROQ_API_KEY", "TAVILY_API_KEY", "HF_TOKEN", "PYANNOTEAI_API_KEY"]


def scrubbed_env(home):
    """Env for subprocesses: no keys, no VERASCOPE_/WHISPERX_ switches, HOME redirected
    (modules call load_dotenv(~/.env), which would silently restore keys)."""
    env = {k: v for k, v in os.environ.items()
           if k not in KEYS and not k.startswith(("VERASCOPE_", "WHISPERX_"))}
    env.update(HOME=str(home), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HTTPS_PROXY="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9")
    return env


def _blocked(*a, **k):
    raise RuntimeError("network blocked in tests")


@pytest.fixture(scope="session", autouse=True)
def offline(tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    mp = pytest.MonkeyPatch()
    for k in list(os.environ):
        if k in KEYS or k.startswith(("VERASCOPE_", "WHISPERX_")):
            mp.delenv(k)
    mp.setenv("HOME", str(home))
    mp.setenv("HF_HUB_OFFLINE", "1")
    mp.setenv("TRANSFORMERS_OFFLINE", "1")
    mp.setattr(socket.socket, "connect", _blocked)
    mp.setattr(socket, "getaddrinfo", _blocked)
    yield home
    mp.undo()


@pytest.fixture
def home(offline):
    return offline
