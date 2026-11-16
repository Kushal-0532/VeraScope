import socket

import pytest


def test_socket_blocked():
    with pytest.raises(RuntimeError, match="network blocked in tests"):
        socket.create_connection(("example.com", 80))


def test_home_redirected(home):
    import os
    assert os.environ["HOME"] == str(home)
    assert "GROQ_API_KEY" not in os.environ
