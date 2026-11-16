"""Every module's `--selfcheck` entry point, run in a scrubbed subprocess."""
import subprocess
import sys

import pytest

from conftest import ROOT, scrubbed_env

MODULES = sorted(p.name for p in ROOT.glob("*.py") if "--selfcheck" in p.read_text())


@pytest.mark.parametrize("mod", MODULES)
def test_selfcheck(mod, home):
    r = subprocess.run([sys.executable, mod, "--selfcheck"], cwd=ROOT, env=scrubbed_env(home),
                       capture_output=True, text=True, timeout=110)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "ok" in r.stdout, r.stdout[-2000:]
