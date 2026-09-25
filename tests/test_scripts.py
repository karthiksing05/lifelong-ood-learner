"""Run every script-style test as a subprocess and assert it exits 0, so `pytest tests -q` covers them.

test_fid pulls Inception through on CPU and takes minutes; the login node has been known to SIGTERM
long torch processes, so run the suite through slurm/jobs/cpu.sbatch when in doubt.
"""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from conftest import SCRIPT_TESTS                                     # noqa: E402


SLOW = {"test_feature_space.py", "test_fid.py"}       # Inception on CPU: minutes on a free node, hours on a shared one


@pytest.mark.parametrize("script", SCRIPT_TESTS)
def test_script_exits_zero(script):
    if script in SLOW and os.environ.get("LOL_SLOW", "1") == "0":
        pytest.skip("LOL_SLOW=0: the Inception-based scripts are skipped")
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"), PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, os.path.join(HERE, script)], capture_output=True, text=True,
                       timeout=3600 if script in SLOW else 1800, env=env, cwd=os.path.dirname(HERE))
    assert r.returncode == 0, f"{script} failed (rc={r.returncode})\n--- stdout ---\n{r.stdout[-3000:]}\n--- stderr ---\n{r.stderr[-3000:]}"
