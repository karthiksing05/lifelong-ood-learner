"""The ledger is the only source a number may be quoted from, so its contract is pinned here.

numpy-only: runs on the login node in under a second.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                       # noqa: E402

BYTES = dict(experts=100, tree=10, codes=1, samples=0, teacher_copies=0, total=111)


def _row(tmp, seed, arm, val, **kw):
    return ledger.record("unit", metrics={"taught_new": val, "nested": {"x": val * 2}}, bytes=BYTES,
                         config={"lr": 1e-4}, seed=seed, bench="colormnist", script="t.py", command="t",
                         keys={"arm": arm}, ledger_dir=tmp, **kw)


def test_refuses_a_row_without_a_full_bytes_block(tmp_path):
    with pytest.raises(ValueError):
        ledger.record("unit", metrics={}, bytes={"total": 1}, config={}, seed=0, bench="b", script="s",
                      command="c", ledger_dir=tmp_path)
    with pytest.raises(ValueError):
        ledger.record("unit", metrics={}, bytes=None, config={}, seed=0, bench="b", script="s",
                      command="c", ledger_dir=tmp_path)
    with pytest.raises(ValueError):
        ledger.record("unit", metrics={}, bytes=dict(BYTES, total=-1), config={}, seed=0, bench="b",
                      script="s", command="c", ledger_dir=tmp_path)


def test_record_writes_schema_and_provenance(tmp_path):
    p = _row(tmp_path, 0, "a", 0.5)
    assert p.exists() and p.parent.name == "unit"
    row = json.loads(p.read_text())
    for k in ("schema", "run_id", "git", "env", "metrics", "bytes", "config", "seed", "started"):
        assert k in row
    assert row["schema"] == ledger.SCHEMA and row["bytes"]["total"] == 111
    assert isinstance(row["git"]["hash"], str) and len(row["git"]["short"]) == 8
    assert row["keys"] == {"arm": "a"} and row["retired"] is None


def test_load_filters_and_mean_with_dotted_keys(tmp_path):
    for s, v in zip((0, 1, 2), (0.70, 0.72, 0.74)):
        _row(tmp_path, s, "a", v)
    _row(tmp_path, 0, "b", 0.10)
    rows = ledger.load("unit", ledger_dir=tmp_path, keys__arm="a")
    assert len(rows) == 3
    m, se, n = ledger.mean("unit", "metrics.taught_new", ledger_dir=tmp_path, keys__arm="a")
    assert n == 3 and abs(m - 0.72) < 1e-12
    assert abs(se - np.std([0.70, 0.72, 0.74], ddof=1) / np.sqrt(3)) < 1e-12
    m2, _, n2 = ledger.mean("unit", "metrics.nested.x", ledger_dir=tmp_path, keys__arm="b")
    assert n2 == 1 and abs(m2 - 0.2) < 1e-12
    assert ledger.mean("unit", "metrics.missing", ledger_dir=tmp_path)[2] == 0
    assert ledger.mean("nope", "metrics.taught_new", ledger_dir=tmp_path)[2] == 0
    assert len(ledger.load("unit", ledger_dir=tmp_path, seed=[0, 1])) == 3


def test_retired_rows_are_hidden_by_default(tmp_path):
    p = _row(tmp_path, 0, "a", 0.5)
    ledger.retire(p, "superseded")
    assert ledger.load("unit", ledger_dir=tmp_path) == []
    assert len(ledger.load("unit", ledger_dir=tmp_path, include_retired=True)) == 1


def test_run_id_is_deterministic_and_keyed(tmp_path):
    a = ledger.run_id("e", "b", "t", 1, "abcdef12", {"arm": "x"})
    b = ledger.run_id("e", "b", "t", 1, "abcdef12", {"arm": "x"})
    c = ledger.run_id("e", "b", "t", 1, "abcdef12", {"arm": "y"})
    assert a == b != c and a.endswith("__s1__abcdef12")
    p1 = _row(tmp_path, 0, "a", 0.5)
    p2 = _row(tmp_path, 0, "a", 0.6)               # same key, same seed: overwrite, one file
    assert p1 == p2 and len(ledger.load("unit", ledger_dir=tmp_path)) == 1


def test_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"lifelong" * 1000)
    assert ledger.sha256_file(f) == hashlib.sha256(b"lifelong" * 1000).hexdigest()
