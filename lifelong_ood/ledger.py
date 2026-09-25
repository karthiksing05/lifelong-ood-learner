"""The results ledger: one small, tracked JSON per run, and the only source a number may be quoted from.

WHY.  `results/*.json` is gitignored and `results/aggregate.json` was rebuilt by hand, so the review of
2026-09-25 found headline numbers in the README that no result file on disk produced, a test suite that
passed against a stale aggregate, and 166 result files that were never under version control.  A ledger
row is written by the experiment itself at the end of a run, carries the git hash, the command, the seed,
the metrics and -- mandatorily -- the bytes the method charged, and is small enough to track.  Tables,
figures and every number in a document are regenerated from `ledger/` alone (see
experiments/audit_numbers.py); the full per-run payload stays in `results/` and is referenced by sha256.

CONVENTIONS.
  * One row per (experiment, bench, tag, seed[, keys]).  `keys` holds small grouping fields such as the
    arm or the expert variant, so a table with several arms per run writes several rows.
  * `bytes` is the accounting block from `lifelong_ood.accounting.charge` and MUST contain every key in
    BYTES_KEYS; `record` refuses a row without it.  Legacy (backfilled) rows say what was not charged.
  * Dotted paths address nested fields everywhere: `mean("c1", "metrics.taught_new", keys__arm="x")`.
    In keyword filters a double underscore stands for the dot.
  * A row is never deleted; `retire(path, reason)` marks it and `load` hides it by default.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import pathlib
import platform
import re
import socket
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "ledger"
RESULTS = ROOT / "results"          # untracked per-run payloads; the ledger references them by sha256
CACHE = RESULTS / "cache"
SCHEMA = 1
BYTES_KEYS = ("experts", "tree", "codes", "samples", "teacher_copies", "total")


# ── provenance ────────────────────────────────────────────────────────────────
def git_info(root=ROOT):
    """-> {hash, short, branch, dirty}.  Never raises: a missing git is recorded, not fatal."""
    def _run(*args):
        try:
            return subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception:                                   # pragma: no cover - no git on the box
            return ""
    h = _run("rev-parse", "HEAD")
    return dict(hash=h or "unknown", short=(h or "unknown")[:8],
                branch=_run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
                dirty=bool(_run("status", "--porcelain", "--untracked-files=no")))


def env_info():
    """Python, torch and device versions plus where the run happened.  torch is imported lazily
    so the ledger stays usable on the login node and in numpy-only tests."""
    out = dict(python=platform.python_version(), host=socket.gethostname(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), torch=None, device=None)
    try:
        import torch                                        # noqa: WPS433
        out["torch"] = torch.__version__
        out["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    return out


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-") or "x"


def run_id(experiment, bench, tag, seed, git_short, keys=None):
    parts = [_slug(experiment), _slug(bench), _slug(tag or "base")]
    if keys:
        parts += [_slug(f"{k}={v}") for k, v in sorted(keys.items())]
    parts += [f"s{int(seed)}", _slug(git_short)]
    return "__".join(parts)


# ── writing ───────────────────────────────────────────────────────────────────
def _check_bytes(b):
    if not isinstance(b, dict):
        raise ValueError("ledger.record: `bytes` must be the accounting dict, got %r" % type(b))
    missing = [k for k in BYTES_KEYS if k not in b]
    if missing:
        raise ValueError(f"ledger.record: `bytes` is missing {missing}; every stored byte must be "
                         "charged (see lifelong_ood.accounting.charge)")
    for k in BYTES_KEYS:
        v = b[k]
        if not isinstance(v, (int, np.integer)) or int(v) < 0:
            raise ValueError(f"ledger.record: bytes[{k!r}] must be a non-negative int, got {v!r}")
    return {k: int(b[k]) for k in BYTES_KEYS} | {k: v for k, v in b.items() if k not in BYTES_KEYS}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pathlib.Path):
        return str(o)
    return o


def record(experiment, *, metrics, bytes, config, seed, bench, script, command, tag="", keys=None,
           stream=None, backbone=None, artifacts=None, started=None, seconds=None, backfilled=False,
           source_commit=None, notes=None, ledger_dir=None, overwrite=True):
    """Write one ledger row and return its path.  Refuses a row without a full `bytes` block."""
    b = _check_bytes(bytes)
    g = git_info()
    if g["dirty"] and not backfilled:
        print(f"[ledger] WARNING: working tree is dirty; row for {experiment} carries dirty=true",
              file=sys.stderr)
    rid = run_id(experiment, bench, tag, seed, g["short"] if not backfilled else (source_commit or g["short"])[:8],
                 keys)
    row = dict(schema=SCHEMA, run_id=rid, experiment=experiment, bench=bench, tag=tag or "", seed=int(seed),
               keys=dict(keys or {}), script=script, command=command, git=g, env=env_info(),
               started=started or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
               seconds=None if seconds is None else float(seconds), config=_jsonable(config),
               stream=_jsonable(stream), backbone=_jsonable(backbone), metrics=_jsonable(metrics), bytes=b,
               artifacts=_jsonable(artifacts or {}), backfilled=bool(backfilled),
               source_commit=source_commit, notes=notes, retired=None)
    d = pathlib.Path(ledger_dir or LEDGER) / _slug(experiment)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{rid}.json"
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.write_text(json.dumps(row, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def retire(path, reason):
    """Mark a row retired (kept on disk, hidden from `load`)."""
    p = pathlib.Path(path)
    row = json.loads(p.read_text(encoding="utf-8"))
    row["retired"] = dict(reason=str(reason), when=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    p.write_text(json.dumps(row, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return p


# ── reading ───────────────────────────────────────────────────────────────────
def get(row, key, default=None):
    """Dotted-path lookup: get(row, "metrics.new-value.joint")."""
    cur = row
    for part in str(key).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _match(row, filters):
    for k, want in filters.items():
        have = get(row, k.replace("__", "."), default=None)
        if isinstance(want, (list, tuple, set)):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


def load(experiment=None, ledger_dir=None, include_retired=False, **filters):
    """Every row of an experiment (or all), newest file last, filtered by dotted keys."""
    base = pathlib.Path(ledger_dir or LEDGER)
    dirs = [base / _slug(experiment)] if experiment else [p for p in base.iterdir() if p.is_dir()] if base.exists() else []
    rows = []
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            row = json.loads(f.read_text(encoding="utf-8"))
            row["_path"] = str(f)
            if row.get("retired") and not include_retired:
                continue
            if _match(row, filters):
                rows.append(row)
    return rows


def mean(experiment, metric, ledger_dir=None, **filters):
    """-> (mean, standard error, n) of a dotted metric over the matching rows (one per seed).
    Missing values are skipped; n counts the rows that carried the metric."""
    vals = [get(r, metric) for r in load(experiment, ledger_dir=ledger_dir, **filters)]
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return (float("nan"), float("nan"), 0)
    v = np.asarray(vals, np.float64)
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return (float(v.mean()), se, int(len(v)))


def fmt(m, se, n, dp=3):
    return "n/a" if n == 0 else f"{m:.{dp}f}" + (f" +/- {se:.{dp}f}" if n > 1 else "") + f" (n={n})"
