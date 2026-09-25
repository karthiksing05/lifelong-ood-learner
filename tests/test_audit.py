"""The number audit passes on the real documents, catches a perturbed number, and covers every
numeral.  numpy + json only."""
import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import audit_coverage as C                                             # noqa: E402
import audit_numbers as A                                              # noqa: E402


def test_number_audit_passes(capsys):
    assert A.main() == 0
    assert "0 discrepancies" in capsys.readouterr().out


def test_perturbed_number_is_caught(tmp_path, monkeypatch):
    src = os.path.join(ROOT, "README.md")
    dst = tmp_path / "README.md"
    text = open(src, encoding="utf-8").read()
    assert "0.771" in text
    dst.write_text(text.replace("0.771", "0.871"), encoding="utf-8")
    monkeypatch.setattr(A, "DOCS", [str(dst), os.path.join(ROOT, "docs", "THE_TASK.md")])
    assert A.run(verbose=False) >= 1


def test_every_numeral_is_covered():
    u = C.uncovered()
    assert not u, "\n".join(f"{p}:{ln}: {tok} | {ctx}" for p, ln, tok, ctx in u)


def test_numbers_json_regenerates_byte_identically(tmp_path, monkeypatch):
    import make_numbers as M
    monkeypatch.setattr(M, "OUT", str(tmp_path / "numbers.json"))
    sys.argv = ["make_numbers.py"]
    assert M.main() == 0
    sys.argv = ["make_numbers.py", "--check"]
    assert M.main() == 0
