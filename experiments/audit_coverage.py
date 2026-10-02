"""Which numerals in the audited documents are NOT re-derived by a check in audit_numbers.py.

audit_numbers answers "does every number I check match?"; this answers "am I checking every number?".
Numerals that are configuration rather than results (years, section numbers, small integers, sizes)
are in IGNORE; anything else uncovered is either a result nobody checks or a number that should not
be in the document.
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import audit_numbers as A                                              # noqa: E402

IGNORE = {
    # confidence levels
    "95",
    # years, dates, section / rung / phase numbers, small counts
    "2026", "09", "25", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "24", "32", "64",
    # sizes and settings stated as such
    "72", "1024", "6.8", "6.8M", "276", "255", "3072", "0.98", "0.05", "0.9", "0.10", "0.20", "0.30", "0.40", "0.55", "0.70",
    "512", "150", "0.1", "50", "1e-5", "800", "0.8", "0.95",
    # the phase-0 criterion and its per-factor retry (settings), the tree's node count (a build setting)
    "104", "0.7", "0.2", "0.5",
    # embodied: noise levels named as settings; simulator-rendered placements per task (a setting)
    "0.4", "400",
    # embodied: attention masses tried as settings (q21_dose)
    "0.85",
}
NUM = re.compile(r"(?<![A-Za-z0-9_./#-])(\d[\d,]*(?:\.\d+)?)(?![A-Za-z0-9_./-])")


def uncovered(paths=None):
    quoted = {q for _, _, q in A.CHECKS}
    out = []
    for p in (paths or A.DOCS):
        if not os.path.exists(p):
            continue
        for ln, line in enumerate(open(p, encoding="utf-8").read().splitlines(), 1):
            if line.strip().startswith(("|---", "```")):
                continue
            for m in NUM.finditer(line):
                tok = m.group(1)
                if tok in IGNORE or tok in quoted or tok.replace(",", "") in IGNORE:
                    continue
                out.append((os.path.relpath(p, ROOT), ln, tok, line.strip()[:100]))
    return out


def main():
    u = uncovered()
    for p, ln, tok, ctx in u:
        print(f"{p}:{ln}: {tok}    | {ctx}")
    print(f"{len(u)} uncovered numerals")
    return 0 if not u else 1


if __name__ == "__main__":
    sys.exit(main())
