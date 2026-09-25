"""Regenerate paper/numbers.json from the ledger: every audited quantity with its value, se and n.

    python experiments/make_numbers.py            # writes paper/numbers.json
    python experiments/make_numbers.py --check    # exits 1 if the file on disk differs
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import audit_numbers as A                                              # noqa: E402

OUT = os.path.join(ROOT, "paper", "numbers.json")


def build():
    d = {}
    for label, fn, quoted in A.CHECKS:
        m, se, n = fn()
        d[label] = dict(value=None if n == 0 else float(m), se=None if n == 0 else float(se), n=int(n),
                        quoted=quoted, formatted=None if n == 0 else A.fmt(m, quoted))
    return d


def main():
    d = build()
    text = json.dumps(d, indent=1, sort_keys=True) + "\n"
    if "--check" in sys.argv:
        cur = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        print("numbers.json up to date" if cur == text else "numbers.json DIFFERS from the ledger")
        return 0 if cur == text else 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(text)
    print(f"wrote {OUT} ({len(d)} quantities)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
