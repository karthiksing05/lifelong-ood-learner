"""Run every `tests/test_*.py` and print one line per test.

`reproduce.sh tests` runs these, but only as part of a much longer pipeline, and several of them need
a GPU -- on this cluster a bare `python tests/test_x.py` on the login node gets its torch process
SIGTERMed, which looks like a hang rather than a failure.  So the suite has to go through the batch
system, and this is the entry point for that:

    sbatch --export=ALL,SCRIPT="tests/run_all.py" experiments/run_gpu.sh

Each test is run in its own process, so one crash does not take the rest of the suite with it, and
the last line of its output is echoed -- these tests print their own verdict as a final line.
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    os.chdir(ROOT)
    py = sys.executable
    tests = sorted(glob.glob(os.path.join("tests", "test_*.py")))
    fails, errs = [], {}
    for t in tests:
        try:
            r = subprocess.run([py, t], capture_output=True, text=True, timeout=2400)
            out = (r.stdout or "").strip().split("\n")
            last = next((l for l in reversed(out) if l.strip()), "(no output)")
            ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            ok, last = False, "TIMEOUT after 40 min"
            r = None
        if not ok:
            fails.append(t)
            errs[t] = (r.stderr or "").strip().split("\n")[-6:] if r else ["timeout"]
        print(f"{'PASS' if ok else 'FAIL'}  {os.path.basename(t):32s} {last[:104]}", flush=True)
    print(f"\n[run_all] {len(tests) - len(fails)}/{len(tests)} passed")
    for t in fails:
        print(f"\n--- {t} stderr tail ---")
        for line in errs[t]:
            print(f"    {line}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
