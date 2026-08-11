"""Is any published figure older than the code that draws it?

This exists because the same failure happened three times on 2026-08-11, each time caught by a
different accident:

  * `make_paper_figures` renamed with `os.replace`, which CONSUMED the source, so a second run had
    nothing left to rename and the paper figure quietly stopped tracking its experiment;
  * two scripts both wrote `f9_case_colormnist.png`, so the README's case study was whichever ran
    last;
  * an n44 process started at 17:30, the figure's caption was edited at 17:38, and the figure it
    wrote at 17:46 carried the OLD caption -- which asserted a claim the documents had just
    retracted, in a figure those documents embed.

The shared shape is an artifact whose content silently diverges from the code meant to produce it.
Nothing in the repository noticed any of them.

This compares each figure's mtime against its generator's and reports anything stale. It is a
warning, not a hard failure: a figure legitimately predates an unrelated edit to its generator, and
GPU figures cannot be regenerated on every commit. What it removes is the SILENCE.

    python tests/test_figures_fresh.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")


def generators():
    """-> {figure basename: [source files that write it]}.  Found by reading the sources."""
    out = {}
    for sub in ("experiments", "tests"):
        d = os.path.join(HERE, sub)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py"):
                continue
            if fn == os.path.basename(__file__):
                continue          # this file NAMES figures in its own docstring; it draws none
            p = os.path.join(d, fn)
            try:
                src = open(p, encoding="utf-8").read()
            except Exception:
                continue
            # only count files that actually WRITE an image
            if "savefig" not in src:
                continue
            # names written via savefig paths, f-strings included: capture the literal stem
            for m in re.findall(r'["\']([a-zA-Z0-9_]*f\d+[a-zA-Z0-9_]*)\{?[^"\']*\.png["\']', src):
                out.setdefault(m, []).append(os.path.join(sub, fn))
    return out


def main():
    gens = generators()
    figs = sorted(f for f in os.listdir(FIG) if f.endswith(".png"))
    stale, unknown, ok = [], [], 0
    for f in figs:
        # generators write `f"f18_celeba_paper{tag}.png"`, so the source literal is a PREFIX of the
        # filename.  Stripping known suffixes was guesswork and missed the very figures that
        # prompted this test; longest matching prefix is the actual relationship.
        base = f[:-4]
        cands = [k for k in gens if base.startswith(k)]
        srcs = gens[max(cands, key=len)] if cands else None
        if not srcs:
            unknown.append(f)
            continue
        ft = os.path.getmtime(os.path.join(FIG, f))
        newer = [s for s in srcs if os.path.getmtime(os.path.join(HERE, s)) > ft]
        if newer:
            stale.append((f, newer))
        else:
            ok += 1

    print(f"[figures] {len(figs)} figures: {ok} current, {len(stale)} STALE, "
          f"{len(unknown)} with no generator found")
    for f, srcs in stale:
        print(f"  STALE  {f:44s} older than {', '.join(srcs)}")
    if unknown:
        print(f"  (no generator matched: {', '.join(unknown[:8])}"
              f"{' ...' if len(unknown) > 8 else ''})")
    print("\n[figures] stale means the drawing code changed after the image was written, so the "
          "image\n          may state something the code no longer does. Regenerate before "
          "citing it.")
    return 0                      # advisory: never fails a suite over a regenerable artifact


if __name__ == "__main__":
    sys.exit(main())
