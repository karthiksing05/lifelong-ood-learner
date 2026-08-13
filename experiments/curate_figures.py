"""Keep `figures/` to the CORE set, put the supporting ones in `figures/additional/`, archive dead.

THREE tiers since 2026-08-13, because "cited somewhere" turned out to be too weak a filter: the
directory still held layout ablations, renderer post-mortems and hierarchy panels that are real
evidence but are not what the work is *about*, and a reader opening `figures/` could not tell the
headline from the footnote.

  CORE        `figures/` -- the graphics the argument is built on: architecture, storage/cost,
              learning curves, composition/OOD, representation, case studies, preservation.
              Enumerated in CORE below and mirrored in docs/OUTLINE.md's "core figure set".
  ADDITIONAL  `figures/additional/` -- still cited, still evidence, but supporting: ablations,
              fault post-mortems, per-experiment intermediates the pipeline copies FROM.
  ARCHIVE     `figures/archive/` -- superseded or uncited; kept because each cost GPU time.

Archive and additional are moves, never deletes.


The directory had grown to 48 PNGs and 41 MB, of which the documents reference 19.  The other 29 are
smoke runs, superseded variants, one-off diagnostics, and intermediates that only exist as inputs to
`make_paper_figures.py`.  A reader opening `figures/` cannot tell which is which, and neither can a
stale-figure check -- `tests/test_figures_fresh.py` reports "11 with no generator found", all of them
in that residue.

Two rules decide what stays, and both are mechanical so the answer does not depend on taste:

  KEEP  the figure is referenced by path from README.md or any docs/*.md
  KEEP  `make_paper_figures.py` names it -- as an input it copies FROM, or as an output it WRITES.
        Inputs have no citations of their own but the published figures are copied from them, so
        archiving one silently breaks regeneration; outputs would simply be recreated by the next
        build, so archiving them makes curation fight the build loop instead of tidying after it.
  else  ARCHIVE

Archive, not delete.  Every one of these took GPU time to make, several came from runs whose
checkpoints are gone, and a figure that turns out to be wanted is then one `git mv` away instead of
one re-run.  `figures/archive/` is excluded from the docs' view but stays in the tree.

    python experiments/curate_figures.py            # dry run, prints the plan
    python experiments/curate_figures.py --apply
"""
import argparse
import glob
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")
ARCH = os.path.join(FIG, "archive")
ADD = os.path.join(FIG, "additional")

# The core set, mirrored in docs/OUTLINE.md. Grouped by what the outline asks for, so that adding a
# figure means deciding which part of the argument it serves rather than dropping it in the pile.
CORE = {
    # architecture
    "f0_architecture.png", "f1_loop.png",
    # storage and cost -- the headline claim
    "f16_addressing.png", "f8_growth.png", "f34_batch_invariance.png",
    # curves
    "f31_learning_curves.png", "f32_mdl_router.png",
    # composition / OOD, on our protocol and on theirs
    "f2_coverage_vs_copying.png", "f26_paper_composition.png",
    # representation
    "f33_representation.png",
    # case studies -- instances at various distances from prototypes
    "f9_case_colormnist.png", "f11_prototype_dial.png",
    # preservation and parity
    "f5_preservation.png", "f6_native_parity.png",
}


def referenced():
    """-> set of figure basenames cited by path from any markdown document."""
    out = set()
    for f in [os.path.join(HERE, "README.md")] + sorted(glob.glob(os.path.join(HERE, "docs/*.md"))):
        with open(f, encoding="utf-8") as fh:
            out |= set(re.findall(r"figures/([A-Za-z0-9_]+\.png)", fh.read()))
    return out


def pipeline_figures():
    """-> every figure basename `make_paper_figures.py` mentions: what it READS and what it WRITES.

    Read out of the source rather than duplicated here: a hardcoded list is exactly the kind of
    second copy of a shared constant that has caused several faults in this repository.

    Both directions matter, and the first version of this only took the sources. That was wrong in
    a way that showed up immediately: `f4_vocabulary` and `f10_case_celeba` are OUTPUTS of the
    pipeline that no document happens to cite, so curation archived them and the next
    `make_paper_figures` run wrote them straight back. A rule that fights the build loop is worse
    than no rule -- anything the pipeline produces is part of the maintained set by definition.
    """
    p = os.path.join(HERE, "experiments", "make_paper_figures.py")
    if not os.path.exists(p):
        return set()
    with open(p, encoding="utf-8") as fh:
        return set(re.findall(r'["\']([A-Za-z0-9_]+\.png)["\']', fh.read()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually move files (default: dry run)")
    args = ap.parse_args()

    keep_ref, keep_src = referenced(), pipeline_figures()
    have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(FIG, "*.png")))

    core = [f for f in have if f in CORE]
    extra = [f for f in have if f not in CORE and (f in keep_ref or f in keep_src)]
    dead = [f for f in have if f not in CORE and f not in keep_ref and f not in keep_src]

    def mb(names, base=FIG):
        return sum(os.path.getsize(os.path.join(base, n)) for n in names) / 1e6

    print(f"[curate] {len(have)} figures in figures/, {mb(have):.1f} MB\n")
    print(f"  CORE {len(core)} ({mb(core):.1f} MB)  -- stay in figures/")
    for f in core:
        print(f"    {f}")
    missing = sorted(CORE - set(have))
    if missing:
        print(f"\n  CORE MISSING {len(missing)} -- declared but not present "
              f"(regenerate, or fix CORE):")
        for f in missing:
            print(f"    {f}")
    print(f"\n  ADDITIONAL {len(extra)} ({mb(extra):.1f} MB)  -> figures/additional/")
    for f in extra:
        why = "cited" if f in keep_ref else "pipeline input/output"
        print(f"    {f:44s} {why}")
    print(f"\n  ARCHIVE {len(dead)} ({mb(dead):.1f} MB)  -> figures/archive/")
    for f in dead:
        print(f"    {f:44s} {os.path.getsize(os.path.join(FIG, f)) / 1e6:6.2f} MB")

    if not args.apply:
        print("\n[curate] dry run: pass --apply to move")
        return 0
    for sub, names in ((ADD, extra), (ARCH, dead)):
        os.makedirs(sub, exist_ok=True)
        for f in names:
            shutil.move(os.path.join(FIG, f), os.path.join(sub, f))
    print(f"\n[curate] moved {len(extra)} to additional/, {len(dead)} to archive/; "
          f"{len(core)} core remain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
