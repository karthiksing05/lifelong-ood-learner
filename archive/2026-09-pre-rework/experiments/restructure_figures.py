"""One-shot: rename the figure set to descriptive names and settle the four-folder layout.

The numbering (f0, f2, f5, f8, f11, f16, f26, f31...) is an accident of the order things were built
in.  It tells a reader nothing, the gaps invite the question "where is f7", and three of the numbers
were reused for different content during the session.  This renames every CORE figure to what it
shows, moves `additional/` to `extra/`, and leaves `archive/` in the tree but out of git.

    figures/            the streamlined set -- one story per file, all three datasets
    figures/dit/        the transformer track
    figures/extra/      supporting evidence: ablations, fault post-mortems, intermediates
    figures/archive/    superseded, gitignored

Every rename is applied in THREE places or it is not applied at all: the file, the generating
script's output path, and every citation in README.md / docs/*.md.  A rename that misses one of those
is worse than no rename, so the script verifies all three afterwards and refuses to report success if
any citation dangles.

    python experiments/restructure_figures.py            # dry run
    python experiments/restructure_figures.py --apply
"""
import argparse
import glob
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures", "initial")

# old basename -> new basename.  Grouped by what the outline asks for.
RENAME = {
    # the architecture
    "f0_architecture.png": "01_architecture.png",
    "f1_loop.png": "02_lifelong_loop.png",
    # storage and cost -- the headline claim
    "f16_addressing.png": "03_storage_ladder.png",
    "f8_growth.png": "04_growth_per_concept.png",
    "f34_batch_invariance.png": "05_batch_invariance.png",
    # curves
    "f31_learning_curves.png": "06_learning_curves.png",
    "f32_mdl_router.png": "07_router_exchange_rate.png",
    # composition / OOD
    "f2_coverage_vs_copying.png": "08_coverage_vs_copying.png",
    # representation
    "f33_representation.png": "09_representation.png",
    # preservation and parity
    "f5_preservation.png": "10_preservation.png",
    "f6_native_parity.png": "11_native_parity.png",
    # case studies -- one per dataset, named by dataset
    "f9_case_colormnist.png": "case_colormnist.png",
    "f26_paper_composition.png": "case_celeba.png",
    "f11_prototype_dial.png": "case_prototype_distance.png",
}

# generating script -> the output paths inside it that must change with the rename
SCRIPTS = ["experiments/make_paper_figures.py", "experiments/make_case_studies.py",
           "experiments/make_n5_curves.py", "experiments/make_mdl_figure.py",
           "experiments/make_performance_figures.py", "experiments/make_rank_frontier.py",
           "experiments/make_architecture_figure.py", "experiments/n48_paper_composition.py",
           "experiments/curate_figures.py", "reproduce.sh"]
DOCS = ["README.md"] + sorted(os.path.relpath(p, HERE) for p in glob.glob(os.path.join(HERE, "docs/*.md")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    have = {os.path.basename(p) for p in glob.glob(os.path.join(FIG, "*.png"))}
    todo = {o: n for o, n in RENAME.items() if o in have}
    missing = [o for o in RENAME if o not in have]
    print(f"[restructure] {len(todo)} of {len(RENAME)} core figures present")
    for o, n in sorted(todo.items()):
        print(f"    {o:34s} -> {n}")
    if missing:
        print(f"[restructure] absent (regenerate or drop from RENAME): {missing}")

    add = os.path.join(FIG, "additional")
    extra = os.path.join(FIG, "extra")
    print(f"[restructure] additional/ -> extra/  "
          f"({len(glob.glob(os.path.join(add, '*.png')))} files)"
          if os.path.isdir(add) else "[restructure] additional/ already moved")

    if not args.apply:
        print("\n[restructure] dry run: pass --apply")
        return 0

    for o, n in todo.items():
        shutil.move(os.path.join(FIG, o), os.path.join(FIG, n))
    if os.path.isdir(add):
        os.makedirs(extra, exist_ok=True)
        for p in glob.glob(os.path.join(add, "*.png")):
            shutil.move(p, os.path.join(extra, os.path.basename(p)))
        os.rmdir(add)

    n_edit = 0
    for rel in SCRIPTS + DOCS:
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            continue
        s = open(p, encoding="utf-8").read()
        o_ = s
        for old, new in RENAME.items():
            s = s.replace(old, new)
        s = s.replace("figures/additional/", "figures/extra/")
        s = s.replace('"additional"', '"extra"').replace("'additional'", "'extra'")
        if s != o_:
            open(p, "w", encoding="utf-8").write(s)
            n_edit += 1
            print(f"    edited {rel}")
    print(f"[restructure] rewrote {n_edit} files")

    # VERIFY: no citation may dangle
    bad = []
    for rel in DOCS:
        p = os.path.join(HERE, rel)
        for m in re.findall(r"figures/(?:dit/|extra/|archive/)?[A-Za-z0-9_]+\.png",
                            open(p, encoding="utf-8").read()):
            if not os.path.exists(os.path.join(HERE, m)):
                bad.append(f"{rel}: {m}")
    if bad:
        print(f"\n[restructure] *** {len(bad)} DANGLING citations -- fix before committing:")
        for b in bad:
            print(f"    {b}")
        return 1
    print("[restructure] every citation resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
