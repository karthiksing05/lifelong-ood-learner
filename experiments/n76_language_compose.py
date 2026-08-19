"""N76 -- does composing in a LANGUAGE embedding space beat the literal phrase? The VLA question, on images.

Phase 4 wants a language-conditioned diffusion policy, and the claim it rests on is not "we can
address a held-out task" -- for a language-conditioned model the held-out task ALREADY has an
address, because you can simply type the sentence. The claim has to be sharper:

    Composing in the model's OWN conditioning geometry -- ideal words fitted over embeddings it was
    actually trained on -- lands closer to the held-out concept than the LITERAL embedding of the
    held-out phrase.

That is the standard compositional-binding failure ("a red cube on a blue plate" rendering with the
colours swapped) stated as a prediction, and it is the single assumption Phase 4 cannot proceed
without. **It should not be tested for the first time inside a robot simulator.**

It does not need a diffusion model at all. CLIP gives a joint image-text space, so the question can
be asked directly: build the vocabulary from text embeddings of SEEN combinations, fit
`c ~ mu + sum_f V_f[value_f]` over them, address a held-out combination, and measure which candidate
embedding is closest to where the held-out concept actually lives -- the mean CLIP embedding of REAL
images of it.

This is also a return to source rather than an extension. `lifelong_ood/neural/idealwords.py` cites
Trager et al., *Linear Spaces of Meanings*, as the basis of the whole method, and that paper's
setting is CLIP embedding space, not a learned class table.

ARMS, and each is a way the claim could be false:

    address        mu + sum_f V_f[value]   fitted over SEEN phrase embeddings only
    literal        CLIP's own embedding of the held-out phrase -- THE control that matters
    nearest        the closest SEEN phrase embedding (retrieval)
    centroid       the mean of all seen phrase embeddings (a "says nothing" floor)

Scored by cosine similarity to the mean CLIP IMAGE embedding of real held-out images, and by
retrieval rank against every candidate class. Text and image live in different cones in CLIP, so an
absolute cosine is meaningless -- every number is read against the LITERAL arm and against the
image-side ceiling (a disjoint half of the real images of that class).

    python experiments/n76_language_compose.py --dataset celeba_grid
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS                                            # noqa: E402

from lifelong_ood.evals import save_json                              # noqa: E402
from lifelong_ood.neural.idealwords import compose, cv_r2, factorize  # noqa: E402

ARMS = ("address", "literal", "nearest", "centroid")

# Phrasings are fixed here rather than tuned: the comparison is between COMPOSITION and the LITERAL
# embedding of the same words, so both arms must be built from the identical vocabulary or the
# result measures prompt engineering.
CELEBA_WORDS = {
    "hair": ["blond hair", "brown hair", "dark hair"],
    "Male": ["a woman", "a man"],
    "Smiling": ["not smiling", "smiling"],
}
SHAPES_WORDS = {
    "floor_hue": [f"a {c} floor" for c in
                  ["red", "orange", "yellow", "chartreuse", "green",
                   "spring green", "cyan", "azure", "blue", "purple"]],
    "object_hue": [f"a {c} object" for c in
                   ["red", "orange", "yellow", "chartreuse", "green",
                    "spring green", "cyan", "azure", "blue", "purple"]],
    "shape": ["a cube", "a cylinder", "a sphere", "a capsule"],
}


def phrase(words, names, factors):
    """One sentence per combination, built from the SAME word list every arm uses."""
    parts = [words[n][int(v) % len(words[n])] for n, v in zip(names, factors)]
    return "a photo of " + ", ".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("celeba_grid", "shapes3d"), default="celeba_grid")
    p.add_argument("--model", default="openai/clip-vit-base-patch32")
    p.add_argument("--n-holdout", type=int, default=4)
    p.add_argument("--stride", type=int, default=1,
                   help="shapes3d value_stride. 1 gives 400 phrases against 25 free vectors "
                        "(ratio 18); 2 gives 100 against 12. The CelebA grid can only reach 8 "
                        "phrases / 5 vectors = 1.6, which is too few to FIT a basis, let alone "
                        "to test one -- the same defect that invalidated 7ee's composition claim.")
    p.add_argument("--n-img", type=int, default=64, help="real images per class, for the target")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    import torch
    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ── data ────────────────────────────────────────────────────────────────
    if args.dataset == "celeba_grid":
        from lifelong_ood.data.celeba import load_celeba_grid
        d = load_celeba_grid(img=64, seed=args.seed, n_per_cell=200,
                             n_holdout=args.n_holdout, max_rows=60000)
        words = CELEBA_WORDS
    else:
        from lifelong_ood.data.shapes3d import load_shapes3d
        d = load_shapes3d(n_per_slot=40, seed=args.seed, value_stride=args.stride,
                          fix_nuisance=["wall_hue"], n_holdout=args.n_holdout)
        words = SHAPES_WORDS
    X = np.asarray(d["X"], np.float32)
    F = np.asarray(d["factors"], int)
    slot = np.asarray(d["slot"], int)
    seen = np.asarray(d["seen"], bool)
    names, sizes = list(d["names"]), [int(s) for s in d["sizes"]]
    img = int(d.get("img", 64))

    seen_slots = sorted(set(slot[seen].tolist()))
    held_slots = [s for s in sorted(set(slot.tolist())) if s not in set(seen_slots)]
    print(f"[n76] {args.dataset}: {len(seen_slots)} seen classes, {len(held_slots)} held out; "
          f"factors {names} sizes {sizes}", flush=True)
    if not held_slots:
        print("[n76] no held-out class -- nothing to compose")
        return 0

    # ── CLIP ────────────────────────────────────────────────────────────────
    model = CLIPModel.from_pretrained(args.model).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(args.model)

    def embed_text(strings):
        with torch.no_grad():
            tok = proc(text=strings, return_tensors="pt", padding=True, truncation=True).to(dev)
            e = model.get_text_features(**tok)
        return (e / e.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float64)

    def embed_images(rows):
        out = []
        with torch.no_grad():
            for i in range(0, len(rows), 64):
                im = np.clip(rows[i:i + 64].reshape(-1, img, img, 3), 0, 1)
                px = proc(images=list((im * 255).astype(np.uint8)), return_tensors="pt")
                e = model.get_image_features(**{k: v.to(dev) for k, v in px.items()})
                out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    # vocabulary: one phrase per SEEN class, and its CLIP embedding
    seen_f = np.stack([F[slot == s][0] for s in seen_slots])
    seen_phrases = [phrase(words, names, f) for f in seen_f]
    C = embed_text(seen_phrases)
    print(f"[n76] vocabulary: {len(C)} seen phrases, e.g. {seen_phrases[0]!r}", flush=True)

    fac = factorize(C, seen_f)
    r2cv, n_cv = cv_r2(C, seen_f)
    n_free = 1 + sum(s - 1 for s in sizes)
    print(f"[n76] ideal words over CLIP TEXT embeddings: R^2 {fac['r2']:.3f} in-sample, "
          f"{r2cv:.3f} leave-one-out; {len(C)} phrases / {n_free} free vectors = "
          f"ratio {len(C) / n_free:.1f}", flush=True)

    rows = []
    for h in held_slots:
        idx = np.where(slot == int(h))[0]
        if len(idx) < 8:
            continue
        tgt = tuple(int(v) for v in F[idx[0]])
        txt = phrase(words, names, tgt)
        E = embed_images(X[idx[:args.n_img]])
        half = max(4, len(E) // 2)
        target_vec = E[:half].mean(0)
        target_vec /= np.linalg.norm(target_vec) + 1e-12
        # THE CEILING: a disjoint half of the SAME real images. Text and image embeddings live in
        # different cones in CLIP, so an absolute cosine says nothing; this is what "as close as the
        # concept is to itself" looks like on the image side.
        other = E[half:].mean(0)
        other /= np.linalg.norm(other) + 1e-12

        cand = {
            "address": compose(fac, tgt),
            "literal": embed_text([txt])[0],
            "nearest": C[int(np.argmax(C @ embed_text([txt])[0]))],
            "centroid": C.mean(0),
        }
        sims = {}
        for a, v in cand.items():
            v = np.asarray(v, np.float64)
            v = v / (np.linalg.norm(v) + 1e-12)
            sims[a] = float(v @ target_vec)
        ceil = float(other @ target_vec)

        # retrieval rank: against every class (seen + held), does this embedding pick the right one?
        all_slots = seen_slots + held_slots
        all_f = np.stack([F[slot == s][0] for s in all_slots])
        all_img = np.stack([
            (lambda e: e / (np.linalg.norm(e) + 1e-12))(
                embed_images(X[np.where(slot == s)[0][:16]]).mean(0)) for s in all_slots])
        ranks = {}
        for a, v in cand.items():
            v = np.asarray(v, np.float64)
            v = v / (np.linalg.norm(v) + 1e-12)
            order = np.argsort(-(all_img @ v))
            ranks[a] = int(np.where(np.asarray(all_slots)[order] == int(h))[0][0]) + 1

        rows.append(dict(slot=int(h), target=list(tgt), phrase=txt, ceiling=ceil,
                         sims=sims, ranks=ranks, n_classes=len(all_slots)))
        print(f"[n76] cell {h} {txt!r}", flush=True)
        print(f"[n76]    cos: " + " | ".join(f"{a} {sims[a]:+.4f}" for a in ARMS)
              + f" | image-half ceiling {ceil:+.4f}", flush=True)
        print(f"[n76]    rank of the true class among {len(all_slots)}: "
              + " | ".join(f"{a} {ranks[a]}" for a in ARMS), flush=True)

    if not rows:
        print("[n76] no scorable held-out class")
        return 0

    def m(a, key="sims"):
        return float(np.mean([r[key][a] for r in rows]))

    print(f"\n[n76] {'=' * 72}")
    print(f"{'arm':12s}{'cos to target':>15s}{'vs literal':>12s}{'mean rank':>11s}{'top-1':>8s}")
    for a in ARMS:
        top1 = float(np.mean([r["ranks"][a] == 1 for r in rows]))
        print(f"{a:12s}{m(a):>15.4f}{m(a) - m('literal'):>+12.4f}"
              f"{m(a, 'ranks'):>11.2f}{top1:>8.2f}")
    print(f"{'-- image half':12s}{float(np.mean([r['ceiling'] for r in rows])):>15.4f}"
          f"{'(ceiling)':>12s}")

    # JUDGE ON RANK, NOT ON RAW COSINE. Measured here and it is not a preference: on Shapes3D the
    # `centroid` arm -- the mean of every seen phrase, which by construction says nothing about the
    # target -- has the HIGHEST cosine of any arm while ranking 389th of 400. CLIP's text embeddings
    # occupy a narrow cone, so a generic "a photo of ..." direction is close to every image, and an
    # absolute image-text cosine mostly measures that offset. Rank is invariant to it.
    gain = m("address") - m("literal")
    r_addr, r_lit, r_cent = m("address", "ranks"), m("literal", "ranks"), m("centroid", "ranks")
    n_cls = rows[0]["n_classes"]
    print("\n[n76] VERDICT")
    if r_cent < 0.25 * n_cls:
        print(f"[n76]   INSTRUMENT WARNING: the centroid arm ranks {r_cent:.1f} of {n_cls}, so this")
        print("[n76]   comparison is not discriminating -- an embedding that says nothing about the")
        print("[n76]   target should rank near the middle. Read nothing below.")
    elif r_addr < r_lit - 0.5:
        print(f"[n76]   COMPOSING BEATS THE LITERAL PHRASE: it ranks the true class {r_addr:.2f} of")
        print(f"[n76]   {n_cls} against the literal embedding's {r_lit:.2f}. That is the assumption")
        print("[n76]   Phase 4 rests on, established in a language embedding space with no diffusion")
        print("[n76]   model involved -- and the cosine column shows why rank is the statistic:")
        print(f"[n76]   the centroid scores {m('centroid'):+.4f} cosine while ranking {r_cent:.0f}.")
    elif r_addr <= r_lit + 0.5:
        print(f"[n76]   COMPOSING TIES THE LITERAL PHRASE ({r_addr:.2f} vs {r_lit:.2f} of {n_cls}).")
        print("[n76]   For a language-conditioned policy there is then no reason to compose an")
        print("[n76]   address rather than type the sentence, and the contribution has to be")
        print("[n76]   STORAGE and ROUTING rather than addressing.")
    else:
        print(f"[n76]   COMPOSING IS WORSE THAN THE LITERAL PHRASE ({r_addr:.2f} vs {r_lit:.2f}).")
        print("[n76]   Phase 4 should use the literal instruction embedding and claim storage only.")
        print("[n76]   Better learned here than after a simulator install.")
    print(f"[n76]   (cosine gain {gain:+.4f}, reported but not the basis of this verdict)")
    save_json(dict(rows=rows, dataset=args.dataset, model=args.model,
                   basis_r2=float(fac["r2"]), basis_r2_cv=float(r2cv), n_cv=int(n_cv),
                   n_seen=len(C), n_free=int(n_free), config=vars(args)),
              os.path.join(RESULTS, f"n76_language_compose{args.tag}.json"), seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
