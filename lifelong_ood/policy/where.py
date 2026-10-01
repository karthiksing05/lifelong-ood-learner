"""WHERE a referent is, as a 16 x 16 map over a camera's image tokens (plan of 2026-10-01 evening, Bind).

Sources, in the order the plan allows them:
    W1  the policy's OWN vision-language backbone: PaliGemma's attention from the referent's word tokens to the image tokens
        while it embeds the prefix (`vlm_grounding`); nothing outside the frozen policy
    W3  an open-vocabulary detector (OWLv2), the general fallback (`OwlDetector`)
    W0  the simulator's per-geom segmentation: the ORACLE, used to score W1 / W3 and for the oracle arm only
Pixel conventions: the client flips LIBERO's 256 render by [::-1, ::-1] and resizes it to 224, so the model's 14-px patch
(i, j) of the 224 image is exactly the 16-px block (i, j) of the flipped 256 render.  Robosuite projects world points to
'opencv' pixels (row from the top) of the UNflipped render, whose rows are bottom-up; in the client's image the same point
is at (row, 255 - col).
"""
from __future__ import annotations

import numpy as np

GRID = 16
PIX = 256


# ── oracle maps from the segmentation dumps (experiments/f0b_firstframes_seg.py) ───────────────────────────────────
def pool(mask256):
    """[256, 256] bool / float (client orientation) -> [16, 16] coverage fraction of each token."""
    m = np.asarray(mask256, np.float32).reshape(GRID, PIX // GRID, GRID, PIX // GRID)
    return m.mean(axis=(1, 3))


def geoms_of(z, name=None, body_substr=None):
    """Geom ids belonging to an object / fixture `name` (by the body tree) or to bodies whose name contains `body_substr`."""
    gb = [str(x) for x in z["geom_body"]]
    bn = [str(x) for x in z["body_names"]]
    root = [str(x) for x in z["body_root"]]
    idx = {b: i for i, b in enumerate(bn)}
    out = []
    for g, b in enumerate(gb):
        i = idx.get(b)
        if i is None:
            continue
        if name is not None and root[i] == name and (body_substr is None or body_substr in b):
            out.append(g)
        elif name is None and body_substr is not None and body_substr in b:
            out.append(g)
    return np.asarray(out, np.int64)


def oracle_map(seg, geoms):
    """seg [256,256] int geom ids (client orientation) + the referent's geom ids -> [16,16] coverage."""
    return pool(np.isin(seg, geoms))


def project(points, K, E, H=PIX, W=PIX):
    """World points [..., 3] -> (row, col) in the CLIENT's 256 image (flipped render), float.  K: 3x3 intrinsics, E: 4x4
    camera pose in the world (robosuite's extrinsic)."""
    P = np.asarray(points, np.float64)
    Rt = np.linalg.inv(np.asarray(E, np.float64))
    Kx = np.eye(4)
    Kx[:3, :3] = np.asarray(K, np.float64)
    M = Kx @ Rt
    hom = np.concatenate([P, np.ones(P.shape[:-1] + (1,))], -1)
    q = hom @ M.T
    q = q[..., :2] / q[..., 2:3]
    row_cv, col_cv = q[..., 1], q[..., 0]
    return np.stack([row_cv, (W - 1) - col_cv], -1)


def box_map(center, xmat, half, K, E):
    """A region site's box (world centre, 3x3 orientation, half sizes) -> [16,16] coverage of its projected bounding rectangle."""
    R = np.asarray(xmat, np.float64).reshape(3, 3)
    s = np.asarray(half, np.float64).reshape(3)
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], np.float64) * s
    pts = np.asarray(center, np.float64) + corners @ R.T
    rc = project(pts, K, E)
    r0, r1 = np.clip([rc[:, 0].min(), rc[:, 0].max()], 0, PIX - 1)
    c0, c1 = np.clip([rc[:, 1].min(), rc[:, 1].max()], 0, PIX - 1)
    m = np.zeros((PIX, PIX), np.float32)
    m[int(np.floor(r0)):int(np.ceil(r1)) + 1, int(np.floor(c0)):int(np.ceil(c1)) + 1] = 1.0
    return pool(m)


def peak(m):
    """[16,16] -> (i, j) of the largest entry."""
    m = np.asarray(m)
    k = int(np.argmax(m))
    return divmod(k, m.shape[1])


def points_at(m_pred, m_true, thr=0.0):
    """Pointing: does the predicted map's peak token overlap the true referent (coverage > thr)?"""
    i, j = peak(m_pred)
    return bool(np.asarray(m_true)[i, j] > thr)


# ── W1: the policy's own grounding ─────────────────────────────────────────────────────────────────────────────────
LANG0 = 768          # first language column of pi0.5's prefix (3 images x 256 tokens)


def phrase_span(tokenizer, prompt, phrase):
    """Token positions (within the language block) of `phrase` inside the tokenised `prompt`, openpi's pi0 format (BOS +
    cleaned text + newline).  -> list of ints (empty if not found)."""
    sp = tokenizer._tokenizer
    clean = lambda s: s.strip().replace("_", " ").replace("\n", " ")       # noqa: E731
    full = sp.encode(clean(prompt), add_bos=True)
    for variant in (" " + clean(phrase), clean(phrase)):
        sub = sp.encode(variant, add_bos=False)
        for i in range(len(full) - len(sub) + 1):
            if full[i:i + len(sub)] == sub:
                return list(range(i, i + len(sub)))
    # fall back to the longest run of the phrase's word pieces that appears contiguously
    words = clean(phrase).split()
    for n in range(len(words), 0, -1):
        for s in range(len(words) - n + 1):
            sub = sp.encode(" " + " ".join(words[s:s + n]), add_bos=False)
            for i in range(len(full) - len(sub) + 1):
                if full[i:i + len(sub)] == sub:
                    return list(range(i, i + len(sub)))
    return []


def vlm_grounding(bk, obs, phrase, layers=None, tokenizer=None):
    """W1.  Embed the prefix of `obs` (its own prompt) with attention capture in PaliGemma's language-model layers, and
    average the attention FROM the phrase's tokens TO each camera's 256 image tokens.  -> {layer: {"agent": [16,16],
    "wrist": [16,16]}} (rows normalised to sum 1 over each camera's tokens)."""
    import torch
    lm = bk.model.paligemma_with_expert.paligemma.language_model
    if tokenizer is None:
        from openpi.models import tokenizer as _tok
        tokenizer = _tok.PaligemmaTokenizer(200)
    span = phrase_span(tokenizer, obs["prompt"], phrase)
    if not span:
        raise ValueError(f"phrase {phrase!r} not found in prompt {obs['prompt']!r}")
    rows = [LANG0 + i for i in span]
    want = set(range(len(lm.layers))) if layers is None else set(layers)
    got = {}

    def hook(l):
        def f(mod, inp, out):
            w = out[1]
            if w is not None:
                got[l] = w[0, :, rows, :512].float().mean(dim=(0, 1)).cpu().numpy()     # heads, phrase rows -> [512]
        return f
    hs = [lm.layers[l].self_attn.register_forward_hook(hook(l)) for l in sorted(want)]
    try:
        with torch.no_grad():
            bk.prefix(bk.observation(obs))
    finally:
        for h in hs:
            h.remove()
    out = {}
    for l, v in got.items():
        a, w = v[:256], v[256:512]
        out[l] = {"agent": (a / max(a.sum(), 1e-12)).reshape(GRID, GRID),
                  "wrist": (w / max(w.sum(), 1e-12)).reshape(GRID, GRID)}
    return out


# ── W3: an open-vocabulary detector (fallback) ─────────────────────────────────────────────────────────────────────
class OwlDetector:
    """OWLv2 (transformers) on the 224 client image: the best box for a text query, rendered as a [16,16] coverage map."""

    def __init__(self, name="google/owlv2-base-patch16-ensemble", device="cuda"):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.proc = Owlv2Processor.from_pretrained(name)
        self.model = Owlv2ForObjectDetection.from_pretrained(name).to(device).eval()
        self.device = device

    def __call__(self, img224, phrase):
        import torch
        inp = self.proc(text=[[phrase]], images=img224, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inp)
        H = W = max(img224.shape[:2])                    # OWLv2 pads to a square
        res = self.proc.post_process_object_detection(out, threshold=0.0, target_sizes=torch.tensor([[H, W]]))[0]
        if len(res["scores"]) == 0:
            self.last_center = None
            return np.zeros((GRID, GRID), np.float32), 0.0
        k = int(torch.argmax(res["scores"]))
        x0, y0, x1, y1 = [float(v) for v in res["boxes"][k]]
        s = PIX / float(img224.shape[0])                  # 224 -> 256 grid of the maps
        m = np.zeros((PIX, PIX), np.float32)
        m[int(max(0, y0 * s)):int(min(PIX, y1 * s)) + 1, int(max(0, x0 * s)):int(min(PIX, x1 * s)) + 1] = 1.0
        # a box map is flat: its POINT is the box centre's token (argmax of a flat map would return the top-left token)
        self.last_center = (min(GRID - 1, int(0.5 * (y0 + y1) * s) // (PIX // GRID)),
                            min(GRID - 1, int(0.5 * (x0 + x1) * s) // (PIX // GRID)))
        return pool(m), float(res["scores"][k])


def unproject(rc, depth, K, E, W=PIX):
    """(row, col) in the CLIENT's 256 image + metric depth there -> world point [3] (inverse of `project`)."""
    r, c = float(rc[0]), float(rc[1])
    u, v = (W - 1) - c, r                                   # opencv pixel of the unflipped render
    K = np.asarray(K, np.float64)
    z = float(depth)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return (np.asarray(E, np.float64) @ np.array([x, y, z, 1.0]))[:3]
