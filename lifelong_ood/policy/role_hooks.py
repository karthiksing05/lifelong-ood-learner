"""RoleHooks: the DEPLOYABLE role editor (plan of 2026-10-01 evening; lifelong_ood/policy/roles.py).

The policy is served the base task prototype's instruction and an observation in which the new task's object and the base's
object exchange appearance.  Where they are comes from an open-vocabulary detector (OWLv2, where.OwlDetector) run on the
policy's own 224 images at every query; nothing from the simulator.  Each object's box is cut out, the holes are inpainted
from the surrounding table (cv2 Telea), and each crop is pasted bottom-aligned at the other's box (no depth: native size).
The oracle counterpart exchanges the meshes in the simulator's renderer (experiments/libero_client.py --swap-mode render).

    hooks = RoleHooks(code, phrases, cams="agent")   # code from roles.role_code; phrases: instance -> the words to detect
"""
from __future__ import annotations

import numpy as np

CAMKEY = {"agent": "observation/image", "wrist": "observation/wrist_image"}


def _box_int(b, H, W):
    x0, y0, x1, y1 = b
    return (int(max(0, np.floor(y0))), int(min(H, np.ceil(y1))), int(max(0, np.floor(x0))), int(min(W, np.ceil(x1))))


def swap_boxes(img, box_a, box_b, pad=2):
    """Exchange two boxes' contents in an HxWx3 uint8 image: inpaint both holes, paste each crop bottom-centred at the other."""
    import cv2
    H, W = img.shape[:2]
    out = img.copy()
    ra, rb = _box_int(box_a, H, W), _box_int(box_b, H, W)
    crops = [img[r[0]:r[1], r[2]:r[3]].copy() for r in (ra, rb)]
    hole = np.zeros((H, W), np.uint8)
    for r in (ra, rb):
        hole[max(0, r[0] - pad):min(H, r[1] + pad), max(0, r[2] - pad):min(W, r[3] + pad)] = 255
    out = cv2.inpaint(out, hole, 5, cv2.INPAINT_TELEA)
    for crop, r in ((crops[1], ra), (crops[0], rb)):            # b's crop goes to a's place and a's to b's
        h, w = crop.shape[:2]
        rbot, cmid = r[1], 0.5 * (r[2] + r[3])
        r0, c0 = rbot - h, int(round(cmid - w / 2.0))
        R0, R1, C0, C1 = max(r0, 0), min(r0 + h, H), max(c0, 0), min(c0 + w, W)
        if R1 > R0 and C1 > C0:
            out[R0:R1, C0:C1] = crop[R0 - r0:R1 - r0, C0 - c0:C1 - c0]
    return out


class RoleHooks:
    def __init__(self, code, phrases, cams="agent", min_score=0.05, device="cuda"):
        from .where import OwlDetector
        self.code = code
        self.prompt = code["prompt"]
        self.pairs = code["pairs"]
        self.phrases = phrases
        self.cams = cams.split("+")
        self.min_score = float(min_score)
        self.owl = OwlDetector(device=device)
        self.log = []

    def _box(self, img, inst):
        import torch
        ph = self.phrases[inst]
        inp = self.owl.proc(text=[[ph]], images=img, return_tensors="pt").to(self.owl.device)
        with torch.no_grad():
            out = self.owl.model(**inp)
        S = max(img.shape[:2])
        res = self.owl.proc.post_process_object_detection(out, threshold=0.0, target_sizes=torch.tensor([[S, S]]))[0]
        if len(res["scores"]) == 0:
            return None, 0.0
        k = int(torch.argmax(res["scores"]))
        return [float(v) for v in res["boxes"][k]], float(res["scores"][k])

    def rewrite(self, obs, meta):
        if any(k.startswith(("seg_", "obj_")) for k in meta):
            raise RuntimeError("the deployable role editor received privileged simulator state")
        obs = dict(obs)
        obs["prompt"] = self.prompt
        rec = {}
        for cam in self.cams:
            img = np.asarray(obs[CAMKEY[cam]])
            for a_, b_ in self.pairs:
                ba, sa = self._box(img, a_)
                bb, sb = self._box(img, b_)
                rec[cam] = (sa, sb)
                if ba is None or bb is None or min(sa, sb) < self.min_score:
                    continue
                img = swap_boxes(img, ba, bb)
            obs[CAMKEY[cam]] = img
        self.log.append(rec)
        return obs

    def __call__(self, obs, pre, meta=None):
        return None, None
