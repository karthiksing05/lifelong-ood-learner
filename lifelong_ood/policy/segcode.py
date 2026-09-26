"""Segment-wise borrowing along the demo's action SEQUENCE.

A demo's chunks are overlapping windows of one T x 7 action sequence (chunk f = actions f..f+9).  Instead of coding every
frame's chunk (policy/dimcode.py), code the sequence once: cut it into segments of L = 10 steps and, for each segment and
action dimension, borrow the closest 10-step trajectory of that dimension from a LEAF of the library -- the policy's own
chunk means at the training frames (the components of its action marginal, the zero-noise level of the hierarchy).
Executed chunk at frame f = the rebuilt sequence's window f..f+9, exactly like a replayed demo; the program stores one
2-byte leaf id per (segment, dimension) and one offset byte per segment (non-zero only for a short tail segment).
"""
from __future__ import annotations

import numpy as np

H, A7 = 10, 7


def sequence(rows):
    """[T, 70] windows of one sequence -> the [T + 9, 7] sequence."""
    m = np.asarray(rows, np.float32).reshape(len(rows), H, A7)
    return np.concatenate([m[:, 0], m[-1, 1:]])


def windows(seq, T):
    return np.asarray(seq)[np.arange(T)[:, None] + np.arange(H)[None]].reshape(T, H * A7)


def encode(leaves, seq, L=10):
    """leaves [N, 10, 7] (normalised), seq [S, 7] -> ids [nseg, 7] int32, offs [nseg] uint8."""
    S = len(seq)
    nseg = int(np.ceil(S / L))
    ids = np.zeros((nseg, A7), np.int32)
    offs = np.zeros(nseg, np.uint8)
    for s in range(nseg):
        a, b = s * L, min((s + 1) * L, S)
        n = b - a
        best = None
        for o in range(0, H - n + 1):
            err = ((leaves[:, o:o + n, :] - seq[a:b][None]) ** 2).sum(1)          # [N, 7]
            i = err.argmin(0)
            e = err[i, np.arange(A7)]
            if best is None:
                best = (e, i, np.full(A7, o))
            else:
                better = e < best[0]
                best = (np.where(better, e, best[0]), np.where(better, i, best[1]), np.where(better, o, best[2]))
        ids[s] = best[1]
        # one offset per segment: take the most common (all zero for full segments)
        offs[s] = int(np.bincount(best[2]).argmax())
        if (best[2] != offs[s]).any():                                             # re-pick with the shared offset
            o = int(offs[s])
            err = ((leaves[:, o:o + n, :] - seq[a:b][None]) ** 2).sum(1)
            ids[s] = err.argmin(0)
    return ids, offs


def decode(leaves, ids, offs, S, L=10):
    seq = np.zeros((S, A7), np.float32)
    for s in range(len(ids)):
        a, b = s * L, min((s + 1) * L, S)
        n, o = b - a, int(offs[s])
        seq[a:b] = leaves[ids[s], o:o + n, np.arange(A7)].T
    return seq


def code_bytes(nseg):
    return int(nseg * A7 * 2 + nseg)
