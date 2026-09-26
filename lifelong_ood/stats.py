"""Statistics for rollout success: Wilson intervals, a two-level (task, then episode) cluster bootstrap that keeps the
pairing of arms on the same init states, the exact sign test on discordant pairs, and a non-inferiority read.

Episodes are Bernoulli; tasks differ a lot in difficulty, so every interval on a pooled rate or a pooled difference
resamples TASKS first and then init states within each task (the same states for both arms of a difference).
"""
import math

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def sign_test(b, c):
    """Exact two-sided binomial test on the discordant pairs (b: A succeeds and B fails, c: the reverse)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def pooled(table, arms, tasks):
    """table[task][arm] = {trial: 0/1}.  Pooled successes / episodes of each arm over `tasks`."""
    out = {}
    for a in arms:
        k = sum(sum(table[t][a].values()) for t in tasks if a in table[t])
        n = sum(len(table[t][a]) for t in tasks if a in table[t])
        out[a] = (k, n)
    return out


def paired(table, a, b, tasks):
    """Per task, the init states both arms ran, as two aligned 0/1 arrays."""
    pairs = {}
    for t in tasks:
        if a in table[t] and b in table[t]:
            s = sorted(set(table[t][a]) & set(table[t][b]))
            if s:
                pairs[t] = (np.array([table[t][a][i] for i in s], float), np.array([table[t][b][i] for i in s], float))
    return pairs


def bootstrap_diff(pairs, B=10000, seed=0):
    """Pooled rate difference (A - B) over tasks, with the two-level paired cluster bootstrap.
    -> dict(diff, lo, hi (95% two-sided), lo90 (the 5th percentile: one-sided 95% lower bound), samples)."""
    rng = np.random.default_rng(seed)
    ts = list(pairs)
    xa = [pairs[t][0] for t in ts]
    xb = [pairs[t][1] for t in ts]
    n = sum(len(x) for x in xa)
    diff = (sum(x.sum() for x in xa) - sum(x.sum() for x in xb)) / n
    sa = np.empty(B)
    for i in range(B):
        pick = rng.integers(0, len(ts), len(ts))
        num, den = 0.0, 0
        for j in pick:
            m = len(xa[j])
            e = rng.integers(0, m, m)
            num += xa[j][e].sum() - xb[j][e].sum()
            den += m
        sa[i] = num / den
    return dict(diff=float(diff), lo=float(np.percentile(sa, 2.5)), hi=float(np.percentile(sa, 97.5)),
                lo90=float(np.percentile(sa, 5)), hi90=float(np.percentile(sa, 95)), n_pairs=int(n), n_tasks=len(ts))


def bootstrap_rate(table, arm, tasks, B=10000, seed=0):
    """Pooled success of one arm with the two-level cluster bootstrap (tasks, then init states)."""
    rng = np.random.default_rng(seed)
    xs = [np.array(list(table[t][arm].values()), float) for t in tasks if arm in table[t]]
    k = sum(x.sum() for x in xs)
    n = sum(len(x) for x in xs)
    s = np.empty(B)
    for i in range(B):
        pick = rng.integers(0, len(xs), len(xs))
        num, den = 0.0, 0
        for j in pick:
            m = len(xs[j])
            num += xs[j][rng.integers(0, m, m)].sum()
            den += m
        s[i] = num / den
    return dict(rate=float(k / n), lo=float(np.percentile(s, 2.5)), hi=float(np.percentile(s, 97.5)), k=int(k), n=int(n), tasks=len(xs))


def discordant(pairs):
    b = int(sum(((x == 1) & (y == 0)).sum() for x, y in pairs.values()))
    c = int(sum(((x == 0) & (y == 1)).sum() for x, y in pairs.values()))
    return b, c
