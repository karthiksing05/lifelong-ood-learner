"""Programs' byte accounting, segment borrowing's exact rebuild, and the rollout statistics."""
import numpy as np

from lifelong_ood import stats
from lifelong_ood.policy import segcode
from lifelong_ood.policy.compose import DemoProgram


def _windows(seq, T):
    return seq[np.arange(T)[:, None] + np.arange(10)[None]].reshape(T, 70).astype(np.float32)


def _program(mu, variant="raw", T=None):
    T = len(mu) if T is None else T
    return DemoProgram(mu=mu, var=np.full_like(mu, 1e-3), key=np.zeros((T, 5), np.float32), codes=[], variant=variant,
                       states=np.zeros((T, 8), np.float32))


def test_a_replayed_demo_is_charged_its_action_sequence():
    rng = np.random.default_rng(0)
    T = 57
    seq = rng.normal(size=(T + 9, 7)).astype(np.float32)
    prog = _program(_windows(seq, T))
    assert prog.windowed()
    assert prog.bytes() == 2 * (T + 9) * 7 + 2 * T * 8                     # the sequence + the fp16 keys
    shuffled = _program(_windows(seq, T)[rng.permutation(T)])
    assert not shuffled.windowed()
    assert shuffled.bytes() == 2 * T * 70 + 2 * T * 8


def test_segment_borrowing_rebuilds_exactly_from_its_ids():
    rng = np.random.default_rng(1)
    leaves = rng.normal(size=(300, 10, 7)).astype(np.float32)
    T = 43
    S = T + 9
    # a sequence made of leaf segments (one leaf per segment and dimension; a short tail at an offset)
    seq = np.zeros((S, 7), np.float32)
    truth = rng.integers(0, 300, size=(int(np.ceil(S / 10)), 7))
    for s, row in enumerate(truth):
        a, b = s * 10, min(s * 10 + 10, S)
        for d in range(7):
            seq[a:b, d] = leaves[row[d], :b - a, d]
    ids, offs = segcode.encode(leaves, seq)
    hat = segcode.decode(leaves, ids, offs, S)
    assert np.abs(hat - seq).max() == 0.0
    assert segcode.code_bytes(len(ids)) == len(ids) * 7 * 2 + len(ids)
    prog = DemoProgram.from_segcode(leaves, _windows(seq, T), np.zeros((T, 8), np.float32))
    assert np.abs(prog.mu - _windows(seq, T)).max() == 0.0
    assert prog.bytes() == segcode.code_bytes(len(ids)) + 2 * T * 8
    # subsampling keeps every segment (the kept windows still read the whole sequence) and 1/k of the keys
    sub = prog.subsample(5)
    assert sub.bytes() == segcode.code_bytes(len(ids)) + 2 * len(sub.mu) * 8


def test_statistics():
    assert stats.sign_test(0, 0) == 1.0
    assert abs(stats.sign_test(10, 0) - 2 / 2 ** 10) < 1e-12
    assert stats.sign_test(3, 7) == stats.sign_test(7, 3)
    lo, hi = stats.wilson(10, 20)
    assert lo < 0.5 < hi
    rng = np.random.default_rng(2)
    table = {str(t): {"a": {i: int(rng.random() < 0.5) for i in range(20)}} for t in range(6)}
    for t in table:
        table[t]["b"] = dict(table[t]["a"])
    d = stats.bootstrap_diff(stats.paired(table, "a", "b", list(table)), B=200)
    assert d["diff"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0
    assert stats.discordant(stats.paired(table, "a", "b", list(table))) == (0, 0)
