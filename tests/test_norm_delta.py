"""Norm with a state-relative action block (pi0_libero's extra_delta_transform): normalise = openpi's DeltaActions then Normalize."""
import numpy as np
import pytest

from lifelong_ood.policy.data import Norm


def _norm(delta, quantile):
    r = np.random.default_rng(0)
    return Norm(q01=-r.random(7).astype(np.float32), q99=r.random(7).astype(np.float32) + 0.5, mean=r.normal(size=7).astype(np.float32),
                std=r.random(7).astype(np.float32) + 0.1, quantile=quantile, delta=delta)


@pytest.mark.parametrize("quantile", [True, False])
def test_delta_matches_openpi_order(quantile):
    r = np.random.default_rng(1)
    acts, st = r.normal(size=(50, 7)).astype(np.float32), r.normal(size=8).astype(np.float32)
    n, plain = _norm(6, quantile), _norm(0, quantile)
    ref = acts.copy()
    ref[:, :6] -= st[:6]                                    # DeltaActions(make_bool_mask(6, -1)): the gripper stays absolute
    np.testing.assert_allclose(n.normalize(acts, state=st), plain.normalize(ref), rtol=1e-6)
    np.testing.assert_allclose(plain.normalize(acts), plain.normalize(acts, state=st), rtol=1e-6)   # pi0.5: the state is ignored
    assert not np.shares_memory(n.normalize(acts, state=st), acts)


def test_delta_needs_state():
    with pytest.raises(AssertionError):
        _norm(6, False).normalize(np.zeros((10, 7), np.float32))
