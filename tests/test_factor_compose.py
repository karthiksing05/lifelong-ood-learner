"""The overhaul's composition and factor parsing (CPU parts; the GPU identities run in experiments/f2_compose_check.py)."""
import numpy as np
import pytest

from lifelong_ood.policy import factors as F
from lifelong_ood.policy.conditionals import Composition


def test_factor_composition_is_affine():
    c = Composition.factor("base", [("ketchup", "alphabet soup"), ("tray", "basket")], [2.0, 0.5])
    assert c.prompts == ["base", "ketchup", "alphabet soup", "tray", "basket"]
    np.testing.assert_allclose(c.coef, [1.0, 2.0, -2.0, 0.5, -0.5])
    np.testing.assert_allclose(c.coef.sum(0), 1.0)


def test_per_dim_gates_stay_affine():
    lam = np.linspace(0, 3, 70)
    c = Composition.factor("base", [("a", "b")], [lam])
    assert c.coef.shape == (3, 70)
    np.testing.assert_allclose(c.coef.sum(0), 1.0, atol=1e-6)


def test_non_affine_rejected():
    with pytest.raises(AssertionError):
        Composition(["a", "b"], np.array([1.0, 0.5], np.float32)).check()


def test_parse_all_tasks():
    ts = F.load()
    assert len(ts) == 130
    by = {(t.suite, t.index): t for t in ts}
    assert by[("libero_90", 48)].slots == [("In", "ketchup", "basket", "contain")]
    assert by[("libero_90", 58)].slots == [("In", "ketchup", "wooden_tray", "contain")]
    assert by[("libero_90", 69)].slots == [("On", "chocolate_pudding", "plate", "left")]
    assert by[("libero_90", 39)].slots == [("Turnoff", "flat_stove", None, None)]
    assert by[("libero_90", 34)].slots == [("On", "white_yellow_mug", "porcelain_mug", "front")]
    assert by[("libero_goal", 0)].language == "open the middle drawer of the cabinet"     # the trained prompt, not the BDDL's


def test_substitution_keeps_the_scenes_phrasing():
    ts = F.load()
    by = {(t.suite, t.index): t for t in ts}
    t48 = by[("libero_90", 48)]
    ph = F.scene_phrases(t48, ts)
    assert F.substitute(t48, "cream_cheese", ph) == "pick up the cream cheese box and put it in the basket"   # task 47's words
    assert F.substitute(t48, "ketchup", ph) == t48.language
    g1 = by[("libero_goal", 1)]
    assert F.substitute(g1, "wine_bottle", F.scene_phrases(g1, ts)) == "put the wine bottle on the stove"
