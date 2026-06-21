import os

import numpy as np
import pytest

from tests.dsv4 import golden


def test_save_and_load_roundtrip(tmp_path):
    p = os.path.join(str(tmp_path), "g.npz")
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    golden.save_golden(p, attn_out=a)
    loaded = golden.load_golden(p)
    np.testing.assert_array_equal(loaded["attn_out"], a)


def test_assert_close_to_golden(tmp_path):
    p = os.path.join(str(tmp_path), "g.npz")
    a = np.ones((4,), dtype=np.float32)
    golden.save_golden(p, x=a)
    g = golden.load_golden(p)
    golden.assert_close_to_golden("x", a + 1e-6, g, rtol=1e-4, atol=1e-4)


def test_assert_close_to_golden_detects_mismatch(tmp_path):
    # Exercise the failure path: a perturbation beyond tolerance must raise.
    p = os.path.join(str(tmp_path), "g.npz")
    a = np.ones((4,), dtype=np.float32)
    golden.save_golden(p, x=a)
    g = golden.load_golden(p)
    perturbed = a.copy()
    perturbed[0] += 1.0
    with pytest.raises(AssertionError):
        golden.assert_close_to_golden("x", perturbed, g, rtol=1e-4, atol=1e-4)


def test_golden_dir_is_absolute_under_tests_dsv4():
    assert os.path.isabs(golden.GOLDEN_DIR)
    assert golden.GOLDEN_DIR.endswith(os.path.join("tests", "dsv4", "goldens"))
