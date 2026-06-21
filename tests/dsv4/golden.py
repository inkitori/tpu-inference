"""Golden capture/replay: persist reference outputs to .npz; TPU tests replay."""
import os

import numpy as np

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")


def save_golden(path: str, **arrays) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})


def load_golden(path: str) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def assert_close_to_golden(name: str, actual, golden: dict, rtol: float,
                           atol: float) -> None:
    np.testing.assert_allclose(golden[name], np.asarray(actual), rtol=rtol,
                               atol=atol)
