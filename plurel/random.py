from __future__ import annotations

import numpy as np

Seed = int | np.random.Generator | None


def generator(seed: Seed) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)
