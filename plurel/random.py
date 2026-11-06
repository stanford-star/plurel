import zlib

import numpy as np

Seed = int | np.random.Generator | None


def generator(seed: Seed) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def lane(root: int, *names: str) -> np.random.Generator:
    """The stream of a named thing: it depends on the root and the names alone, so adding a node,
    a column or a key moves no other draw."""
    return np.random.default_rng([root, *(zlib.crc32(name.encode()) for name in names)])
