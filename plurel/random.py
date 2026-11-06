import hashlib

import numpy as np

Seed = int | np.random.Generator | None


def generator(seed: Seed) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def lane(root: int, *names: str) -> np.random.Generator:
    """The stream of a named thing: it depends on the root and the names alone, so adding a node,
    a column or a key moves no other draw."""
    keys = (hashlib.blake2b(name.encode(), digest_size=8).digest() for name in names)
    return np.random.default_rng([root, *(int.from_bytes(key) for key in keys)])
