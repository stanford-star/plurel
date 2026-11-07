"""Near-duplicate schemas: two records about the same row entity with the same or an equivalent target.
The score is the smaller of two Jaccard similarities, one over record tokens (id and the description's entity
clause) and one over target tokens (target name and class names); pairs above the threshold are candidates a
reader resolves in the schemas."""

from __future__ import annotations

import re

import numpy as np
from scipy import sparse

from plurel import Table
from plurel.world.authoring.realism import ENTITY
from plurel.world.record import level_names

STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "in",
        "at",
        "on",
        "for",
        "to",
        "by",
        "with",
        "per",
        "within",
        "from",
        "over",
        "under",
        "one",
        "each",
        "row",
        "is",
    ]
)
TOKEN = re.compile(r"[a-z0-9]+")
THRESHOLD = 0.6  # the smaller of the record and target Jaccards: a duplicate shares both the entity and the outcome


def _tokens(text: str) -> frozenset[str]:
    words = TOKEN.findall(text.lower().replace("_", " "))
    return frozenset(
        w[:-1] if w.endswith("s") and len(w) > 3 else w
        for w in words
        if w not in STOP and not w.isdigit()
    )


def record_tokens(schema: Table) -> frozenset[str]:
    """The row entity: the id and the description's entity clause."""
    entity = ENTITY.search(schema.doc)
    return _tokens(" ".join([schema.name, entity.group(1) if entity else ""]))


def target_tokens(schema: Table) -> frozenset[str]:
    """The outcome: the target name and its class names."""
    return _tokens(" ".join([schema.target, *level_names(schema.declared[schema.target])]))


def _jaccard(sets: list[frozenset[str]]) -> np.ndarray:
    """Pairwise Jaccard similarity of token sets, through a sparse presence matrix: a schema holds
    a few tokens of a vocabulary of thousands, and a dense integer product of that matrix with its
    transpose would run outside BLAS for minutes."""
    vocab = {t: i for i, t in enumerate(sorted(set().union(*sets)))}
    rows = np.repeat(np.arange(len(sets)), [len(toks) for toks in sets])
    cols = np.fromiter((vocab[t] for toks in sets for t in toks), dtype=np.int64, count=len(rows))
    presence = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(sets), len(vocab)))
    inter = (presence @ presence.T).toarray()
    size = np.asarray(presence.sum(axis=1)).ravel()
    union = size[:, None] + size[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def candidates(
    schemas, *, threshold: float, among: set[str] | None
) -> list[tuple[str, str, float]]:
    """(schema_id, schema_id, score) pairs above the threshold, highest first; with ``among``, only pairs that
    involve one of those ids. A pair is resolved in the schemas (one deleted, or the records told apart), never listed away."""
    schemas = list(schemas)
    score = np.minimum(
        _jaccard([record_tokens(s) for s in schemas]), _jaccard([target_tokens(s) for s in schemas])
    )
    rows, cols = np.triu_indices(len(schemas), k=1)
    keep = score[rows, cols] >= threshold
    out = []
    for i, j in zip(rows[keep], cols[keep], strict=True):
        a, b = schemas[i].name, schemas[j].name
        if among is None or a in among or b in among:
            out.append((a, b, float(score[i, j])))
    return sorted(out, key=lambda x: -x[2])


def report(pairs: list[tuple[str, str, float]], schemas) -> str:
    """One line per candidate pair with both files, for the reader who confirms or clears them."""
    where = {s.name: "/".join(getattr(s, "taxonomy", ("?",))) for s in schemas}
    return (
        "\n".join(f"{score:.2f}  {a}  ({where[a]})  vs  {b}  ({where[b]})" for a, b, score in pairs)
        or "no candidate pairs"
    )
