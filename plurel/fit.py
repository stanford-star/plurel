"""Realizing fits a database to one draw of its rows: each scored column's bias to its shares, then
every level cut pinned where that draw puts it."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from plurel.columns import Column
from plurel.graph import LookupEdge, Node, level_edges
from plurel.random import lane

CALIBRATION_ROWS = 4096  # rows of the draw that realizing fits a table's edges and scores to
CALIBRATION_PASSES = 100
CALIBRATION_TOLERANCE = 0.01  # the largest bias step, in noise units, at which a fit stops


def fit_score(table, name: str, latents: Mapping[str, np.ndarray], rng: np.random.Generator):
    """The table with a bias under which its scored column `name` meets its shares: a fixed point
    on one draw and fresh Gumbel noise, each pass shifting a level by the noise scale times the log
    of its wanted over its seen share."""
    column = table.columns[name]
    node, n = column.node, len(latents[name])
    scores = sum(edge.apply(latents[edge.parent]) for edge in node.edges)
    noise = rng.gumbel(0.0, node.noise.scale, size=(n, node.dim))
    wanted, biases = np.asarray(column.fit_shares), np.zeros(node.dim)
    for _ in range(CALIBRATION_PASSES):
        codes = np.argmax(scores + biases + noise, axis=1)
        shares = np.bincount(codes, minlength=node.dim) / n
        step = node.noise.scale * np.log(wanted / np.maximum(shares, 1.0 / n))
        if np.abs(step).max() < CALIBRATION_TOLERANCE * node.noise.scale:
            break
        biases += step
    bias = tuple(float(b) for b in biases - biases.mean())
    return table.like({**table.table, name: replace(column, node=replace(node, bias=bias))})


def cut(table, latents: Mapping[str, np.ndarray]):
    """The table with every binned column cut at fixed edges, its latent's quantiles at its shares,
    and what reads its levels cut there too. Rows sharing a parent are not independent, so a column
    that depends on a key keeps cutting each draw at its shares."""
    edges = {
        name: tuple(float(e) for e in level_edges(latents[name], column.category_probabilities))
        for name, column in table.columns.items()
        if column.kind == "categorical" and column.values is None and column.node.dim == 1
        if name not in table.crossing
    }

    def fixed(node: Node) -> Node:
        reads = (
            replace(e, edges=edges[e.parent])
            if isinstance(e, LookupEdge) and e.parent in edges
            else e
            for e in node.edges
        )
        return replace(node, edges=tuple(reads))

    entries: dict[str, object] = {}
    for name, entry in table.table.items():
        if isinstance(entry, Column) and entry.values is None:
            entry = replace(entry, node=fixed(entry.node), binning=edges.get(name))
        elif isinstance(entry, Node):
            entry = fixed(entry)
        entries[name] = entry
    return table.like(entries)


def fits(table) -> bool:
    """Whether realizing fits anything of a table to a draw: a scored column, or a level cut of a
    generated table's categorical that depends on no key, as `cut` pins."""
    for name, column in table.columns.items():
        if not isinstance(column, Column) or column.values is not None:
            continue
        cuttable = column.kind == "categorical" and column.node.dim == 1
        if column.fit_shares or table.size is None and cuttable and name not in table.crossing:
            return True
    return False


def calibration_rows(schema, given: Mapping[str, int] | None) -> dict[str, int]:
    """Row counts of the calibration draw: those given, else each table's given rows, else
    `CALIBRATION_ROWS`; a table whose rows a key counts takes what the counts give, and one with a
    key unique on its own takes at most its parent's rows, a counted parent then taking the total
    its counts give on average."""
    given = dict(given or {})
    rows = {
        t: CALIBRATION_ROWS if scm.size is None else scm.size
        for t, scm in schema.tables.items()
        if t not in schema.sized
    }
    for fk in schema.keys.values():
        if fk.values is not None and schema.parent_keys(fk) is None and fk.values.notna().any():
            rows[fk.parent] = max(rows[fk.parent], int(fk.values.max()) + 1)
    rows |= given
    for (table, _), fk in schema.keys.items():
        if fk.unique is not True or fk.nullable or table in given or table not in rows:
            continue
        if fk.parent not in rows and fk.parent in schema.sized:
            counts = schema.keys[fk.parent, schema.sized[fk.parent]]
            if counts.parent in rows:
                total = round(counts.counts.mean * rows[counts.parent])
                total = max(total, counts.counts.least * rows[counts.parent])
                if counts.counts.most is not None:
                    total = min(total, counts.counts.most * rows[counts.parent])
                rows[fk.parent] = max(1, min(total, CALIBRATION_ROWS))
        if fk.parent in rows:
            rows[table] = min(rows[table], rows[fk.parent])
    return rows


def fitted(schema, seed: int, given: Mapping[str, int] | None = None):
    """The schema with its scores and level cuts fit to one draw, each scored column after those
    it reads; a schema with nothing to fit is itself. The draw takes `calibration_rows`. Every fit
    draws from streams named by its table and column, so adding a column moves no other fit. Given
    rows cut each draw: they are all the rows there are."""
    if not any(fits(scm) for scm in schema.tables.values()):
        return schema
    tables, rebuilt = dict(schema.tables), type(schema)
    rows = calibration_rows(schema, given)

    def draw(stream):
        try:
            return rebuilt(tables).sample_with_latents(rows, seed=stream)[1]
        except ValueError as error:
            raise ValueError(
                f"realizing fits to one draw of {rows} rows, which failed: {error}. If those"
                " sizes cannot be drawn, pass feasible ones to realize(seed, rows)"
            ) from error

    for table, scm in schema.tables.items():
        for name in scm.order:
            if getattr(scm.columns.get(name), "fit_shares", None):
                identity = schema.identity[table]
                latents = draw(lane(seed, "fit", identity, name))
                noise = lane(seed, "fit noise", identity, name)
                tables[table] = fit_score(tables[table], name, latents[table], noise)
    latents = draw(lane(seed, "fit cuts"))
    return rebuilt(
        {t: scm if scm.size is not None else cut(scm, latents[t]) for t, scm in tables.items()}
    )
