"""One draw of a compiled schema: links every key, evaluates every node and records every column in
the schema's one topological order, each from its own named stream, then assembles the frames.
`plurel.schema` compiles and checks a schema; this module only executes it."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from plurel.columns import Column
from plurel.declare import CODE, ForeignKey
from plurel.graph import AGGREGATES, COMPLETE, Edge, Foreign, Node, Recorded, Summary
from plurel.links import counted, distinct, joint_codes, matched, repeats
from plurel.random import lane


@dataclass(frozen=True)
class Linked:
    """The vertex of a key's links: the parent row each row of its table points at."""

    column: str


def checked(name: str, node: Node, latent: np.ndarray, n: int) -> np.ndarray:
    if latent.shape != (n, node.dim):
        raise ValueError(f"{name!r} produced {latent.shape}, declared {(n, node.dim)}")
    if not np.isfinite(latent).all():
        raise ValueError(f"{name!r} produced non-finite values")
    return latent


def numbers(recorded: pd.Series, where: str) -> np.ndarray:
    """A record as floats, a time as epoch seconds and a blank as NaN."""
    if getattr(recorded.dtype, "tz", None) is not None:
        raise ValueError(
            f"{where} holds times in a time zone, which a time read from it would lose: give it"
            " naive times, e.g. converted with .dt.tz_convert(None)"
        )
    if pd.api.types.is_datetime64_any_dtype(recorded):
        seconds = recorded.to_numpy("datetime64[ns]").astype(np.int64) / 1e9
        return np.where(recorded.isna().to_numpy(), np.nan, seconds)
    return recorded.to_numpy(dtype=float, na_value=np.nan)


def filled(values: np.ndarray, tail: Recorded, where: str) -> np.ndarray:
    """Recorded values with the tail's fill where there are none."""
    blank = np.isnan(values)
    if not blank.any():
        return values
    if tail.fill is None:
        raise ValueError(
            f"{where} has rows without a recorded value, blank or with a key pointing at nothing:"
            " give Recorded a fill"
        )
    return np.where(blank, tail.fill, values)


def per(fk: ForeignKey) -> str:
    return f" per {', '.join(fk.together)}" if fk.together else ""


class Run:
    """One draw of `schema` at `rows` from `root`, and what it has made so far: latents, records
    and records as numbers by table, and links by table and key. A key that counts its table's
    rows sets that table's size in `rows`."""

    def __init__(self, schema, rows: dict[str, int], root: int) -> None:
        self.schema, self.rows, self.root = schema, rows, root
        self.latents: dict[str, dict[str, np.ndarray]] = {t: {} for t in schema.tables}
        self.records: dict[str, dict[str, pd.Series]] = {t: {} for t in schema.tables}
        self.recorded: dict[str, dict[str, np.ndarray]] = {t: {} for t in schema.tables}
        self.links: dict[tuple[str, str], np.ndarray] = {}

    def stream(self, kind: str, table: str, name: str) -> np.random.Generator:
        return lane(self.root, kind, self.schema.identity[table], name)

    def propagate(self, held: Mapping[str, Mapping[str, np.ndarray]]) -> "Run":
        """Link every key, evaluate every node and record every column once, in topological order:
        a held node takes its values, any other its edges plus noise."""
        for table, name in self.schema.order:
            scm = self.schema.tables[table]
            if isinstance(name, Linked):
                self.links[table, name.column] = self.link(table, name.column)
                continue
            n = self.rows[table]
            if isinstance(name, Recorded):
                column = name.column
                record = scm.record(
                    column,
                    self.latents[table],
                    self.records[table],
                    self.stream("column", table, column),
                    n,
                )
                self.records[table][column] = record
                if (table, column) in self.schema.read:
                    self.recorded[table][column] = numbers(record, f"{table}.{column}")
                continue
            node = scm.nodes[name]
            if name in held.get(table, {}):
                self.latents[table][name] = checked(name, node, held[table][name], n)
                continue
            edges = self.schema.crossings.get((table, name), ())
            value = self.evaluate(table, node, edges, self.stream("noise", table, name))
            self.latents[table][name] = checked(name, node, value, n)
        return self

    def evaluate(
        self, table: str, node: Node, edges: tuple[Edge, ...], stream: np.random.Generator
    ) -> np.ndarray:
        """A node's value over the rows of `table`: its edges, those the schema resolves among
        them, plus its noise."""
        exogenous, across = node.sample_noise(self.rows[table], stream), None
        for edge in edges:
            term = edge.apply(self.resolve(table, edge.parent))
            across = term if across is None else across + term
        parents = {parent: self.latents[table][parent] for parent in node.parents}
        return node.evaluate(parents, exogenous, across)

    def resolve(self, table: str, tail: Foreign | Summary | Recorded) -> np.ndarray:
        """The array a tail the schema resolves contributes to `table`, one row per row of it."""
        n = self.rows[table]
        if isinstance(tail, Recorded):
            return filled(self.recorded[table][tail.column], tail, f"{table}.{tail.column}")[
                :, None
            ]
        if isinstance(tail, Foreign):
            fk = self.schema.keys[table, tail.key]
            indices = self.links[table, tail.key]
            if isinstance(tail.node, Recorded):
                values, linked = np.full(n, np.nan), indices >= 0
                values[linked] = self.recorded[fk.parent][tail.node.column][indices[linked]]
                where = f"{tail.key}.{tail.node.column} of {table!r}"
                return filled(values, tail.node, where)[:, None]
            source = self.latents[fk.parent][tail.node]
            linked = indices >= 0
            if linked.all():
                return source.take(indices, axis=0).astype(float, copy=False)
            fill = -1.0 if tail.node.endswith(CODE) else fk.fill  # no level, so no effect
            if fill is None:
                raise ValueError(f"key {tail.key!r} of {table!r} has null rows; set its fill")
            out = np.full((n, source.shape[1]), float(fill))
            out[linked] = source[indices[linked]]
            return out
        indices = self.links[tail.table, tail.key]
        linked = indices >= 0
        if isinstance(tail.node, Recorded):
            read = self.recorded[tail.table][tail.node.column]
            linked &= ~np.isnan(read)
            values = read[linked][:, None]
        elif tail.node is None:
            values = np.empty((int(linked.sum()), 1))
        else:
            values = self.latents[tail.table][tail.node][linked]
        out = AGGREGATES[tail.how](values, indices[linked], n)
        empty = np.bincount(indices[linked], minlength=n) == 0
        if empty.any() and tail.how not in COMPLETE:
            if tail.fill is None:
                read = getattr(tail.node, "column", tail.node)
                raise ValueError(
                    f"summary of {tail.table}.{read} met rows with nothing to aggregate; set fill"
                )
            out[empty] = tail.fill
        return out

    def link(self, table: str, column: str) -> np.ndarray:
        """The parent row each row of `table` points at through `column`: given, counted per
        parent, matched among eligible parents by preference, or drawn by its link."""
        fk, rows = self.schema.keys[table, column], self.rows
        if fk.values is not None:
            indices = self.given(table, column, fk)
            if fk.unique and repeats(self.unique_groups(table, fk), indices).any():
                raise ValueError(
                    f"given values of {column!r} in {table!r} repeat a parent{per(fk)}"
                )
            return indices
        stream = self.stream("link", table, column)
        log_weights = np.zeros(rows[fk.parent])
        if (table, column) in self.schema.choosers:
            node, edges = self.schema.choosers[table, column]
            value = self.evaluate(fk.parent, node, edges, self.stream("key", table, column))
            log_weights = checked(f"{column} weights", node, value, rows[fk.parent])[:, 0]
        if fk.counts is not None:
            weights = np.exp(log_weights - log_weights.max()) if len(log_weights) else log_weights
            counts = fk.counts
            drawn = counted(
                weights, counts.mean, rows.get(table), stream, counts.most, counts.least
            )
            rows[table] = len(drawn)
        elif fk.match or fk.prefer or fk.unique:
            child, parent = self.groups(table, fk)
            if fk.unique:
                group = self.unique_groups(table, fk)
                try:
                    leave = fk.unmatched == "null"
                    drawn = distinct(child, parent, log_weights, group, stream, leave)
                except ValueError as error:
                    raise ValueError(
                        f"{column!r} of {table!r} must not repeat a parent{per(fk)}: {error}"
                    ) from error
            else:
                drawn = matched(child, parent, log_weights, stream)
            if fk.unmatched == "raise" and (unmatched := int((drawn < 0).sum())):
                raise ValueError(
                    f"{unmatched} rows of {table!r} match no row of {fk.parent!r} on"
                    f' {dict(fk.match)}: pass unmatched="null" to {column!r} to leave them'
                    " pointing at nothing"
                )
        else:
            drawn = np.asarray(fk.link.sample(rows[table], rows[fk.parent], stream))
        if drawn.shape != (rows[table],) or not np.issubdtype(drawn.dtype, np.integer):
            raise ValueError(f"link {column!r} must return one integer per child row")
        if len(drawn) and (drawn.min() < -1 or drawn.max() >= rows[fk.parent]):
            raise ValueError(f"link {column!r} points outside the parent table")
        indices = drawn.astype(np.int64)
        if fk.nullable:
            indices[stream.random(len(indices)) < fk.nullable] = -1
        return indices

    def given(self, table: str, column: str, fk: ForeignKey) -> np.ndarray:
        """Row positions in the parent of given key values; a blank points at nothing."""
        keys = self.schema.parent_keys(fk)
        index = pd.RangeIndex(self.rows[fk.parent]) if keys is None else pd.Index(keys)
        indices = index.get_indexer(fk.values)
        if ((indices < 0) & fk.values.notna().to_numpy()).any():
            raise ValueError(f"key {column!r} of {table!r} holds values its parent lacks")
        return indices.astype(np.int64)

    def groups(self, table: str, fk: ForeignKey) -> tuple[np.ndarray, np.ndarray]:
        """One integer per child row and per parent row, equal where every matched column agrees,
        -1 where one is blank."""
        n = self.rows[table]
        columns = [
            np.concatenate([self.values(table, mine), self.values(fk.parent, theirs)])
            for mine, theirs in fk.match.items()
        ]
        joint = joint_codes(columns, n + self.rows[fk.parent])
        return joint[:n], joint[n:]

    def unique_groups(self, table: str, fk: ForeignKey) -> np.ndarray:
        """The groups whose rows a unique key must not point at one parent twice."""
        columns = [self.values(table, column) for column in fk.together]
        return joint_codes(columns, self.rows[table])

    def values(self, table: str, column: str) -> np.ndarray:
        """A matched or grouping column's values, NaN where blank: a key's parent rows or a
        categorical's levels."""
        if column in self.schema.tables[table].fkeys:
            indices = self.links[table, column]
            return np.where(indices < 0, np.nan, indices).astype(object)
        return self.records[table][column].astype(object).to_numpy()

    def frames(self) -> dict[str, pd.DataFrame]:
        """One frame per table of its records, in the order the table declares them, with its key
        columns."""
        frames = {}
        for table, scm in self.schema.tables.items():
            records = self.records[table]
            framed = {
                name: c.framed(records[name]) if isinstance(c, Column) else records[name]
                for name, c in scm.columns.items()
            }
            frame = pd.DataFrame(framed, index=range(self.rows[table]))
            for column, fk in scm.fkeys.items():
                indices, given = self.links[table, column], self.schema.parent_keys(fk)
                if given is None:
                    frame[column] = pd.Series(indices, dtype="Int64").mask(indices < 0)
                else:
                    keys = given.astype("Int64") if pd.api.types.is_integer_dtype(given) else given
                    # by label, so a row pointing at nothing reads a blank, even of no parent rows
                    frame[column] = keys.reindex(indices).reset_index(drop=True)
            for name in scm.derived:
                frame[name] = records[name]
            frames[table] = frame[[name for name in scm.table if name in frame]]
        return frames
