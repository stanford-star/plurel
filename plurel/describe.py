"""A plain description of a table or schema as resolved: built of dicts, lists, strings and
numbers, so it serializes to JSON and depends on no class of plurel. Each column says what it
records and which causes it reads, as latents or as recorded values; hidden helpers, the products
and level codes a declaration compiles to, are folded back into the columns that own them."""

import dataclasses
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from plurel.declare import CODE, Declaration, Derived, ForeignKey, PrimaryKey
from plurel.graph import (
    Foreign,
    LevelEdge,
    LinearEdge,
    MatrixEdge,
    Node,
    Recorded,
    Summary,
)
from plurel.random import Range


def plain(value: object) -> object:
    """A value as plain data: a dataclass as its type and fields, a range as its bounds."""
    if isinstance(value, Range):
        return {"range": [plain(value.low), plain(value.high)]}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
        return {"type": type(value).__name__, **fields}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: plain(item) for key, item in value.items()}
        return [{"key": plain(key), "value": plain(item)} for key, item in value.items()]
    if isinstance(value, tuple | list):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"array": list(value.shape)} if value.size > 16 else plain(value.tolist())
    if isinstance(value, pd.Series):
        return {"given": len(value)}
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if callable(value):
        return getattr(value, "__name__", type(value).__name__)
    return value


def reading(tail: object) -> dict[str, object]:
    """What a tail reads and in which values: a column of this table, one across a key, or a
    summary of the rows pointing back."""
    if isinstance(tail, Recorded):
        return {"reads": tail.column, "values": "recorded", "fill": tail.fill}
    if isinstance(tail, Foreign):
        inner = reading(tail.node)
        return {**inner, "reads": f"{tail.key}.{inner['reads']}"}
    if isinstance(tail, Summary):
        inner = {} if tail.node is None else reading(tail.node)
        return {
            "summary": tail.table,
            "key": tail.key,
            "how": tail.how,
            "reads": inner.get("reads"),
            "values": inner.get("values", "rows"),
            "fill": tail.fill,
        }
    return {"reads": str(tail).removesuffix(CODE), "values": "latent"}


def cause(edge: object, helpers: Mapping[str, Node], categories: tuple | None) -> dict[str, object]:
    """One edge as a cause: what it reads and its effect; a product a declaration compiled to a
    helper is read as the pair it multiplies."""
    if isinstance(edge, LinearEdge) and edge.parent in helpers:
        first, second = helpers[edge.parent].edges
        described = {"product": [reading(first.parent)["reads"], reading(second.parent)["reads"]]}
        if isinstance(first, LevelEdge):
            slopes = {str(level): effect for level, effect in first.effects}
            return {**described, "slopes": slopes, "values": "latent"}
        return {
            **described,
            "weight": edge.weight,
            "transform": plain(first.transform),
            "values": "latent",
        }
    described = reading(edge.parent)
    if isinstance(edge, LinearEdge):
        return {**described, "weight": edge.weight, "transform": plain(edge.transform)}
    if isinstance(edge, LevelEdge):
        return {**described, "effects": {str(k): v for k, v in edge.effects}}
    if isinstance(edge, MatrixEdge) and categories is not None and edge.matrix.shape[0] == 1:
        return {**described, "scores": dict(zip(map(str, categories), plain(edge.matrix[0])))}
    return {**described, "edge": type(edge).__name__, "dim": edge.dim}


def mechanism(node: Node, helpers: Mapping[str, Node], categories: tuple | None = None) -> dict:
    return {
        "causes": [cause(edge, helpers, categories) for edge in node.edges],
        "how": node.op,
        "noise": plain(node.noise),
    }


def column(name: str, entry: object, helpers: Mapping[str, Node]) -> dict[str, object]:
    """One authored entry of a table, resolved."""
    if isinstance(entry, PrimaryKey):
        return {"kind": "primary key", "given": entry.values is not None}
    if isinstance(entry, ForeignKey):
        return {
            "kind": "foreign key",
            "parent": entry.parent,
            "given": entry.values is not None,
            "link": plain(entry.link),
            "nullable": entry.nullable,
            "fill": entry.fill,
            "match": dict(entry.match),
            "prefer": plain(dict(entry.prefer)),
            "counts": plain(entry.counts),
            "unique": list(entry.together) if not isinstance(entry.unique, bool) else entry.unique,
        }
    if isinstance(entry, Derived):
        return {"kind": "derived", "inputs": list(entry.inputs), "function": plain(entry.function)}
    if isinstance(entry, Node):
        return {"kind": "hidden", "dim": entry.dim, **mechanism(entry, helpers)}
    out: dict[str, object] = {"kind": entry.kind, "given": entry.values is not None}
    if entry.values is not None:
        return out | ({"levels": plain(entry.categories)} if entry.kind == "categorical" else {})
    out |= mechanism(entry.node, helpers, entry.categories)
    if entry.kind == "categorical":
        shares = entry.fit_shares or entry.category_probabilities
        out |= {
            "levels": plain(entry.categories),
            "shares": plain(shares),
            "ordered": entry.ordered,
        }
    else:
        out["marginal"] = plain(entry.marginal)
        options = {"clip": entry.clip, "round": entry.round, "zeros": entry.zeros}
        out |= {key: plain(value) for key, value in options.items() if value is not None}
        if entry.by_level is not None:
            parent, ranges = entry.by_level
            out["by_level"] = {
                "parent": parent,
                "ranges": {str(k): [lo, hi] for k, lo, hi in ranges},
            }
    out["missing"] = entry.missing
    if entry.missing_by is not None:
        out["missing_by"] = list(entry.missing_by)
    calibrated = {}
    if entry.binning is not None:
        calibrated["cuts"] = plain(entry.binning)
    if entry.fit_shares is not None and entry.node.bias is not None:
        calibrated["bias"] = plain(entry.node.bias)
    if calibrated:
        out["calibrated"] = calibrated
    return out


def describe_table(table: object) -> dict[str, object]:
    """A table as resolved: its metadata, then each authored entry. A declaration is kept as
    written under `declared`, ranges and all, beside what it resolved to."""
    helpers = {
        name: node
        for name, node in table.nodes.items()
        if name not in table.authored and node.op == "product" and len(node.edges) == 2
    }
    entries = {}
    for name, authored in table.authored.items():
        described = column(name, table.table[name], helpers)
        if name in table.docs:
            described["doc"] = table.docs[name]
        if (
            isinstance(authored, Declaration)
            or isinstance(authored, ForeignKey)
            and authored.family
        ):
            described["declared"] = plain(authored)
        entries[name] = described
    return {
        "name": table.name,
        "doc": table.doc,
        "target": table.target,
        "sources": table.sources,
        "time_column": table.time_column,
        "primary_key": table.pkey_column,
        "rows": table.size,
        "columns": entries,
    }


def describe_schema(schema: object) -> dict[str, object]:
    """Every table of a schema, and for each the tables it points at and those holding events."""
    return {
        "tables": {name: describe_table(table) for name, table in schema.tables.items()},
        "keys": [
            {"table": table, "column": column, "parent": fk.parent}
            for (table, column), fk in schema.keys.items()
        ],
        "events": sorted(schema.eventful),
    }


def said(cause: Mapping[str, object]) -> str:
    """A cause as a short phrase."""
    if "summary" in cause:
        what = cause["how"] + ("" if cause["reads"] is None else f" of {cause['reads']}")
        read = f"{what} over {cause['summary']}.{cause['key']}"
    elif "product" in cause:
        read = " * ".join(cause["product"])
    else:
        read = cause["reads"]
    if cause.get("values") == "recorded":
        read = f"recorded {read}"
    if "weight" in cause:
        transform = cause.get("transform", "identity")
        return f"{cause['weight']:g} * {read}" + (
            "" if transform == "identity" else f" ({transform})"
        )
    for key in ("effects", "slopes", "scores"):
        if key in cause:
            return f"{read} by {key[:-1] if key != 'slopes' else 'slope'}: {cause[key]}"
    return f"{read} ({cause.get('edge', 'edge')})"


def explain(description: Mapping[str, object]) -> str:
    """A description, schema or table, as readable lines."""
    tables = description.get("tables", {None: description})
    lines = []
    for name, table in tables.items():
        notes = [f"{key} {table[key]}" for key in ("time_column", "target") if table.get(key)]
        lines.append(" ".join(filter(None, [name or table.get("name") or "table", *notes])))
        if table.get("doc"):
            lines.append(f"  {table['doc']}")
        for column, entry in table["columns"].items():
            head = entry["kind"] if not entry.get("given") else f"{entry['kind']}, given"
            if entry["kind"] == "foreign key":
                head += f" -> {entry['parent']}"
                chosen = [key for key in ("match", "prefer", "counts", "unique") if entry[key]]
                head += "".join(f", {key} {entry[key]}" for key in chosen)
            lines.append(f"  {column}: {head}")
            if entry.get("doc"):
                lines.append(f"    {entry['doc']}")
            if entry.get("marginal"):
                lines.append(f"    marginal {entry['marginal']['type']}")
            if entry.get("levels") and not entry.get("given"):
                lines.append(f"    levels {dict(zip(map(str, entry['levels']), entry['shares']))}")
            for each in entry.get("causes", ()):
                lines.append(f"    <- {said(each)}")
            if entry["kind"] == "derived":
                lines.append(f"    = {entry['function']}({', '.join(entry['inputs'])})")
            if entry.get("calibrated"):
                lines.append(f"    calibrated {entry['calibrated']}")
    return "\n".join(lines)
