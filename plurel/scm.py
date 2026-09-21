from collections.abc import Hashable, Mapping
from graphlib import CycleError, TopologicalSorter

from plurel.columns import Column
from plurel.mechanisms import Node


def topological[T: Hashable](parents: Mapping[T, tuple[T, ...]]) -> tuple[T, ...]:
    try:
        return tuple(TopologicalSorter(parents).static_order())
    except CycleError as error:
        raise ValueError("nodes must form a directed acyclic graph") from error


class SCM:
    """A table's DAG and the columns that observe it; a Schema executes it. Edge tails that
    cross keys are the table's `inputs`."""

    def __init__(
        self,
        nodes: Mapping[str, Node],
        columns: Mapping[str, Column],
        time_column: str | None = None,
    ) -> None:
        self.nodes = dict(nodes)
        inputs: set[Hashable] = set()
        for child, node in self.nodes.items():
            local = {parent for parent in node.parents if isinstance(parent, str)}
            if unknown := local - set(self.nodes):
                raise ValueError(f"{child!r} refers to unknown parents {sorted(unknown)}")
            inputs |= set(node.parents) - local
        self.inputs = frozenset(inputs)
        parents = {
            name: tuple(parent for parent in node.parents if isinstance(parent, str))
            for name, node in self.nodes.items()
        }
        self.order = topological(parents)
        self.columns = dict(columns)
        for name, column in self.columns.items():
            if column.kind == "key":
                continue
            needed = {column.node}
            if isinstance(column.missing, str):
                needed.add(column.missing)
            if unknown := needed - set(self.nodes):
                raise ValueError(f"column {name!r} refers to unknown nodes {sorted(unknown)}")
        kinds = {name: column.kind for name, column in self.columns.items()}
        for name, column in self.columns.items():
            if column.after is not None and (
                column.after == name or kinds.get(column.after) != "timestamp"
            ):
                raise ValueError(f"column {name!r} must come after another timestamp column")
        keys = [name for name, kind in kinds.items() if kind == "key"]
        if len(keys) > 1:
            raise ValueError("a table has at most one key column")
        if time_column is not None and kinds.get(time_column) != "timestamp":
            raise ValueError(f"time column {time_column!r} must be a timestamp column")
        self.pkey_column = keys[0] if keys else None
        self.time_column = time_column
        self.timestamp_nodes = frozenset(
            column.node for column in self.columns.values() if column.kind == "timestamp"
        )
