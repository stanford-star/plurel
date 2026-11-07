"""A table read as a catalog's record: its recorded columns, its causal graph and the rules a record
keeps on top of what a table's constructor already checks."""

from collections.abc import Mapping

from plurel import ByLevel, Categorical, Derived, Hidden, Numeric, Scaled, Shares, Table


def features(table: Table) -> list[str]:
    """The recorded columns but the target, in authored order."""
    return [
        name
        for name, entry in table.declared.items()
        if name != table.target and not isinstance(entry, Hidden)
    ]


def level_names(entry: Categorical) -> list:
    return list(entry.levels.levels if isinstance(entry.levels, Shares) else entry.levels)


def shares(entry: Categorical) -> Mapping | None:
    """The authored shares of a categorical entry; `None` when they are drawn or equal."""
    levels = entry.levels.levels if isinstance(entry.levels, Shares) else entry.levels
    return levels if isinstance(levels, Mapping) else None


def is_numeric(entry: object) -> bool:
    return isinstance(entry, Numeric | Derived)


def parents(table: Table, name: str) -> list[str]:
    """Every column or cause `name` reads: its causes, a product's two, what it scores or is
    computed from and what sets its ranges."""
    entry, found = table.declared[name], []
    for cause in getattr(entry, "causes", {}):
        found.extend(cause if isinstance(cause, tuple) else [cause])
    for scored in (getattr(entry, "scores", None) or {}).values():
        found.extend(scored)
    if isinstance(entry, Derived):
        found.extend(entry.inputs)
    if isinstance(getattr(entry, "marginal", None), ByLevel):
        found.append(entry.marginal.parent)
    return list(dict.fromkeys(found))


def children(table: Table) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {name: [] for name in table.declared}
    for child in table.declared:
        for parent in parents(table, child):
            out[parent].append(child)
    return out


def ancestors(table: Table, name: str) -> set[str]:
    seen, frontier = set(), parents(table, name)
    while frontier:
        node = frontier.pop()
        if node not in seen:
            seen.add(node)
            frontier.extend(parents(table, node))
    return seen


def pairs(entry: object) -> list[tuple[str, str]]:
    return [cause for cause in getattr(entry, "causes", {}) if isinstance(cause, tuple)]


def domain(table: Table) -> str:
    """A placed record's subject line: its group's title in lower case, else its name."""
    title = getattr(table, "title", None)
    return table.name if title is None else title.lower()


def _weighted_transforms(entry: object) -> list[object]:
    """Each weighted cause's transform, None for a bare weight; causes by level have none."""
    effects = (e for c, e in entry.causes.items() if not isinstance(c, tuple))
    pairs_of = [
        e if isinstance(e, tuple) else (e, None) for e in effects if not isinstance(e, Mapping)
    ]
    return [transform for _, transform in pairs_of if not isinstance(transform, Mapping)]


def validate(table: Table) -> None:
    """Raises naming the first rule the record breaks."""

    def check(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"{table.name}: {message}")

    if not (table.name or "").strip():
        raise ValueError("a record needs a name")
    check(bool((table.doc or "").strip()), "needs a table doc")
    check(table.target is not None, "needs a target")
    check(not table.fkeys, "a catalog record is one table")
    target = table.declared[table.target]
    check(isinstance(target, Categorical), "the target must be categorical")
    check(bool(target.causes or target.scores), "the target needs causes or scores")
    check(shares(target) is not None, "the target authors its shares")
    check(not target.missing, "the target is never blank")
    below = children(table)
    check(not below[table.target], "the target must have no children (label leakage)")
    for name, entry in table.declared.items():
        check(
            bool((getattr(entry, "doc", None) or "").strip()), f"{name!r}: every column needs a doc"
        )
        if isinstance(entry, Hidden):
            check(not entry.causes, f"{name!r}: a hidden cause is a root")
            check(len(below[name]) >= 2, f"{name!r}: a hidden cause feeds two or more columns")
            continue
        if isinstance(entry, Derived):
            continue
        check(
            isinstance(entry, Numeric | Categorical), f"{name!r}: a record's columns are authored"
        )
        if isinstance(entry, Numeric):
            check(
                isinstance(entry.marginal, Scaled | ByLevel),
                f"{name!r}: a number names its units and range with Scaled or ByLevel",
            )
        scored = isinstance(entry, Categorical) and entry.scores is not None
        if getattr(entry, "causes", None):
            check(entry.noise is not None, f"{name!r}: a column with causes authors its noise")
        if entry.how != "sum":
            enough = len(parents(table, name)) >= 2 and not scored
            check(enough, f"{name!r}: {entry.how} combines two or more causes")
        if entry.how == "rules":
            steps = all(transform == "step" for transform in _weighted_transforms(entry))
            check(steps, f"{name!r}: rules read thresholds: every weighted cause has `step`")
        for a, b in pairs(entry):
            check(
                a in entry.causes and b in entry.causes, f"{name!r}: {(a, b)} must both be causes"
            )
            if name == table.target:
                recorded = not isinstance(table.declared[a], Hidden) and not isinstance(
                    table.declared[b], Hidden
                )
                check(recorded, f"{name!r}: the target's product reads recorded columns")
        if scored:
            check(len(level_names(entry)) >= 3, f"{name!r}: scores need three or more levels")
            check(any(entry.scores.values()), f"{name!r}: scores need a driving column")
