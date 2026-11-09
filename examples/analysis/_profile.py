"""The profile `diversity.py` and `realism.py` measure on a database: its structure, read off a
description such as `Schema.describe()`, and its data, read off its frames. Imported, not run."""

import numpy as np
import pandas as pd

RECORDED = ("numeric", "categorical", "timestamp")


def entries(described: dict, *kinds: str):
    for name, table in described["tables"].items():
        for c, column in table["columns"].items():
            if column["kind"] in kinds:
                yield name, c, column


def structure(described: dict) -> dict[str, float]:
    """What the generator is, read off `describe()`: no data needed. Causes and edge kinds leave
    out the read through which a column records its own node."""
    kinds = [column["kind"] for _, _, column in entries(described, *RECORDED)]
    edges = [
        cause.get("edge")
        for _, _, column in entries(described, *RECORDED, "hidden")
        for cause in column.get("causes", [])
        if cause.get("edge") not in (None, "ReadEdge")
    ]
    return {
        "tables": len(described["tables"]),
        "columns": len(kinds),
        "categorical share": kinds.count("categorical") / len(kinds) if kinds else np.nan,
        "time share": kinds.count("timestamp") / len(kinds) if kinds else np.nan,
        "keys": len(described["keys"]),
        "causes": len(edges),
        "edge kinds": len(set(edges)),
        "top edge share": max(map(edges.count, set(edges))) / len(edges) if edges else np.nan,
    }


def gini(counts: np.ndarray) -> float:
    """0 when every parent has as many children, towards 1 when a few have them all."""
    if counts.sum() == 0:
        return 0.0
    ordered = np.sort(counts)
    n = len(ordered)
    return float((2 * np.arange(1, n + 1) - n - 1) @ ordered / (n * ordered.sum()))


def median(values: list[float]) -> float:
    return float(np.median(values)) if len(values) else np.nan


def data(frames: dict[str, pd.DataFrame], described: dict) -> dict[str, float]:
    """What a dataset looks like: medians over its columns, tables and keys."""
    skews, missing, balance, dependence, coverage = [], [], [], [], []
    for name, frame in frames.items():
        columns = described["tables"][name]["columns"]
        numbers = []
        for c, column in columns.items():
            if column["kind"] not in RECORDED or c not in frame:
                continue
            series = frame[c]
            missing.append(series.isna().mean())
            if column["kind"] == "categorical":
                shares = series.value_counts(normalize=True).to_numpy()
                shares = shares[shares > 0]  # levels never drawn
                entropy = -(shares * np.log(shares)).sum()
                balance.append(entropy / np.log(len(shares)) if len(shares) > 1 else 0.0)
            elif column["kind"] == "numeric":
                values = series.dropna().astype(float)
                if values.nunique() > 1:
                    skews.append(abs(values.skew()))
                    numbers.append(c)
            elif "start" in (calendar := column.get("noise") or {}):
                start, end = pd.Timestamp(calendar["start"]), pd.Timestamp(calendar["end"])
                coverage.append((series.max() - series.min()) / (end - start))
        if len(numbers) > 1:
            ranks = frame[numbers].astype(float).corr("spearman").abs().to_numpy()
            dependence.append(np.nanmean(ranks[np.triu_indices(len(numbers), 1)]))
    spread = []
    for key in described["keys"]:
        parent = described["tables"][key["parent"]]["primary_key"]
        children = frames[key["table"]][key["column"]].dropna()
        ids = frames[key["parent"]][parent]
        spread.append(gini(children.value_counts().reindex(ids, fill_value=0).to_numpy()))
    return {
        "numeric skew": median(skews),
        "missing share": float(np.mean(missing)) if missing else np.nan,
        "class balance": median(balance),
        "dependence": median(dependence),
        "children skew": median(spread),
        "time coverage": median(coverage),
    }
