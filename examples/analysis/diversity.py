"""How diverse are the databases a random prior draws, and where does the diversity come from?

Draws 40 generators from `SchemaPrior` (mechanism seeds 0 to 39) and samples each once; samples the
first 8 of them 5 times more (data seeds 1 to 5). Every dataset gets a profile: its structure
(tables, columns, kinds, keys, causes, edge kinds, the share of causes through the most used kind)
and its data (shape of the numbers, missingness, class balance, dependence, how unevenly children
spread over parents). Then:

1. Mechanism seeds against data seeds: how much each measure moves across generators, against
   across datasets of one generator, pooled over the 8. Structure cannot move with the data seed.
2. Near-duplicates: generator pairs whose profiles are closer than two datasets of one generator.
3. Degenerate output: constant columns, categoricals with one level, empty tables.

A measure a dataset has nothing to measure on, such as class balance without categoricals, is
missing, not zero: it is left out of medians and spreads and counts as average in distances.
Row counts are kept small so this runs in about ten seconds; they are part of the prior's knobs,
so the counts measured here are the ones set below, not the prior's defaults."""

import numpy as np
import pandas as pd

from plurel import LogIntegersRange
from plurel.prior import SchemaPrior

if not __debug__:
    raise SystemExit("the checks are assert statements, which python -O drops: run without -O")

GENERATORS = 40
REPEATED, DATA_SEEDS = 8, 5  # generators sampled again, and how many more times each
PRIOR = SchemaPrior(
    entity_row_count=LogIntegersRange(250, 500), activity_row_count=LogIntegersRange(1000, 3000)
)
RECORDED = ("numeric", "categorical", "timestamp")
STRUCTURE = (
    "tables",
    "columns",
    "categorical share",
    "time share",
    "keys",
    "causes",
    "edge kinds",
    "top edge share",
)
DATA = ("numeric skew", "missing share", "class balance", "dependence", "children skew")


def structure(described: dict) -> dict[str, float]:
    """What the generator is, read off `describe()`: no data needed. Causes and edge kinds leave
    out the read through which a column records its own node."""
    columns = [
        column
        for table in described["tables"].values()
        for column in table["columns"].values()
        if column["kind"] in RECORDED
    ]
    edges = [
        cause.get("edge")
        for table in described["tables"].values()
        for column in table["columns"].values()
        for cause in column.get("causes", [])
        if cause.get("edge") not in (None, "ReadEdge")
    ]
    kinds = [column["kind"] for column in columns]
    return {
        "tables": len(described["tables"]),
        "columns": len(columns),
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
    return float(np.median(values)) if values else np.nan


def data(frames: dict[str, pd.DataFrame], described: dict) -> dict[str, float]:
    """What a dataset looks like: medians over its columns, tables and keys."""
    skews, missing, balance, dependence = [], [], [], []
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
    }


def degenerate(frames: dict[str, pd.DataFrame], described: dict) -> dict[str, int]:
    found = {"constant columns": 0, "one-level categoricals": 0, "empty tables": 0}
    for name, frame in frames.items():
        found["empty tables"] += frame.empty
        for c, column in described["tables"][name]["columns"].items():
            if c in frame and column["kind"] in RECORDED and frame[c].nunique() <= 1:
                kind = column["kind"] == "categorical"
                found["one-level categoricals" if kind else "constant columns"] += 1
    return found


def profiles(mechanism: int, data_seeds) -> list[tuple[dict, dict]]:
    """Each data seed's profile and degenerate output, the generator realized once."""
    generator = generators[mechanism]
    described = generator.describe()
    rows = PRIOR.rows(generator, mechanism)
    out = []
    for seed in data_seeds:
        frames = generator.sample(rows, seed=seed)
        out.append((structure(described) | data(frames, described), degenerate(frames, described)))
    return out


generators = [PRIOR.realize(seed) for seed in range(GENERATORS)]
once = [profiles(seed, [0])[0] for seed in range(GENERATORS)]
again = {i: profiles(i, range(1, DATA_SEEDS + 1)) for i in range(REPEATED)}
by_generator = pd.DataFrame([profile for profile, _ in once])
within = {i: pd.DataFrame([p for p, _ in runs]) for i, runs in again.items()}

# 1. what moves with the mechanism seed, what with the data seed
pooled = np.sqrt(pd.concat([frame.var() for frame in within.values()], axis=1).mean(axis=1))
pooled = pooled.where(pooled > 1e-9, 0.0)  # rounding left over from equal shares
moves = pd.DataFrame(
    {
        "median": by_generator.median(),
        "spread, generators": by_generator.std(),
        "spread, data seeds": pooled,
    }
)
moves["ratio"] = moves["spread, generators"] / moves["spread, data seeds"].replace(0, np.nan)
shown = moves.round(3).astype(object).where(moves.notna(), "–")  # a data seed cannot move it
print(f"1. Spread (standard deviation) of each measure over {GENERATORS} generators, one dataset")
print(f"   each, against over {DATA_SEEDS} datasets of one generator, pooled over {REPEATED}:\n")
print(shown.to_string(), "\n")
for i, frame in within.items():
    fixed = frame[list(STRUCTURE)].nunique(dropna=False) == 1
    assert fixed.all(), f"a data seed changed generator {i}'s {list(fixed.index[~fixed])}"
varied = by_generator[list(STRUCTURE)].nunique() > 1
assert varied.all(), f"{list(varied.index[~varied])} did not vary across mechanism seeds"
data_moves = moves.loc[list(DATA)]
assert (data_moves["spread, generators"] > data_moves["spread, data seeds"]).all()

# 2. near-duplicates: generators closer to each other than two datasets of one generator are
center, scale = by_generator.mean(), by_generator.std().replace(0, 1.0)


def standardized(frame: pd.DataFrame) -> np.ndarray:
    return ((frame - center) / scale).fillna(0.0).to_numpy()  # missing counts as average


def distances(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, None] - points[None], axis=-1)[np.triu_indices(len(points), 1)]


between = distances(standardized(by_generator))
same = np.concatenate([distances(standardized(frame)) for frame in within.values()])
threshold = np.quantile(same, 0.95)
close = int((between < threshold).sum())
print("2. Distance between profiles, in standard deviations of each measure:")
print(f"   two generators: median {np.median(between):.2f}, closest {between.min():.2f}")
print(f"   two datasets of one generator: median {np.median(same):.2f}, 95% below {threshold:.2f}")
print(f"   generator pairs closer than that: {close} of {len(between)}\n")
fingerprints = {str(g.describe()) for g in generators}
assert len(fingerprints) == GENERATORS, "two mechanism seeds drew the same generator"

# 3. degenerate output
found = pd.DataFrame([counts for _, counts in once])
print(f"3. Degenerate output over the {GENERATORS} generators (count, and generators with any):")
for name, column in found.items():
    print(f"   {name}: {int(column.sum())}, in {int((column > 0).sum())} generators")
print("\ndiversity: structure moves only with the mechanism seed; data measures move more with it")
