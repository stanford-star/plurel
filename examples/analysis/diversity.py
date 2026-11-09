"""How diverse are the databases a random prior draws, and where does the diversity come from?

Draws 40 generators from `SchemaPrior` (mechanism seeds 0 to 39), samples each once, and samples
the first generator 40 times (data seeds 0 to 39). Every dataset gets a profile: its structure
(tables, columns, kinds, keys, causes, edges) and its data (shape of the numbers, missingness,
class balance, dependence, how unevenly children spread over parents). Then:

1. Mechanism seeds against data seeds: how much each measure moves when the generator changes,
   against when only the data seed does. Structure cannot move with the data seed.
2. Near-duplicates: generator pairs whose profiles are closer than two datasets of one generator.
3. Degenerate output: constant columns, categoricals with one level, empty tables.

Row counts are kept small so this runs in about a minute; they are part of the prior's knobs, so
the counts measured here are the ones set below, not the prior's defaults."""

import statistics

import numpy as np
import pandas as pd

from plurel import LogIntegersRange
from plurel.prior import SchemaPrior

if not __debug__:
    raise SystemExit("the checks are assert statements, which python -O drops: run without -O")

GENERATORS = 40
PRIOR = SchemaPrior(
    entity_row_count=LogIntegersRange(250, 500), activity_row_count=LogIntegersRange(1000, 3000)
)
STRUCTURE = ("tables", "columns", "categorical share", "time share", "keys", "causes", "edge kinds")
DATA = (
    "numeric skew",
    "missing share",
    "class balance",
    "dependence",
    "children skew",
)


def structure(described: dict) -> dict[str, float]:
    """What the generator is, read off `describe()`: no data needed."""
    columns = [
        column
        for table in described["tables"].values()
        for column in table["columns"].values()
        if column["kind"] in ("numeric", "categorical", "timestamp")
    ]
    causes = [
        cause
        for table in described["tables"].values()
        for column in table["columns"].values()
        for cause in column.get("causes", [])
    ]
    kinds = [column["kind"] for column in columns]
    return {
        "tables": len(described["tables"]),
        "columns": len(columns),
        "categorical share": kinds.count("categorical") / len(kinds),
        "time share": kinds.count("timestamp") / len(kinds),
        "keys": len(described["keys"]),
        "causes": len(causes),
        "edge kinds": len({cause.get("edge") for cause in causes}),
    }


def gini(counts: np.ndarray) -> float:
    """0 when every parent has as many children, towards 1 when a few have them all."""
    if counts.sum() == 0:
        return 0.0
    ordered = np.sort(counts)
    n = len(ordered)
    return float((2 * np.arange(1, n + 1) - n - 1) @ ordered / (n * ordered.sum()))


def data(frames: dict[str, pd.DataFrame], described: dict) -> dict[str, float]:
    """What a dataset looks like: medians over its columns, tables and keys."""
    skews, missing, balance, dependence = [], [], [], []
    for name, frame in frames.items():
        recorded = [
            c
            for c, column in described["tables"][name]["columns"].items()
            if column["kind"] in ("numeric", "categorical", "timestamp") and c in frame
        ]
        numbers = []
        for c in recorded:
            series = frame[c]
            missing.append(series.isna().mean())
            if described["tables"][name]["columns"][c]["kind"] == "categorical":
                shares = series.value_counts(normalize=True).to_numpy()
                shares = shares[shares > 0]  # levels never drawn
                entropy = -(shares * np.log(shares)).sum()
                balance.append(entropy / np.log(len(shares)) if len(shares) > 1 else 0.0)
            elif described["tables"][name]["columns"][c]["kind"] == "numeric":
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
        counts = children.value_counts().reindex(ids, fill_value=0).to_numpy()
        spread.append(gini(counts))
    return {
        "numeric skew": statistics.median(skews) if skews else 0.0,
        "missing share": statistics.mean(missing) if missing else 0.0,
        "class balance": statistics.median(balance) if balance else 1.0,
        "dependence": statistics.median(dependence) if dependence else 0.0,
        "children skew": statistics.median(spread) if spread else 0.0,
    }


def degenerate(frames: dict[str, pd.DataFrame], described: dict) -> dict[str, int]:
    found = {"constant columns": 0, "one-level categoricals": 0, "empty tables": 0}
    for name, frame in frames.items():
        found["empty tables"] += frame.empty
        for c, column in described["tables"][name]["columns"].items():
            if c not in frame or column["kind"] not in ("numeric", "categorical", "timestamp"):
                continue
            if frame[c].nunique() <= 1:
                key = (
                    "one-level categoricals"
                    if column["kind"] == "categorical"
                    else "constant columns"
                )
                found[key] += 1
    return found


def draw(mechanism: int, data_seed: int) -> tuple[dict, dict, dict]:
    generator = PRIOR.realize(mechanism)
    described = generator.describe()
    frames = generator.sample(PRIOR.rows(generator, mechanism), seed=data_seed)
    profile = structure(described) | data(frames, described)
    return profile, degenerate(frames, described), described


across_generators = [draw(seed, 0) for seed in range(GENERATORS)]
across_data = [draw(0, seed) for seed in range(GENERATORS)]
by_generator = pd.DataFrame([profile for profile, _, _ in across_generators])
by_data = pd.DataFrame([profile for profile, _, _ in across_data])

# 1. what moves with the mechanism seed, what with the data seed
moves = pd.DataFrame(
    {
        "median": by_generator.median(),
        "spread, generators": by_generator.std(),
        "spread, data seeds": by_data.std(),
    }
)
moves["ratio"] = moves["spread, generators"] / moves["spread, data seeds"].replace(0, np.nan)
shown = moves.round(3).astype(object).where(moves.notna(), "–")  # a data seed cannot move it
print("1. Spread (standard deviation) of each measure over 40 generators, one dataset each,")
print("   against over 40 datasets of one generator:\n")
print(shown.to_string(), "\n")
assert (by_data[list(STRUCTURE)].nunique() == 1).all(), "a data seed changed the structure"
assert (by_generator[list(STRUCTURE)].nunique() > 1).all(), "every generator has one structure"
data_moves = moves.loc[list(DATA)]
assert (data_moves["spread, generators"] > data_moves["spread, data seeds"]).all()

# 2. near-duplicates: generators closer to each other than two datasets of one generator are
scale = by_generator.std().replace(0, 1.0)
points = ((by_generator - by_generator.mean()) / scale).to_numpy()
same = ((by_data - by_generator.mean()) / scale).to_numpy()
pairs = np.linalg.norm(points[:, None] - points[None], axis=-1)[np.triu_indices(GENERATORS, 1)]
within = np.linalg.norm(same[:, None] - same[None], axis=-1)[np.triu_indices(GENERATORS, 1)]
threshold = np.quantile(within, 0.95)
close = int((pairs < threshold).sum())
print("2. Distance between profiles, in standard deviations of each measure:")
print(f"   two generators: median {np.median(pairs):.2f}, closest {pairs.min():.2f}")
print(
    f"   two datasets of one generator: median {np.median(within):.2f}, 95% below {threshold:.2f}"
)
print(f"   generator pairs closer than that: {close} of {len(pairs)}\n")
fingerprints = {str(described) for _, _, described in across_generators}
assert len(fingerprints) == GENERATORS, "two mechanism seeds drew the same generator"

# 3. degenerate output
found = pd.DataFrame([counts for _, counts, _ in across_generators])
print("3. Degenerate output over the 40 generators (count, and generators with any):")
for name, column in found.items():
    print(f"   {name}: {int(column.sum())}, in {int((column > 0).sum())} generators")
print("\ndiversity: structure moves only with the mechanism seed; data measures move more with it")
