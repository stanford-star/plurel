"""How diverse are the databases a random prior draws, and where does the diversity come from?

Draws 40 generators from `SchemaPrior` (mechanism seeds 0 to 39) and samples each once; samples the
first 8 of them 5 times more (data seeds 1 to 5). Every dataset gets a profile of four groups:

- structure: tables, columns, kinds, keys, causes, edge kinds, the share of causes through the
  most used kind;
- graph: the depth of each table's hidden causal graph, causes per hidden node, the depth of the
  table graph and how many keys point into each referenced table;
- parameters: linear weights, noise scales, and how many noise, source and marginal families and
  calendars a generator uses, all read off `describe()`;
- data: shape of the numbers, missingness, class balance, dependence, how unevenly children
  spread over parents, and the share of its calendar an event table's timestamps span.

Then:

1. Mechanism seeds against data seeds: how much each measure moves across generators, against
   across datasets of one generator, pooled over the 8. Only data can move with the data seed.
2. Families: how many generators use each noise, source and marginal family the prior offers.
3. Near-duplicates: generator pairs whose profiles are closer than two datasets of one generator.
4. Degenerate and collapsed output: constant columns, categoricals with one level, empty tables,
   and categoricals one level dominates, by whether they record a categorical node or bin a
   numeric one.
5. Time: keys and summaries never read events, and a cut at any time leaves every key valid.
6. `TablePrior` alone: the same profile, spread, near-duplicates and collapsed output over 40
   single tables, half with a time column, and how often two of them draw the same timestamps.

A measure a dataset has nothing to measure on, such as class balance without categoricals, is
missing, not zero: it is left out of medians and spreads and counts as average in distances.
Row counts are kept small so this runs in well under a minute; they are part of the prior's
knobs, so the counts measured here are the ones set below, not the prior's defaults.

Graph and parameter measures read `describe()` as it is laid out today (noise and marginal types,
the width a column reads, calendar bounds); its layout is not part of the compatibility promise,
so this script follows it."""

from collections import Counter
from functools import cache

import numpy as np
import pandas as pd

from plurel import LogIntegersRange
from plurel.prior import SchemaPrior
from plurel.schema import Schema

if not __debug__:
    raise SystemExit("the checks are assert statements, which python -O drops: run without -O")

GENERATORS = 40
REPEATED, DATA_SEEDS = 8, 5  # generators sampled again, and how many more times each
PRIOR = SchemaPrior(
    entity_row_count=LogIntegersRange(250, 500), activity_row_count=LogIntegersRange(1000, 3000)
)
TABLE_ROWS = 1000
DOMINANT = 0.9  # share of its most common level past which a categorical counts as collapsed
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
GRAPH = ("node depth", "node in-degree", "table depth", "key fan-out")
PARAMETERS = (
    "weight spread",
    "noise scale (log10)",
    "noise families",
    "source families",
    "marginal families",
    "calendars",
)
DATA = ("numeric skew", "missing share", "class balance", "dependence", "children skew")
TIME = ("time coverage",)
FIXED = STRUCTURE + GRAPH + PARAMETERS
SCHEMA_ONLY = ("tables", "keys", "table depth", "key fan-out", "calendars")  # one per table


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


def longest(parents: dict[str, list[str]]) -> int:
    """Edges on the longest path of an acyclic graph given by each vertex's parents."""

    @cache
    def depth(vertex: str) -> int:
        return max((1 + depth(p) for p in parents[vertex]), default=0)

    return max(map(depth, parents), default=0)


def graph(described: dict) -> dict[str, float]:
    """The hidden causal graph inside each table, through local causes only, and the table graph
    through keys, self-references left out."""
    depths, degrees = [], []
    for table in described["tables"].values():
        hidden = {c: e for c, e in table["columns"].items() if e["kind"] == "hidden"}
        local = {
            c: [cause["reads"] for cause in e["causes"] if "summary" not in cause]
            for c, e in hidden.items()
        }
        depths.append(longest({c: [p for p in ps if p in hidden] for c, ps in local.items()}))
        degrees += [len(e["causes"]) for e in hidden.values()]
    keys = [key for key in described["keys"] if key["table"] != key["parent"]]
    references = {
        name: [k["parent"] for k in keys if k["table"] == name] for name in described["tables"]
    }
    fan_out = Counter(key["parent"] for key in keys)
    return {
        "node depth": median(depths),
        "node in-degree": float(np.mean(degrees)) if degrees else np.nan,
        "table depth": longest(references),
        "key fan-out": float(np.mean(list(fan_out.values()))) if fan_out else np.nan,
    }


def family(distribution: dict | None) -> str:
    """A distribution's family: an affine map is its inner family's, a mixture is named by its
    parts."""
    if distribution is None:
        return "latent"
    if distribution["type"] == "Affine":
        return family(distribution["distribution"])
    if distribution["type"] == "Mixture":
        return "Mixture(" + ", ".join(map(family, distribution["components"])) + ")"
    return distribution["type"]


def drawn(described: dict) -> dict[str, list]:
    """The parameters a generator drew: linear weights, the scaled noise of numeric nodes with
    local causes, the exogenous noise of local sources (which crossing edges may still feed), the
    marginals of numeric columns and the calendars of time columns. Categorical nodes, whose noise
    is always Gumbel, are left out."""
    found = {kind: [] for kind in ("weights", "scales", "noises", "sources")}
    for _, _, column in entries(described, "hidden"):
        if column["noise"]["type"] == "Gumbel":
            continue
        found["weights"] += [cause["weight"] for cause in column["causes"] if "weight" in cause]
        if column["noise"]["type"] == "Affine":
            found["scales"].append(column["noise"]["scale"])
            found["noises"].append(family(column["noise"]))
        else:
            found["sources"].append(family(column["noise"]))
    found["marginals"] = [family(c.get("marginal")) for _, _, c in entries(described, "numeric")]
    found["calendars"] = [str(c["noise"]) for _, _, c in entries(described, "timestamp")]
    return found


def parameters(described: dict) -> dict[str, float]:
    found = drawn(described)
    weights, scales = found["weights"], found["scales"]
    return {
        "weight spread": float(np.std(weights)) if len(weights) > 1 else np.nan,
        "noise scale (log10)": median(list(np.log10(scales))),
        "noise families": len(set(found["noises"])) or np.nan,
        "source families": len(set(found["sources"])) or np.nan,
        "marginal families": len(set(found["marginals"])) or np.nan,
        "calendars": len(set(found["calendars"])) or np.nan,
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


def degenerate(frames: dict[str, pd.DataFrame], described: dict) -> dict[str, int]:
    """Constant and empty output, and categoricals one level dominates: those recording a
    categorical node read whole, and those binning one slot of a numeric node."""
    found = {
        "constant columns": 0,
        "one-level categoricals": 0,
        "empty tables": 0,
        "dominated categorical nodes": 0,
        "dominated binned numerics": 0,
        "categorical nodes": 0,
        "binned numerics": 0,
        "top shares, categorical nodes": [],
        "top shares, binned numerics": [],
    }
    for name, frame in frames.items():
        found["empty tables"] += frame.empty
        for c, column in described["tables"][name]["columns"].items():
            if c not in frame or column["kind"] not in RECORDED:
                continue
            if frame[c].nunique() <= 1:
                kind = column["kind"] == "categorical"
                found["one-level categoricals" if kind else "constant columns"] += 1
            elif column["kind"] == "categorical":
                causes = column.get("causes") or [{}]
                binned = causes[0].get("dim") == 1
                kind = "binned numerics" if binned else "categorical nodes"
                top = frame[c].value_counts(normalize=True).iloc[0]
                found[kind] += 1
                found[f"dominated {kind}"] += top > DOMINANT
                found[f"top shares, {kind}"].append(top)
    return found


def time_violations(frames: dict[str, pd.DataFrame], described: dict) -> dict[str, int]:
    """What would leak or dangle in time. Keys into a table with a time column could point at a
    row from after the event; summaries over events would let a static row see the future; and
    cutting at the median timestamp must leave every key of the rows kept pointing at a kept row."""
    events = set(described["events"])
    summaries = [
        (name, cause["summary"])
        for name, _, column in entries(described, *RECORDED, "hidden")
        for cause in column.get("causes", [])
        if "summary" in cause
    ]
    times = {
        name: frames[name][table["time_column"]]
        for name, table in described["tables"].items()
        if table["time_column"]
    }
    cut = pd.concat(times.values()).median() if times else None
    kept = {
        name: frame[times[name] <= cut] if name in times else frame
        for name, frame in frames.items()
    }
    dangling = 0
    for key in described["keys"]:
        ids = kept[key["parent"]][described["tables"][key["parent"]]["primary_key"]]
        dangling += int((~kept[key["table"]][key["column"]].dropna().isin(ids)).sum())
    return {
        "keys into events": sum(key["parent"] in events for key in described["keys"]),
        "summaries over or into events": sum(a in events or b in events for a, b in summaries),
        "dangling keys after a cut": dangling,
    }


def profiles(generator: Schema, rows: dict[str, int], data_seeds) -> list[tuple[dict, dict]]:
    """Each data seed's profile and its degenerate, collapsed and leaking output."""
    described = generator.describe()
    fixed = structure(described) | graph(described) | parameters(described)
    out = []
    for seed in data_seeds:
        frames = generator.sample(rows, seed=seed)
        counts = degenerate(frames, described) | time_violations(frames, described)
        out.append((fixed | data(frames, described), counts))
    return out


def survey(generators: list[Schema], rows) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """One profile per generator, more per repeated generator, and the counts of each dataset."""
    once = [profiles(g, rows(g, i), [0])[0] for i, g in enumerate(generators)]
    again = {
        i: profiles(generators[i], rows(generators[i], i), range(1, DATA_SEEDS + 1))
        for i in range(REPEATED)
    }
    within = {i: pd.DataFrame([p for p, _ in runs]) for i, runs in again.items()}
    return pd.DataFrame([p for p, _ in once]), within, pd.DataFrame([c for _, c in once])


def spread(
    by_generator: pd.DataFrame, within: dict[int, pd.DataFrame], skip: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Print how much each measure moves across generators and across data seeds, but those in
    `skip`; check that only data moves with the data seed, and moves more across generators."""
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
    groups = {"structure": STRUCTURE, "graph": GRAPH, "parameters": PARAMETERS, "data": DATA + TIME}
    for group, measures in groups.items():
        print(f"   {group}")
        rows = [m for m in measures if m not in skip]
        print(shown.loc[rows].to_string(header=group == "structure"), "\n")
    for i, frame in within.items():
        fixed = frame[list(FIXED)].nunique(dropna=False) == 1
        assert fixed.all(), f"a data seed changed generator {i}'s {list(fixed.index[~fixed])}"
    data_moves = moves.loc[list(DATA)].dropna()
    assert (data_moves["spread, generators"] > data_moves["spread, data seeds"]).all()
    return moves


def near_duplicates(by_generator: pd.DataFrame, within: dict[int, pd.DataFrame]) -> None:
    """Generators closer to each other than two datasets of one generator are. Time coverage is
    left out: it is one up to rounding everywhere, so standardizing it only scales up noise."""
    by_generator = by_generator.drop(columns=list(TIME))
    within = {i: frame.drop(columns=list(TIME)) for i, frame in within.items()}
    center, scale = by_generator.mean(), by_generator.std().replace(0, 1.0).fillna(1.0)

    def standardized(frame: pd.DataFrame) -> np.ndarray:
        return ((frame - center) / scale).fillna(0.0).to_numpy()  # missing counts as average

    def distances(points: np.ndarray) -> np.ndarray:
        gaps = np.linalg.norm(points[:, None] - points[None], axis=-1)
        return gaps[np.triu_indices(len(points), 1)]

    between = distances(standardized(by_generator))
    same = np.concatenate([distances(standardized(frame)) for frame in within.values()])
    threshold = np.quantile(same, 0.95)
    close = int((between < threshold).sum())
    print("   distance between profiles, in standard deviations of each measure:")
    print(f"   two generators: median {np.median(between):.2f}, closest {between.min():.2f}")
    print(
        f"   two datasets of one generator: median {np.median(same):.2f}, 95% below {threshold:.2f}"
    )
    print(f"   generator pairs closer than that: {close} of {len(between)}\n")


def collapsed(counts: pd.DataFrame) -> None:
    print(f"   count over the {len(counts)} generators, and generators with any:")
    for name in ("constant columns", "one-level categoricals", "empty tables"):
        print(
            f"   {name}: {int(counts[name].sum())}, in {int((counts[name] > 0).sum())} generators"
        )
    for kind in ("categorical nodes", "binned numerics"):
        dominated, total = counts[f"dominated {kind}"], counts[kind]
        print(
            f"   {kind} with one level over {DOMINANT:.0%}: {int(dominated.sum())} of"
            f" {int(total.sum())}, in {int((dominated > 0).sum())} generators;"
            f" median top share {median(counts[f'top shares, {kind}'].explode().dropna()):.2f}"
        )
    print()


generators = [PRIOR.realize(seed) for seed in range(GENERATORS)]
by_generator, within, counts = survey(generators, PRIOR.rows)
fingerprints = {str(g.describe()) for g in generators}
assert len(fingerprints) == GENERATORS, "two mechanism seeds drew the same generator"

print(f"1. Spread (standard deviation) of each measure over {GENERATORS} generators, one dataset")
print(f"   each, against over {DATA_SEEDS} datasets of one generator, pooled over {REPEATED}:\n")
spread(by_generator, within)
constant = [m for m in FIXED if by_generator[m].nunique() <= 1]
print(f"   the same in every generator: {', '.join(constant) or 'none'}\n")

print(f"2. Families: how many of the {GENERATORS} generators use each one")
found = [drawn(g.describe()) for g in generators]
for kind in ("noises", "sources", "marginals"):
    used = Counter(name for f in found for name in set(f[kind]))
    print(f"   {kind}: {', '.join(f'{name} {n}' for name, n in used.most_common())}")
weights = np.concatenate([f["weights"] for f in found])
print(f"   linear weights, all generators pooled: std {weights.std():.2f} over {len(weights)}\n")

print("3. Near-duplicates")
near_duplicates(by_generator, within)

print("4. Degenerate and collapsed output")
collapsed(counts)

print("5. Time")
leaks = counts[["keys into events", "summaries over or into events", "dangling keys after a cut"]]
for name, column in leaks.items():
    print(f"   {name}: {int(column.sum())}")
assert (leaks == 0).all().all(), "a key or summary reads the future"
print()

tables = [  # named apart: a table draws its data by its name
    Schema({f"t{seed}": PRIOR.table_prior.realize(seed, time=seed % 2 == 0)})
    for seed in range(GENERATORS)
]
table_profiles, table_within, table_counts = survey(
    tables, lambda g, i: dict.fromkeys(g.tables, TABLE_ROWS)
)
assert len({str(t.describe()) for t in tables}) == GENERATORS, "two mechanism seeds drew one table"
print(f"6. TablePrior alone: {GENERATORS} tables of {TABLE_ROWS} rows, half with a time column\n")
spread(table_profiles, table_within, skip=SCHEMA_ONLY + ("children skew",))
constant = [m for m in FIXED if m not in SCHEMA_ONLY and table_profiles[m].nunique() <= 1]
print(f"   the same in every table: {', '.join(constant) or 'none'}\n")
near_duplicates(table_profiles, table_within)
collapsed(table_counts)
timed = [(name, t) for t in tables for name in t.eventful]
stamps = [tuple(t.sample({name: TABLE_ROWS}, seed=0)[name]["time"]) for name, t in timed]
calendars = [str(t.describe()["tables"][name]["columns"]["time"]["noise"]) for name, t in timed]
pairs = [(a, b) for a in range(len(timed)) for b in range(a + 1, len(timed))]
same = sum(stamps[a] == stamps[b] for a, b in pairs)
same_calendar = sum(calendars[a] == calendars[b] for a, b in pairs)
print(f"   pairs of the {len(timed)} timed tables with identical timestamps at one data seed:")
print(f"   {same} of {len(pairs)}, against {same_calendar} pairs that drew the same calendar\n")
print("diversity: structure moves only with the mechanism seed; data measures move more with it")
