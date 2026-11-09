"""Where does a database drawn from `SchemaPrior` look like a real one, and where does it not?

Loads one real relbench database (`rel-f1` by default, relbench's name for the copy it hosts at
`stanford-star/relbench-v1/rel-f1`; relbench downloads it on first use, so this needs network
access) and draws 60 databases from `SchemaPrior` (mechanism seeds 0 to 59, data seed 0). The
prior keeps its defaults except row counts, which span the real database's: referenced tables
take the range of the real referenced tables' sizes, the others the range of the real event
tables'. Row counts below the prior's floor, the HSBM link's most clusters, are raised to it.

Every database gets the profile `examples/analysis/_profile.py` defines, the one `diversity.py`
measures, plus the shape of the numbers (heavy tails, zeros, integers), category counts, children
per parent and how timestamps cover time. A real column's kind comes from its dtype: datetimes are
timestamps, other numbers numeric, everything else categorical. Code columns are categorical:
integer columns named like keys (`statusId`) and a short list per dataset (`number`, rel-f1's car
number); ordinal positions and counts stay numeric. Keys are left out, as in the generated profile.
Then:

1. Internal consistency: rules the generator imposes on what it authors, asserted on every draw,
   and whether the real database keeps them. Holding them says the generator is coherent, not
   that it is realistic.
2. Empirical similarity: for each measure, where the real value falls among the draws, as a
   percentile, and whether any draw reaches it. Causes and edge kinds are left out: a real
   database has no mechanism to read them from. Measures marked size-dependent move with row
   counts; matching the real database's sizes is what makes them comparable at all.

Every claim printed is measured on this one real database: one sample of real data, which says
where the prior misses it, not how real databases at large behave.

`pixi run recipes` runs `--self-test` instead, with no download: a small generated database,
exported to relbench as `create_database` does, stands in for the real one and must profile as
its own description does."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import relbench

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _profile import data, gini, median, structure  # noqa: E402

from plurel import LogIntegersRange, create_database  # noqa: E402
from plurel.prior import SchemaPrior  # noqa: E402

if not __debug__:
    raise SystemExit("the checks are assert statements, which python -O drops: run without -O")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--dataset", default="rel-f1", help="a relbench name, Hub path or folder")
parser.add_argument("--generators", type=int, default=60)
parser.add_argument("--self-test", action="store_true", help="a generated stand-in, no download")
args = parser.parse_args()
CODES = {"rel-f1": ("number",)}  # integer columns that name things rather than count them
TIME_BINS = 50
ZERO_HEAVY = 0.1  # share of zeros that makes a numeric column zero-heavy
COMPARED = (
    "tables",
    "columns",
    "categorical share",
    "time share",
    "keys",
    "numeric skew",
    "missing share",
    "class balance",
    "dependence",
    "children skew",
    "numeric kurtosis",
    "zero-heavy share",
    "integer share",
    "category levels",
    "most levels",
    "children per parent",
    "childless parents",
    "time span (years)",
    "distinct times",
    "time concentration",
)
SIZE_DEPENDENT = (
    "class balance",
    "dependence",
    "children skew",
    "most levels",
    "children per parent",
    "childless parents",
    "distinct times",
)


def kind(name: str, series: pd.Series, codes: tuple[str, ...]) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp"
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return "categorical"
    integers = pd.api.types.is_integer_dtype(series)
    return "categorical" if integers and (name.endswith("Id") or name in codes) else "numeric"


def described(db: relbench.base.Database, codes: tuple[str, ...] = ()) -> dict:
    """The parts of `Schema.describe()` the profile reads, for a relbench database."""
    tables, keys = {}, []
    for name, table in db.table_dict.items():
        fkeys = table.fkey_col_to_pkey_table
        keys += [{"table": name, "column": c, "parent": p} for c, p in fkeys.items()]
        tables[name] = {
            "primary_key": table.pkey_col,
            "time_column": table.time_col,
            "columns": {
                c: {"kind": kind(c, table.df[c], codes)}
                for c in table.df
                if c != table.pkey_col and c not in fkeys
            },
        }
    return {"tables": tables, "keys": keys}


def columns(frames: dict[str, pd.DataFrame], described: dict, wanted: str):
    for name, frame in frames.items():
        for c, column in described["tables"][name]["columns"].items():
            if column["kind"] == wanted and c in frame:
                yield frame[c].dropna()


def shape(frames: dict[str, pd.DataFrame], described: dict) -> dict[str, float]:
    """Marginals, category counts, children per parent and time coverage: medians over columns
    and keys."""
    numbers = [s.astype(float) for s in columns(frames, described, "numeric") if s.nunique() > 1]
    levels = [s.nunique() for s in columns(frames, described, "categorical") if len(s)]
    per_parent, childless = [], []
    for key in described["keys"]:
        parent = frames[key["parent"]][described["tables"][key["parent"]]["primary_key"]]
        counts = frames[key["table"]][key["column"]].dropna().value_counts()
        counts = counts.reindex(parent, fill_value=0).to_numpy()
        per_parent.append(counts.mean())
        childless.append((counts == 0).mean())
    span, distinct, concentration = [], [], []
    for times in columns(frames, described, "timestamp"):
        if times.nunique() < 2:
            continue
        span.append((times.max() - times.min()).days / 365.25)
        distinct.append(times.nunique() / len(times))
        bins = np.histogram(times.astype("int64"), TIME_BINS)[0]
        concentration.append(gini(bins))
    return {
        "numeric kurtosis": median([s.kurt() for s in numbers]),
        "zero-heavy share": float(np.mean([(s == 0).mean() >= ZERO_HEAVY for s in numbers]))
        if numbers
        else np.nan,
        "integer share": float(np.mean([(s % 1 == 0).all() for s in numbers]))
        if numbers
        else np.nan,
        "category levels": median(levels),
        "most levels": float(max(levels)) if levels else np.nan,
        "children per parent": median(per_parent),
        "childless parents": median(childless),
        "time span (years)": median(span),
        "distinct times": median(distinct),
        "time concentration": median(concentration),
    }


def profile(frames: dict[str, pd.DataFrame], described: dict) -> dict[str, float]:
    return structure(described) | data(frames, described) | shape(frames, described)


def referenced(described: dict) -> set[str]:
    return {k["parent"] for k in described["keys"] if k["parent"] != k["table"]}


def ordered(frames: dict[str, pd.DataFrame], described: dict) -> dict[str, float]:
    """Per key between two timed tables, the share of child rows no earlier than their parent."""
    out = {}
    for key in described["keys"]:
        parent = described["tables"][key["parent"]]
        child_time = described["tables"][key["table"]]["time_column"]
        if parent["time_column"] and child_time and key["parent"] != key["table"]:
            ids = frames[key["parent"]].set_index(parent["primary_key"])[parent["time_column"]]
            children = frames[key["table"]].dropna(subset=[key["column"]])
            started = ids.reindex(children[key["column"]]).to_numpy()
            out[f"{key['table']}.{key['column']}"] = (children[child_time] >= started).mean()
    return out


def rules(frames: dict[str, pd.DataFrame], described: dict) -> dict[str, float]:
    """Shares of tables or keys keeping a rule the generator authors; nan when there is nothing
    the rule applies to."""
    resolved, unique = [], []
    for key in described["keys"]:
        ids = frames[key["parent"]][described["tables"][key["parent"]]["primary_key"]]
        values = frames[key["table"]][key["column"]].dropna()
        resolved.append(values.isin(ids).all())
    for name, table in described["tables"].items():
        if table["primary_key"]:
            unique.append(frames[name][table["primary_key"]].is_unique)
    timed = {name for name, table in described["tables"].items() if table["time_column"]}
    parents = referenced(described)
    after = list(ordered(frames, described).values())
    return {
        "unique primary keys": float(np.mean(unique)),
        "foreign keys that resolve": float(np.mean(resolved)) if resolved else np.nan,
        "referenced tables that are static": 1 - len(timed & parents) / len(parents)
        if parents
        else np.nan,
        "children no earlier than parents": float(np.mean([a == 1 for a in after]))
        if after
        else np.nan,
    }


def percentile(draws: pd.Series, value: float) -> float:
    draws = draws.dropna()
    if np.isnan(value) or draws.empty:
        return np.nan
    return 100 * float((draws < value).mean() + 0.5 * (draws == value).mean())


def reach(draws: pd.Series, value: float) -> str:
    draws = draws.dropna()
    if np.isnan(value) or draws.empty:
        return "nothing to measure"
    if value < draws.min():
        return "below every draw"
    if value > draws.max():
        return "above every draw"
    return "reached"


def matched(described: dict, frames: dict[str, pd.DataFrame]) -> tuple[SchemaPrior, list[str]]:
    """The default prior with row counts spanning the real tables', and the tables it cannot
    match because they are smaller than its floor."""
    prior = SchemaPrior()
    floor = prior.link_cluster_count.high**prior.link_level_count.high
    parents = referenced(described)
    sizes = {name: len(frame) for name, frame in frames.items()}
    small = [f"{name} ({rows} rows)" for name, rows in sizes.items() if rows < floor]

    def spanning(names: list[str], default: LogIntegersRange) -> LogIntegersRange:
        if not names:
            return default
        rows = [max(sizes[name], floor) for name in names]
        return LogIntegersRange(min(rows), max(rows))

    entity = spanning(sorted(parents), prior.entity_row_count)
    activity = spanning(sorted(set(sizes) - parents), prior.activity_row_count)
    return SchemaPrior(entity_row_count=entity, activity_row_count=activity), small


if args.self_test:
    NAME, generators = "self-test", 5
    small = SchemaPrior(
        entity_row_count=LogIntegersRange(250, 500), activity_row_count=LogIntegersRange(1000, 3000)
    )
    stand_in = small.realize(1000)
    stand_in_frames = stand_in.sample(small.rows(stand_in, 1000), seed=0)
    db = create_database(stand_in, stand_in_frames)
    expected = profile(stand_in_frames, stand_in.describe())
else:
    NAME, generators = Path(args.dataset.rstrip("/")).name, args.generators
    db = relbench.load_dataset(args.dataset).get_db(upto_test_timestamp=False)
real_described = described(db, CODES.get(NAME, ()))
real_frames = {name: table.df for name, table in db.table_dict.items()}
real = profile(real_frames, real_described)
real_rules = rules(real_frames, real_described)
if args.self_test:
    got, want = pd.Series(real)[list(COMPARED)], pd.Series(expected)[list(COMPARED)]
    assert np.allclose(got, want, equal_nan=True), "a relbench export profiles unlike its schema"

prior, unmatched = matched(real_described, real_frames)
drawn, drawn_rules = [], []
for seed in range(generators):
    generator = prior.realize(seed)
    frames = generator.sample(prior.rows(generator, seed), seed=0)
    description = generator.describe()
    drawn.append(profile(frames, description))
    drawn_rules.append(rules(frames, description))
drawn = pd.DataFrame(drawn)[list(COMPARED)]
drawn_rules = pd.DataFrame(drawn_rules)
entity, activity = prior.entity_row_count, prior.activity_row_count
print(f"Draws at {NAME}'s sizes: referenced tables {entity.low} to {entity.high} rows, the others")
print(
    f"{activity.low} to {activity.high}; below the prior's floor: {', '.join(unmatched) or 'none'}\n"
)

# 1. internal consistency
print(f"1. Internal consistency: rules the generator authors, over {generators} draws and")
print(f"   on {NAME} (share of the tables or keys keeping each; – where none apply):\n")
consistency = pd.DataFrame(
    {
        "draws, lowest": drawn_rules.min(),
        "draws applying": drawn_rules.notna().sum(),
        NAME: pd.Series(real_rules),
    }
)
print(consistency.round(3).astype(object).where(consistency.notna(), "–").to_string(), "\n")
for rule in ("unique primary keys", "foreign keys that resolve"):
    assert (drawn_rules[rule].dropna() == 1).all(), f"a draw broke {rule}"
assert (drawn_rules["referenced tables that are static"].dropna() == 1).all()
assert drawn_rules["children no earlier than parents"].isna().all(), "a timed table is referenced"
timed_parents = sorted(
    name for name in referenced(real_described) if real_described["tables"][name]["time_column"]
)
if timed_parents:
    early = {key: share for key, share in ordered(real_frames, real_described).items() if share < 1}
    print(f"   {NAME} references tables that hold events, {timed_parents}, which the prior never")
    print("   draws. Keys whose child rows can be earlier than their parent (share no earlier):")
    print("".join(f"   {key}: {share:.3f}\n" for key, share in early.items()))

# 2. empirical similarity
compared = pd.DataFrame(
    {
        NAME: pd.Series(real)[list(COMPARED)],
        "draws, 5%": drawn.quantile(0.05),
        "median": drawn.median(),
        "95%": drawn.quantile(0.95),
        "percentile": [percentile(drawn[m], real[m]) for m in COMPARED],
    }
)
compared["size-dependent"] = ["yes" if m in SIZE_DEPENDENT else "" for m in COMPARED]
compared["reach"] = [reach(drawn[m], real[m]) for m in COMPARED]
shown = compared.copy()
numbers = compared.columns[:-2]
shown[numbers] = compared[numbers].round(3)
print(f"2. Empirical similarity, measured on {NAME} only: each measure on {NAME}, its spread")
print(f"   over the {generators} draws, and where {NAME} falls among them:\n")
print(shown.astype(object).where(compared.notna(), "–").to_string(), "\n")
missed = compared.index[compared["reach"].str.endswith("every draw")]
edges = compared.index[compared["percentile"].sub(50).abs().ge(45) & ~compared.index.isin(missed)]
print(f"   never reached by the prior on {NAME}: {', '.join(missed) or 'none'}")
print(f"   reached only at the edge (below 5% or above 95%): {', '.join(edges) or 'none'}")
print(
    f"   of the never reached, size-dependent: {', '.join(missed.intersection(SIZE_DEPENDENT)) or 'none'}\n"
)
assert compared[NAME].loc[["tables", "columns", "keys"]].notna().all()

print(f"realism: {len(missed)} of {len(COMPARED)} measures on {NAME} lie outside every draw")
