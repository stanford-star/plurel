"""How learnable are the columns a random prior draws, and how hard are they to predict?

Draws 8 databases from `SchemaPrior` and 12 tables from `TablePrior` (every other one with a time
column), with fixed seeds, and picks up to two targets per table at random: a categorical column
to classify and a numeric one to regress. A target's features are the other recorded columns of
its row and, for a child table, the columns of each parent joined through its foreign key (a
self-reference is not joined). Event tables are split by time, training on the rows up to the
70% quantile of their time and testing on the later ones; static tables by row, a row being an
entity. No feature holds the target or anything recorded after it: a parent column is withheld
when it summarizes the target's table, directly or through further summaries and gathers, and
parents are static, so none records a later event.

Each target is predicted by a linear model (logistic or ridge regression) and a gradient-boosted
tree, and by baselines: the training majority class or mean and, for a child table, the majority
or mean per parent. Loss is the error rate, or the squared error of a numeric target's normal
scores in the training distribution, so a few outliers do not decide. The lift is the share of
the best baseline's test loss the better learner removes, with a paired bootstrap interval that
resamples dependent test rows together: 20 blocks of consecutive test times in an event table, a parent's
children in a static child table, single rows otherwise. Bands: no signal (the interval reaches 0), hard (lift below 0.1), moderate (below
0.4), easy (below 0.8) and trivial. Easy and trivial targets are kept and reported.

A learner within 2% of perfect is flagged as leakage-like. An exact copy is a column of the row
the target is a function of (one category per category, a monotone map, a binning), which the
prior makes once a table has more columns than its nodes offer. A flagged target without one
gets the best lift one feature reaches alone; a near-copy is a feature that alone is near-perfect,
such as a column reading a node through very little noise. A control refits a sample of the
targets on shuffled training labels. The checks assert only the analysis: no target among its
features, training times before test times, and no signal in the control; what the prior draws
is printed, not asserted.

Row counts are small, as in `diversity.py`, so this runs in about a minute single-threaded; the
bands are those of these row counts, not of the prior's defaults. With PLUREL_QUICK=1, as
`scripts/run_recipes.py` sets, it draws 1 database and 4 tables, with 40 boosting rounds instead of 100."""

import os
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from plurel import LogIntegersRange
from plurel.prior import SchemaPrior, TablePrior

if not __debug__:
    raise SystemExit("the checks are assert statements, which python -O drops: run without -O")

warnings.filterwarnings("ignore", category=ConvergenceWarning)
QUICK = os.environ.get("PLUREL_QUICK") == "1"  # scripts/run_recipes.py sets it
DATABASES, TABLES = (1, 4) if QUICK else (8, 12)
TABLE_ROWS, BLOCKS, ROUNDS = 1000, 20, 40 if QUICK else 100
PRIOR = SchemaPrior(
    entity_row_count=LogIntegersRange(250, 500), activity_row_count=LogIntegersRange(1000, 3000)
)
RECORDED = ("numeric", "categorical", "timestamp")
TRAIN, RESAMPLES, CONTROLS = 0.7, 300, 10 if QUICK else 30
BANDS = ("no signal", "hard", "moderate", "easy", "trivial")
EDGES = (0.1, 0.4, 0.8)
NEAR_PERFECT = 0.02  # learner loss under this share of the constant baseline's


def informed(described: dict) -> Callable[[str, str], set[str]]:
    """The tables a column or node summarizes, through summaries, gathers and reads."""
    parents = {(key["table"], key["column"]): key["parent"] for key in described["keys"]}
    memo: dict[tuple[str, str], set[str]] = {}

    def tables(table: str, node: str) -> set[str]:
        if (table, node) not in memo:
            found = set()
            for cause in described["tables"][table]["columns"][node].get("causes", []):
                if "summary" in cause:
                    found |= {cause["summary"]} | tables(cause["summary"], cause["reads"])
                elif "." in cause["reads"]:
                    key, read = cause["reads"].split(".", 1)
                    found |= tables(parents[(table, key)], read)
                else:
                    found |= tables(table, cause["reads"])
            memo[table, node] = found
        return memo[table, node]

    return tables


def recorded(frame: pd.DataFrame, columns: dict) -> pd.DataFrame:
    out = {}
    for c, column in columns.items():
        if column["kind"] in RECORDED and c in frame:
            series = frame[c]
            out[c] = series.astype("int64") / 1e9 if column["kind"] == "timestamp" else series
    return pd.DataFrame(out, index=frame.index)


def determines(feature: pd.Series, target: pd.Series) -> bool:
    """Whether the target is exactly a function of the feature where both are recorded: one
    category per feature category, a monotone map of a number, or a binning of one."""
    both = feature.notna() & target.notna()
    feature, target = feature[both], target[both]
    if feature.nunique() < 2:
        return False
    if isinstance(feature.dtype, pd.CategoricalDtype):
        return bool(target.groupby(feature, observed=True).nunique().max() == 1)
    if isinstance(target.dtype, pd.CategoricalDtype):
        ordered = target.cat.codes.to_numpy()[np.lexsort((target.cat.codes, feature))]
        return int((np.diff(ordered) != 0).sum()) + 1 == target.nunique()
    return bool(abs(feature.corr(target.astype(float), method="spearman")) > 1 - 1e-9)


def clusters(
    frame: pd.DataFrame, time: str | None, keys: list[dict], test: np.ndarray
) -> np.ndarray:
    """Bootstrap units: blocks of consecutive times over the test period for an event table, the
    parent for a child table (a row without one on its own), the row otherwise."""
    if time is not None:
        order = frame[time].where(test).rank(method="first").to_numpy() - 1
        return np.where(test, order * BLOCKS // max(test.sum(), 1), -1).astype(int)
    if keys:
        parent = frame[keys[0]["column"]].astype("float64").to_numpy()
        return np.where(np.isnan(parent), -1.0 - np.arange(len(frame)), parent)
    return np.arange(len(frame))


def tasks(name: str, frames: dict, described: dict, rng: np.random.Generator) -> list[dict]:
    """Up to one classification and one regression per table, with features and a split."""
    summarized = informed(described)
    out = []
    for table, frame in frames.items():
        info = described["tables"][table]
        own = recorded(frame, info["columns"])
        keys = [k for k in described["keys"] if k["table"] == table and k["parent"] != table]
        joined, withheld, offered = [], 0, 0
        for key in keys:
            parent = key["parent"]
            columns = recorded(
                frames[parent].set_index(described["tables"][parent]["primary_key"]),
                described["tables"][parent]["columns"],
            )
            leaks = [c for c in columns if table in summarized(parent, c)]
            withheld, offered = withheld + len(leaks), offered + columns.shape[1]
            columns = columns.drop(columns=leaks)
            ids = frame[key["column"]].astype("float64")
            joined.append(
                columns.reindex(ids).set_axis(frame.index).add_prefix(f"{key['column']}.")
            )
        time = info["time_column"]
        for kind in ("categorical", "numeric"):
            candidates = [
                c
                for c in own
                if info["columns"][c]["kind"] == kind and c != time and own[c].nunique() > 1
            ]
            if not candidates:
                continue
            target = candidates[rng.integers(len(candidates))]
            features = pd.concat([own.drop(columns=target), *joined], axis=1)
            labelled = frame[target].notna().to_numpy()
            if time is None:
                train = rng.random(len(frame)) < TRAIN
            else:
                cutoff = frame[time].quantile(TRAIN)
                train = (frame[time] <= cutoff).to_numpy()
                assert frame[time][train].max() <= frame[time][~train].min()
            units = clusters(frame, time, keys, ~train)
            out.append(
                {
                    "source": name,
                    "table": table,
                    "target": target,
                    "task": "classification" if kind == "categorical" else "regression",
                    "split": "time" if time else "row",
                    "child": bool(keys),
                    "x": features[labelled],
                    "y": frame[target][labelled],
                    "groups": [frame[k["column"]][labelled] for k in keys],
                    "train": train[labelled],
                    "withheld": (withheld, offered),
                    "clusters": units[labelled],
                    "copies": [c for c in own if c != target and determines(own[c], own[target])],
                }
            )
    return out


def learners(task: str, rounds: int) -> dict:
    linear = make_column_transformer(
        (
            make_pipeline(SimpleImputer(add_indicator=True), StandardScaler()),
            make_column_selector(dtype_include="number"),
        ),
        (
            make_pipeline(
                SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")
            ),
            make_column_selector(dtype_include="category"),
        ),
    )
    tree = dict(max_iter=rounds, categorical_features="from_dtype", random_state=0)
    if task == "classification":
        return {
            "linear": make_pipeline(linear, LogisticRegression(max_iter=300)),
            "tree": HistGradientBoostingClassifier(**tree),
        }
    return {
        "linear": make_pipeline(linear, Ridge()),
        "tree": HistGradientBoostingRegressor(**tree),
    }


def losses(task: str, y: pd.Series, predicted) -> np.ndarray:
    if task == "classification":
        return (np.asarray(predicted, dtype=object) != y.to_numpy(dtype=object)).astype(float)
    return (np.asarray(predicted, dtype=float) - y.to_numpy(dtype=float)) ** 2


def baselines(task: str, y_train: pd.Series, y_test: pd.Series, groups) -> dict:
    center = y_train.mode().iloc[0] if task == "classification" else y_train.mean()
    out = {"constant": losses(task, y_test, np.full(len(y_test), center, dtype=object))}
    for i, (train, test) in enumerate(groups):
        by = y_train.groupby(train.to_numpy(), observed=True)
        per = by.agg(lambda s: s.mode().iloc[0]) if task == "classification" else by.mean()
        guess = test.map(per).astype(object).where(test.map(per).notna(), center)
        out[f"per parent {i}"] = losses(task, y_test, guess.to_numpy())
    return out


def fit(
    item: dict, y_train: pd.Series, only: list[str] | None = None, rounds: int = ROUNDS
) -> dict[str, np.ndarray]:
    train = item["train"]
    x = item["x"] if only is None else item["x"][only]
    cast = str if item["task"] == "classification" else float
    models = learners(item["task"], rounds)
    if only is not None:
        models = {"tree": models["tree"]}
    out = {}
    for name, model in models.items():
        model.fit(x[train], y_train.astype(cast))
        out[name] = losses(item["task"], item["y"][~train].astype(cast), model.predict(x[~train]))
    return out


def lift(
    learned: dict, base: np.ndarray, units: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """The better learner's share of the baseline loss removed, and its bootstrap interval over
    resampled units of test rows."""
    stacked = np.stack([*learned.values(), base])
    point = 1 - stacked[:-1].mean(axis=1).min() / max(base.mean(), 1e-12)
    _, unit = np.unique(units, return_inverse=True)
    k = unit.max() + 1
    sums = np.stack([np.bincount(unit, weights=values, minlength=k) for values in stacked])
    sizes = np.bincount(unit, minlength=k)
    drawn = np.stack([np.bincount(d, minlength=k) for d in rng.integers(k, size=(RESAMPLES, k))])
    means = (drawn @ sums.T) / (drawn @ sizes)[:, None]
    resampled = 1 - means[:, :-1].min(axis=1) / np.maximum(means[:, -1], 1e-12)
    low, high = np.quantile(resampled, [0.025, 0.975])
    return point, low, high


def band(point: float, low: float) -> str:
    if low <= 0:
        return "no signal"
    return BANDS[1 + int(np.searchsorted(EDGES, point, side="right"))]


def single(item: dict) -> tuple[str, float]:
    """The feature whose tree alone removes the most of the constant baseline's loss."""
    train, y = item["train"], item["y"]
    constant = baselines(item["task"], y[train], y[~train], [])["constant"].mean()

    def score(feature: str, rounds: int) -> float:
        return 1 - fit(item, y[train], [feature], rounds)["tree"].mean() / max(constant, 1e-12)

    screened = sorted(item["x"], key=lambda c: score(c, ROUNDS // 4), reverse=True)[:3]
    scores = {c: score(c, ROUNDS) for c in screened}
    best = max(scores, key=scores.get)
    return best, scores[best]


def normal_scores(y: pd.Series, train: np.ndarray) -> pd.Series:
    """Values as normal scores of the training rows' distribution, so outliers cannot decide."""
    known = np.sort(y[train].to_numpy(dtype=float))
    below = np.searchsorted(known, y.to_numpy(dtype=float), side="left")
    above = np.searchsorted(known, y.to_numpy(dtype=float), side="right")
    return pd.Series(norm.ppf((below + above + 1) / (2 * len(known) + 2)), index=y.index)


def evaluate(item: dict, rng: np.random.Generator, shuffle: bool = False) -> dict:
    train, y = item["train"], item["y"]
    if item["task"] == "regression":
        y = normal_scores(y, train)
        item = item | {"y": y}
    y_train = y[train]
    if shuffle:
        y_train = pd.Series(rng.permutation(y_train.to_numpy()), index=y_train.index)
    groups = [(g[train], g[~train]) for g in item["groups"]]
    base = baselines(item["task"], y_train, y[~train], groups)
    best = min(base, key=lambda b: base[b].mean())
    learned = fit(item, y_train)
    point, low, high = lift(learned, base[best], item["clusters"][~train], rng)
    record = {
        "source": item["source"],
        "table": item["table"],
        "target": item["target"],
        "task": item["task"],
        "split": item["split"],
        "child": item["child"],
        "best baseline": "constant" if best == "constant" else "per parent",
        "lift": point,
        "low": low,
        "high": high,
        "band": band(point, low),
    }
    constant = base["constant"].mean()
    for name, values in learned.items():
        record[name] = 1 - values.mean() / max(constant, 1e-12)
    record["near perfect"] = min(v.mean() for v in learned.values()) < NEAR_PERFECT * constant
    return record


rng = np.random.default_rng(0)
items = []
for seed in range(DATABASES):
    schema = PRIOR.realize(seed)
    frames = schema.sample(PRIOR.rows(schema, seed), seed=0)
    items += tasks(f"schema {seed}", frames, schema.describe(), rng)
for seed in range(TABLES):
    table = TablePrior().realize(seed, time=seed % 2 == 1)
    frames = {"t0": table.sample(TABLE_ROWS, seed=0)}
    items += tasks(f"table {seed}", frames, {"tables": {"t0": table.describe()}, "keys": []}, rng)
for item in items:
    assert item["target"] not in item["x"], "a target is among its own features"
    assert item["train"].any() and (~item["train"]).any()
with threadpool_limits(1):  # small fits: threads only contend
    results = pd.DataFrame([evaluate(item, rng) for item in items])
    controls = pd.DataFrame(
        [
            evaluate(item, rng, shuffle=True)
            for item in items[:: max(1, len(items) // CONTROLS)][:CONTROLS]
        ]
    )
    flagged = results.index[results["near perfect"]]
    alone = {i: single(items[i]) for i in flagged if not items[i]["copies"]}
near = [i for i, (_, best) in alone.items() if best > 1 - NEAR_PERFECT]

# 1. difficulty bands
results["group"] = results["task"] + ", " + results["split"] + " split"
counts = pd.crosstab(results["group"], results["band"]).reindex(columns=list(BANDS), fill_value=0)
counts.loc["all"] = counts.sum()
copied = results.index[[bool(item["copies"]) for item in items]]
counts.loc["all without an exact copy"] = (
    results.drop(copied)["band"].value_counts().reindex(BANDS, fill_value=0)
)
print(f"1. Difficulty of {len(results)} targets from {DATABASES} databases and {TABLES} tables:")
print("   lift = share of the best baseline's test loss the better learner removes\n")
print(counts.to_string(), "\n")
print(results.groupby("group")["lift"].describe()[["25%", "50%", "75%"]].round(2).to_string(), "\n")

# 2. learners and baselines
child = results[results["child"]]
parent_wins = int((child["best baseline"] == "per parent").sum())
tree_wins = results["tree"] > results["linear"]
skill = results[["linear", "tree"]].clip(lower=-1).median().round(2)
print("2. Learners and baselines:")
print(f"   median lift over the constant baseline: linear {skill['linear']}, tree {skill['tree']}")
print(f"   the tree beats the linear model on {int(tree_wins.sum())} of {len(results)} targets")
print(f"   the mean per parent beats the constant on {parent_wins} of {len(child)} child targets")
joins = {(i["source"], i["table"]): (i["split"], *i["withheld"]) for i in items if i["child"]}
joins = pd.DataFrame(joins.values(), columns=["split", "withheld", "offered"])
withheld = joins.groupby("split")[["withheld", "offered"]].sum()
print("   parent columns withheld as summaries of the child, and offered, by split:")
print(withheld.to_string().replace("\n", "\n   ").join(["   ", "\n"]))

# 3. leakage-like results
copies = flagged.intersection(copied)
print(f"3. Near-perfect targets: {len(flagged)}; with an exact copy among the features (a column")
print(f"   the target is a function of): {len(copies)}, of {len(copied)} targets that have one;")
print(
    f"   with a near-copy instead (one feature alone near-perfect): {len(near)}. Those without an"
)
print("   exact copy, and the best lift one feature reaches alone:")
for i, (feature, best) in alone.items():
    row = results.loc[i]
    print(
        f"   {items[i]['source']} {row['table']}.{row['target']} ({row['task']}):"
        f" lift {row['lift']:.3f}; {feature} alone {best:.3f}{' (near-copy)' if i in near else ''}"
    )
print()

# 4. shuffled control
false = int((controls["band"] != "no signal").sum())
print(f"4. Control, shuffled training targets: signal found on {false} of {len(controls)}\n")
assert false <= (0.3 if QUICK else 0.1) * len(controls), "the harness finds signal in shuffled data"
print(f"difficulty: {len(results)} targets banded; shuffled targets show no signal")
