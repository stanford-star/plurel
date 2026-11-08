"""The benchmark's workloads: each builds a declaration and the row counts that draw about `rows`
rows from it. They cover single tables narrow and wide, ladders of columns, causes and tables,
random-prior databases, the domain recipes with their retained rows, and keys that match, prefer,
count and stay unique."""

import sys
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

import pandas as pd

from plurel import (
    Categorical,
    Counts,
    ForeignKey,
    IntegersRange,
    LogNormal,
    Numeric,
    PrimaryKey,
    Recorded,
    Schema,
    Shares,
    Table,
    Timestamp,
    Uniform,
)
from plurel.prior import SchemaPrior, TablePrior

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "generation"))

Build = Callable[[int], tuple[Schema, dict[str, int]]]


def narrow(rows: int) -> tuple[Schema, dict[str, int]]:
    """One table of five columns, a short chain of causes."""
    table = Table(
        {
            "a": Numeric(Uniform(0, 1)),
            "b": Numeric(Uniform(0, 1), causes={"a": 1.0}),
            "c": Categorical(Shares(("x", "y", "z")), causes={"b": 1.0}),
            "d": Numeric(causes={"a": 0.5, "c": {"y": 1.0}}),
            "e": Numeric(causes={"d": 1.0}, missing=0.1),
        },
        name="narrow",
    )
    return Schema({"narrow": table}), {"narrow": rows}


def chained(width: int, causes: int) -> dict[str, object]:
    """`width` numbers, each caused by the `causes` before it."""
    columns: dict[str, object] = {"x0": Numeric(Uniform(0, 1))}
    for i in range(1, width):
        columns[f"x{i}"] = Numeric(
            Uniform(0, 1), causes={f"x{j}": 0.3 for j in range(max(0, i - causes), i)}
        )
    return columns


def wide(rows: int) -> tuple[Schema, dict[str, int]]:
    """One table of 50 numbers, each caused by the three before it, and a categorical."""
    columns = chained(50, 3) | {"c": Categorical(Shares(("a", "b", "c")), causes={"x49": 1.0})}
    return Schema({"wide": Table(columns, name="wide")}), {"wide": rows}


def shaped(width: int, causes: int) -> Build:
    """One table of `width` numbers, each caused by the `causes` before it."""

    def build(rows: int) -> tuple[Schema, dict[str, int]]:
        return Schema({"t": Table(chained(width, causes), name="t")}), {"t": rows}

    build.__doc__ = f"one table of {width} numbers, each caused by up to {causes} before it"
    return build


def tables(count: int) -> Build:
    """A chain of `count` tables of five columns, each keyed to the one before and caused by it,
    the rows split evenly, none below 256."""

    def build(rows: int) -> tuple[Schema, dict[str, int]]:
        declared = {"t0": Table({"id": PrimaryKey()} | chained(4, 1))}
        for i in range(1, count):
            columns = {"id": PrimaryKey(), "parent": ForeignKey(f"t{i - 1}")} | chained(4, 1)
            columns["x0"] = Numeric(Uniform(0, 1), causes={"parent.x3": 1.0})
            declared[f"t{i}"] = Table(columns)
        return Schema(declared), {t: max(256, rows // count) for t in declared}

    build.__doc__ = f"a chain of {count} keyed tables"
    return build


def prior(seed: int) -> Build:
    """A random-prior database, its tables scaled so the largest takes `rows`, none below 256."""

    def build(rows: int) -> tuple[Schema, dict[str, int]]:
        drawn = SchemaPrior()
        schema = drawn.realize(seed)
        sizes = drawn.rows(schema, seed)
        scale = rows / max(sizes.values())
        # a link's clusters need rows: no table below 256, or its own drawn size if smaller
        return schema, {table: max(min(n, 256), round(n * scale)) for table, n in sizes.items()}

    build.__doc__ = f"a SchemaPrior database, seed {seed}"
    return build


def prior_table(rows: int) -> tuple[Schema, dict[str, int]]:
    """One table of 20 columns drawn by TablePrior, the same table at every size."""
    table = TablePrior(column_count=IntegersRange(20, 20)).realize(0)
    return Schema({"t": table}), {"t": rows}


def mixed(rows: int) -> tuple[Schema, dict[str, int]]:
    """Hand-authored customers and orders around a product table drawn by TablePrior: orders
    point at both, their amount caused across both keys. A twentieth of the rows are products,
    a tenth customers, the rest orders."""
    products = TablePrior(column_count=IntegersRange(10, 10)).realize(0)
    customers = Table(
        {
            "id": PrimaryKey(),
            "segment": Categorical({"gold": 0.2, "silver": 0.3, "bronze": 0.5}),
            "income": Numeric(LogNormal(), causes={"segment": {"gold": 1.0}}),
        }
    )
    orders = Table(
        {
            "id": PrimaryKey(),
            "customer_id": ForeignKey("customers"),
            "product_id": ForeignKey("products"),
            "time": Timestamp("2020-01-01", "2024-01-01"),
            "amount": Numeric(
                LogNormal(), causes={"customer_id.income": 1.0, "product_id.col0": 0.5}
            ),
        },
        time_column="time",
    )
    schema = Schema({"products": products, "customers": customers, "orders": orders})
    share = {"products": 0.05, "customers": 0.1, "orders": 0.85}
    return schema, {t: max(256, round(rows * s)) for t, s in share.items()}


def keyed(rows: int) -> tuple[Schema, dict[str, int]]:
    """Students counted per real course level, matched to courses, unique per student."""
    courses = Table.from_frame(
        pd.DataFrame({"id": range(40), "level": ["intro", "core", "advanced", "elective"] * 10}),
        primary_key="id",
        extra={"size": Numeric(Uniform(10, 300))},
    )
    students = Table({"id": PrimaryKey(), "year": Numeric(Uniform(1, 4), round=1)})
    enrollments = Table(
        {
            "student_id": ForeignKey(
                "students", counts=Counts(3.0, {"year": 0.4}, least=1, most=8)
            ),
            "level": Categorical(
                ("intro", "core", "advanced", "elective"), causes={"student_id.year": 1.0}
            ),
            "course_id": ForeignKey(
                "courses",
                match={"level": "level"},
                prefer={Recorded("size"): 0.01},
                unique="student_id",
            ),
            "grade": Numeric(causes={Recorded("student_id.year"): 5.0}, noise=5.0),
        }
    )
    schema = Schema({"courses": courses, "students": students, "enrollments": enrollments})
    return schema, {"students": max(2, rows // 3)}


def retail(rows: int) -> tuple[Schema, dict[str, int]]:
    """The retail recipe, its retained customers grown with the size: about 380 rows per 40."""
    recipe = import_module("retail")
    return recipe.build(*recipe.fixtures(max(1, round(rows / 380)))), {}


def visits(rows: int) -> tuple[Schema, dict[str, int]]:
    """The visits recipe: its retained lookups stay small, and patients take the size."""
    recipe = import_module("visits")
    return recipe.build(*recipe.fixtures()), {"patients": max(2, rows // 3)}


def enrollments(rows: int) -> tuple[Schema, dict[str, int]]:
    """The enrollments recipe, its retained students grown with the size: about 430 rows per 150."""
    recipe = import_module("enrollments")
    return recipe.build(*recipe.fixtures(max(1, round(rows / 430)))), {}


WORKLOADS: dict[str, Build] = {
    "narrow": narrow,
    "wide": wide,
    "columns-10": shaped(10, 3),
    "columns-200": shaped(200, 3),
    "dense": shaped(40, 40),
    "tables-2": tables(2),
    "tables-8": tables(8),
    "tables-32": tables(32),
    "prior-table": prior_table,
    "prior-0": prior(0),
    "prior-1": prior(1),
    "prior-2": prior(2),
    "mixed": mixed,
    "keyed": keyed,
    "retail": retail,
    "visits": visits,
    "enrollments": enrollments,
}

# the size above which a workload is not attempted, kept for the report: 200 columns at 1M rows is
# 200M cells, beyond the memory of the machines measured so far
LARGEST: dict[str, int] = {"columns-200": 100_000}
