"""How far two platforms' seeded draws differ, from what `test/fingerprints.py --save` writes.

    python scripts/compare_draws.py draws/darwin-arm64 draws/linux-x86_64-X86_V3

Per draw: whether its bytes are the same (the digests saved beside it), whether both drew the same
tables and every table as many rows, the share of cells that differ, the largest numeric
difference in the column's standard deviations, the cells blank on one side only, the share of
categorical cells that changed, the cells whose rank in their column changed, and how far the
marginals (means in deviations, category shares) and the dependence between numbers (Spearman
correlation) moved. Numbers include keys and times in seconds. A draw saved on one side only is
listed as missing."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def numeric(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(series):
        return np.where(
            series.isna(), np.nan, series.to_numpy("datetime64[ns]").astype(np.int64) / 1e9
        )
    return series.to_numpy(dtype=float, na_value=np.nan)


def is_number(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)


def deviation(values: np.ndarray) -> float:
    spread = np.nanstd(values)
    return spread if spread > 0 else 1.0


def reranked(u: np.ndarray, v: np.ndarray) -> int:
    """Rows whose rank in the column differs, ties and blanks ranked by position."""
    first, second = (pd.Series(x).rank(method="first", na_option="bottom") for x in (u, v))
    return int((first != second).sum())


class Tally:
    """Differences summed over a draw's tables and columns."""

    def __init__(self) -> None:
        self.cells = self.differing = self.blank = self.reordered = 0
        self.changed = self.categorical = 0
        self.largest = self.mean_shift = self.share_shift = self.dependence = 0.0

    def number(self, u: np.ndarray, v: np.ndarray, aligned: bool) -> None:
        scale = deviation(u)
        self.mean_shift = max(self.mean_shift, abs(np.nanmean(u) - np.nanmean(v)) / scale)
        if not aligned:
            return
        blank_u, blank_v = np.isnan(u), np.isnan(v)
        flips = blank_u != blank_v
        gap = np.abs(u - v)[~blank_u & ~blank_v]
        self.blank += int(flips.sum())
        self.differing += int(flips.sum() + (gap > 0).sum())
        self.largest = max(self.largest, float(gap.max()) / scale if gap.size else 0.0)
        self.reordered += reranked(u, v)

    def levels(self, x: pd.Series, y: pd.Series, aligned: bool) -> None:
        shares = pd.concat(
            [x.value_counts(normalize=True), y.value_counts(normalize=True)], axis=1
        ).fillna(0.0)
        self.share_shift = max(
            self.share_shift, float((shares.iloc[:, 0] - shares.iloc[:, 1]).abs().max())
        )
        if aligned:
            first, second = x.astype(object), y.astype(object)
            blank = first.isna() != second.isna()
            moved = int((first.fillna("<NA>") != second.fillna("<NA>")).sum())
            self.blank += int(blank.sum())
            self.changed += moved
            self.differing += moved
            self.categorical += len(x)

    def table(self, a: pd.DataFrame, b: pd.DataFrame) -> bool:
        aligned = len(a) == len(b) and list(a.columns) == list(b.columns)
        for column in a.columns:
            if column not in b:
                continue
            if is_number(a[column]):
                self.number(numeric(a[column]), numeric(b[column]), aligned)
            else:
                self.levels(a[column], b[column], aligned)
            self.cells += len(a)
        numbers = [c for c in a.columns if is_number(a[c])]
        if aligned and len(numbers) > 1:
            ranks = [
                pd.DataFrame({c: numeric(frame[c]) for c in numbers}).corr("spearman").to_numpy()
                for frame in (a, b)
            ]
            self.dependence = max(self.dependence, float(np.nanmax(np.abs(ranks[0] - ranks[1]))))
        return aligned


def share(count: int, total: int) -> str:
    return f"{count / total:.1%}" if total else "–"


def compare(first: Path, second: Path) -> dict[str, str]:
    (a, a_digest), (b, b_digest) = read(first), read(second)
    tally, aligned = Tally(), set(a) == set(b)
    for table in sorted(set(a) & set(b)):
        aligned &= tally.table(a[table], b[table])
    shown = (lambda text: text) if aligned else (lambda text: "–")
    return {
        "same bytes": "yes" if a_digest == b_digest else "no",
        "same tables and rows": "yes" if aligned else "no",
        "cells differing": shown(share(tally.differing, tally.cells)),
        "largest number difference, sd": shown(f"{tally.largest:.2g}"),
        "blank on one side": shown(str(tally.blank)),
        "categorical cells changed": shown(share(tally.changed, tally.categorical)),
        "cells reranked": shown(str(tally.reordered)),
        "mean shift, sd": f"{tally.mean_shift:.2g}",
        "category share shift": f"{tally.share_shift:.2g}",
        "Spearman shift": shown(f"{tally.dependence:.2g}"),
    }


def read(folder: Path) -> tuple[dict[str, pd.DataFrame], str]:
    """A saved draw's tables, their categorical levels restored, and its digest."""
    record = json.loads((folder / "draw.json").read_text())
    frames = {}
    for path in sorted(folder.glob("*.parquet")):
        frame = pd.read_parquet(path)
        for column, kind in record["categoricals"].get(path.stem, {}).items():
            frame[column] = pd.Categorical(
                frame[column], categories=kind["levels"], ordered=kind["ordered"]
            )
        frames[path.stem] = frame
    return frames, record["digest"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    names = sorted(
        {f.name for root in (args.first, args.second) for f in root.iterdir() if f.is_dir()}
    )
    rows = {}
    for name in names:
        sides = [root / name for root in (args.first, args.second)]
        missing = [str(side.parent) for side in sides if not side.is_dir()]
        rows[name] = {"same bytes": f"missing from {', '.join(missing)}"} if missing else None
        rows[name] = rows[name] or compare(*sides)
    header = ["draw", *max(rows.values(), key=len)]
    print("| " + " | ".join(header) + " |")
    print("|---|" + "---:|" * (len(header) - 1))
    for name, row in rows.items():
        cells = [row.get(column, "") for column in header[1:]]
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
