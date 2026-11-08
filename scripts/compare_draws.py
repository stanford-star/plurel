"""How far two platforms' seeded draws differ, from the parquet `test/fingerprints.py --save` writes.

    python scripts/compare_draws.py draws/darwin-arm64 draws/linux-x86_64

Per draw: whether every table drew as many rows, the share of cells that differ, the largest
numeric difference in the column's standard deviations, the share of categorical cells that
changed, and how far the marginals (means in deviations, category shares) and the dependence
between numbers (Spearman correlation) moved. Numbers include keys and times in seconds."""

import argparse
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


def compare(first: dict[str, pd.DataFrame], second: dict[str, pd.DataFrame]) -> dict[str, object]:
    same_rows, cells, differing = True, 0, 0
    largest, changed, categorical = 0.0, 0, 0
    mean_shift, share_shift, dependence = 0.0, 0.0, 0.0
    for table, a in first.items():
        b = second[table]
        aligned = len(a) == len(b)
        same_rows &= aligned
        for column in a.columns:
            x, y = a[column], b[column]
            if is_number(x):
                u, v = numeric(x), numeric(y)
                scale = deviation(u)
                mean_shift = max(mean_shift, abs(np.nanmean(u) - np.nanmean(v)) / scale)
                if aligned:
                    both = ~(np.isnan(u) & np.isnan(v))
                    gap = np.abs(np.nan_to_num(u - v, nan=np.inf))[both]
                    differing += int((gap > 0).sum())
                    finite = gap[np.isfinite(gap)]
                    largest = max(largest, float(finite.max()) / scale if finite.size else 0.0)
            else:
                shares = pd.concat(
                    [x.value_counts(normalize=True), y.value_counts(normalize=True)], axis=1
                ).fillna(0.0)
                share_shift = max(
                    share_shift, float((shares.iloc[:, 0] - shares.iloc[:, 1]).abs().max())
                )
                if aligned:
                    moved = int(
                        (x.astype(object).fillna("<NA>") != y.astype(object).fillna("<NA>")).sum()
                    )
                    changed += moved
                    differing += moved
                    categorical += len(x)
            cells += len(x)
        numbers = [c for c in a.columns if is_number(a[c])]
        if aligned and len(numbers) > 1:
            ranks = [
                pd.DataFrame({c: numeric(frame[c]) for c in numbers}).corr("spearman").to_numpy()
                for frame in (a, b)
            ]
            dependence = max(dependence, float(np.nanmax(np.abs(ranks[0] - ranks[1]))))
    return {
        "same rows": "yes" if same_rows else "no",
        "cells differing": f"{differing / cells:.1%}" if same_rows else "–",
        "largest number difference, sd": f"{largest:.2g}" if same_rows else "–",
        "categorical cells changed": f"{changed / categorical:.1%}"
        if categorical and same_rows
        else "–",
        "mean shift, sd": f"{mean_shift:.2g}",
        "category share shift": f"{share_shift:.2g}",
        "Spearman shift": f"{dependence:.2g}" if same_rows else "–",
    }


def read(folder: Path) -> dict[str, pd.DataFrame]:
    return {path.stem: pd.read_parquet(path) for path in sorted(folder.glob("*.parquet"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    rows = {
        folder.name: compare(read(folder), read(args.second / folder.name))
        for folder in sorted(args.first.iterdir())
        if folder.is_dir()
    }
    header = ["draw", *next(iter(rows.values()))]
    print("| " + " | ".join(header) + " |")
    print("|---|" + "---:|" * (len(header) - 1))
    for name, row in rows.items():
        print(f"| {name} | " + " | ".join(str(v) for v in row.values()) + " |")


if __name__ == "__main__":
    main()
