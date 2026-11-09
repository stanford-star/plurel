"""The benchmark report's figures, drawn from its results with matplotlib: each draw's end-to-end
time against the rows it drew, and its peak memory against its cells, tables and databases in two
validated colors, on log scales."""

from pathlib import Path

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#3d3c39", "#e6e5e1", "#a8a7a1"
COLORS = {"table": "#2a78d6", "database": "#eb6834"}
LABELS = {"table": "random tables", "database": "random databases"}
KINDS = ("time", "memory")


def style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 12,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 11.5,
        }
    )


def amount(value: float, _=None) -> str:
    for unit, scale in (("M", 1e6), ("k", 1e3)):
        if value >= scale:
            return f"{value / scale:g}{unit}"
    return f"{value:g}"


def scatter(plt, ticker, runs: list[dict], x: str, y: str, labels, title: str, path: Path):
    figure, axis = plt.subplots(figsize=(9, 5.6))
    for kind, color in COLORS.items():
        points = [(r[x], r[y]) for r in runs if r["kind"] == kind and "failure" not in r]
        if points:
            axis.scatter(
                *zip(*points),
                s=34,
                color=color,
                edgecolor=SURFACE,
                linewidth=0.8,
                alpha=0.9,
                label=f"{LABELS[kind]} ({len(points)})",
            )
    axis.set(xscale="log", yscale="log", xlabel=labels[0], ylabel=labels[1])
    axis.xaxis.set_major_formatter(ticker.FuncFormatter(amount))
    axis.yaxis.set_major_formatter(ticker.FuncFormatter(labels[2]))
    axis.grid(True, which="major")
    axis.set_axisbelow(True)
    axis.legend(loc="upper left")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def draw(runs: list[dict], machine: str, results: Path) -> list[Path]:
    """The figures beside the results they are drawn from, named after them, or none without
    matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import ticker
    except ImportError:
        return []
    style(plt)
    paths = {kind: results.with_name(f"{results.stem}-{kind}.png") for kind in KINDS}
    scatter(
        plt,
        ticker,
        runs,
        "rows",
        "total_seconds",
        ("rows drawn", "seconds, end to end", lambda v, _: f"{v:g} s"),
        f"Time of each draw · {machine}",
        paths["time"],
    )
    scatter(
        plt,
        ticker,
        runs,
        "cells",
        "peak_mib",
        (
            "cells drawn",
            "peak memory",
            lambda v, _: f"{v / 1024:g} GiB" if v >= 1024 else f"{v:g} MiB",
        ),
        f"Peak memory of each draw · {machine}",
        paths["memory"],
    )
    return list(paths.values())
