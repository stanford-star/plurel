"""The benchmark report's figures, drawn from its summary with matplotlib: time and peak memory
against rows drawn, one line per workload, and where the time goes at the largest size.

Colors follow a validated categorical order, assigned to workloads in a fixed order and never
cycled; three of them sit below 3:1 contrast on the surface, so the legend names every line and the
report's tables give every number."""

import statistics
from pathlib import Path

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#3d3c39", "#e6e5e1", "#a8a7a1"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")
ORDER = ("table", "database-2", "database-8", "database-32", "database-128")
KINDS = ("time", "memory", "phases")


def label(workload: str) -> str:
    """What a workload is, for readers; its short name stays in the results."""
    if workload == "table":
        return "random table, 20 columns"
    if workload.startswith("database-"):
        return f"random database, {workload.split('-')[1]} tables"
    return workload


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


def ordered(summary: dict) -> list[str]:
    measured = {w for w, _ in summary}
    return [w for w in ORDER if w in measured] + sorted(measured - set(ORDER))


def rows_text(value: float, _=None) -> str:
    if value >= 1e6:
        return f"{value / 1e6:g}M"
    return f"{value / 1e3:g}k" if value else "0"


def memory_text(value: float, _=None) -> str:
    return f"{value / 1024:g} GiB" if value >= 1024 else f"{value:g} MiB"


def scaling(plt, ticker, summary: dict, field: str, axis_label: str, title: str, path: Path):
    figure, axis = plt.subplots(figsize=(9, 5.6))
    for color, workload in zip(SERIES, ordered(summary)):
        points = sorted(
            (s["rows"], statistics.median(s[field]))
            for (w, _), s in summary.items()
            if w == workload
        )
        axis.plot(
            *zip(*points),
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=label(workload),
        )
    axis.set_xlim(0, None)
    axis.set_ylim(0, None)
    axis.xaxis.set_major_formatter(ticker.FuncFormatter(rows_text))
    if field == "peak":
        axis.yaxis.set_major_locator(ticker.MultipleLocator(512))
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(memory_text))
    else:
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g} s"))
    axis.set_xlabel("rows drawn")
    axis.set_ylabel(axis_label)
    axis.grid(True)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def phases(plt, summary: dict, phase_names: tuple[str, ...], title: str, path: Path):
    """Each workload at the largest size, its time split by phase as shares of the total."""
    largest = max(n for _, n in summary)
    order = [w for w in ordered(summary) if (w, largest) in summary]
    figure, axis = plt.subplots(figsize=(12, 0.62 * len(order) + 2.2))
    for row, workload in enumerate(order):
        s = summary[workload, largest]
        times = [statistics.median(s["phases"][p]) for p in phase_names]
        total = sum(times)
        start = 0.0
        for color, phase, time in zip(SERIES, phase_names, times):
            axis.barh(
                row,
                time / total,
                left=start,
                height=0.66,
                color=color,
                edgecolor=SURFACE,
                linewidth=1.5,
                label=phase if row == 0 else None,
            )
            start += time / total
        axis.text(1.01, row, f"{total:.2g} s", va="center", color=INK, fontsize=11.5)
    axis.set_yticks(range(len(order)), [label(w) for w in order])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.set_xlabel(f"share of the time to draw, save and read back {rows_text(largest)} rows")
    axis.tick_params(axis="y", length=0)
    axis.spines["left"].set_visible(False)
    axis.legend(ncol=len(phase_names), loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1)
    figure.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 0.94, 0.98))
    figure.savefig(path, dpi=200)
    plt.close(figure)


def draw(summary: dict, phase_names: tuple[str, ...], machine: str, results: Path) -> list[Path]:
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
    scaling(
        plt,
        ticker,
        summary,
        "total",
        "seconds, end to end",
        f"Time against rows drawn · {machine}",
        paths["time"],
    )
    scaling(
        plt,
        ticker,
        summary,
        "peak",
        "peak memory",
        f"Peak memory against rows drawn · {machine}",
        paths["memory"],
    )
    phases(plt, summary, phase_names, f"Where the time goes · {machine}", paths["phases"])
    return list(paths.values())
