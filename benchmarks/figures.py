"""The benchmark report's figures, drawn from its summary with matplotlib: time and memory against
rows drawn, one panel per kind of schema, and where the time goes at each workload's largest size.

Colors follow a validated categorical order, assigned by position within a panel and never cycled;
three of them sit below 3:1 contrast on the surface, so every line also carries a direct label."""

import math
import statistics
from pathlib import Path

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#b9b8b2"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")

# panels in reading order, each a kind of schema and its workloads in color order
GROUPS = {
    "Purely synthetic": ("prior-table", "prior-1", "prior-0", "prior-2"),
    "Hand-authored, one table": ("narrow", "columns-10", "dense", "wide", "columns-200"),
    "Hand-authored, many tables": ("tables-2", "tables-8", "tables-32", "keyed"),
    "Hand-authored with synthetic or real tables": ("mixed", "retail", "visits", "enrollments"),
}


def panels(workloads: set[str]) -> dict[str, list[str]]:
    """GROUPS restricted to the workloads measured, and any others in a last panel."""
    grouped = {title: [w for w in names if w in workloads] for title, names in GROUPS.items()}
    known = {w for names in GROUPS.values() for w in names}
    grouped["Other"] = sorted(workloads - known)
    return {title: names for title, names in grouped.items() if names}


def style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def label(workload: str, tables: int) -> str:
    return f"{workload} · {tables} tables" if tables > 1 else workload


def rows_text(value: float, _=None) -> str:
    for unit, scale in (("M", 1e6), ("k", 1e3)):
        if value >= scale:
            return f"{value / scale:g}{unit}"
    return f"{value:g}"


def seconds_text(value: float, _=None) -> str:
    return f"{value:g} s"


def memory_text(value: float, _=None) -> str:
    return f"{value / 1024:g} GiB" if value >= 1024 else f"{value:g} MiB"


def spaced(heights: list[float], gap: float) -> list[float]:
    """Label heights in log10 space, moved apart by at least `gap`, in their order."""
    order = sorted(range(len(heights)), key=heights.__getitem__)
    placed = list(heights)
    for previous, current in zip(order, order[1:]):
        placed[current] = max(placed[current], placed[previous] + gap)
    return placed


def scaling(plt, ticker, summary: dict, field: str, axis_label: str, title: str, path: Path):
    groups = panels({w for w, _ in summary})
    columns = 2
    lines = math.ceil(len(groups) / columns)
    figure, axes = plt.subplots(
        lines, columns, figsize=(10, 3.6 * lines + 0.8), sharex=True, sharey=True, squeeze=False
    )
    floor = min(statistics.median(s["peak"]) for s in summary.values())
    values = [statistics.median(s[field]) for s in summary.values()]
    gap = 0.05 * math.log10(max(values) / min(values))  # labels a twentieth of the span apart
    for axis, (panel, workloads) in zip(axes.flat, groups.items()):
        ends = []
        for color, workload in zip(SERIES, workloads):
            points = sorted(
                (s["rows"], statistics.median(s[field]), s["tables"])
                for (w, _), s in summary.items()
                if w == workload
            )
            x, y, tables = zip(*points)
            axis.plot(
                x,
                y,
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=4.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                label=label(workload, tables[-1]),
                zorder=3,
            )
            ends.append((x[-1], y[-1], label(workload, tables[-1])))
        heights = spaced([math.log10(y) for _, y, _ in ends], gap)
        for (x, y, text), height in zip(ends, heights):
            axis.annotate(
                text,
                (x, y),
                xytext=(x * 1.3, 10**height),
                color=MUTED,
                fontsize=7.5,
                va="center",
                arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.6, "shrinkA": 0},
            )
        if field == "peak":
            axis.axhline(floor, color=AXIS, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        axis.set_title(panel)
        axis.grid(True, which="major")
        axis.set_axisbelow(True)
    for axis in axes.flat[len(groups) :]:
        axis.set_visible(False)
    first = axes.flat[0]
    first.set(xscale="log", yscale="log")
    widest = max(s["rows"] for s in summary.values())
    first.set_xticks([10**k for k in range(3, math.floor(math.log10(widest)) + 1)])
    first.set_xlim(right=widest * 9)  # room for the direct labels, no ticks past the data
    first.xaxis.set_major_formatter(ticker.FuncFormatter(rows_text))
    first.xaxis.set_minor_locator(ticker.NullLocator())
    if field == "peak":
        steps = [100, 200, 300, 500, 1024, 2048, 3072, 5120, 10240, 20480]
        first.set_yticks([v for v in steps if min(values) / 1.2 <= v <= max(values) * 1.2])
    first.yaxis.set_major_formatter(
        ticker.FuncFormatter(memory_text if field == "peak" else seconds_text)
    )
    first.yaxis.set_minor_formatter(ticker.NullFormatter())
    for axis in axes[-1]:
        axis.set_xlabel("rows drawn")
    for axis in axes[:, 0]:
        axis.set_ylabel(axis_label)
    figure.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    if field == "peak":
        figure.text(
            0.01,
            0.005,
            f"Dashed: {memory_text(round(floor))}, the interpreter and imports alone.",
            color=MUTED,
            fontsize=8,
        )
    figure.tight_layout(rect=(0, 0.02, 1, 0.97))
    figure.savefig(path, dpi=200)
    plt.close(figure)


def phases(plt, summary: dict, phase_names: tuple[str, ...], title: str, path: Path):
    """Each workload's time at its largest size, split by phase, as shares of its total."""
    order = [w for names in panels({w for w, _ in summary}).values() for w in names]
    largest = {w: max(n for v, n in summary if v == w) for w in order}
    figure, axis = plt.subplots(figsize=(10, 0.36 * len(order) + 1.6))
    for row, workload in enumerate(order):
        s = summary[workload, largest[workload]]
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
        axis.text(
            1.01,
            row,
            f"{seconds_text(float(f'{total:.2g}'))} · {rows_text(float(f'{s["rows"]:.2g}'))} rows",
            va="center",
            color=MUTED,
            fontsize=8,
        )
    axis.set_yticks(range(len(order)), [f"{w} ({rows_text(largest[w])})" for w in order])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.set_xlabel("share of end-to-end time")
    axis.tick_params(axis="y", length=0)
    axis.spines["left"].set_visible(False)
    axis.legend(ncol=len(phase_names), loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1)
    figure.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 0.9, 0.98))
    figure.savefig(path, dpi=200)
    plt.close(figure)


def draw(summary: dict, phase_names: tuple[str, ...], machine: str, out: Path) -> list[str]:
    """The figures beside `out`, named after it, or none without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import ticker
    except ImportError:
        return []
    style(plt)
    stem = out.stem.removeprefix("REPORT").strip("-").lower()
    names = {
        kind: f"{stem}-{kind}.png" if stem else f"{kind}.png"
        for kind in ("time", "memory", "phases")
    }
    scaling(
        plt,
        ticker,
        summary,
        "total",
        "end-to-end time",
        f"End-to-end time against rows drawn · {machine}",
        out.parent / names["time"],
    )
    scaling(
        plt,
        ticker,
        summary,
        "peak",
        "peak memory",
        f"Peak memory against rows drawn · {machine}",
        out.parent / names["memory"],
    )
    phases(
        plt,
        summary,
        phase_names,
        f"Where the time goes at each workload's largest size · {machine}",
        out.parent / names["phases"],
    )
    return list(names.values())
