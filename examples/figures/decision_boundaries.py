"""Decision boundaries of tables drawn by `TablePrior`, as in the README.

For each table seed, takes the first categorical column with two numeric scalar causes upstream
whose map has 2 to 6 classes, none above 80%, changes along both axes and little speckle. The two
causes are held on a grid over their 2-98% quantiles and every other source node at its median, so
the map is the mechanism's own boundary, not a fitted classifier; each cell shows the class drawn
most often over several data seeds. Pairs are searched on a coarse grid and drawn on a fine one.
Needs matplotlib; takes about two minutes for the default eight seeds."""

import argparse
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from plurel import TablePrior

SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
SEARCH, DRAW = (120, 15), (300, 25)


def majority(labels: np.ndarray, size: int) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(labels, size // 2, mode="edge"), (size, size)
    ).reshape(*labels.shape, size * size)
    return np.stack([(windows == c).sum(-1) for c in range(-1, labels.max() + 1)]).argmax(0) - 1


def ancestors(table, name: str) -> set[str]:
    found, stack = set(), [name]
    while stack:
        for parent in table.nodes[stack.pop()].parents:
            if parent not in found:
                found.add(parent)
                stack.append(parent)
    return found


def classes(table, column, a, b, natural, sources, grid, draws) -> np.ndarray:
    xs, ys = (np.linspace(*np.quantile(natural[n][:, 0], [0.02, 0.98]), grid) for n in (a, b))
    held = {
        s: np.repeat(np.median(natural[s], axis=0, keepdims=True), grid * grid, axis=0)
        for s in sources
    }
    held[a], held[b] = (v.reshape(-1, 1) for v in np.meshgrid(xs, ys))
    levels = None
    votes = []
    for seed in range(draws):
        values = table.sample_with_latents(grid * grid, seed=seed, held=held)[0][column].astype(
            "category"
        )
        levels = list(values.cat.categories) if levels is None else levels
        votes.append(values.cat.set_categories(levels).cat.codes.to_numpy())
    counts = np.stack([(np.stack(votes) == c).sum(0) for c in range(len(levels))])
    mode = np.where(counts.max(0) > 0, counts.argmax(0), -1)
    return mode.reshape(grid, grid)


def clear(labels: np.ndarray) -> bool:
    known = labels[labels >= 0]
    shares = np.bincount(known) / max(known.size, 1)
    smooth = majority(labels, 3)
    across, along = ((np.diff(smooth, axis=k) != 0).mean() for k in (1, 0))
    return (
        2 <= (shares > 0.03).sum() <= 6
        and shares.max() < 0.8
        and min(across, along) > 0.2 * max(across, along)
        and (smooth != labels).mean() < 0.06
    )


def boundary(seed: int) -> np.ndarray | None:
    table = TablePrior().realize(seed)
    natural = table.sample_with_latents(4000, seed=0)[1]
    sources = [n for n, node in table.nodes.items() if not node.edges]
    for column, entry in table.columns.items():
        if getattr(entry, "kind", None) != "categorical":
            continue
        upstream = ancestors(table, entry.node.edges[0].parent)
        scalars = sorted(
            n for n in upstream if table.nodes[n].dim == 1 and not table.nodes[n].onehot
        )
        for a, b in itertools.combinations(scalars, 2):
            if clear(classes(table, column, a, b, natural, sources, *SEARCH)):
                return classes(table, column, a, b, natural, sources, *DRAW)
    return None


def image(labels: np.ndarray, tint: float = 0.22) -> np.ndarray:
    palette = np.array([to_rgb(c) for c in SERIES]) * (1 - tint) + tint
    labels = majority(labels, 3)
    rgb = np.full((*labels.shape, 3), to_rgb(SURFACE))
    rgb[labels >= 0] = palette[labels[labels >= 0] % len(SERIES)]
    smooth = majority(labels, 5)
    edge = np.zeros(labels.shape, bool)
    edge[:, 1:] |= smooth[:, 1:] != smooth[:, :-1]
    edge[1:, :] |= smooth[1:, :] != smooth[:-1, :]
    rgb[edge] = 1.0
    return rgb


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--seeds", type=int, nargs="+", default=[45, 102, 225, 115, 47, 87, 123, 262])
parser.add_argument("--out", default="decision_boundaries.png")
args = parser.parse_args()

maps = [m for m in map(boundary, args.seeds) if m is not None]
if not maps:
    raise ValueError(f"no table among seeds {args.seeds} has a clear boundary")
cols = min(4, len(maps))
rows = -(-len(maps) // cols)
fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), facecolor=SURFACE, squeeze=False)
for ax in axes.flat:
    ax.set_axis_off()
for ax, labels in zip(axes.flat, maps):
    ax.imshow(image(labels), origin="lower", interpolation="nearest")
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.04, hspace=0.04)
fig.savefig(args.out, dpi=220, facecolor=SURFACE)
print(f"{len(maps)} of {len(args.seeds)} tables drawn to {args.out}")
