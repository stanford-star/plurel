"""World knowledge for a catalog's schemas: the text a reader gets instead of a schema's own, a
``description`` and a note per column. It carries two things a table's rows cannot give at few rows:
what a large sample of the schema's population shows (:func:`study`: how common each outcome is, how
much each column tells about it, where its values sit per outcome), and what only text can say
(:func:`roles`: which columns cause the outcome, which follow from it or are computed from others, which
are unrelated, whether an unrecorded cause matters; and the domain's own meaning of its entities, which
the writer brings). :func:`brief` is what the writer reads, :func:`lint` checks what it returns and
:func:`load_knowledge` loads what is written (``<directory>/<sector>.json``, keyed by schema id)."""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from plurel.world.compiler import realize_world
from plurel.world.schema import WorldSchema

# realizations pooled, so the study holds for every table of the schema; realize tables under other
# seeds and the study never sees their rows
STUDY_SEEDS = (0, 1, 2, 3)
STUDY_ROWS = 2048  # rows per realization: 8192 in all
STUDY_DRAW = 2  # apart from the draws tables take first (0 and 1)
MIN_ROWS = (
    50  # an outcome or category level with fewer rows (or none recorded) gets no typical value
)
VOICES = ("first person", "impersonal", "second person")
# words that give the records away as generated; "driver" in the sense of what moves the outcome is
# for the review (a person who drives, or miles driven, is domain language)
BANNED = (r"\bsynthetic\b", r"\bsimulat")
DASHES = ("\u2014", "\u2013")  # em and en dash
# openings that become a frame when many schemas share them
STOCK_OPENING = r"^\W*((each|every) (single )?(row|record|entry|line)\b|you are looking at|this (table|dataset|file|log)\b)"


def load_knowledge(directory: Path) -> dict[str, dict]:
    """Every schema's written world knowledge in ``directory``, by schema id."""
    out: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        out.update(json.loads(path.read_text()))
    return out


def _ancestors(schema: WorldSchema, name: str) -> set[str]:
    seen, frontier = set(), list(schema.parent_names(name))
    while frontier:
        node = frontier.pop()
        if node not in seen:
            seen.add(node)
            frontier.extend(schema.parent_names(node))
    return seen


def roles(schema: WorldSchema) -> dict[str, str]:
    """Each column's place in the causal graph relative to the outcome."""
    upstream = _ancestors(schema, schema.target)
    out = {}
    for name in schema.feature_names():
        mine = _ancestors(schema, name)
        if inputs := schema._formula_inputs(name):
            out[name] = f"computed from {', '.join(inputs)}"
        elif name in upstream:
            out[name] = "a cause of the outcome"
        elif schema.target in mine:
            out[name] = "a consequence of the outcome"
        elif mine & upstream:
            out[name] = "shares a cause with the outcome"
        else:
            out[name] = "unrelated to the outcome"
    return out


def _information(x: np.ndarray, y: np.ndarray) -> float:
    """The share of the outcome's uncertainty that knowing ``x`` (discrete) removes."""
    _, y_codes = np.unique(y, return_inverse=True)
    counts = pd.crosstab(x, y_codes).to_numpy(dtype=float)
    joint = counts / counts.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    mutual = float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())
    entropy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    return mutual / entropy if entropy > 0 else 0.0


def study(schema: WorldSchema) -> dict:
    """What a large sample of the schema's population shows, pooled over ``STUDY_SEEDS``
    realizations: each outcome's share, and per column how much it alone tells about the outcome
    (``information``, 0 to 1) and where its values sit per outcome: a numeric column's median
    percentile (within its own realization, so rescaled units never mix), a categorical one's
    outcome shares per level. Columns most informative first."""
    target, frames = schema.target, []
    for seed in STUDY_SEEDS:
        frame = realize_world(schema, seed).rows(n=STUDY_ROWS, draw=STUDY_DRAW)
        numeric = [
            c for c in frame.columns if c != target and pd.api.types.is_numeric_dtype(frame[c])
        ]
        frame[numeric] = frame[numeric].rank(pct=True) * 100
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    y = rows[target].astype(str).to_numpy()
    outcomes = [
        v
        for v in (schema.nodes[target].levels or schema.nodes[target].class_scores)
        if (y == v).any()
    ]
    columns = {}
    for name in schema.feature_names():
        values = rows[name]
        if pd.api.types.is_numeric_dtype(values):
            bins = np.where(values.isna(), -1, np.minimum(values.fillna(0).to_numpy() // 10, 9))
            medians = {
                v: float(values[y == v].median()) for v in outcomes if (y == v).sum() >= MIN_ROWS
            }
            where = {
                v: round(m) for v, m in medians.items() if np.isfinite(m)
            }  # all blank: no value
            columns[name] = {
                "information": _information(bins, y),
                "typical_percentile_by_outcome": where,
            }
        else:
            cells = values.astype(str).where(values.notna(), "missing").to_numpy()
            shares = {level: {v: round(float((y[cells == level] == v).mean()), 2) for v in outcomes}
                      for level in sorted(set(cells)) if (cells == level).sum() >= MIN_ROWS}  # fmt: skip
            columns[name] = {
                "information": _information(cells, y),
                "outcome_shares_by_value": shares,
            }
    for entry in columns.values():
        entry["information"] = round(entry["information"], 3)
    return {
        "outcome_shares": {v: round(float((y == v).mean()), 2) for v in outcomes},
        "columns": dict(sorted(columns.items(), key=lambda kv: -kv[1]["information"])),
    }


def brief(schema: WorldSchema) -> dict:
    """What the writer reads for one schema: the schema's own words, the study, each column's role,
    whether an unrecorded cause matters, and a voice drawn from the schema id."""
    found, role = study(schema), roles(schema)
    target = schema.nodes[schema.target]
    rng = np.random.default_rng(zlib.crc32(schema.schema_id.encode()))
    return {
        "schema_id": schema.schema_id,
        "voice": str(rng.choice(VOICES)),
        "domain": schema.description,
        "outcome": {
            "column": schema.target,
            "description": target.desc,
            "values": list(found["outcome_shares"]),
            "shares": found["outcome_shares"],
        },
        "columns": {
            name: {
                "description": schema.nodes[name].desc,
                **(
                    {"values": list(schema.nodes[name].levels)} if schema.nodes[name].levels else {}
                ),
                "role": role[name],
                **measured,
            }
            for name, measured in found["columns"].items()
        },
        "unrecorded_cause": any(schema.nodes[n].hidden for n in _ancestors(schema, schema.target)),
    }


def lint(written: dict, schema: WorldSchema, brief: dict) -> list[str]:
    """What is wrong with a schema's written world knowledge; empty when it may ship."""
    problems = []
    if not isinstance(written.get("description"), str) or not written["description"].strip():
        problems.append("description is required")
    elif re.search(STOCK_OPENING, written["description"], re.IGNORECASE):
        problems.append("stock opening: say what a record stands for in the domain's own way")
    expected = {*schema.feature_names(), schema.target}
    notes = written.get("columns", {})
    if set(notes) != expected:
        problems.append(f"columns must be exactly {sorted(expected)}")
    problems += [f"empty note for {c}" for c, note in notes.items() if not isinstance(note, str) or not note.strip()]  # fmt: skip
    text = json.dumps(written, ensure_ascii=False)
    if any(d in text for d in DASHES):
        problems.append("dash")
    # the schema's own column names may hold a banned word (a simulation_runs column is domain language)
    prose = re.sub("|".join(map(re.escape, sorted(expected, key=len, reverse=True))), "", text)
    problems += [f"banned wording: {w}" for w in BANNED if re.search(w, prose, re.IGNORECASE)]
    hidden = [n for n, node in schema.nodes.items() if node.hidden]
    low = text.lower()
    problems += [f"names the unrecorded cause {n}" for n in hidden if n.lower() in low or n.lower().replace("_", " ") in low]  # fmt: skip
    known = set(re.findall(r"\d+", json.dumps(brief)))
    problems += [
        f"number not in the brief: {n}" for n in sorted(set(re.findall(r"\d+", text)) - known)
    ]
    return problems
