"""Realism rules for world schemas and the census that reports them over a catalog. ``validation`` returns the
defects that make a schema unusable, ``hygiene`` the authoring findings, ``check`` both (the gate on a new file);
``phenomena`` names the record phenomena it carries; ``census`` tabulates every schema."""

from __future__ import annotations

import re

import pandas as pd

from plurel import Categorical, Derived, Hidden, Numeric, Table
from plurel.world.realize import realize_world
from plurel.world.record import features, level_names, pairs, shares

ID_NAME = re.compile(
    r"(_id|_number|_num|_code|_serial|_ticket|_locator|_tag|_key|_ref|_barcode)$|^(id|ref)_"
)
QUANT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|percent|per cent|per (?:thousand|million|hundred|\d[\d,]*)"
    r"|in (?:a|one|ten|every) (?:hundred|thousand|million)|in \d+|out of \d+)"
)
SOURCE_WORD = re.compile(
    r"\b(?:statistics?|reports?|surveys?|registry|registers?|census|yearbook|bulletin|annual|published|stud(?:y|ies)"
    r"|standards?|guidelines?|indicators?|agency|office|ministry|department|regulator|association|federation)\b",
    re.I,
)
SOURCE_NAME = re.compile(
    r"\b(?:NIBRS|NCVS|Eurostat|OECD|WHO|UNODC|BJS|ONS|NHTSA|NFPA|USFA|IRS|SSA|DWP|FCA|Federal Reserve|UK Finance|ABI"
    r"|Insurance Europe|ILO|UNECE|World Bank|IRTAD|HMCTS|Council of Europe|Data Book|Government at a Glance"
    r"|Society at a Glance|Education at a Glance|LFS|SILC|ATUS|HETUS|NCES|NHS|CDC|BLS|FBI|NHTS|UNICEF|FAO|IEA|EEA|EPA"
    r"|NOAA|NASA|USDA|CMS|HESA|DfE|Ofsted|Ofcom|IAB|MRC|UEFA|FIFA|IOC|ISO|IEC|ASTM|IPCC|IUCN|EMA|FDA|ECB|BIS|IMF|EU|UN)\b"
)
BAND_TARGET = re.compile(r"_(?:band|quintile|decile|tercile|bucket)$")
PRECISE_PERCENT = re.compile(r"(\d+)\.(\d+) ?(?:%|percent)")
HEDGES = re.compile(r"\b(?:about|roughly|around|near|nearly|under|over|some)\b[^.]{0,24}$")
PUBLIC_DATA = re.compile(
    r"\b(?:public dataset|microdata|public[ -]use (?:file|sample)|PUF|kaggle|openml|uci (?:machine learning )?repository|recode file|supplement file|open data)\b",
    re.I,
)
SHARE_QUANTUM = (
    0.005  # target shares finer than half a percent point, two or more, are a copied table row
)
SHARE_PRECISION_FLOOR = 0.05
PRECISE_INT = 10  # ten percent or more: a decimal wants a hedge word and two decimals never pass; one to ten: two decimals want a hedge
PHENOMENA = (
    "mar_or_mnar",
    "censor_or_rounding",
    "zero_inflation",
    "hidden_cause",
    "non_additive_mechanism",
    "interaction",
)  # structural, read off the Table
ROWS = 512  # rows of a realization a class must reach
MIN_CLASS_SHARE = 0.002  # rarer than 1 in 500 cannot be realized in a pool of ROWS rows
MIN_CLASSES = 2
MAX_CLASSES = 10  # a target with more is usually a code list, not what a process decides
MIN_FACTS = 2
SHARE_CONCENTRATION = 100.0  # realization at the authored shares
CONSTANT_CHECK_ROWS = 400
CLASS_CHECK_ROWS = 2000


def sourced(text: str) -> bool:
    return bool(SOURCE_WORD.search(text) or SOURCE_NAME.search(text))


def target_kind(entry: Categorical) -> str:
    return "argmax" if entry.scores else ("graded" if entry.ordered else "nominal")


def phenomena(schema: Table) -> set[str]:
    """The record phenomena the schema carries, by name (see PHENOMENA): each is a mechanism of the Table."""
    entries = schema.declared.values()
    numbers = [e for e in entries if isinstance(e, Numeric)]
    carried = {
        "mar_or_mnar": any(getattr(e, "missing_by", None) is not None for e in entries),
        "censor_or_rounding": any(n.clip is not None or n.round is not None for n in numbers),
        "zero_inflation": any(n.zeros for n in numbers),
        "hidden_cause": any(isinstance(e, Hidden) for e in entries),
        "non_additive_mechanism": any(getattr(e, "how", "sum") != "sum" for e in entries),
        "interaction": any(pairs(e) for e in entries),
    }
    return {name for name in PHENOMENA if carried[name]}


def validation(schema: Table, *, realize: bool) -> list[str]:
    """Defects that make a schema unusable as training data; empty when it passes. ``realize`` also draws
    rows to catch absent classes and constant columns. Read off the mechanism, never off names."""
    t, feats = schema.declared[schema.target], features(schema)
    reasons: list[str] = []
    classes = len(level_names(t))
    if not MIN_CLASSES <= classes <= MAX_CLASSES:
        reasons.append(
            f"target has {classes} classes; {MIN_CLASSES} to {MAX_CLASSES}: merge the rarest into their real parent category or predict the decision the record holds"
        )
    probs = shares(t) or {}
    if probs:
        values = list(probs.values())
        if max(values) - min(values) < 1e-9:
            reasons.append("target shares are uniform")
        if min(values) < MIN_CLASS_SHARE:
            reasons.append(
                f"a class share under {MIN_CLASS_SHARE:.1%} cannot be realized in a {ROWS}-row pool: {probs}; condition the rows so the event is at least that common"
            )
    thin = [
        k
        for k in feats
        if isinstance(n := schema.declared[k], Categorical)
        and not n.causes
        and not n.scores
        and shares(n) is None
        and len(level_names(n)) >= 4
    ]
    if thin:
        reasons.append(
            f"categorical roots with 4 or more levels and no level_probs (uniform by omission): {thin}"
        )
    if realize:
        try:
            population = realize_world(schema, seed=0, share_concentration=SHARE_CONCENTRATION)
            frame = population.rows(n=CONSTANT_CHECK_ROWS, draw=0)
            seen = set(population.rows(n=CLASS_CHECK_ROWS, draw=1)[schema.target].astype(str))
            absent = [c for c in level_names(t) if str(c) not in seen]
            if absent:
                reasons.append(f"classes absent from {CLASS_CHECK_ROWS} rows: {absent}")
            flat = [c for c in feats if frame[c].nunique(dropna=True) < 2]
            if flat:
                reasons.append(f"constant feature columns: {flat}")
        except Exception as e:  # noqa: BLE001
            reasons.append(f"realize failed: {type(e).__name__}: {str(e)[:120]}")
    return reasons


def hygiene(schema: Table) -> list[str]:
    """Authoring rules for a new file: the record's framing, the copy guards and the provenance of its
    numbers. The gate enforces them on new work; over the catalog they are reported, not blocking."""
    feats = features(schema)
    reasons: list[str] = []
    if not schema.doc.startswith("Each row is"):
        reasons.append("description must open with 'Each row is'")
    ids = [k for k in feats if _numeric(schema, k) and ID_NAME.search(k)]
    if ids:
        reasons.append(f"identifiers rendered as numeric features: {ids}")
    if BAND_TARGET.search(schema.target):
        reasons.append(
            f"target {schema.target!r} is a measurement cut into bins; predict the event or decision the record holds and keep the measurement as a column"
        )
    reasons.extend(copied_precision(schema))
    words = public_data_words(schema)
    if words:
        reasons.append(
            f"description or rationale names a public dataset rather than the organisation's own record: {words}"
        )
    rationale = schema.sources or ""
    facts = QUANT.findall(rationale)
    if len(facts) < MIN_FACTS:
        reasons.append(
            f"rationale anchors {len(facts)} quantitative facts, wanted {MIN_FACTS} or more (shares, rates, typical values with % or per)"
        )
    if not sourced(rationale):
        reasons.append(
            "rationale names no statistical source (a registry, survey, report, standard or agency)"
        )
    return reasons


def check(schema: Table, *, realize: bool) -> list[str]:
    """The gate on a new file: validation, then hygiene."""
    return validation(schema, realize=realize) + hygiene(schema)


def _numeric(schema: Table, name: str) -> bool:
    return isinstance(schema.declared[name], Numeric | Derived)


def placed(schema: Table) -> tuple[str, str, str]:
    return getattr(schema, "taxonomy", ("", "", ""))


ENTITY = re.compile(
    r"Each row is (?:one |a |an |the )?(?:single )?([A-Za-z0-9][A-Za-z0-9' -]*?)(?:[,;.:(]| that| who| which| whose| at | in | on | for | from | with | over | during )"
)


def census(schemas) -> pd.DataFrame:
    """One row per schema: taxonomy, target kind, widths, shares, the phenomena it carries, its static validation
    defects and its hygiene findings."""
    rows = []
    for s in schemas:
        t, feats = s.declared[s.target], features(s)
        numeric = [k for k in feats if _numeric(s, k)]
        m = ENTITY.search(s.doc)
        probs = shares(t) or {}
        rationale = s.sources or ""
        rows.append(dict(
            schema_id=s.name, sector=placed(s)[0], division=placed(s)[1],
            entity=m.group(1).strip() if m else "", n_features=len(feats), n_numeric=len(numeric),
            target_kind=target_kind(t),
            n_classes=len(level_names(t)), majority=max((probs or {0: float("nan")}).values()),
            round_shares=bool(probs) and all(abs(p * 20 - round(p * 20)) < 1e-9 for p in probs.values()),
            identifiers=sum(bool(ID_NAME.search(k)) for k in numeric),
            facts=len(QUANT.findall(rationale)), sourced=sourced(rationale),
            phenomena="|".join(sorted(phenomena(s))), invalid=len(validation(s, realize=False)), hygiene=len(hygiene(s)),
        ))  # fmt: skip
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame) -> str:
    """The census in a few lines: shares of schemas with each defect, valid, clean, and each phenomenon."""
    n = len(table)
    lines = [f"schemas {n}", f"round target shares {table.round_shares.mean():.2f}", f"sourced rationale {table.sourced.mean():.2f}",
             f"rationales with 2+ facts {(table.facts >= MIN_FACTS).mean():.2f}", f"schemas with identifier features {(table.identifiers > 0).mean():.2f}",
f"schemas valid on the static rules {(table.invalid == 0).mean():.2f}", f"schemas clean on hygiene {(table.hygiene == 0).mean():.2f}"]  # fmt: skip
    for name in PHENOMENA:
        lines.append(f"phenomenon {name} {table.phenomena.str.contains(name).mean():.2f}")
    return "\n".join(lines)


def copied_precision(schema: Table) -> list[str]:
    """Signs that a published table was copied to the decimal: two or more target shares above the floor finer
    than half a percent point, a percentage with two decimals, or one of ten or more quoted to a decimal without a
    hedge word before it."""
    out: list[str] = []
    probs = shares(schema.declared[schema.target]) or {}
    rationale = schema.sources or ""
    precise = [
        p
        for p in probs.values()
        if p >= SHARE_PRECISION_FLOOR and abs(p / SHARE_QUANTUM - round(p / SHARE_QUANTUM)) > 1e-6
    ]
    if len(precise) >= 2:
        out.append(f"target shares {precise} are finer than {SHARE_QUANTUM:.3f}")
    for m in PRECISE_PERCENT.finditer(rationale):
        hedged = HEDGES.search(rationale[max(0, m.start() - 40) : m.start()])
        whole, decimals = int(m.group(1)), len(m.group(2))
        if (
            decimals >= 3
            or (whole >= PRECISE_INT and (decimals >= 2 or not hedged))
            or (whole >= 1 and decimals >= 2 and not hedged)
        ):
            out.append(f"rationale quotes {m.group(0)!r} to the decimal without 'about'")
    return out


def public_data_words(schema: Table) -> list[str]:
    """Words that say the record is a public dataset rather than an organisation's own file."""
    return sorted(
        {
            m.group(0).lower()
            for text in (schema.doc or "", schema.sources or "")
            for m in PUBLIC_DATA.finditer(text)
        }
    )
