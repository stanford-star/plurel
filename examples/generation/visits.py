"""Administrative clinic visits: patients matched to clinics in their region, visit counts that
follow the patient, and times that keep their order.

The mechanisms are illustrative and authored, not a clinically validated model of care. Run it with
`python examples/generation/visits.py`; `--scale` grows the patients.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from plurel import (
    Categorical,
    Counts,
    Exponential,
    ForeignKey,
    Numeric,
    PrimaryKey,
    Range,
    Recorded,
    Schema,
    Table,
    Timestamp,
    Uniform,
    create_database,
    read_database,
    split_timestamps,
    write_database,
)

DAY, HOUR = 86400, 3600  # a timestamp without a span counts in epoch seconds


def fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small synthetic stand-ins for a retained region lookup and clinic directory."""
    rng = np.random.default_rng(0)
    regions = pd.DataFrame({"region_id": ["R1", "R2", "R3"], "name": ["north", "south", "east"]})
    clinics = pd.DataFrame(
        {
            "clinic_id": [f"K{i:02d}" for i in range(12)],
            "region_id": np.repeat(regions.region_id.to_numpy(), 4),
            "beds": rng.integers(10, 120, 12).astype(float),
        }
    )
    return regions, clinics


def build(regions: pd.DataFrame, clinics: pd.DataFrame) -> Schema:
    region_table = Table.from_frame(regions, primary_key="region_id")
    clinic_table = Table.from_frame(
        clinics,
        primary_key="clinic_id",
        foreign_keys={"region_id": "regions"},
        docs={"beds": "staffed beds"},
    )
    patients = Table(
        {
            "patient_id": PrimaryKey(),
            "region_id": ForeignKey("regions"),
            "age": Numeric(Uniform(0, 95), round=1, doc="years"),
            "registered": Timestamp("2022-01-01", "2023-01-01"),
            # a clinic of the patient's own region, matched on the region key both point at,
            # larger ones preferred by their recorded beds
            "clinic_id": ForeignKey(
                "clinics", match={"region_id": "region_id"}, prefer={Recorded("beds"): 0.02}
            ),
        },
        name="patients",
        doc="one registered patient per row",
        docs={"region_id": "where the patient lives"},
    )
    visits = Table(
        {
            "visit_id": PrimaryKey(),
            # about two visits each, more for older patients (a latent effect), at most six;
            # realizing draws how strongly age matters
            "patient_id": ForeignKey(
                "patients", counts=Counts(2.0, causes={"age": Range(0.3, 0.8)}, most=6)
            ),
            # a positive delay after the recorded registration, so a visit never precedes it
            "start": Timestamp(
                causes={Recorded("patient_id.registered"): 1.0}, noise=Exponential(60 * DAY)
            ),
            "completed": Timestamp(causes={Recorded("start"): 1.0}, noise=Exponential(2 * HOUR)),
            "urgent": Categorical({"no": 0.85, "yes": 0.15}, causes={"patient_id.age": 0.5}),
        },
        name="visits",
        doc="one visit per row, an event table",
        time_column="start",
        target="urgent",
    )
    return Schema(
        {"regions": region_table, "clinics": clinic_table, "patients": patients, "visits": visits}
    )


def check(frames: dict[str, pd.DataFrame], regions: pd.DataFrame, clinics: pd.DataFrame) -> None:
    """The invariants the declarations promise."""
    patients, visits = frames["patients"], frames["visits"]
    assert frames["regions"].equals(regions) and frames["clinics"].equals(clinics)
    placed = patients.merge(clinics, on="clinic_id", suffixes=("", "_clinic"))
    assert len(placed) == len(patients) and (placed.region_id == placed.region_id_clinic).all()
    assert visits.patient_id.isin(patients.patient_id).all()
    assert visits.patient_id.value_counts().max() <= 6
    timed = visits.merge(patients, on="patient_id")
    assert (timed.start > timed.registered).all(), "a visit comes after registration"
    assert (visits.completed > visits.start).all(), "a visit completes after it starts"


def main(mechanism_seed: int, data_seed: int, scale: int, out: Path | None) -> None:
    regions, clinics = fixtures()
    generator = build(regions, clinics).realize(mechanism_seed)
    print(generator.explain())
    frames = generator.sample({"patients": 200 * scale}, seed=data_seed)
    check(frames, regions, clinics)
    db = create_database(generator, frames)
    val_timestamp, test_timestamp = split_timestamps(db)  # events need a temporal split to save
    with tempfile.TemporaryDirectory() as scratch:
        path = write_database(
            db,
            (out or Path(scratch)) / "visits",
            name="visits",
            val_timestamp=val_timestamp,
            test_timestamp=test_timestamp,
            overwrite=True,
        )
        loaded = read_database(path)
    for name, table in db.table_dict.items():
        assert len(loaded.table_dict[name].df) == len(table.df)
        assert loaded.table_dict[name].time_col == table.time_col
    sizes = {name: len(frame) for name, frame in frames.items()}
    print(f"visits: {sizes}; every invariant holds and the database round-trips")


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("the checks are assert statements, which python -O drops: run without -O")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mechanism-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--out", type=Path, help="keep the saved database here")
    args = parser.parse_args()
    main(args.mechanism_seed, args.data_seed, args.scale, args.out)
