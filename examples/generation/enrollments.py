"""Student enrollments: retained students and courses, capped enrollment counts, courses matched
to a level, one enrollment per student, term and course, and grades read from both sides.

The mechanisms are authored, not fit: the dataset shows how the declarations compose, not how
students choose or score. Run it with `python examples/generation/enrollments.py`; `--scale` grows
the students.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from plurel import (
    Categorical,
    Counts,
    ForeignKey,
    Numeric,
    Range,
    Recorded,
    Schema,
    Table,
    create_database,
    read_database,
    write_database,
)

LEVELS = ("intro", "intermediate", "advanced")


def fixtures(scale: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small synthetic stand-ins for retained student and course records."""
    rng = np.random.default_rng(0)
    n = 150 * scale
    students = pd.DataFrame(
        {
            "student_id": [f"S{i:05d}" for i in range(n)],
            "year": rng.integers(1, 5, n).astype(float),
            "gpa": np.round(rng.uniform(2.0, 4.0, n), 2),
        }
    )
    courses = pd.DataFrame(
        {
            "course_id": [f"C{i:03d}" for i in range(30)],
            "level": pd.Categorical(np.repeat(LEVELS, 10), categories=LEVELS),
            "difficulty": np.round(rng.uniform(1, 5, 30), 1),
        }
    )
    return students, courses


def build(students: pd.DataFrame, courses: pd.DataFrame) -> Schema:
    student_table = Table.from_frame(
        students,
        primary_key="student_id",
        docs={"year": "year of study", "gpa": "grade point average, 0 to 4"},
    )
    course_table = Table.from_frame(
        courses, primary_key="course_id", docs={"difficulty": "rated 1 to 5"}
    )
    enrollments = Table(
        {
            # one to six enrollments per student, more in later years (a latent effect)
            "student_id": ForeignKey(
                "students", counts=Counts(3.0, causes={"year": 0.4}, least=1, most=6)
            ),
            "term": Categorical(("fall", "spring")),
            # later years take higher levels: a weight on the year's latent, cut at the shares
            "level": Categorical(
                {"intro": 0.4, "intermediate": 0.35, "advanced": 0.25},
                causes={"student_id.year": 1.5},
            ),
            # a course of that level, never twice for one student in one term
            "course_id": ForeignKey(
                "courses", match={"level": "level"}, unique=("student_id", "term")
            ),
            # in grade points: recorded GPA and difficulty in their units, plus noise of 6 points;
            # realizing draws how much difficulty costs
            "grade": Numeric(
                causes={
                    Recorded("student_id.gpa"): 20.0,
                    Recorded("course_id.difficulty"): Range(-6.0, -2.0),
                },
                noise=6.0,
                clip=(0.0, 100.0),
                round=1,
                doc="final grade out of 100",
            ),
        },
        name="enrollments",
        doc="one enrollment per row, a static snapshot",
        target="grade",
    )
    return Schema({"students": student_table, "courses": course_table, "enrollments": enrollments})


def check(frames: dict[str, pd.DataFrame], students: pd.DataFrame, courses: pd.DataFrame) -> None:
    """The invariants the declarations promise."""
    taken = frames["enrollments"]
    assert frames["students"].equals(students) and frames["courses"].equals(courses)
    per = taken.student_id.value_counts().reindex(students.student_id, fill_value=0)
    assert per.between(1, 6).all(), "every student takes 1 to 6 enrollments"
    assert not taken.duplicated(["student_id", "term", "course_id"]).any(), "unique triples"
    matched = taken.merge(courses, on="course_id", suffixes=("", "_course"))
    assert (
        len(matched) == len(taken)
        and (matched.level.astype(str) == matched.level_course.astype(str)).all()
    )
    assert taken.grade.between(0, 100).all()
    graded = taken.merge(students, on="student_id")
    assert graded.grade.corr(graded.gpa) > 0, "better students score higher"


def main(mechanism_seed: int, data_seed: int, scale: int, out: Path | None) -> None:
    students, courses = fixtures(scale)
    generator = build(students, courses).realize(mechanism_seed)
    print(generator.explain())
    frames = generator.sample(seed=data_seed)
    check(frames, students, courses)
    db = create_database(generator, frames)
    with tempfile.TemporaryDirectory() as scratch:
        path = write_database(
            db, (out or Path(scratch)) / "enrollments", name="enrollments", overwrite=True
        )
        loaded = read_database(path)
    for name, table in db.table_dict.items():
        assert len(loaded.table_dict[name].df) == len(table.df)
        assert loaded.table_dict[name].fkey_col_to_pkey_table == table.fkey_col_to_pkey_table
    sizes = {name: len(frame) for name, frame in frames.items()}
    print(f"enrollments: {sizes}; every invariant holds and the database round-trips")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mechanism-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--out", type=Path, help="keep the saved database here")
    args = parser.parse_args()
    main(args.mechanism_seed, args.data_seed, args.scale, args.out)
