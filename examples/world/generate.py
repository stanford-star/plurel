"""One authored table, end to end: check the schema, realize it under a seed, draw its rows, and read the
world knowledge written for it.

    python examples/world/generate.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))  # the example's catalog is a package beside this script

from catalog import SCHEMAS  # noqa: E402

from plurel.world import brief, lint, load_knowledge, realize_world, roles  # noqa: E402
from plurel.world.authoring.realism import check  # noqa: E402


def main() -> None:
    schema = next(s for s in SCHEMAS if s.schema_id == "repair_cafe_item_visit_outcome")
    print(
        f"{schema.schema_id} in {'/'.join(schema.taxonomy)}: gate {check(schema, realize=True) or 'OK'}"
    )

    # the seed draws the magnitudes, the draw the rows; both repeat anywhere
    population = realize_world(schema, seed=0)
    rows = population.rows(n=2000, draw=0)
    print(rows.head().to_string())
    print(rows["outcome"].value_counts(normalize=True).round(2).to_string())

    # an intervention sets a column's latent value before the draw
    older = population.rows(n=2000, draw=0, interventions={"item_age_years": 2.0})
    print(f"repair_failed {(rows['outcome'] == 'repair_failed').mean():.2f} as drawn, "
          f"{(older['outcome'] == 'repair_failed').mean():.2f} with every item old")  # fmt: skip

    print(roles(schema))
    written = load_knowledge(HERE / "world_knowledge")[schema.schema_id]
    print(f"world knowledge lint: {lint(written, schema, brief(schema)) or 'OK'}")
    print(written["description"])


if __name__ == "__main__":
    main()
