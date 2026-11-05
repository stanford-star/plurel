"""Repair cafes: item visits. Holds repair_cafe_item_visit_outcome."""

from plurel.world.schema import Edge, Node, WorldSchema

REPAIR_CAFE_ITEM_VISIT = WorldSchema(
    schema_id="repair_cafe_item_visit_outcome",
    target="outcome",
    description=(
        "Each row is one household item brought to a volunteer repair cafe; outcome records how "
        "the visit ended."
    ),
    rationale=(
        "The outcome's shares follow the 2022 RepairMonitor report of the Repair Cafe "
        "International Foundation: about 61 percent of repairs succeeded, about 14 percent ended "
        "with advice and about 25 percent failed. The columns and their dependencies are "
        "illustrative: older items and worse faults take longer and fail more often, and a thin "
        "volunteer turnout lengthens both the wait and the repair."
    ),
    nodes={
        # a cause the record does not hold: simulated, never rendered
        "volunteer_cover": Node(
            kind="numeric", hidden=True, desc="how many skilled volunteers were on hand that session",
        ),
        "item_age_years": Node(
            kind="numeric", units="duration", dist="right", scale=(0, 40), rounding=1,
            desc="age of the item in years",
        ),
        "item_type": Node(
            kind="categorical", levels=["small_appliance", "electronics", "textile", "furniture"],
            level_probs={"small_appliance": 0.4, "electronics": 0.3, "textile": 0.2, "furniture": 0.1},
            desc="what kind of item it is",
        ),
        "fault_severity": Node(
            parents={"item_age_years": +0.5},
            kind="categorical", ordinal=True, levels=["minor", "moderate", "major"],
            level_probs={"minor": 0.5, "moderate": 0.35, "major": 0.15},
            noise_scale="medium", noise_dist="gaussian",
            desc="the fault as assessed at the intake desk",
        ),
        "wait_minutes": Node(
            parents={"volunteer_cover": -1},
            kind="numeric", units="duration", scale=(0, 120), rounding=5,
            noise_scale="low", noise_dist="skewed",
            desc="minutes between arrival and a volunteer starting on the item",
        ),
        # a nominal parent enters by a per-level offset; the record leaves some rows blank
        "repair_minutes": Node(
            parents={
                "volunteer_cover": -0.5,
                "fault_severity": +1,
                "item_type": {"small_appliance": 0.0, "electronics": 0.6, "textile": -0.5, "furniture": 0.3},
            },
            kind="numeric", units="duration", scale=(5, 180), missing=0.08,
            noise_scale="medium", noise_dist="laplace",
            desc="minutes a volunteer spent on the item",
        ),
        # a derived column: exact arithmetic on the rendered values
        "total_minutes": Node(
            kind="numeric",
            formula=lambda wait_minutes, repair_minutes: wait_minutes + repair_minutes,
            desc="minutes from arrival to the end of the repair",
        ),
        "outcome": Node(
            parents={
                "fault_severity": Edge(effect=+1),
                "item_age_years": Edge(effect=+0.5, shape="hinge"),
                "repair_minutes": Edge(effect=+0.5, shape="saturating"),
            },
            interactions=[("item_age_years", "repair_minutes")],
            kind="categorical", levels=["repaired", "advice_given", "repair_failed"],
            level_probs={"repaired": 0.61, "advice_given": 0.14, "repair_failed": 0.25},
            noise_scale="medium", noise_dist="gaussian",
            desc="how the visit ended",
        ),
    },
)  # fmt: skip
