"""Repair cafes: item visits. Holds repair_cafe_item_visit_outcome."""

from plurel import (
    LOW,
    MEDIUM,
    STRONG,
    WEAK,
    Categorical,
    Derived,
    Hidden,
    Noise,
    Numeric,
    Range,
    Scaled,
    Table,
)

REPAIR_CAFE_ITEM_VISIT = Table(
    {
        # a cause the record does not hold: simulated, never recorded
        "volunteer_cover": Hidden(doc="how many skilled volunteers were on hand that session"),
        "item_age_years": Numeric(
            Scaled("duration", 0, 40, shape="right"), round=1, doc="age of the item in years"
        ),
        "item_type": Categorical(
            {"small_appliance": 0.4, "electronics": 0.3, "textile": 0.2, "furniture": 0.1},
            doc="what kind of item it is",
        ),
        "fault_severity": Categorical(
            {"minor": 0.5, "moderate": 0.35, "major": 0.15},
            ordered=True,
            causes={"item_age_years": WEAK},
            noise=Noise("gaussian", MEDIUM),
            doc="the fault as assessed at the intake desk",
        ),
        "wait_minutes": Numeric(
            Scaled("duration", 0, 120),
            round=5,
            causes={"volunteer_cover": -STRONG},
            noise=Noise("skewed", LOW),
            doc="minutes between arrival and a volunteer starting on the item",
        ),
        # an unordered cause moves a column level by level; the record leaves some rows blank
        "repair_minutes": Numeric(
            Scaled("duration", 5, 180),
            missing=0.08,
            causes={
                "volunteer_cover": -WEAK,
                "fault_severity": STRONG,
                "item_type": (
                    Range(0.7, 1.6),
                    {"small_appliance": 0.0, "electronics": 0.6, "textile": -0.5, "furniture": 0.3},
                ),
            },
            noise=Noise("laplace", MEDIUM),
            doc="minutes a volunteer spent on the item",
        ),
        # a derived column: exact arithmetic on the recorded values
        "total_minutes": Derived(
            lambda wait_minutes, repair_minutes: wait_minutes + repair_minutes,
            doc="minutes from arrival to the end of the repair",
        ),
        "outcome": Categorical(
            {"repaired": 0.61, "advice_given": 0.14, "repair_failed": 0.25},
            causes={
                "fault_severity": STRONG,
                "item_age_years": (WEAK, "relu"),
                "repair_minutes": (WEAK, "tanh"),
                ("item_age_years", "repair_minutes"): Range(0.8, 2.5),
            },
            noise=Noise("gaussian", MEDIUM),
            doc="how the visit ended",
        ),
    },
    name="repair_cafe_item_visit_outcome",
    target="outcome",
    doc=(
        "Each row is one household item brought to a volunteer repair cafe; outcome records how "
        "the visit ended."
    ),
    sources=(
        "The outcome's shares follow the 2022 RepairMonitor report of the Repair Cafe "
        "International Foundation: about 61 percent of repairs succeeded, about 14 percent ended "
        "with advice and about 25 percent failed. The columns and their dependencies are "
        "illustrative: older items and worse faults take longer and fail more often, and a thin "
        "volunteer turnout lengthens both the wait and the repair."
    ),
)
