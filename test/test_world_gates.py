from dataclasses import replace

from plurel.world import Edge, Node, WorldSchema
from plurel.world.authoring.dedupe import THRESHOLD, candidates, report
from plurel.world.authoring.realism import census, check, hygiene, phenomena, summary, validation

CASE = WorldSchema(
    schema_id="clinic_referral_outcome",
    target="outcome",
    description="Each row is one referral a clinic received; outcome records how it ended.",
    rationale="About 30 percent of referrals are accepted and about 1 in 10 are returned, by a regulator's annual report.",
    nodes={
        "age": Node(kind="numeric", units="duration", dist="right", scale=(0, 90), desc="the patient's age"),
        "wait_days": Node(
            kind="numeric", units="duration", scale=(0, 60), noise_scale="low", noise_dist="gaussian",
            parents={"age": Edge(effect=+0.5)}, desc="days waited",
        ),
        "kind": Node(kind="categorical", levels=["a", "b"], level_probs={"a": 0.6, "b": 0.4}, desc="the kind"),
        "outcome": Node(
            kind="categorical", levels=["declined", "accepted"], level_probs={"declined": 0.7, "accepted": 0.3},
            parents={"wait_days": Edge(effect=+1), "kind": {"a": -0.5, "b": 0.5}},
            noise_scale="medium", noise_dist="gaussian", desc="how it ended",
        ),
    },
)  # fmt: skip


def with_nodes(**nodes) -> WorldSchema:
    return replace(CASE, nodes={**CASE.nodes, **nodes})


def test_a_well_made_schema_passes_the_gate():
    assert check(CASE, realize=True) == []


def test_validation_reads_the_target():
    uniform = with_nodes(outcome=replace(CASE.nodes["outcome"], level_probs={"declined": 0.5, "accepted": 0.5}))  # fmt: skip
    assert "target shares are uniform" in validation(uniform, realize=False)
    rare = with_nodes(outcome=replace(CASE.nodes["outcome"], level_probs={"declined": 0.999, "accepted": 0.001}))  # fmt: skip
    assert any("cannot be realized" in r for r in validation(rare, realize=False))
    eleven = [f"c{i}" for i in range(11)]
    many = with_nodes(outcome=replace(CASE.nodes["outcome"], levels=eleven, level_probs={c: 0.2 if i == 0 else 0.08 for i, c in enumerate(eleven)}))  # fmt: skip
    assert any("2 to 10" in r for r in validation(many, realize=False))


def test_validation_asks_wide_roots_for_their_shares():
    wide = Node(kind="categorical", levels=["a", "b", "c", "d"], desc="a wide root")
    assert any("uniform by omission" in r for r in validation(with_nodes(wide=wide), realize=False))


def test_validation_realizes_rows_to_find_constant_columns():
    flat = Node(kind="numeric", units="score", dist="symmetric", scale=(0, 10), censor=(9.99, None), rounding=5.0, desc="flat")  # fmt: skip
    schema = with_nodes(flat=flat)
    assert validation(schema, realize=False) == []
    assert any("constant feature columns" in r for r in validation(schema, realize=True))


def test_hygiene_reads_the_authoring_rules():
    assert hygiene(CASE) == []
    assert any("Each row is" in r for r in hygiene(replace(CASE, description="One referral.")))
    ids = Node(kind="numeric", units="count", dist="counts", scale=(1, 999), desc="a code")
    assert any("identifiers rendered" in r for r in hygiene(with_nodes(site_code=ids)))
    assert any("quantitative facts" in r for r in hygiene(replace(CASE, rationale="A report.")))
    no_source = "About 30 percent are accepted and about 1 in 10 are returned."
    assert any("no statistical source" in r for r in hygiene(replace(CASE, rationale=no_source)))


def test_hygiene_flags_a_binned_target_and_a_copied_table():
    nodes = {k: v for k, v in CASE.nodes.items() if k != "outcome"} | {
        "wait_band": CASE.nodes["outcome"]
    }
    banded = replace(CASE, target="wait_band", nodes=nodes)
    assert any("cut into bins" in r for r in hygiene(banded))
    copied = replace(
        CASE, rationale=CASE.rationale + " The public use file shows 43.3 percent of clinics."
    )
    reasons = hygiene(copied)
    assert any("to the decimal" in r for r in reasons) and any(
        "public dataset" in r for r in reasons
    )
    precise = with_nodes(outcome=replace(CASE.nodes["outcome"], level_probs={"declined": 0.6837, "accepted": 0.3163}))  # fmt: skip
    assert any("finer than" in r for r in hygiene(precise))


def test_phenomena_are_read_off_the_mechanism():
    assert phenomena(CASE) == set()
    schema = with_nodes(
        age=replace(CASE.nodes["age"], rounding=1.0, missing=0.1, missing_by=("age", "high")),
        wait_days=replace(CASE.nodes["wait_days"], parents={"age": +0.5, "kind": {"a": 0.1, "b": -0.1}}, mechanism="max"),
    )  # fmt: skip
    assert phenomena(schema) == {"mar_or_mnar", "censor_or_rounding", "non_additive_mechanism"}


def test_the_census_tabulates_a_catalog():
    table = census([CASE, replace(CASE, schema_id="other", description="One referral.")])
    assert len(table) == 2 and set(table.schema_id) == {"clinic_referral_outcome", "other"}
    row = table.set_index("schema_id").loc["clinic_referral_outcome"]
    assert (row.n_features, row.n_numeric, row.target_kind, row.n_classes) == (3, 2, "nominal", 2)
    assert row.entity == "referral a clinic received" and row.invalid == 0 and row.hygiene == 0
    text = summary(table)
    assert text.startswith("schemas 2") and "phenomenon interaction" in text


def test_dedupe_finds_a_schema_authored_twice():
    other = replace(CASE, schema_id="garden_soil_sample_grade", target="grade", description="Each row is one soil sample.",
                    nodes={k: v for k, v in CASE.nodes.items() if k != "outcome"} | {"grade": replace(CASE.nodes["outcome"], levels=["poor", "rich"], level_probs={"poor": 0.5, "rich": 0.5})})  # fmt: skip
    twin = replace(CASE, schema_id="clinic_referral_final_outcome")
    pairs = candidates([CASE, twin, other], threshold=THRESHOLD, among=None)
    assert [pair[:2] for pair in pairs] == [
        ("clinic_referral_outcome", "clinic_referral_final_outcome")
    ]
    assert (
        candidates([CASE, twin, other], threshold=THRESHOLD, among={"garden_soil_sample_grade"})
        == []
    )
    assert candidates([CASE, other], threshold=THRESHOLD, among=None) == []
    assert "clinic_referral_final_outcome" in report(pairs, [CASE, twin, other])
