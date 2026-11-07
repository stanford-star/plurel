from dataclasses import replace

from plurel import LOW, MEDIUM, STRONG, WEAK, Categorical, Noise, Numeric, Scaled, Shares, Table
from plurel.world.authoring.dedupe import THRESHOLD, candidates, report
from plurel.world.authoring.realism import census, check, hygiene, phenomena, summary, validation

CASE = Table(
    {
        "age": Numeric(Scaled("duration", 0, 90, shape="right"), doc="the patient's age"),
        "wait_days": Numeric(
            Scaled("duration", 0, 60), causes={"age": WEAK},
            noise=Noise("gaussian", LOW), doc="days waited",
        ),
        "kind": Categorical({"a": 0.6, "b": 0.4}, doc="the kind"),
        "outcome": Categorical(
            {"declined": 0.7, "accepted": 0.3},
            causes={"wait_days": STRONG, "kind": {"a": -0.5, "b": 0.5}},
            noise=Noise("gaussian", MEDIUM), doc="how it ended",
        ),
    },
    name="clinic_referral_outcome",
    target="outcome",
    doc="Each row is one referral a clinic received; outcome records how it ended.",
    sources="About 30 percent of referrals are accepted and about 1 in 10 are returned, by a regulator's annual report.",
)  # fmt: skip


def like(table: Table = CASE, **fields) -> Table:
    return Table(fields.pop("entries", table.declared), **{**table.metadata, **fields})


def with_entries(**entries) -> Table:
    return like(entries={**CASE.declared, **entries})


def with_shares(shares: dict) -> Table:
    return with_entries(outcome=replace(CASE.declared["outcome"], levels=shares))


def test_a_well_made_schema_passes_the_gate():
    assert check(CASE, realize=True) == []


def test_validation_reads_the_target():
    assert "target shares are uniform" in validation(with_shares({"declined": 0.5, "accepted": 0.5}), realize=False)  # fmt: skip
    rare = with_shares({"declined": 0.999, "accepted": 0.001})
    assert any("cannot be realized" in r for r in validation(rare, realize=False))
    many = with_shares({f"c{i}": 0.2 if i == 0 else 0.08 for i in range(11)})
    assert any("2 to 10" in r for r in validation(many, realize=False))


def test_validation_asks_wide_roots_for_their_shares():
    wide = Categorical(Shares(["a", "b", "c", "d"]), doc="a wide root")
    assert any("uniform by omission" in r for r in validation(with_entries(wide=wide), realize=False))  # fmt: skip


def test_validation_realizes_rows_to_find_constant_columns():
    flat = Numeric(Scaled("score", 0, 10, shape="symmetric"), clip=(9.0, 9.5), doc="flat", round=9.0)  # fmt: skip
    schema = with_entries(flat=flat)
    assert validation(schema, realize=False) == []
    assert any("constant feature columns" in r for r in validation(schema, realize=True))


def test_hygiene_reads_the_authoring_rules():
    assert hygiene(CASE) == []
    assert any("Each row is" in r for r in hygiene(like(doc="One referral.")))
    ids = Numeric(Scaled("count", 1, 999, shape="counts"), doc="a code")
    assert any("identifiers rendered" in r for r in hygiene(with_entries(site_code=ids)))
    assert any("quantitative facts" in r for r in hygiene(like(sources="A report.")))
    no_source = "About 30 percent are accepted and about 1 in 10 are returned."
    assert any("no statistical source" in r for r in hygiene(like(sources=no_source)))


def test_hygiene_flags_a_binned_target_and_a_copied_table():
    entries = {k: v for k, v in CASE.declared.items() if k != "outcome"} | {
        "wait_band": CASE.declared["outcome"]
    }
    assert any("cut into bins" in r for r in hygiene(like(entries=entries, target="wait_band")))
    copied = like(sources=CASE.sources + " The public use file shows 43.3 percent of clinics.")
    reasons = hygiene(copied)
    assert any("to the decimal" in r for r in reasons) and any(
        "public dataset" in r for r in reasons
    )
    assert any("finer than" in r for r in hygiene(with_shares({"declined": 0.6837, "accepted": 0.3163})))  # fmt: skip


def test_phenomena_are_read_off_the_mechanism():
    assert phenomena(CASE) == set()
    schema = with_entries(
        age=replace(CASE.declared["age"], round=1.0, missing=0.1, missing_by=("age", "high")),
        wait_days=replace(CASE.declared["wait_days"], causes={"age": WEAK, "kind": {"a": 0.1, "b": -0.1}}, how="max"),
    )  # fmt: skip
    assert phenomena(schema) == {"mar_or_mnar", "censor_or_rounding", "non_additive_mechanism"}


def test_the_census_tabulates_a_catalog():
    table = census([CASE, like(name="other", doc="One referral.")])
    assert len(table) == 2 and set(table.schema_id) == {"clinic_referral_outcome", "other"}
    row = table.set_index("schema_id").loc["clinic_referral_outcome"]
    assert (row.n_features, row.n_numeric, row.target_kind, row.n_classes) == (3, 2, "nominal", 2)
    assert row.entity == "referral a clinic received" and row.invalid == 0 and row.hygiene == 0
    text = summary(table)
    assert text.startswith("schemas 2") and "phenomenon interaction" in text


def test_dedupe_finds_a_schema_authored_twice():
    entries = {k: v for k, v in CASE.declared.items() if k != "outcome"} | {
        "grade": replace(CASE.declared["outcome"], levels={"poor": 0.5, "rich": 0.5})
    }
    other = like(entries=entries, name="garden_soil_sample_grade", target="grade", doc="Each row is one soil sample.")  # fmt: skip
    twin = like(name="clinic_referral_final_outcome")
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
