import json
from dataclasses import replace

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
    Shares,
    Table,
)
from plurel.world import brief, lint, load_knowledge, roles, study
from plurel.world.record import features

CASE = Table(
    {
        "cause": Numeric(Scaled("score", 0, 10, shape="symmetric"), doc="a root cause"),
        "other": Numeric(Scaled("count", 0, 50, shape="counts"), doc="a second root"),
        "mediator": Numeric(
            Scaled("percent", 0, 100), causes={"cause": STRONG, "other": (-WEAK, "relu")},
            noise=Noise("gaussian", LOW), doc="a mediator",
        ),
        "aftermath": Numeric(
            Scaled("amount", 0, 1000), causes={"mediator": WEAK},
            noise=Noise("laplace", MEDIUM), doc="a consequence of the mediator",
        ),
        "ticket": Numeric(Scaled("count", 1000, 9999, shape="counts"), doc="a bookkeeping id"),
        "outcome": Categorical(
            {"no": 0.7, "yes": 0.3},
            causes={"mediator": STRONG, "cause": -WEAK, ("mediator", "cause"): Range(0.8, 2.5)},
            noise=Noise("gaussian", MEDIUM), doc="the outcome",
        ),
    },
    name="ledger_case",
    target="outcome",
    doc="Each row is one case.",
)  # fmt: skip
GOOD = {
    "description": "A case that escalated is a yes. Cases with a high mediator usually escalate.",
    "columns": {c: "a note" for c in [*features(CASE), "outcome"]},
}


def with_entries(**entries) -> Table:
    return Table({**CASE.declared, **entries}, **CASE.metadata)


def caused(name: str, **causes):
    entry = CASE.declared[name]
    return replace(entry, causes={**entry.causes, **causes})


def with_frailty() -> Table:
    return with_entries(
        frailty=Hidden(doc="unrecorded frailty"),
        mediator=caused("mediator", frailty=STRONG),
        outcome=caused("outcome", frailty=STRONG),
    )


def test_roles_read_each_columns_place_in_the_graph():
    assert roles(CASE) == {
        "cause": "a cause of the outcome",
        "other": "a cause of the outcome",
        "mediator": "a cause of the outcome",
        "aftermath": "shares a cause with the outcome",
        "ticket": "unrelated to the outcome",
    }
    total = Derived(lambda cause, other: cause + other, doc="their total")
    assert roles(with_entries(total=total))["total"] == "computed from cause, other"


def test_the_study_pools_the_population_most_informative_column_first():
    found = study(CASE)
    columns = found["columns"]
    assert list(columns)[0] == "mediator" and columns["ticket"]["information"] < 0.01
    where = columns["mediator"]["typical_percentile_by_outcome"]
    assert where["yes"] > where["no"]
    assert set(found["outcome_shares"]) == {"no", "yes"}


def test_the_study_reads_a_categorical_column_by_its_outcome_shares():
    site = Categorical(Shares(["north", "south"]), doc="the site")
    found = study(with_entries(site=site, outcome=caused("outcome", site={"north": -1.5, "south": 1.5})))  # fmt: skip
    shares = found["columns"]["site"]["outcome_shares_by_value"]
    assert shares["south"]["yes"] > shares["north"]["yes"]


def test_a_brief_holds_the_schemas_words_its_study_and_its_graph():
    found = brief(CASE)
    assert found["schema_id"] == "ledger_case" and found["voice"] in ("first person", "impersonal", "second person")  # fmt: skip
    assert found["outcome"]["description"] == "the outcome" and not found["unrecorded_cause"]
    assert found["columns"]["mediator"]["role"] == "a cause of the outcome"
    assert brief(CASE) == found  # the voice and the study are seeded
    assert brief(with_frailty())["unrecorded_cause"]


def test_the_lint_passes_knowledge_that_notes_every_column():
    assert lint(GOOD, CASE, brief(CASE)) == []


def test_the_lint_refuses_stock_openings():
    found = brief(CASE)
    for stock in ("Each row is one case.", "Every record here is a case.", "You are looking at cases."):  # fmt: skip
        assert lint({**GOOD, "description": stock}, CASE, found)[0].startswith("stock opening")


def test_the_lint_names_what_is_wrong():
    found = brief(CASE)
    bad = {"description": "In this synthetic data the mediator matters — in 4321 of cases.", "columns": {"mediator": "a note"}}  # fmt: skip
    problems = " | ".join(lint(bad, CASE, found))
    assert "columns must be exactly" in problems and "dash" in problems
    assert "number not in the brief: 4321" in problems and "banned wording" in problems
    assert lint({**GOOD, "columns": {**GOOD["columns"], "outcome": " "}}, CASE, found) == ["empty note for outcome"]  # fmt: skip
    assert lint({"columns": GOOD["columns"]}, CASE, found) == ["description is required"]


def test_the_lint_lets_a_column_name_hold_a_banned_word():
    runs = Numeric(Scaled("count", 0, 50, shape="counts"), doc="runs logged")
    schema = with_entries(simulation_runs=runs)
    found = brief(schema)
    written = {
        "description": "A yes is a case that escalated; simulation_runs carries little.",
        "columns": {c: "a note" for c in [*features(schema), "outcome"]},
    }
    assert lint(written, schema, found) == []
    leaky = {**written, "description": written["description"] + " These records are simulated."}
    assert lint(leaky, schema, found) == ["banned wording: \\bsimulat"]


def test_the_lint_refuses_naming_the_unrecorded_cause():
    schema = with_frailty()
    written = {**GOOD, "description": GOOD["description"] + " Frailty matters too."}
    assert "names the unrecorded cause frailty" in lint(written, schema, brief(schema))


def test_written_knowledge_loads_by_schema_id(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"one": GOOD}))
    (tmp_path / "b.json").write_text(json.dumps({"two": GOOD}))
    (tmp_path / "ignored.txt").write_text("not knowledge")
    assert set(load_knowledge(tmp_path)) == {"one", "two"}
    assert load_knowledge(tmp_path / "missing") == {}
