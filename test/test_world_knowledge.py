import json
from dataclasses import replace

from plurel.world import Edge, Node, WorldSchema, brief, lint, load_knowledge, roles, study

CASE = WorldSchema(
    schema_id="ledger_case",
    target="outcome",
    description="Each row is one case.",
    nodes={
        "cause": Node(kind="numeric", units="score", dist="symmetric", scale=(0, 10), desc="a root cause"),
        "other": Node(kind="numeric", units="count", dist="counts", scale=(0, 50), desc="a second root"),
        "mediator": Node(
            kind="numeric", units="percent", scale=(0, 100), noise_scale="low", noise_dist="gaussian",
            parents={"cause": Edge(effect=+1), "other": Edge(effect=-0.5, shape="hinge")}, desc="a mediator",
        ),
        "aftermath": Node(
            kind="numeric", units="amount", scale=(0, 1000), noise_scale="medium", noise_dist="laplace",
            parents={"mediator": Edge(effect=+0.5)}, desc="a consequence of the mediator",
        ),
        "ticket": Node(kind="numeric", units="count", dist="counts", scale=(1000, 9999), desc="a bookkeeping id"),
        "outcome": Node(
            kind="categorical", levels=["no", "yes"], level_probs={"no": 0.7, "yes": 0.3},
            parents={"mediator": Edge(effect=+1), "cause": Edge(effect=-0.5)},
            interactions=[("mediator", "cause")], noise_scale="medium", noise_dist="gaussian", desc="the outcome",
        ),
    },
)  # fmt: skip
GOOD = {
    "description": "A case that escalated is a yes. Cases with a high mediator usually escalate.",
    "columns": {c: "a note" for c in [*CASE.feature_names(), "outcome"]},
}


def test_roles_read_each_columns_place_in_the_graph():
    assert roles(CASE) == {
        "cause": "a cause of the outcome",
        "other": "a cause of the outcome",
        "mediator": "a cause of the outcome",
        "aftermath": "shares a cause with the outcome",
        "ticket": "unrelated to the outcome",
    }
    total = Node(kind="numeric", formula=lambda cause, other: cause + other, desc="their total")
    assert (
        roles(replace(CASE, nodes={**CASE.nodes, "total": total}))["total"]
        == "computed from cause, other"
    )


def test_the_study_pools_the_population_most_informative_column_first():
    found = study(CASE)
    columns = found["columns"]
    assert list(columns)[0] == "mediator" and columns["ticket"]["information"] < 0.01
    where = columns["mediator"]["typical_percentile_by_outcome"]
    assert where["yes"] > where["no"]
    assert set(found["outcome_shares"]) == {"no", "yes"}


def test_the_study_reads_a_categorical_column_by_its_outcome_shares():
    site = Node(kind="categorical", levels=["north", "south"], desc="the site")
    outcome = replace(CASE.nodes["outcome"], parents={**CASE.nodes["outcome"].parents, "site": {"north": -1.5, "south": 1.5}})  # fmt: skip
    found = study(replace(CASE, nodes={**CASE.nodes, "site": site, "outcome": outcome}))
    shares = found["columns"]["site"]["outcome_shares_by_value"]
    assert shares["south"]["yes"] > shares["north"]["yes"]


def test_a_brief_holds_the_schemas_words_its_study_and_its_graph():
    found = brief(CASE)
    assert found["schema_id"] == "ledger_case" and found["voice"] in ("first person", "impersonal", "second person")  # fmt: skip
    assert found["outcome"]["description"] == "the outcome" and not found["unrecorded_cause"]
    assert found["columns"]["mediator"]["role"] == "a cause of the outcome"
    assert brief(CASE) == found  # the voice and the study are seeded
    hidden = Node(kind="numeric", hidden=True, desc="unrecorded frailty")
    nodes = {
        **CASE.nodes,
        "frailty": hidden,
        "mediator": replace(CASE.nodes["mediator"], parents={**CASE.nodes["mediator"].parents, "frailty": +1}),
        "outcome": replace(CASE.nodes["outcome"], parents={**CASE.nodes["outcome"].parents, "frailty": +1}),
    }  # fmt: skip
    assert brief(replace(CASE, nodes=nodes))["unrecorded_cause"]


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
    runs = Node(kind="numeric", units="count", dist="counts", scale=(0, 50), desc="runs logged")
    schema = replace(CASE, nodes={**CASE.nodes, "simulation_runs": runs})
    found = brief(schema)
    written = {
        "description": "A yes is a case that escalated; simulation_runs carries little.",
        "columns": {c: "a note" for c in [*schema.feature_names(), "outcome"]},
    }
    assert lint(written, schema, found) == []
    leaky = {**written, "description": written["description"] + " These records are simulated."}
    assert lint(leaky, schema, found) == ["banned wording: \\bsimulat"]


def test_the_lint_refuses_naming_the_unrecorded_cause():
    hidden = Node(kind="numeric", hidden=True, desc="unrecorded frailty")
    nodes = {
        **CASE.nodes,
        "frailty": hidden,
        "mediator": replace(CASE.nodes["mediator"], parents={**CASE.nodes["mediator"].parents, "frailty": +1}),
        "outcome": replace(CASE.nodes["outcome"], parents={**CASE.nodes["outcome"].parents, "frailty": +1}),
    }  # fmt: skip
    schema = replace(CASE, nodes=nodes)
    written = {**GOOD, "description": GOOD["description"] + " Frailty matters too."}
    assert "names the unrecorded cause frailty" in lint(written, schema, brief(schema))


def test_written_knowledge_loads_by_schema_id(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"one": GOOD}))
    (tmp_path / "b.json").write_text(json.dumps({"two": GOOD}))
    (tmp_path / "ignored.txt").write_text("not knowledge")
    assert set(load_knowledge(tmp_path)) == {"one", "two"}
    assert load_knowledge(tmp_path / "missing") == {}
