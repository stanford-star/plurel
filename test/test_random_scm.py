import hashlib
from dataclasses import replace

import plurel.random_scm as random_scm
from plurel.causal import node_of
from plurel.random_scm import realize_random


def digest(frame) -> str:
    return hashlib.sha256(frame.round(4).to_csv(index=False).encode()).hexdigest()[:16]


def test_a_random_table_is_a_pure_function_of_its_seed():
    population = realize_random(3, columns=(5, 8))
    rows = population.rows(40, 0)
    assert rows.equals(population.rows(40, 0)) and not rows.equals(population.rows(40, 1))
    assert list(rows.columns) == [*population.features, population.target]
    assert 5 <= len(rows.columns) <= 8
    assert population.id == "random--s3"


def test_the_target_is_categorical_with_parents_and_never_blank():
    population = realize_random(3, columns=(5, 8))
    plan = population.plan
    assert plan.columns[population.target].kind == "categorical"
    assert plan.columns[population.target].missing == 0.0
    if any(plan.nodes[c.node].edges for c in plan.columns.values() if c.kind == "categorical"):
        assert plan.nodes[node_of(plan, population.target)].edges
    assert all(node_of(plan, name) in plan.nodes for name in plan.columns)


def test_the_column_count_is_pinned_or_bounded():
    assert len(realize_random(7, columns=12).rows(5, 0).columns) == 12
    assert 3 <= len(realize_random(1, columns=(3, 24)).rows(5, 0).columns) <= 24


def test_the_draws_are_pinned():
    """A change that moves the prior's draws must say so here."""
    assert digest(realize_random(0, columns=8).rows(200, 0)) == "edacbbc8414339fb"
    assert digest(realize_random(1, columns=(3, 24)).rows(200, 0)) == "0d7584dc77798fd7"
    assert digest(realize_random(7, columns=12).rows(200, 0)) == "b98ca2e29ef62a8c"


def test_the_prior_caps_a_concat_node_at_128_wide(monkeypatch):
    """Concat widths compound through the DAG: this seed is 226 wide without the cap, and keeps its
    narrower concat nodes with it."""
    seed = 2991418007096968182
    capped = realize_random(seed, columns=(4, 24)).plan.nodes.values()
    assert max(node.dim for node in capped if not node.onehot) <= 128
    assert max(node.dim for node in capped if node.op == "concat") > 64
    monkeypatch.setattr(random_scm, "PRIOR", replace(random_scm.PRIOR, node_max_width=None))
    free = realize_random(seed, columns=(4, 24)).plan.nodes.values()
    assert max(node.dim for node in free if not node.onehot) > 128
