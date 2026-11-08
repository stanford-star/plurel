import json

import fingerprints


def test_seeded_draws_keep_their_stored_bytes():
    stored = json.loads(fingerprints.STORED.read_text())
    drawn = fingerprints.draw()
    assert set(drawn) == set(stored), "workloads and stored fingerprints differ: run --update"
    moved = sorted(name for name in drawn if drawn[name] != stored[name])
    assert not moved, (
        f"draws moved for {moved}. If the change means to move them, run"
        " `pixi run fingerprints` and say so in the PR; COMPATIBILITY.md says what that breaks"
    )


def test_a_generator_and_its_draws_are_rebuilt_from_their_seeds():
    rebuilt = {name: generator for name, generator, _ in fingerprints.workloads()}
    for name, generator, rows in fingerprints.workloads():
        again = rebuilt[name]
        assert generator.describe() == again.describe(), name
        first = fingerprints.digest(generator.sample(rows, seed=3))
        assert first == fingerprints.digest(again.sample(rows, seed=3)), name
        assert first != fingerprints.digest(generator.sample(rows, seed=4)), name
