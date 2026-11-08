import json

import fingerprints


def test_seeded_draws_keep_their_stored_bytes_on_this_platform():
    drawn = fingerprints.draw()
    stored = fingerprints.stored()
    assert fingerprints.PLATFORM in stored, (
        f"no fingerprints stored for {fingerprints.PLATFORM}; to add them, put this under that key"
        f" in test/fingerprints.json: {json.dumps(drawn, sort_keys=True)}"
    )
    mine = stored[fingerprints.PLATFORM]
    assert set(drawn) == set(mine), "workloads and stored fingerprints differ: run --update"
    moved = sorted(name for name in drawn if drawn[name] != mine[name])
    assert not moved, (
        f"draws moved on {fingerprints.PLATFORM} for {moved}. If the change means to move them,"
        " run `pixi run fingerprints` on each tested platform and say so in the PR;"
        " COMPATIBILITY.md says what that breaks"
    )


def test_a_generator_and_its_draws_are_rebuilt_from_their_seeds():
    rebuilt = {name: generator for name, generator, _ in fingerprints.workloads()}
    for name, generator, rows in fingerprints.workloads():
        again = rebuilt[name]
        assert generator.describe() == again.describe(), name
        first = fingerprints.digest(generator.sample(rows, seed=3))
        assert first == fingerprints.digest(again.sample(rows, seed=3)), name
        assert first != fingerprints.digest(generator.sample(rows, seed=4)), name
