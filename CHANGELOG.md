# Changelog

Changes to PluRel's public API, newest first. **Breaking** marks a change that needs a migration,
given beside it; a change that moves seeded draws says so and updates `test/fingerprints.json`
(see [COMPATIBILITY.md](COMPATIBILITY.md)). Every PR that breaks the API or moves draws adds a line
here.

## v2, unreleased (the `v2` branch)

- **Moves draws** of `SchemaPrior`: it warps its table prior once per database, and each table
  warps around that, since warping a warped `Range` now centres the new preference on a draw from
  the old one. Databases differ more as wholes, in their mechanism mix, missingness and kinds,
  while their tables stay related. Hand-authored schemas and a default `TablePrior` alone draw as
  before; a prior given a shaped `Range` now warps around that shape instead of discarding it.
- **Moves draws** of realized schemas with a real table of more than 4096 rows: realizing
  calibrates on a random 4096 of them, its real children keeping the rows that point at kept
  parents, instead of drawing the whole database. Realizing 1M real customers falls from about 7 s
  to 0.04 s. Other schemas draw as before; the fingerprints do not move.
- **#150** The two steps are named in docs, docstrings and errors: `realize` makes a generator, its
  seed the mechanism seed; `sample` draws a dataset of it, its seed the data seed. Three domain
  recipes in `examples/generation`.
- **#146** `Counts(least=...)` floors each parent's count. An empty real parent leaves children
  with null keys instead of raising.
- **#141** `write_database` saves a static database without split timestamps; one with events
  still needs them.
- **#140** **Breaking:** `nullable` now controls only random missingness. A key that matches or is
  unique and should leave rows without a parent pointing at nothing adds `unmatched="null"`; before,
  any `nullable` rate did that too. `Table.from_frame` takes `name`, `doc`, `target`, `sources` and
  `docs`; `Derived(..., columns={parameter: column})` reads any column name.
- **#139** **Breaking for internal callers:** draws run in `plurel.draw.Run`; `Schema.resolve`,
  `Schema.link`, `Schema.propagate` and `Schema.observe` are gone. The public API is unchanged.
- **#133** Public types are checked by pyright (`Table` takes `Mapping[str, Entry]`); CI runs lint,
  types, tests on Linux and macOS, the README and a clean wheel install.
- **#131** `realize(seed, rows=...)` sets the calibration draw's sizes; calibration is skipped when
  nothing is fit, and counted and unique keys get feasible sizes. `Recorded` reads in `prefer` and
  `counts` are checked when the schema is built.
- **#122** `describe()` and `explain()` on tables and schemas.
- **#121** The README states what a draw repeats and what depends on its size.
- **#120** `ForeignKey(unique=...)` draws keys without repeating a parent, alone or per group;
  `Counts(most=...)` caps each parent's count.
- **#119** `ForeignKey(match=..., prefer=..., counts=Counts(...))` choose parents by their columns
  and set a table's size from per-parent counts. **Breaking for internal callers:** `Schema.links`
  is gone and `propagate` returns the draw's state.
- **#118** `Recorded(...)` reads what a column records, in its units, as a cause, across a key or
  in a `Summary`; a `Timestamp` without a span is its causes in epoch seconds, after a recorded
  time. Bare names still read latents and existing draws were unchanged.
- **#117** `write_database` publishes atomically: the path holds a complete dataset or none.
- **#116** **Moves draws:** a table draws by its name, else its key, so a named table draws the
  same in any schema. Tables sharing a name raise. Derived columns may read derived columns, and a
  target must be a recorded column.

Earlier v2 work, the redesign around one node type and the table declarations, is in the commit
history up to #87; the v1 generator is at the `v1.1.0` and `v1.0.0` tags.
