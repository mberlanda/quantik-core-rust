# Changelog

All notable changes to `quantik-core` are documented here.

## Unreleased

## 1.2.0 - 2026-07-18

### Added

- Event-based search telemetry (`SearchTelemetry`) for the MCTS, beam, and
  minimax engines, and a `search-summary.v1` JSONL exporter
  (`bench::contracts::search_summary_row`).

### Changed

- Minimax `expanded_nodes` is now counted at the successor-set computation, so a
  no-legal-moves node is both expanded and terminal (matching the normative
  counter semantics).
- Adopted the registered `search-summary.v1` schema label.
- Bumped the crate version and all `*_CONTRACT_VERSION` constants to `1.2.0`,
  tracking the contracts `1.2.0` release (which adds `search-summary.v1` and
  `model-checkpoint.v1`).

## 1.1.0 - 2026-07-14

### Added

- Added `opening-book-summary.v1` export support through
  `bench_bfs_inspect summary-json` and `scripts/inspect_opening_book.sh
  summary-json`, allowing Rust-generated depth books to be compared against
  Python consumers through the contracts `v1.1.0` consistency action.
- Added a GitHub Actions opening-book consistency job that generates a
  deterministic depth-4 book, writes the Rust summary, asks the Python stack to
  consume the same SQLite book, and compares both summaries with
  `mberlanda/quantik-core-contracts/actions/opening-book-consistency@v1.1.0`.

## 1.0.0 - 2026-07-13

First published release, mirroring the versioning of the sibling Python
`quantik-core` package rather than starting pre-1.0 — everything below
accumulated since this crate's creation.

### Added

- Added crates.io publication metadata to `crates/quantik-core/Cargo.toml`
  (license, keywords, categories, repository/homepage/documentation URLs),
  a repo-root `LICENSE` (MIT, mirroring the Python package's), a crate-local
  `README.md` for the crates.io/docs.rs package page, and a
  trusted-publishing GitHub Actions job (`publish-crate`, triggered on a
  published GitHub Release, using `rust-lang/crates-io-auth-action` — no
  static token) so `quantik-core` can be published to crates.io the same
  way the Python package publishes to PyPI.
- Updated the root `README.md` to describe this crate as the companion to
  [`quantik-core` on PyPI](https://pypi.org/project/quantik-core/), with
  guidance on when to reach for the Rust crate (bulk self-play/training-data
  generation, exhaustive search, opening-book construction — backed by this
  week's measured 200-900x wall-clock advantage) versus the Python package
  (model training, exploratory/notebook work), plus a clarifying note on
  the "Known Position Counts" table distinguishing ongoing vs. terminal
  canonical positions.
- Added `docs/superpowers/plans/2026-07-13-crates-io-packaging-and-ml-data-pipeline.md`,
  the implementation plan this change executes the first part of.
- Added `evaluation` module: a fitted-linear feature vector (`features()`) and
  weighted evaluator (`evaluate()`) ported from the Python engine, covering
  own/opponent/shared 3-line threats, mobility difference, and build-two /
  build-one features.
- Added `MinimaxEngine`, an exact alpha-beta negamax solver with iterative
  deepening, a canonical-key (`State::canonical_key()`) transposition table,
  sibling move dedup by child canonical key, an optional wall-clock search
  budget, and a `solve()` convenience that runs to the full 16-ply depth with
  no time limit.
- Added `BeamSearchEngine`, a level-by-level beam search that dedups
  candidates per depth via canonical key while accumulating symmetry
  multiplicity (path-count) on repeat occurrences, always preserves terminal
  leaves regardless of beam width, supports depth-dependent beam and rollout
  schedules plus a wall-clock time budget, and exposes `ranked_root_moves()`
  for comparing root-level candidates.
- Added `MCTSConfig::time_limit_s`, an optional wall-clock search budget
  checked after each completed iteration, and made
  `MCTSConfig::use_transposition_table` actually consulted: a canonical-key
  transposition table merges symmetric children when enabled, with
  path-based backpropagation to support nodes with multiple parents.
- Added the cross-engine benchmark harness: a checksummed shared position
  dataset (`benchmarks/positions-v1.json`) interoperable with the Python
  implementation's artifacts via a byte-exact canonical JSON encoder, an
  exact reference solver, uniform engine adapters (minimax/MCTS/beam/random)
  with a correctness preflight, agreement/cost/stability/head-to-head
  aggregation, reproducible result bundles, generated Markdown reports, and
  a `dataset`/`run`/`report`/`export-book` CLI (`cross_engine_benchmark`
  binary), documented in `docs/BENCHMARKS.md`.
- Added crash-safe checkpoint/resume to benchmark runs: `run --checkpoint
  <path>` streams every completed observation row and head-to-head game to
  a JSON Lines file as it happens, and `--resume` continues an interrupted
  run from that file instead of restarting it, guarded by a dataset checksum
  and config fingerprint so runs are never silently mixed.
- Added parallel `--workers N` for benchmark `run`: agreement observations
  and head-to-head games run on an N-thread rayon pool, with results
  streamed to the checkpoint/bundle in the same task order as `--workers 1`
  (byte-identical content too, for adapters whose behavior doesn't depend
  on real wall-clock CPU time).
- Added opening-book persistence of exactly-solved benchmark references:
  solved positions are upserted into the existing SQLite opening book keyed
  by the 18-byte canonical key, with reads and writes both restricted to
  boards that are their own canonical representative (the book does not yet
  record the symmetry transform needed to translate stored moves to other
  orientations); pre-existing book databases are migrated in place via an
  idempotent `ALTER TABLE`. The schema stays byte- and family-compatible
  with the Python `opening_book.py`, so a book file is readable from either
  language.

### Changed

- **Breaking:** replaced the single-file `.ckpt` benchmark checkpoint format
  (`run --checkpoint <path>`) with a Python-compatible checkpoint
  *directory* (`run --checkpoint-dir <dir>`: `manifest.json` written
  atomically + `observations.jsonl` + `h2h.jsonl`, both compact sorted-key
  JSON Lines). A checkpoint directory written by this crate now
  loads/reports correctly in Python's `benchmarks.checkpoint` module and
  vice versa; resume *validation* stays intra-language, since the two
  CLIs' config dictionaries differ in shape (documented in
  `docs/BENCHMARKS.md`). There is no migration path from the old
  single-file format — a `.ckpt` file cannot be resumed under the new
  layout. `report --input <dir>` now accepts a checkpoint directory
  directly and renders a partial-state report (a `"checkpoint": {status,
  counts}` bundle block, surfaced as two extra Markdown lines) from
  whatever rows/games have completed so far, without waiting for the run
  to finish. New `run` flags: `--checkpoint-dir`, `--checkpoint-every`
  (manifest/progress update cadence), `--workers`; removed: `--checkpoint`.
  The Rust-only `bundle["resumed"]` boolean from the old format is gone —
  the `"checkpoint"` block's `status`/`counts` supersede it, matching the
  Python bundle shape.

### Fixed

- Fixed MCTS UCB1 selection and the reported `win_probability`, which used
  player 0's win count regardless of which player was actually choosing at a
  given node, systematically starving player 1's best replies — the same
  class of perspective bug already fixed in the Python engine.
