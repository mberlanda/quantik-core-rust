# Changelog

All notable changes to `quantik-core` are documented here.

## Unreleased

### Added

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
