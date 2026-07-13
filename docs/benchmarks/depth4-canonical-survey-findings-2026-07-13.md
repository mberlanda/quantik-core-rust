# Depth-4 canonical sampled survey: findings

The sampled survey decided on in
[`depth4-canonical-solve-cost-2026-07-13.md`](depth4-canonical-solve-cost-2026-07-13.md)
(Option A: bounded-budget sample rather than the ~12.7-day full exact
solve) ran to **200 of its planned 250 positions** before being
unexpectedly killed by the environment; both the JSON checkpoint and the
SQLite opening book (`benchmarks/results/depth4.db`) persisted everything
completed up to that point, so nothing solved was lost. 200 positions
sampled from 10,946 canonical depth-4 states (seed 20260712, `--budget-s
45`) is a 1.8% sample of the full space — real findings, held to
appropriately modest confidence.

## Headline

| | |
|---|---|
| Sampled | 200 / 250 planned (1.8% of the 10,946 canonical depth-4 states) |
| Solved within 45s budget | **171 (85.5%)** |
| Hit the budget (cutoff, unsolved) | 29 (14.5%) — every one at exactly 45.0-45.1s |

The 45s budget was chosen after a pessimistic 2-datapoint pilot (0/10
solved at 10s; 1/1 solved at 120s, taking 100s) — the full sample shows
that pilot was a poor predictor: **most positions solve well under
budget** (median 19.4s), and only a minority are genuinely hard. This is
a useful correction to keep in mind for any future budget-sizing
decision on this dataset.

## Solve cost distribution (among the 171 solved)

| | Solve time | Nodes |
|---|---|---|
| min | 4.20s | 1,503,092 |
| p50 | 19.36s | 6,202,930 |
| p95 | 39.62s | 14,426,190 |
| max | 45.00s | 22,286,926 |
| mean | 20.75s | 7,434,769 |

## Value distribution — the "proper probability" question, answered

Every canonical depth-4 position has player 0 to move (piece counts are
always 2-2 at depth 4), so the solved `value` field is directly "does P0
win with best play from here."

**Raw, one vote per canonical state** (n=171):

```
P0 wins: 97 (56.7%, Wilson 95% CI [49.2%, 63.9%])
P1 wins: 74 (43.3%)
```

**Multiplicity-weighted** — weighting each canonical state by how many
raw (non-canonical) 4-ply move sequences fold onto it, i.e. "if the
first 4 plies were played uniformly at random, what fraction of games
are already decided in P0's favor":

```
P0 wins: 60,416 of 99,584 weighted mass (60.7%)
P1 wins: 39,168 of 99,584 weighted mass (39.3%)
```

The two numbers disagree by 4 points (56.7% vs 60.7%) — canonical states
that are P0-favorable tend to have somewhat *higher* path-multiplicity
(more raw move orderings reach them) than P1-favorable ones. This is
exactly the distinction raised earlier in this investigation: a plain
per-canonical-state tally undercounts positions with many symmetric/
transposable paths relative to positions that are reached more narrowly,
and the multiplicity weighting (already computed for free during the
canonical-fold enumeration, no extra solving needed) corrects for it.

Caveat: this is a 1.8% sample (200 of 10,946), and the *cutoff* 29
positions are excluded from the value tally entirely — if hard-to-solve
positions are systematically more likely to favor one side (plausible:
sharper, harder-to-refute positions might correlate with a particular
kind of advantage), this could bias the estimate. No evidence either way
from this sample; flagged as a real caveat, not resolved.

## Coverage sanity check

The 200-position sample's multiplicity mass (116,864 raw 4-ply sequences)
is 85.2% covered by the solved subset (99,584) — the solved positions
aren't a weirdly narrow slice of the sample's raw-sequence mass, so the
value split above is reasonably representative of the sample as a whole,
not just of whichever states happened to be cheap to solve.

## What's in the book

`benchmarks/results/depth4.db` now holds 171 solved canonical depth-4
positions with their exact game value, optimal move set, node count, and
solver metadata — a real, if partial, opening book fragment, keyed by
the same 18-byte canonical key used everywhere else in this project, so
it's readable from Python via `opening_book.py` without any format
changes.

## If more coverage is wanted later

Resuming isn't automatic (this was a one-off script, not the checkpoint-
directory infrastructure built for the benchmark harness), but the book
itself is durable and additive: rerunning `examples/depth4_survey` with a
different `--seed` (or a higher `--sample`) will solve a *different*
200-ish positions and add them to the same book via `INSERT OR REPLACE`,
so repeated runs accumulate coverage rather than redo work — the 171
positions already solved will simply be skipped (root-level book
short-circuit in `solve_position_with_book`).

## Reproduction

```bash
cd quantik-core-rust
cargo build --release --example depth4_survey
./target/release/examples/depth4_survey \
  --budget-s 45 --sample 250 --seed 20260712 \
  --db benchmarks/results/depth4.db \
  --out benchmarks/results/depth4-survey.json
```
