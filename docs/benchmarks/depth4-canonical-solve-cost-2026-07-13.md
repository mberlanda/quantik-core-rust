# Depth-4 canonical exact-solve cost: findings, tradeoffs, and projections

## Background

The Python worktree's `benchmarks/depth4-canonical/` ships a 1,000-position
*sample* of canonical, nonterminal depth-4 Quantik positions, all with
`reference: null` — the worktree's own `benchmarks/README.md` documents
that "exact agreement tables require a reference augmentation pass first,"
i.e. nobody has actually run the exact solver over this dataset yet, in
either language.

The goal of this investigation: enumerate the **full** canonical depth-4
state space (not just the 1,000-sample) in Rust, exactly solve it, and use
the result both as a probability-distribution / opening-book finding and as
a cross-language validation point.

## What was measured

**Enumeration** (canonical-key BFS from the empty board, 4 plies, folding
every legal child onto its canonical representative and summing
multiplicity — the same accumulation technique `docs/BEAM_SEARCH.md`
documents for beam search): **10,946 canonical nonterminal depth-4
states**, computed in 0.08–0.14s. This matches the Python worktree's
documented count of 10,946 exactly — a clean, free cross-language parity
check, independent of the solving question below.

**Exact-solve cost pilot**, before committing to a full run:

| Budget/position | Sample | Result |
|---|---|---|
| 10 s | 10 positions | **0/10 solved** — every position hit cutoff |
| 120 s | 1 position | **1/1 solved, in 100.043 s** |

## Root cause

`bench::reference::solve_position`'s per-child solver
(`score_child`) allocates a **fresh** `MinimaxEngine` (empty
transposition table) for *every one* of a position's ~30–40 legal root
moves, and `solve_position_with_book` only checks the persistent SQLite
book at the **root** of each call — not at any of the positions visited
during the internal 12-ply recursive search. So a depth-4 position (12
plies remaining) pays the full search cost from scratch, on every one of
its own children, with zero cache reuse either within one position's own
search or across the 10,946 independent root positions. This is fine for
the shallower phases the existing 36-position benchmark dataset actually
solves (early_mid/late_mid/endgame have far fewer remaining plies), but
depth-4 sits exactly where the benchmark's own "opening" phase (0–4
pieces) was deliberately excluded from reference solving for cost reasons
— this investigation empirically confirms why.

## Projection

Extrapolating the single observed data point (~100 s/position,
single-threaded, no cross-position caching) across all 10,946 states:

```
10,946 positions x ~100 s/position ≈ 1,094,600 s ≈ 304 hours ≈ ~12.7 days
```

This is a rough, single-sample extrapolation — actual per-position cost
almost certainly varies widely (some positions are far more forcing/
constrained than others) — but it's the only empirical anchor available,
and it's consistent with the "0/10 solved at 10s" pilot: whatever the true
distribution, a meaningful fraction of positions clearly needs on the
order of a minute or more.

## Options considered

| Option | Projected cost | What you get |
|---|---|---|
| **A. Sampled survey** (chosen) | Bounded by `sample x budget`, e.g. 250 x 45s = 11,250s (~3.1h) worst case | Honest solved-fraction + value distribution over a representative sample; no code changes; doesn't cover the full 10,946 |
| **B. Shared transposition caching** | 1–2 orders of magnitude cheaper if it works (share one TT across a position's sibling `score_child` calls, and persist it across positions via the book), then a full run in hours instead of ~12.7 days | Full coverage, but requires implementing + reviewing changes to `minimax.rs`/`reference.rs` before any run starts |
| **C. Bounded-depth evaluation** | Seconds, total, for all 10,946 | Full coverage, but heuristic scores, not proven win/loss/draw values — a different kind of finding |
| **D. Full exact solve as-is** | ~12.7 days, single background job | Full coverage, ground-truth values, no engineering work — just a very long wait |

## Decision

**Option A — sampled survey**, run now: `examples/depth4_survey --sample
250 --budget-s 45 --seed 20260712`, worst-case ~3.1 hours, running in the
background against a fresh SQLite book at `benchmarks/results/depth4.db`.
Findings will be written up separately once it completes (solved
fraction, value distribution — both raw and multiplicity-weighted by the
4-ply path-count accumulated during enumeration — solve-time distribution,
and node counts).

Options B and C remain open follow-ups if full 10,946-state coverage is
wanted later: B is the higher-leverage engineering investment (also
directly useful for any future deeper-canonical-depth book-building, not
just depth-4), C is the fast/cheap fallback if only a heuristic signal is
needed.

## Follow-up: is there unused symmetry-based pruning at depth 4?

Two more questions came up: does the current 192-element symmetry group
(8 board symmetries x 24 shape relabelings) already capture everything
available, and — separately, not a pruning question — what does the
orbit-size distribution of the 10,946 states actually look like?

**Board rotation/reflection and shape relabeling are already complete.**
`SymmetryHandler::find_canonical` takes the minimum over all 8 D4 board
symmetries x 24 shape permutations = 192 combinations. D4 is the full
symmetry group of a 4x4 grid whose rows, columns, *and* 2x2 zones must
all map onto rows/columns/zones simultaneously — there is no larger
geometric symmetry group for this board shape. All 4! shape relabelings
are already enumerated too. Neither axis has headroom left.

**Color swap (relabeling player 0 <-> player 1) was investigated and
empirically refuted as a usable symmetry.** The hypothesis: since
Quantik's win condition is color-blind (`has_winning_line` unions both
players' pieces per shape) and the blocking rule is symmetric in
`player`/`opponent`, relabeling P0<->P1 throughout a position looked like
it should negate the game value and preserve the optimal-move set — and
depth-4 positions are always at even ply (2 pieces each), which is
exactly the case where the relabeled position remains a structurally
valid state (piece-count parity checks out). This turned out to be wrong
in a way only caught by testing it: relabeling a fixed 4-piece
*configuration* is not the same as relabeling the *move history* that
produced it, and a careful reachability argument shows the two diverge —
the fourth move's legality under the relabeled order depends on a
constraint (shapes at moves 3 and 4 not sharing a line) that the original
game never had to satisfy, because move 4 happens after move 3 either
way. Verified empirically over 15 solved even-ply pairs
(`examples/color_swap_check.rs`, since deleted): **0/15 had matching
optimal-move sets, and only ~7/15 (chance level) even had a negated
value.** Color swap is not a valid canonicalization axis for this
ruleset — the existing 192-element group appears to be the ceiling for
symmetry-based deduplication.

**Orbit-size distribution** (transposition count per canonical state,
under the existing 192-element group — pure symmetry computation, no
solving, via `SymmetryHandler::orbit_size`):

| Orbit size | Canonical states | Share | Raw boards |
|---|---|---|---|
| 8 | 2 | 0.02% | 16 |
| 16 | 15 | 0.14% | 240 |
| 24 | 8 | 0.07% | 192 |
| 32 | 80 | 0.73% | 2,560 |
| 48 | 208 | 1.90% | 9,984 |
| 96 | 3,771 | 34.45% | 361,916 |
| 192 | 6,862 | 62.69% | 1,317,504 |
| **Total** | **10,946** | 100% | **1,692,512** |

Every orbit size is a divisor of 192 (the orbit–stabilizer theorem: orbit
size = |group| / |stabilizer|), so only these 7 values are possible at
all. Almost two-thirds of canonical depth-4 states have the maximal
192-element orbit (no symmetry stabilizes them — generic, "lopsided"
configurations), and 97%+ have orbit size 96 or 192; only 25 states
(0.23%) have any symmetry beyond a 48-element stabilizer. This confirms
the current canonicalization is already doing real, substantial work: on
average each canonical state stands in for ~154.6 raw boards, so the
"true" reachable depth-4 state space (1,692,512 raw boards) is already
~155x larger than what we solve. There just isn't a *further*
undiscovered symmetry axis to exploit beyond what's already implemented.

Interactive chart: https://claude.ai/code/artifact/afcf3dab-d679-48ef-9013-a05157f757fd
