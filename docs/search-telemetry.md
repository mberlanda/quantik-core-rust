# Search Telemetry

This document describes the `search_telemetry` surface shared by the MCTS,
beam, and minimax engines in this crate: what each event counter means, how
each engine maps its internal values onto a common `[-1, 1]` scale, when a
telemetry record's root-move identity is trustworthy, and how to export
`search-summary.v1` JSONL rows for offline analysis.

## 1. Purpose

`search-summary.v1` is a registered data contract
(`quantik-core-contracts` `schemas/search-summary-v1.json`) for search
diagnostics (event counters, root-move statistics, principal variation) emitted
by any of this crate's search engines. The Rust surface, the Python mirror, and
the contract registration all landed; this crate provides the telemetry types,
the per-engine instrumentation, and the JSONL exporter that emits the stable
`search-summary.v1` label.

See also:
- Design spec: `docs/superpowers/specs/2026-07-16-search-telemetry-design.md`
  (lands via the companion docs PR
  [#34](https://github.com/mberlanda/quantik-core-rust/pull/34); path is
  dangling until that merges)
- Contract doc (sibling repo `quantik-core-contracts`, path
  `docs/search-summary-v1.md`): the registration target this surface feeds.

## 2. Normative event semantics

The counters are defined by **search events**, not by whichever variables an
engine already happens to have. A definition names one event; every engine
increments the counter at exactly the code path where that event occurs.
These definitions are normative for every engine in this crate and for the
Python mirror in quantik-core-py.

| Counter | Event (same in every engine) |
| --- | --- |
| `expanded_nodes` | A state's successor set was computed by the search. |
| `generated_nodes` | A successor state was constructed. |
| `transposition_hits` | A cached search result or subtree was reused via state-keyed lookup instead of being searched again. |
| `canonical_dedup_hits` | A generated state was merged with, or skipped in favor of, an already-present duplicate — without reusing any search result. |
| `terminal_hits` | A state was determined terminal during tree search. Rollout outcomes are excluded in every engine. |
| `tablebase_hits` | A value/policy result was obtained from an external probe artifact instead of search. Always 0 until such an artifact exists. |

Counters are not mutually exclusive: a state whose enumeration finds zero
legal moves is both expanded and terminal. Identical semantics do not imply
comparable magnitudes: MCTS expands incrementally under an iteration budget,
minimax expands exhaustively to a depth, beam prunes by design.
`expanded_nodes` measures the same event everywhere, but cross-engine
workload comparison belongs to `elapsed_ms` only.

## 3. Per-engine hook mapping

| Counter | MCTS | Beam | Minimax |
| --- | --- | --- | --- |
| `expanded_nodes` | +1 per node created (root push in `search()`, child push in `expand()`) — the state's successor set was just computed via `generate_legal_moves`. | +1 per frontier entry processed in `generate_candidates`, including the no-legal-moves case. | +1 per successor-set computation: once for the root moves in `search`, then once right after `generate_legal_moves` in each `negamax` node (the depth-0 leaf and the no-legal-moves node included; a `has_winning_line` node returns before enumeration and is not counted). |
| `generated_nodes` | +1 per `apply_move` in `expand()`. | +1 per `apply_move` on a candidate move. | += `moves.len()` inside `children()` (every move is applied before dedup filtering). |
| `transposition_hits` | +1 in the TT early-return branch of `expand()`. | Always 0 — beam never reuses search results. | +1 at each TT early-return: `Bound::Exact` and the narrowed `alpha >= beta` return. |
| `canonical_dedup_hits` | Not applicable at the counter level (MCTS canonical merging is tracked via `root_identity_preserved`, not this counter). | +1 at the dedup-merge branch; a private `root_dedup_hits` also increments only when `depth == 1`. | +1 per dedup skip inside `children()`. |
| `terminal_hits` | +1 when a created node (root or child) has `is_terminal == true`. `simulate()`/rollouts are never instrumented. | +1 for each terminal child (winner branch) and each no-legal-moves frontier entry. | +1 at the `has_winning_line` return and the `moves.is_empty()` return in `negamax`. |
| `tablebase_hits` | Always 0. | Always 0. | Always 0. |

Distinctions worth calling out explicitly:

- **Result reuse vs. duplicate merging.** `transposition_hits` requires that a
  previously computed result or subtree was reused. Beam canonical dedup and
  minimax child dedup merge duplicates but re-derive nothing, so they land in
  `canonical_dedup_hits` instead — beam dedup must never masquerade as
  transposition reuse.
- **Tree scope for terminals.** MCTS and beam both run rollouts; both exclude
  rollout terminals from `terminal_hits`, so the counter means the same thing
  in the two sampling engines and in minimax.

## 4. Value semantics

Invariant: `root_value` and every `RootMoveStat::q_value` lie in `[-1.0,
1.0]`, positive is good for the root player, and `|v| = 1.0` only for proven
results (terminal nodes, mates). Every unproven (sampled or heuristic)
estimate is clamped to `[-UNPROVEN_VALUE_BOUND, UNPROVEN_VALUE_BOUND]` (`1.0 -
1e-6`) via `clamp_unproven` in `crates/quantik-core/src/search_telemetry.rs`,
so a sampled or heuristic value can never be mistaken for a proven `±1.0`.

Per-engine mapping:

- **MCTS**: win probability `p` for the root player maps to `2p - 1`. A
  terminal child (and a terminal best child's `root_value`) is a PROVEN
  result: its value is derived directly from the node's `terminal_value`
  (P0-perspective; negated for the root's perspective when the root mover is
  player 1) and reported as exact `±1.0`. Every non-terminal child's
  rollout-sampled `2p - 1` goes through `clamp_unproven` instead.
- **Minimax**: `minimax_q_from_score(score, win)` in
  `crates/quantik-core/src/minimax.rs`. Mate scores are `±(win - ply)` with
  `ply <= 16`, so a proven result satisfies `|score| >= win - 16.0` and maps
  to exactly `±1.0`. Everything else is squashed with the smooth,
  monotonic, sign-preserving `score / (1.0 + score.abs())`, which stays
  strictly inside `(-1, 1)`.
- **Beam**: a ranked root move's `q_value` is exact `1.0` only when
  `RankedRootMove::has_terminal_win` is set and `best_value >= 1.0` — a
  proven root-player win via that move. `RankedRootMove` carries no
  equivalent flag for a proven *loss*: once a terminal loss and a sampled
  loss both collapse to `best_value == -1.0`, they are indistinguishable, so
  every other case (including a proven loss) goes through `clamp_unproven`.
  This is a deliberate, documented conservatism: a proven loss is reported
  as `-UNPROVEN_VALUE_BOUND` rather than exactly `-1.0`. `root_value` follows
  the same rule at the `best_leaf` level: exact `±1.0` when `best_leaf` is
  terminal (its `value` is `±1.0` by construction), `clamp_unproven`
  otherwise.

`PolicyMassKind` per engine: MCTS = `Visits` (true root visit counts), beam =
`Multiplicity` (leaf multiplicity grouped by first move), minimax = `None`
(`root_moves` carry exact `q_value`s from the per-move score vector,
`policy_mass = 0`).

## 5. Root identity

`root_identity_preserved` is `false` whenever canonical/transposition merging
may have collapsed distinct root moves onto shared statistics:

- **MCTS**: preserved iff the transposition table is off
  (`use_transposition_table: false`). With it on, symmetric positions (e.g.
  the empty board's 64 legal first moves) collapse onto their canonical
  representatives.
- **Minimax**: preserved iff `dedup_children: false`.
- **Beam**: best-effort — preserved iff no depth-1 canonical dedup occurred.
  Symmetric positions such as the empty board may still skip even with a
  "default" configuration, since beam dedup is a property of the search
  outcome, not a single config flag.

The exporter (`search_summary_row` in
`crates/quantik-core/src/bench/contracts.rs`) returns `Ok(None)` — a
legitimate skip, not an error — for any row whose telemetry has
`root_identity_preserved == false`.

For telemetry-quality runs intended for export, configure: MCTS with
`use_transposition_table: false`, minimax with `dedup_children: false`, and
treat beam skips as expected rather than a bug to chase.

## 6. Exporter usage

Run the exporter example against a small fixed position set (the empty
board plus two mid-game positions), across all three engines:

```sh
cargo run -p quantik-core --example search_summary_export -- --out <path>
```

This writes one JSON line per completed root search whose root identity was
preserved, using the registered schema label `search-summary.v1`
(`SEARCH_SUMMARY_SCHEMA` in `bench::contracts`). Rows that are skipped
for an unpreserved root identity are logged to stderr, not written.

The contract is registered in `quantik-core-contracts`
(`schemas/search-summary-v1.json`), so downstream consumers can rely on the
stable `search-summary.v1` label.
