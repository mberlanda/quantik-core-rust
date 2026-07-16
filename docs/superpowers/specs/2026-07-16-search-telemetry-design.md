# Search Telemetry Surface — Design

Date: 2026-07-16
Status: approved design, pre-implementation
Workstream: `search-summary.v1` registration path (quantik-core-contracts
`docs/search-summary-v1.md`)

## Context

`search-summary.v1` is proposed but not registered. Its registration gates
require that Rust and Python expose the same observable semantics for search
diagnostics before any artifact carries the contract label. Neither stack
currently counts transposition hits or terminal hits anywhere; MCTS has no
result struct, no principal variation, and its root-move identity collapses
under canonical merging; minimax computes exact per-root-move scores and
discards them.

This design adds an explicit telemetry surface to the three engines. It is
PR 1 of a three-PR workstream:

1. **quantik-core-rust** (this spec, detailed): telemetry types, event-based
   counters, per-engine surfaces, draft JSONL exporter.
2. **quantik-core-py**: mirror of the same types and semantics; the semantics
   sections of this spec are normative for both stacks.
3. **quantik-core-contracts**: register `search-summary.v1` with schema,
   fixtures, and cross-stack parity validation, then flip the draft schema
   label.

## Event-Based Counter Semantics (normative for both stacks)

The counters are defined by **search events**, not by whichever variables an
engine already happens to have. A definition names one event; every engine
increments the counter at exactly the code path where that event occurs. This
is what makes the semantics portable: two stacks agree because they count the
same thing happening, not because two unrelated numbers were given the same
field name.

Vocabulary follows the standard search-literature distinction: a state is
**generated** when it is constructed as a successor of another state, and
**expanded** when its own successors are enumerated.

| Counter | Event (same in every engine) |
| --- | --- |
| `expanded_nodes` | A state's successor set was computed by the search. |
| `generated_nodes` | A successor state was constructed. |
| `transposition_hits` | A cached **search result or subtree** was reused via state-keyed lookup instead of being searched again. |
| `canonical_dedup_hits` | A generated state was merged with, or skipped in favor of, an already-present duplicate — **without reusing any search result**. |
| `terminal_hits` | A state was determined terminal during tree search. Rollout outcomes are excluded in every engine. |
| `tablebase_hits` | A value/policy result was obtained from an external probe artifact instead of search. Always `0` until such an artifact exists. |

Per-engine hook mapping (Rust sites from the 2026-07-16 audit; Python mirrors
the same events at its equivalent sites):

| Counter | MCTS | Beam | Minimax |
| --- | --- | --- | --- |
| `expanded_nodes` | each `_expand`/expansion of a leaf | each surviving beam node whose children are generated for the next level | each negamax call that enumerates children (non-terminal, not cut off before enumeration) |
| `generated_nodes` | children created during expansion | candidate children constructed (`candidates_generated`) | children constructed in `children()` |
| `transposition_hits` | hit in the `transpositions` map (existing early-return path) | `0` — beam never reuses search results | TT-probe success (existing hit branches) |
| `canonical_dedup_hits` | pack-level merges in `add_child_node` | `candidates_deduped` | `dedup_children` skips |
| `terminal_hits` | terminal children encountered during expansion (never rollout terminals) | terminal leaves inserted | terminal leaves evaluated in negamax |

Distinctions that the definitions make structural rather than documentary:

- **Result reuse vs. duplicate merging.** `transposition_hits` requires that a
  previously computed result or subtree was reused. Beam canonical dedup and
  minimax child dedup merge duplicates but re-derive nothing, so they land in
  `canonical_dedup_hits`. This resolves the contract gate's explicit concern
  that beam dedup must never masquerade as transposition reuse.
- **Tree scope for terminals.** MCTS and beam both run rollouts; both exclude
  rollout terminals from `terminal_hits`. The counter therefore means the same
  thing in the two sampling engines and in minimax.
- **Comparability caveat (goes into all documentation verbatim).** Identical
  semantics do not imply comparable magnitudes. MCTS expands incrementally
  under an iteration budget, minimax expands exhaustively to a depth, beam
  prunes by design. `expanded_nodes` measures the same event everywhere, but
  cross-engine workload comparison belongs to `elapsed_ms` only.

Backward compatibility: minimax's existing `MinimaxResult.nodes` (negamax call
count) is untouched; telemetry counts its own events with its own fields.

## Core Types

New module `crates/quantik-core/src/search_telemetry.rs`:

```rust
pub enum EngineKind { Mcts, Beam, Minimax }
pub enum PolicyMassKind { Visits, Multiplicity, None }

pub struct RootMoveStat {
    pub mv: Move,
    pub action_index: u8,        // shape * 16 + position (action-index.v1)
    pub policy_mass: u64,        // semantics per PolicyMassKind; 0 when None
    pub q_value: Option<f64>,    // see Value Semantics
}

pub struct SearchTelemetry {
    pub engine_kind: EngineKind,
    pub root_value: f64,
    pub policy_mass_kind: PolicyMassKind,
    pub root_moves: Vec<RootMoveStat>,
    pub root_identity_preserved: bool,
    pub principal_variation: Vec<Move>,
    pub expanded_nodes: u64,
    pub generated_nodes: u64,
    pub transposition_hits: u64,
    pub canonical_dedup_hits: u64,
    pub terminal_hits: u64,
    pub tablebase_hits: u64,
    pub elapsed_ms: u64,
    pub depth_reached: u32,
    pub seed: Option<u64>,
}
```

Every field carries rustdoc stating its normative definition; the module-level
doc holds the event-semantics table and the comparability caveat.

## Value Semantics (normative for both stacks)

Invariant for `root_value` and every `q_value`: the value lies in `[-1.0, 1.0]`,
positive is good for the root player, and `|v| = 1.0` only for proven results.

Per-engine mapping:

- **MCTS**: win probability `p` for the root player maps to `2p - 1`.
- **Beam**: evaluator values already conform (verify during implementation;
  clamp if a scheduled evaluator can exceed the range).
- **Minimax**: the implementation must inspect the actual evaluation scale
  (`eval_config` heuristic range and any mate-magnitude convention) and define
  a documented squash that preserves the invariant, mapping proven wins/losses
  to exactly `±1.0`. The mapping is engine-specific; the invariant is not.

`PolicyMassKind` per engine: MCTS = `Visits` (true root visit counts),
beam = `Multiplicity` (leaf multiplicity grouped by first move), minimax =
`None` (`root_moves` carry exact `q_value`s from the currently discarded
`search_root` score vector, `policy_mass = 0`).

## Root Identity

No change to search behavior. Telemetry reports what the engine actually did:

- `root_identity_preserved = false` whenever canonical/transposition merging
  may have collapsed distinct root moves onto shared nodes (MCTS with
  `use_transposition_table: true`; minimax with `dedup_children: true`).
- The exporter skips rows where identity is not preserved. Telemetry-quality
  runs configure MCTS with the transposition table off (following the existing
  `examples/selfplay_export.rs` precedent, rationale documented there at
  lines 96-113) and minimax with `dedup_children: false`; the exporter example
  sets both explicitly.
- The regression case is pinned by test: MCTS with TT on, empty board, must
  report `root_identity_preserved: false` (the documented 64-moves-collapse-
  to-3 case).

## Engine Surfaces

- **MCTS**: `search` gains internal elapsed measurement and the new counters;
  a new `telemetry(&self) -> SearchTelemetry` accessor derives `root_moves`
  from root children (visits and win-rate mapped to `[-1,1]`) and produces a
  principal variation by max-visit descent (new traversal, tie-broken by
  lowest action index for determinism).
- **Beam**: telemetry is derived from `BeamSearchResult` — `root_moves` from
  `ranked_root_moves` (mass = `total_multiplicity`, `q_value` = `best_value`),
  PV = `best_leaf.moves`, counters from `BeamStats` plus the new event
  counters; elapsed measured in `search`.
- **Minimax**: `search_root` keeps its per-move `scored` vector and threads it
  into telemetry; PV and elapsed already exist; new counters at the TT-hit,
  terminal, and dedup branches.

Each engine exposes the same shape: run the search, then obtain one
`SearchTelemetry` for that root search.

## Draft JSONL Exporter (bench feature)

`search_summary_row(...)` in `bench/contracts.rs`, mirroring
`observation_v1_row`: canonical position key, qfen, bitboards, legal action
mask, engine config block (`search_depth`, `rollouts`, `beam_width`,
`node_budget`, `time_budget_ms`, `seed`), and the telemetry fields, including
dense `policy_visits[64]` / `root_q_values[64]` arrays built from
`root_moves` (unfilled legal slots are `0` mass / `null` q).

The schema label is **`search-summary.v1-draft`**. The contract doc forbids
emitting `search-summary.v1` before registration; PR 3 flips the label after
the contract is registered. `examples/search_summary_export.rs` runs all
three engines over a small position set and writes JSONL through the existing
`canonical_json` path.

## Testing

- Per-engine counter tests on constructed positions with known transpositions
  and terminals (e.g., a position pair reachable by two move orders must
  produce `transposition_hits > 0` in minimax with TT on).
- Invariant tests: mass only on legal actions; every `q_value` and
  `root_value` in range; PV non-empty when a best move exists, starts with the
  best move, and is a legal line from the root.
- Rollout-exclusion test: an MCTS run whose rollouts certainly reach terminals
  must not count them in `terminal_hits` (compare against expansion-only
  terminal count).
- The `root_identity_preserved` regression test above.
- Exporter row-shape test validating the draft row against the intended field
  list, and that identity-unpreserved rows are skipped.

## Documentation Deliverables

The user has flagged documentation quality as a priority for this slice:

- Module-level rustdoc in `search_telemetry.rs` carrying the full normative
  event-semantics table, the result-reuse vs. dedup distinction, the rollout
  exclusion rule, and the comparability caveat.
- `docs/search-telemetry.md` in quantik-core-rust: the same semantics in prose
  for non-Rust readers, plus per-engine hook mapping and exporter usage. This
  file is the source the Python mirror and the contracts registration will
  cite.
- `scripts/README.md` / example docs updated for the new exporter example.
- After merge, a small quantik-core-contracts doc PR updates
  `docs/search-summary-v1.md` "Current Implementation State" to record the
  Rust surface (held for manual review, per workflow).

## Out of Scope

- Python mirror (PR 2) — bound by the normative sections here.
- Contract registration, schema, fixtures, parity tests (PR 3).
- Any tablebase/probe implementation (`tablebase_hits` stays 0).
- Changes to search behavior, move ordering, or existing result types.
