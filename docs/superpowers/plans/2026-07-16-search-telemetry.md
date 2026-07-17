# Search Telemetry Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an event-based `SearchTelemetry` surface to the MCTS, beam, and minimax engines plus a draft JSONL exporter, per `docs/superpowers/specs/2026-07-16-search-telemetry-design.md`.

**Architecture:** A new `search_telemetry` module defines the shared types and normative event semantics. Each engine gains event counters at its existing enumeration/hit/terminal code paths and a `telemetry()` accessor that assembles one `SearchTelemetry` per root search. A bench-side `search_summary_row` mirrors `observation_v1_row` and an example binary emits draft JSONL rows.

**Tech Stack:** Rust (quantik-core crate), serde_json for the exporter, existing bench helpers (`action_index`, `legal_action_mask`, `canonical_key_hex`, `canonical_json`).

## Global Constraints

- Repo: `quantik-core-rust`, branch `search-telemetry` (exists; spec is committed on it).
- Working directory for all commands: `crates/quantik-core` unless stated.
- The spec's "Event-Based Counter Semantics" and "Value Semantics" sections are normative. When this plan and the spec disagree, the spec wins; flag the conflict.
- No changes to search behavior, move ordering, or existing public result types (`MinimaxResult`, `BeamSearchResult`, `BeamStats` stay as-is; new fields/methods only).
- Commit messages: NO Co-Authored-By trailer, no tool prefixes.
- `action_index = shape * 16 + position` (0..63), always.
- Value invariant: `root_value` and every `q_value` in `[-1.0, 1.0]`, positive good for the root player, `|v| = 1.0` only for proven results.
- Draft schema label is exactly `search-summary.v1-draft`. Never emit `search-summary.v1`.
- Every task ends with `cargo fmt --all && cargo clippy --all-targets -- -D warnings && cargo test -p quantik-core` green before its commit.
- No absolute machine paths in code, docs, or scripts.

## Normative Event Semantics (copy into module docs verbatim where indicated)

| Counter | Event (same in every engine) |
| --- | --- |
| `expanded_nodes` | A state's successor set was computed by the search. |
| `generated_nodes` | A successor state was constructed. |
| `transposition_hits` | A cached search result or subtree was reused via state-keyed lookup instead of being searched again. |
| `canonical_dedup_hits` | A generated state was merged with, or skipped in favor of, an already-present duplicate — without reusing any search result. |
| `terminal_hits` | A state was determined terminal during tree search. Rollout outcomes are excluded in every engine. |
| `tablebase_hits` | A value/policy result was obtained from an external probe artifact instead of search. Always 0 until such an artifact exists. |

Counters are not mutually exclusive: a state whose enumeration finds zero legal moves is both expanded and terminal. Identical semantics do not imply comparable magnitudes: MCTS expands incrementally under an iteration budget, minimax expands exhaustively to a depth, beam prunes by design. `expanded_nodes` measures the same event everywhere, but cross-engine workload comparison belongs to `elapsed_ms` only.

---

### Task 1: `search_telemetry` module — shared types and normative docs

**Files:**
- Create: `crates/quantik-core/src/search_telemetry.rs`
- Modify: `crates/quantik-core/src/lib.rs` (add `pub mod search_telemetry;` in alphabetical order)

**Interfaces:**
- Consumes: `crate::moves::Move` (`{player: u8, shape: u8, position: u8}`).
- Produces (used by Tasks 2-5): `EngineKind` (+ `as_str()`), `PolicyMassKind` (+ `as_str()`), `SearchEventCounters`, `RootMoveStat` (+ `RootMoveStat::new`), `SearchTelemetry`.

- [ ] **Step 1: Write failing tests** (inside the new module, `#[cfg(test)] mod tests`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::Move;

    #[test]
    fn root_move_stat_computes_action_index() {
        let stat = RootMoveStat::new(Move::new(0, 2, 5), 7, Some(0.25));
        assert_eq!(stat.action_index, 2 * 16 + 5);
        assert_eq!(stat.policy_mass, 7);
        assert_eq!(stat.q_value, Some(0.25));
    }

    #[test]
    fn engine_kind_strings_match_bench_conventions() {
        assert_eq!(EngineKind::Mcts.as_str(), "mcts");
        assert_eq!(EngineKind::Beam.as_str(), "beam");
        assert_eq!(EngineKind::Minimax.as_str(), "minimax");
    }

    #[test]
    fn event_counters_default_to_zero() {
        let c = SearchEventCounters::default();
        assert_eq!(
            (c.expanded_nodes, c.generated_nodes, c.transposition_hits,
             c.canonical_dedup_hits, c.terminal_hits, c.tablebase_hits),
            (0, 0, 0, 0, 0, 0)
        );
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p quantik-core search_telemetry`
Expected: compile error, module does not exist.

- [ ] **Step 3: Implement the module**

Module-level `//!` docs MUST contain, verbatim, the "Normative Event Semantics" table and the two paragraphs after it (non-exclusivity; comparability caveat), introduced as: "These definitions are normative for every engine in this crate and for the Python mirror in quantik-core-py."

```rust
use crate::moves::Move;

/// Which engine produced a [`SearchTelemetry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineKind {
    Mcts,
    Beam,
    Minimax,
}

impl EngineKind {
    /// Stable lowercase label matching bench row `engine_kind` conventions.
    pub fn as_str(&self) -> &'static str {
        match self {
            EngineKind::Mcts => "mcts",
            EngineKind::Beam => "beam",
            EngineKind::Minimax => "minimax",
        }
    }
}

/// What `policy_mass` means for this engine's [`RootMoveStat`]s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyMassKind {
    /// True root visit counts (MCTS).
    Visits,
    /// Leaf multiplicity grouped by first move (beam).
    Multiplicity,
    /// No mass; `policy_mass` is 0 and only `q_value` is meaningful (minimax).
    None,
}

impl PolicyMassKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            PolicyMassKind::Visits => "visits",
            PolicyMassKind::Multiplicity => "multiplicity",
            PolicyMassKind::None => "none",
        }
    }
}

/// The six event counters. Field docs carry each counter's one-line
/// normative definition from the module docs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchEventCounters {
    pub expanded_nodes: u64,
    pub generated_nodes: u64,
    pub transposition_hits: u64,
    pub canonical_dedup_hits: u64,
    pub terminal_hits: u64,
    pub tablebase_hits: u64,
}

/// Per-root-move statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct RootMoveStat {
    pub mv: Move,
    /// `shape * 16 + position` (action-index.v1).
    pub action_index: u8,
    /// Semantics per the telemetry's [`PolicyMassKind`]; 0 when `None`.
    pub policy_mass: u64,
    /// `[-1, 1]`, root-player perspective; `None` when the engine has no
    /// value estimate for this move (e.g. an unvisited MCTS child).
    pub q_value: Option<f64>,
}

impl RootMoveStat {
    pub fn new(mv: Move, policy_mass: u64, q_value: Option<f64>) -> Self {
        Self {
            mv,
            action_index: mv.shape * 16 + mv.position,
            policy_mass,
            q_value,
        }
    }
}

/// One telemetry record for one completed root search.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchTelemetry {
    pub engine_kind: EngineKind,
    /// `[-1, 1]`, root-player perspective; `|v| = 1` only for proven results.
    pub root_value: f64,
    pub policy_mass_kind: PolicyMassKind,
    pub root_moves: Vec<RootMoveStat>,
    /// `false` whenever canonical/transposition merging may have collapsed
    /// distinct root moves onto shared statistics (MCTS with the
    /// transposition table on; minimax with `dedup_children`; beam when any
    /// depth-1 canonical dedup occurred). Exporters skip such rows.
    pub root_identity_preserved: bool,
    pub principal_variation: Vec<Move>,
    pub counters: SearchEventCounters,
    pub elapsed_ms: u64,
    pub depth_reached: u32,
    pub seed: Option<u64>,
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p quantik-core search_telemetry`
Expected: 3 passed.

- [ ] **Step 5: fmt, clippy, full suite, commit**

```bash
cargo fmt --all && cargo clippy --all-targets -- -D warnings && cargo test -p quantik-core
git add crates/quantik-core/src/search_telemetry.rs crates/quantik-core/src/lib.rs
git commit -m "Add search telemetry types with normative event semantics"
```

---

### Task 2: MCTS instrumentation and telemetry

**Files:**
- Modify: `crates/quantik-core/src/mcts.rs`
- Tests: same file, `mod tests`.

**Interfaces:**
- Consumes: Task 1 types.
- Produces: `MCTSEngine::telemetry(&self) -> Option<SearchTelemetry>`.

**Hook rules (from the normative table):**
- `expanded_nodes += 1` each time a node's successor set is computed via `generate_legal_moves` at node creation: the root push in `search()` (mcts.rs:98-108) and the child push in `expand()` (mcts.rs:240+).
- `generated_nodes += 1` at each `apply_move` in `expand()` (mcts.rs:229).
- `transposition_hits += 1` in the TT early-return branch of `expand()` (mcts.rs:232-237).
- `terminal_hits += 1` when a created node (root or child) has `is_terminal == true`. `simulate()` is NOT instrumented (rollout exclusion).
- `tablebase_hits` stays 0.

- [ ] **Step 1: Add engine fields and reset**

Add to `MCTSEngine`: `counters: SearchEventCounters`, `elapsed_ms: u64`, `max_depth_reached: u32`. Initialize in `new()` (`SearchEventCounters::default()`, 0, 0). At the top of `search()` alongside the existing clears: reset all three. In `search()`, capture `let started = Instant::now();` before the iteration loop and set `self.elapsed_ms = started.elapsed().as_millis() as u64;` right before `self.best_move(bb)`. Update `self.max_depth_reached = self.max_depth_reached.max(path.len() as u32 - 1);` after each iteration's `select` (path includes the root, so depth = len - 1).

- [ ] **Step 2: Write failing tests**

```rust
#[test]
fn telemetry_none_before_search() {
    let engine = MCTSEngine::new(MCTSConfig::default());
    assert!(engine.telemetry().is_none());
}

#[test]
fn telemetry_tt_on_empty_board_collapses_and_flags_identity() {
    let mut engine = MCTSEngine::new(MCTSConfig {
        max_iterations: 200,
        seed: Some(11),
        ..Default::default()
    });
    engine.search(&Bitboard::EMPTY).unwrap();
    let t = engine.telemetry().unwrap();
    assert!(!t.root_identity_preserved);
    // The documented collapse: 64 legal first moves, 3 canonical children.
    assert!(t.root_moves.len() < 64);
    assert!(t.counters.transposition_hits > 0);
}

#[test]
fn telemetry_tt_off_preserves_identity_and_invariants() {
    let mut engine = MCTSEngine::new(MCTSConfig {
        max_iterations: 500,
        seed: Some(11),
        use_transposition_table: false,
        ..Default::default()
    });
    let (best, _) = engine.search(&Bitboard::EMPTY).unwrap();
    let t = engine.telemetry().unwrap();
    assert!(t.root_identity_preserved);
    assert_eq!(t.policy_mass_kind, PolicyMassKind::Visits);
    assert_eq!(t.engine_kind, EngineKind::Mcts);
    assert_eq!(t.counters.transposition_hits, 0);
    assert!(t.counters.expanded_nodes > 0);
    assert!(t.counters.generated_nodes >= t.counters.expanded_nodes - 1);
    // Mass lands only on legal root actions and every q is in range.
    let legal: std::collections::HashSet<u8> = generate_legal_moves(&Bitboard::EMPTY)
        .iter().map(|m| m.shape * 16 + m.position).collect();
    for stat in &t.root_moves {
        assert!(legal.contains(&stat.action_index));
        if let Some(q) = stat.q_value {
            assert!((-1.0..=1.0).contains(&q));
        }
    }
    assert!((-1.0..=1.0).contains(&t.root_value));
    // PV starts with the best move and is a legal line from the root.
    assert_eq!(t.principal_variation.first(), Some(&best));
    let mut bb = Bitboard::EMPTY;
    for mv in &t.principal_variation {
        assert!(generate_legal_moves(&bb).contains(mv));
        bb = apply_move(&bb, mv);
    }
}

#[test]
fn rollout_terminals_are_excluded_from_terminal_hits() {
    // 3 iterations: the tree cannot reach a terminal at depth <= 3 (no
    // Quantik game ends before ply 7), but every rollout finishes a game.
    let mut engine = MCTSEngine::new(MCTSConfig {
        max_iterations: 3,
        seed: Some(5),
        use_transposition_table: false,
        ..Default::default()
    });
    engine.search(&Bitboard::EMPTY).unwrap();
    let t = engine.telemetry().unwrap();
    assert_eq!(t.counters.terminal_hits, 0);
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p quantik-core mcts::tests::telemetry`
Expected: compile error, `telemetry` not found.

- [ ] **Step 4: Implement counters and `telemetry()`**

Counter insertions per the hook rules. Root push in `search()`:

```rust
        self.counters.expanded_nodes += 1;
        if is_terminal {
            self.counters.terminal_hits += 1;
        }
```

In `expand()` after `let new_bb = apply_move(&parent_bb, &mv);`:

```rust
        self.counters.generated_nodes += 1;
```

In the TT hit branch before `return existing;`:

```rust
                self.counters.transposition_hits += 1;
```

After computing `is_terminal` for the new child (before the push):

```rust
        self.counters.expanded_nodes += 1;
        if is_terminal {
            self.counters.terminal_hits += 1;
        }
```

The accessor (place after `nodes_created`):

```rust
    /// Telemetry for the most recent `search()` call; `None` if `search`
    /// has not run or found no legal moves. See the `search_telemetry`
    /// module docs for the normative counter semantics.
    pub fn telemetry(&self) -> Option<SearchTelemetry> {
        let root = self.nodes.first()?;
        if root.children.is_empty() {
            return None;
        }
        let root_bb = root.bb;
        let mover = current_player(&root_bb).unwrap_or(0);
        let q_of = |node: &MCTSNode| -> Option<f64> {
            if node.visit_count == 0 {
                return None;
            }
            let wins = if mover == 0 { node.win_count_p0 } else { node.win_count_p1 };
            Some(2.0 * (wins as f64 / node.visit_count as f64) - 1.0)
        };
        let root_moves: Vec<RootMoveStat> = root
            .children
            .iter()
            .map(|&idx| {
                let child = &self.nodes[idx];
                RootMoveStat::new(
                    child.mv.expect("child node always has a move"),
                    child.visit_count as u64,
                    q_of(child),
                )
            })
            .collect();
        let (_, win_rate) = self.best_move(&root_bb)?;
        Some(SearchTelemetry {
            engine_kind: EngineKind::Mcts,
            root_value: 2.0 * win_rate - 1.0,
            policy_mass_kind: PolicyMassKind::Visits,
            root_moves,
            root_identity_preserved: !self.config.use_transposition_table,
            principal_variation: self.principal_variation(),
            counters: self.counters,
            elapsed_ms: self.elapsed_ms,
            depth_reached: self.max_depth_reached,
            seed: self.config.seed,
        })
    }

    /// Max-visit descent from the root; ties break on the lowest
    /// `action_index` for determinism. Bounded by 16 plies (a full game).
    fn principal_variation(&self) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut node_id = 0usize;
        for _ in 0..16 {
            let node = &self.nodes[node_id];
            let best = node
                .children
                .iter()
                .filter(|&&c| self.nodes[c].visit_count > 0)
                .min_by_key(|&&c| {
                    let child = &self.nodes[c];
                    let mv = child.mv.expect("child node always has a move");
                    (std::cmp::Reverse(child.visit_count), mv.shape * 16 + mv.position)
                });
            let Some(&best) = best else { break };
            pv.push(self.nodes[best].mv.expect("child node always has a move"));
            node_id = best;
        }
        pv
    }
```

Add imports: `use crate::search_telemetry::{EngineKind, PolicyMassKind, RootMoveStat, SearchEventCounters, SearchTelemetry};`

- [ ] **Step 5: Run tests**

Run: `cargo test -p quantik-core mcts`
Expected: all pass, including pre-existing MCTS tests (behavior unchanged).

- [ ] **Step 6: fmt, clippy, full suite, commit**

```bash
cargo fmt --all && cargo clippy --all-targets -- -D warnings && cargo test -p quantik-core
git add crates/quantik-core/src/mcts.rs
git commit -m "Instrument MCTS with event counters and telemetry accessor"
```

---

### Task 3: Beam instrumentation and telemetry

**Files:**
- Modify: `crates/quantik-core/src/beam_search.rs`

**Interfaces:**
- Consumes: Task 1 types; `BeamSearchResult::ranked_root_moves`.
- Produces: `BeamSearchEngine::telemetry(&self, result: &BeamSearchResult) -> SearchTelemetry`.

**Hook rules:**
- `expanded_nodes += 1` per frontier entry processed in `generate_candidates` (its successor set is computed), including the no-legal-moves case.
- `generated_nodes += 1` per `apply_move` on a candidate move.
- `terminal_hits += 1` for each terminal child (winner branch) and each no-legal-moves frontier entry.
- `canonical_dedup_hits += 1` at the dedup-merge branch; additionally track a private `root_dedup_hits` incremented only when `depth == 1` — `root_identity_preserved = root_dedup_hits == 0`.
- `transposition_hits` and `tablebase_hits` stay 0 (beam never reuses results).
- Elapsed: measure in `search()` (`Instant::now()` at entry, millis at every return path that produces a result).

Engine gains fields `counters: SearchEventCounters`, `root_dedup_hits: u64`, `elapsed_ms: u64`, reset at the top of `search()`. `generate_candidates` needs the current `depth` and `&mut self` counter access — it already takes `&mut self` context via the calling loop in `search()`; thread `depth` through its signature if not already present.

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn beam_empty_board_depth1_dedup_flags_identity() {
    let mut engine = BeamSearchEngine::new(BeamSearchConfig {
        beam_width: 64,
        max_depth: 2,
        rollouts_per_candidate: 1,
        random_seed: Some(3),
        ..Default::default()
    })
    .unwrap();
    let result = engine.search(&Bitboard::EMPTY).unwrap();
    let t = engine.telemetry(&result);
    // 64 symmetric first moves collapse under canonical dedup at depth 1.
    assert!(!t.root_identity_preserved);
    assert!(t.counters.canonical_dedup_hits > 0);
    assert_eq!(t.counters.transposition_hits, 0);
    assert_eq!(t.engine_kind, EngineKind::Beam);
    assert_eq!(t.policy_mass_kind, PolicyMassKind::Multiplicity);
}

#[test]
fn beam_telemetry_invariants_and_pv() {
    let mut engine = BeamSearchEngine::new(BeamSearchConfig {
        beam_width: 8,
        max_depth: 4,
        rollouts_per_candidate: 2,
        random_seed: Some(3),
        ..Default::default()
    })
    .unwrap();
    let result = engine.search(&Bitboard::EMPTY).unwrap();
    let t = engine.telemetry(&result);
    assert!((-1.0..=1.0).contains(&t.root_value));
    assert!(t.counters.expanded_nodes > 0);
    assert!(t.counters.generated_nodes >= t.counters.expanded_nodes);
    let legal: std::collections::HashSet<u8> = generate_legal_moves(&Bitboard::EMPTY)
        .iter().map(|m| m.shape * 16 + m.position).collect();
    for stat in &t.root_moves {
        assert!(legal.contains(&stat.action_index));
        assert!(stat.policy_mass > 0);
        if let Some(q) = stat.q_value {
            assert!((-1.0..=1.0).contains(&q));
        }
    }
    // PV mirrors the best leaf's line.
    assert_eq!(
        t.principal_variation,
        result.best_leaf.as_ref().map(|l| l.moves.clone()).unwrap_or_default()
    );
    assert!(t.elapsed_ms < 60_000);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p quantik-core beam_search::tests::beam_telemetry`
Expected: compile error, `telemetry` not found.

- [ ] **Step 3: Implement**

Counter insertions per hook rules (sites: no-moves branch beam_search.rs:440-452, `candidates_generated` line ~455, winner branch ~460-477, dedup branch ~479-483). `telemetry`:

```rust
    /// Telemetry for `result` (from this engine's most recent `search`).
    /// Root data derives from `ranked_root_moves(None)`; counters and
    /// elapsed come from that run. See the `search_telemetry` module docs
    /// for the normative counter semantics.
    pub fn telemetry(&self, result: &BeamSearchResult) -> SearchTelemetry {
        let ranked = result.ranked_root_moves(None);
        let root_moves: Vec<RootMoveStat> = ranked
            .iter()
            .map(|r| {
                RootMoveStat::new(
                    r.mv,
                    r.total_multiplicity,
                    Some(r.best_value.clamp(-1.0, 1.0)),
                )
            })
            .collect();
        let root_value = result
            .best_leaf
            .as_ref()
            .map(|leaf| {
                let v = if result.root_player == 0 { leaf.value } else { -leaf.value };
                v.clamp(-1.0, 1.0)
            })
            .unwrap_or(0.0);
        SearchTelemetry {
            engine_kind: EngineKind::Beam,
            root_value,
            policy_mass_kind: PolicyMassKind::Multiplicity,
            root_moves,
            root_identity_preserved: self.root_dedup_hits == 0,
            principal_variation: result
                .best_leaf
                .as_ref()
                .map(|l| l.moves.clone())
                .unwrap_or_default(),
            counters: self.counters,
            elapsed_ms: self.elapsed_ms,
            depth_reached: result.max_depth_reached,
            seed: self.config.random_seed,
        }
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p quantik-core beam_search`
Expected: all pass, pre-existing beam tests unchanged.

- [ ] **Step 5: fmt, clippy, full suite, commit**

```bash
cargo fmt --all && cargo clippy --all-targets -- -D warnings && cargo test -p quantik-core
git add crates/quantik-core/src/beam_search.rs
git commit -m "Instrument beam search with event counters and telemetry accessor"
```

---

### Task 4: Minimax instrumentation and telemetry

**Files:**
- Modify: `crates/quantik-core/src/minimax.rs`

**Interfaces:**
- Consumes: Task 1 types.
- Produces: `MinimaxEngine::telemetry(&self) -> Option<SearchTelemetry>`; `minimax_q_from_score(score: f64, win: f64) -> f64` (pub(crate), used by tests).

**Hook rules:**
- `expanded_nodes += 1` per `children()` invocation on a state (root call in `search_root` and each call in `negamax`).
- `generated_nodes += moves.len() as u64` inside `children()` (every move is applied before dedup filtering).
- `canonical_dedup_hits += 1` per dedup skip inside `children()`.
- `transposition_hits += 1` at each TT early-return: the `Bound::Exact` return and the `alpha >= beta` narrowed return (minimax.rs:339, 352).
- `terminal_hits += 1` at the `has_winning_line` return and the `moves.is_empty()` return in `negamax` (minimax.rs:312, 320).
- `tablebase_hits` stays 0.
- `children()` becomes an instance method `fn children(&mut self, bb, moves, dedup) -> Vec<ChildEntry>` so it can count (update both call sites).

**Value mapping (documented on the function):** minimax mate scores are `±(win - ply)` with `ply <= 16`, so proven results satisfy `|score| >= win - 16.0`. Map proven to exactly `±1.0`; squash heuristic scores with the smooth, monotonic, sign-preserving `score / (1.0 + score.abs())`, which is strictly inside `(-1, 1)`:

```rust
/// Map a root-perspective minimax score onto the telemetry value scale.
/// Proven results (mate scores, `|score| >= win - 16`) map to exactly
/// `±1.0`; heuristic scores squash smoothly into `(-1, 1)`.
pub(crate) fn minimax_q_from_score(score: f64, win: f64) -> f64 {
    if score >= win - 16.0 {
        1.0
    } else if score <= -(win - 16.0) {
        -1.0
    } else {
        score / (1.0 + score.abs())
    }
}
```

**State kept for telemetry:** engine fields `counters: SearchEventCounters`, `last_root_scored: Vec<(Move, f64)>`, `last_result_meta: Option<(Vec<Move>, u64, u32)>` — or simpler: store `last_pv: Vec<Move>`, `last_elapsed_ms: u64`, `last_depth: u32`, `last_root_value: f64`, populated where `search()` builds `MinimaxResult`. `search_root` additionally writes `self.last_root_scored = scored.iter().map(|(m, v, _)| (*m, *v)).collect();` before returning. Reset all in `search()` alongside `self.nodes = 0`.

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn q_mapping_proven_and_heuristic() {
    let win = 10_000.0;
    assert_eq!(minimax_q_from_score(win - 7.0, win), 1.0);
    assert_eq!(minimax_q_from_score(-(win - 7.0), win), -1.0);
    let q = minimax_q_from_score(3.0, win);
    assert!(q > 0.0 && q < 1.0);
    let q_neg = minimax_q_from_score(-3.0, win);
    assert!((q + q_neg).abs() < 1e-12); // odd symmetry
    assert!(minimax_q_from_score(5.0, win) > minimax_q_from_score(3.0, win));
}

#[test]
fn minimax_telemetry_dedup_and_identity() {
    let mut engine = MinimaxEngine::new(MinimaxConfig {
        max_depth: 3,
        ..Default::default()
    });
    engine.search(&State::new(Bitboard::EMPTY)).unwrap();
    let t = engine.telemetry().unwrap();
    assert!(!t.root_identity_preserved); // dedup_children defaults to true
    assert!(t.counters.canonical_dedup_hits > 0);
    assert_eq!(t.engine_kind, EngineKind::Minimax);
    assert_eq!(t.policy_mass_kind, PolicyMassKind::None);
    for stat in &t.root_moves {
        assert_eq!(stat.policy_mass, 0);
        let q = stat.q_value.expect("minimax scores every searched root move");
        assert!((-1.0..=1.0).contains(&q));
    }
}

#[test]
fn minimax_transposition_hits_require_tt() {
    // Depth >= 4 from a quiet position revisits transposed states.
    let state = State::new(Bitboard::EMPTY);
    let mut with_tt = MinimaxEngine::new(MinimaxConfig {
        max_depth: 4,
        dedup_children: false,
        ..Default::default()
    });
    with_tt.search(&state).unwrap();
    let hits_with = with_tt.telemetry().unwrap().counters.transposition_hits;
    let mut without_tt = MinimaxEngine::new(MinimaxConfig {
        max_depth: 4,
        dedup_children: false,
        use_transposition_table: false,
        ..Default::default()
    });
    without_tt.search(&state).unwrap();
    let hits_without = without_tt.telemetry().unwrap().counters.transposition_hits;
    assert!(hits_with > 0);
    assert_eq!(hits_without, 0);
    // Identity is preserved without dedup.
    assert!(without_tt.telemetry().unwrap().root_identity_preserved);
}

#[test]
fn minimax_telemetry_pv_matches_result() {
    let mut engine = MinimaxEngine::new(MinimaxConfig {
        max_depth: 3,
        ..Default::default()
    });
    let result = engine.search(&State::new(Bitboard::EMPTY)).unwrap();
    let t = engine.telemetry().unwrap();
    assert_eq!(t.principal_variation, result.pv);
    assert_eq!(t.depth_reached, result.depth_reached);
    assert!((-1.0..=1.0).contains(&t.root_value));
    // No Quantik game ends before ply 7, so a depth-3 tree has no terminals.
    assert_eq!(t.counters.terminal_hits, 0);
    assert!(t.counters.expanded_nodes > 0);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p quantik-core minimax::tests::minimax_telemetry`
Expected: compile error.

- [ ] **Step 3: Implement** counters, method-ized `children`, score squash, state capture, and:

```rust
    /// Telemetry for the most recent completed `search()`/`solve()`;
    /// `None` before the first search. See the `search_telemetry` module
    /// docs for the normative counter semantics.
    pub fn telemetry(&self) -> Option<SearchTelemetry> {
        if self.last_root_scored.is_empty() {
            return None;
        }
        let win = self.config.eval_config.win;
        let root_moves = self
            .last_root_scored
            .iter()
            .map(|&(mv, score)| RootMoveStat::new(mv, 0, Some(minimax_q_from_score(score, win))))
            .collect();
        Some(SearchTelemetry {
            engine_kind: EngineKind::Minimax,
            root_value: minimax_q_from_score(self.last_root_value, win),
            policy_mass_kind: PolicyMassKind::None,
            root_moves,
            root_identity_preserved: !self.config.dedup_children,
            principal_variation: self.last_pv.clone(),
            counters: self.counters,
            elapsed_ms: self.last_elapsed_ms,
            depth_reached: self.last_depth,
            seed: self.config.random_seed,
        })
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p quantik-core minimax`
Expected: all pass, pre-existing minimax tests unchanged.

- [ ] **Step 5: fmt, clippy, full suite, commit**

```bash
cargo fmt --all && cargo clippy --all-targets -- -D warnings && cargo test -p quantik-core
git add crates/quantik-core/src/minimax.rs
git commit -m "Instrument minimax with event counters, root scores, and telemetry"
```

---

### Task 5: Draft JSONL exporter and example

**Files:**
- Modify: `crates/quantik-core/src/bench/contracts.rs`
- Create: `crates/quantik-core/examples/search_summary_export.rs`

**Interfaces:**
- Consumes: `SearchTelemetry` (Tasks 1-4); existing `action_index(shape, position)`, `legal_action_mask(&Bitboard)`, `canonical_key_hex(&State)`, `bench::canonical::canonical_json`.
- Produces:

```rust
pub const SEARCH_SUMMARY_DRAFT_SCHEMA: &str = "search-summary.v1-draft";

/// Engine run configuration echoed into the row; `None` maps to JSON null.
pub struct SearchSummaryRunConfig<'a> {
    pub config_label: &'a str,
    pub search_depth: Option<u32>,
    pub rollouts: Option<u64>,
    pub beam_width: Option<u64>,
    pub node_budget: Option<u64>,
    pub time_budget_ms: Option<u64>,
}

/// One draft search-summary row, or `Ok(None)` when
/// `telemetry.root_identity_preserved` is false (such rows are skipped, per
/// the design spec's Root Identity section).
pub fn search_summary_row(
    row_id: u64,
    run_id: &str,
    qfen: &str,
    telemetry: &SearchTelemetry,
    run_config: &SearchSummaryRunConfig,
) -> Result<Option<Value>, String>
```

Row fields (mirroring `observation_v1_row` at contracts.rs:2015-2074): `schema` (draft label), `contract_version: "1.1.0"`, `run_id`, `row_id`, `position_key`, `ply`, `side_to_move`, `bitboards`, `qfen`, `legal_action_mask`, `engine_kind` (`telemetry.engine_kind.as_str()`), `engine_version` (`env!("CARGO_PKG_VERSION")`), `engine_checkpoint: null`, `config_label`, `search_depth`, `rollouts`, `beam_width`, `node_budget`, `time_budget_ms`, `seed`, `root_value`, `policy_mass_kind`, `policy_visits` (dense `[u64; 64]` from `root_moves` mass, zeros elsewhere), `root_q_values` (dense 64-array, `null` for actions without a q), `principal_variation` (array of action indices from root order), `expanded_nodes`, `generated_nodes`, `transposition_hits`, `canonical_dedup_hits`, `terminal_hits`, `tablebase_hits`, `elapsed_ms`, `depth_reached`.

- [ ] **Step 1: Write failing tests** (in contracts.rs `mod tests`)

```rust
#[test]
fn search_summary_row_shape_and_mask_consistency() {
    let mut engine = crate::mcts::MCTSEngine::new(crate::mcts::MCTSConfig {
        max_iterations: 50,
        seed: Some(7),
        use_transposition_table: false,
        ..Default::default()
    });
    engine.search(&Bitboard::EMPTY).unwrap();
    let telemetry = engine.telemetry().unwrap();
    let run_config = SearchSummaryRunConfig {
        config_label: "test-mcts",
        search_depth: None,
        rollouts: Some(50),
        beam_width: None,
        node_budget: None,
        time_budget_ms: None,
    };
    let row = search_summary_row(0, "run-test", "4x4-empty-qfen-here", &telemetry, &run_config)
        .unwrap()
        .expect("identity preserved rows are emitted");
    assert_eq!(row["schema"], SEARCH_SUMMARY_DRAFT_SCHEMA);
    assert_eq!(row["engine_kind"], "mcts");
    assert_eq!(row["policy_visits"].as_array().unwrap().len(), 64);
    assert_eq!(row["root_q_values"].as_array().unwrap().len(), 64);
    // Mass only on legal actions.
    let mask = row["legal_action_mask"].as_u64().unwrap();
    for (i, v) in row["policy_visits"].as_array().unwrap().iter().enumerate() {
        if v.as_u64().unwrap() > 0 {
            assert!(mask & (1u64 << i) != 0);
        }
    }
    assert!(row["expanded_nodes"].as_u64().unwrap() > 0);
}

#[test]
fn search_summary_row_skips_unpreserved_identity() {
    let mut engine = crate::mcts::MCTSEngine::new(crate::mcts::MCTSConfig {
        max_iterations: 50,
        seed: Some(7),
        ..Default::default() // TT on -> identity not preserved
    });
    engine.search(&Bitboard::EMPTY).unwrap();
    let telemetry = engine.telemetry().unwrap();
    let run_config = SearchSummaryRunConfig {
        config_label: "test-mcts-tt",
        search_depth: None,
        rollouts: Some(50),
        beam_width: None,
        node_budget: None,
        time_budget_ms: None,
    };
    let row = search_summary_row(0, "run-test", "4x4-empty-qfen-here", &telemetry, &run_config).unwrap();
    assert!(row.is_none());
}
```

Replace `"4x4-empty-qfen-here"` with the empty-board QFEN used by existing tests in this file (search `contracts.rs` tests for an existing empty/QFEN constant and reuse it; `State::new(Bitboard::EMPTY).to_qfen()` is acceptable if no constant exists).

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p quantik-core search_summary_row`
Expected: compile error.

- [ ] **Step 3: Implement** `search_summary_row` next to `observation_v1_row`, following its `json!` style; parse the qfen with `State::from_qfen`, derive `bb`, `ply` (piece counts), `side_to_move` (`current_player`), `position_key` (`canonical_key_hex`), `legal_action_mask(&bb)`. PV encodes as `telemetry.principal_variation.iter().map(|m| action_index(m.shape, m.position)).collect::<Vec<u8>>()`.

- [ ] **Step 4: Write the example** `examples/search_summary_export.rs`: for the empty board plus two mid-game QFENs (reuse positions from existing examples/tests), run all three engines — MCTS with `use_transposition_table: false`, minimax with `dedup_children: false`, beam as configured (its row may legitimately skip when depth-1 dedup fires; print a note when skipped) — and `writeln!` each `canonical_json` row to `--out <path>` (default `search-summaries.jsonl`). Follow `examples/selfplay_export.rs` for arg parsing and file writing. Every engine gets a fixed seed so output is reproducible.

- [ ] **Step 5: Run tests and the example**

```bash
cargo test -p quantik-core search_summary
cargo run -p quantik-core --example search_summary_export -- --out /tmp/search-summaries.jsonl
```
Expected: tests pass; example writes >= 6 rows (3 positions x >= 2 emitting engines), each line parses as JSON with `schema == "search-summary.v1-draft"`.

- [ ] **Step 6: fmt, clippy, full suite, commit**

```bash
cargo fmt --all && cargo clippy --all-targets -- -D warnings && cargo test -p quantik-core
git add crates/quantik-core/src/bench/contracts.rs crates/quantik-core/examples/search_summary_export.rs
git commit -m "Add draft search-summary JSONL exporter and example"
```

---

### Task 6: Documentation

**Files:**
- Create: `docs/search-telemetry.md` (repo root `docs/`)
- Modify: `scripts/README.md` (if it indexes examples; otherwise the crate-level README that does)

**Interfaces:** none (prose).

- [ ] **Step 1: Write `docs/search-telemetry.md`** with these sections:
  1. **Purpose** — the search-summary.v1 registration path; link the spec and quantik-core-contracts `docs/search-summary-v1.md`.
  2. **Normative event semantics** — the table and both caveat paragraphs from the plan header, verbatim, plus the sentence: "These definitions are normative for every engine in this crate and for the Python mirror in quantik-core-py."
  3. **Per-engine hook mapping** — the three engines' hook rules from Tasks 2-4, as a table.
  4. **Value semantics** — the `[-1, 1]` invariant, per-engine mappings including `minimax_q_from_score` (proven threshold `win - 16`, smooth squash) and the MCTS `2p - 1` mapping.
  5. **Root identity** — when each engine preserves it, what the exporter does, and the configuration for telemetry-quality runs (MCTS TT off, minimax `dedup_children: false`, beam best-effort).
  6. **Exporter usage** — the example command, the draft schema label, and the explicit warning that `search-summary.v1` must not be emitted before contract registration.
- [ ] **Step 2: Cross-link** — add the example to whichever README indexes examples, one line.
- [ ] **Step 3: Verify docs contain no absolute machine paths**

Run: `grep -rn '/Users/\|/private/tmp' docs/search-telemetry.md`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add docs/search-telemetry.md scripts/README.md
git commit -m "Document the search telemetry surface and event semantics"
```
