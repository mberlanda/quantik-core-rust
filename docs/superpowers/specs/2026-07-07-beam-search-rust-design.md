# Beam Search for quantik-core-rust — Design Spec

**Date:** 2026-07-07 · **Status:** Approved for next-session implementation
**Source:** Port of Python PR https://github.com/mberlanda/quantik-core-py/pull/15
(branch `feat/mcts-beam-search`, unmerged as of 2026-07-07).
**Reference material (vendored, authoritative for algorithm semantics):**
- `docs/superpowers/reference/beam-search-py/beam_search.py` (355 lines — the Python implementation)
- `docs/superpowers/reference/beam-search-py/test_beam_search.py` (25 tests)
- `docs/superpowers/reference/beam-search-py/design-spec.md` (original Python design)
- `docs/superpowers/reference/beam-search-py/BEAM_SEARCH.md` (user docs)

## Goal

A parametrizable beam search engine in `crates/quantik-core/src/beam_search.rs`
that complements `MCTSEngine`: descends level-by-level to true terminal leaves
(up to the full 16-ply game) with memory bounded at O(beam_width × depth),
symmetry-aware dedup, adversarial (mover-relative) pruning, and replayable
principal variations.

## Key delta vs the Python version

The Python engine inserts survivors into a shared `CompactGameTree`
(64-byte nodes, transposition merge). **quantik-core-rust has no
`CompactGameTree`** — `MCTSEngine` uses a private `Vec<MCTSNode>`.
**Decision: standalone module, no shared tree (YAGNI).** Keep the counters
(`nodes_inserted` etc.) so the memory-bound contract is still testable;
sharing a tree with MCTS is an explicit non-goal / possible follow-up.
All other semantics (terminal handling, dedup, mover-relative pruning,
root-perspective ranking, private seeded RNG) are ported faithfully.

## Rust API (crate conventions: `Result<_, String>` errors like `State::unpack`; `rand::StdRng` like `MCTSEngine`)

```rust
// crates/quantik-core/src/beam_search.rs
use crate::bitboard::Bitboard;
use crate::moves::Move;

pub type Evaluator = Box<dyn Fn(&Bitboard) -> f64>; // returns value in [-1,1], P0 perspective

pub struct BeamSearchConfig {
    pub beam_width: usize,          // default 64,  must be >= 1
    pub max_depth: u32,             // default 16,  must be 1..=16
    pub rollouts_per_candidate: u32,// default 8,   must be >= 1
    pub seed: Option<u64>,          // default None; seeds a PRIVATE StdRng
}
impl Default for BeamSearchConfig { /* values above */ }

#[derive(Clone, Debug, PartialEq)]
pub struct BeamLeaf {
    pub moves: Vec<Move>,   // principal variation from the root, replayable via apply_move
    pub value: f64,         // P0 perspective; ±1.0 for terminal leaves
    pub depth: u32,         // == moves.len()
    pub is_terminal: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeamSearchStats {
    pub candidates_generated: u64, // children produced by expansion
    pub candidates_deduped: u64,   // dropped by canonical-payload dedup
    pub nodes_inserted: u64,       // survivors kept across all levels (≤ beam_width × depth)
    pub nodes_pruned: u64,         // scored candidates cut by the beam
    pub evaluations: u64,          // evaluator invocations
}

pub struct BeamSearchResult {
    pub best_leaf: Option<BeamLeaf>,    // best from the ROOT player's perspective
    pub terminal_leaves: Vec<BeamLeaf>, // all terminals found, best-first (root perspective)
    pub reached_terminal: bool,         // any terminal leaf discovered
    pub max_depth_reached: u32,
    pub stats: BeamSearchStats,
}

pub struct BeamSearchEngine { /* config, rng: StdRng, evaluator: Option<Evaluator>, stats */ }

impl BeamSearchEngine {
    pub fn new(config: BeamSearchConfig) -> Result<Self, String>;   // validates config
    pub fn with_evaluator(self, evaluator: Evaluator) -> Self;      // builder-style override
    pub fn search(&mut self, root: &Bitboard) -> Result<BeamSearchResult, String>;
    pub fn stats(&self) -> &BeamSearchStats;
}
```

Export: add `pub mod beam_search;` to `crates/quantik-core/src/lib.rs`
(alphabetical position: line 2, after `bitboard`).

## Algorithm (per depth level; port of `beam_search.py:112-300`)

Frontier entry: `(bb: Bitboard, moves: Vec<Move>, value: f64)` — value is the
candidate's P0-perspective evaluation, kept so max-depth frontier entries can
become leaves without re-evaluating.

Root validation (`search`): `check_winner(root) != NoWin` → `Err("root state is already terminal")`;
`current_player(root) == None` → `Err(...)`; no legal moves → `Err("root has no legal moves")`.

For `depth` in `1..=max_depth`, with `mover = current_player(frontier bb)`
(identical for every entry at a level — players strictly alternate):

1. **Expand:** for each frontier entry, `generate_legal_moves`; if empty, the
   entry itself is terminal — mover is blocked and **loses** (value `-1.0` if
   mover is P0 else `+1.0`); record it as a terminal leaf and drop it.
   Otherwise `apply_move` each legal move and classify with `check_winner`.
2. **Terminals:** children with `Player0Wins`/`Player1Wins` become terminal
   leaves (value `+1.0`/`-1.0`), are **not** carried forward.
3. **Dedup:** non-terminal children deduped within the level by
   `SymmetryHandler::canonical_payload(&bb)` (`[u8; 16]`, `HashSet`); first
   path encountered wins. (Empty board: 64 depth-1 moves → 3 canonical states.)
4. **Score:** evaluate each survivor → `v ∈ [-1,1]` P0 perspective (clamp
   defensively). Rank by **mover-relative** score: `v` if mover is P0, `-v`
   if mover is P1. This is the adversarial pruning invariant — pinned by a
   mutation-killing test in Python (`test_pruning_uses_mover_relative_score`).
5. **Prune:** stable sort descending by score (`sort_by(|a,b| b.score.total_cmp(&a.score))`
   on a stable sort preserves insertion order for ties → seeded determinism),
   `truncate(beam_width)`. Survivors form the next frontier;
   `nodes_inserted += survivors`, `nodes_pruned += cut`.
6. Stop when the frontier is empty or `max_depth` is reached. Track
   `max_depth_reached` as the deepest level that produced any leaf or survivor.

**Result ranking:** collected leaves = all terminal leaves + (if the loop ended
at `max_depth` with a live frontier) the final frontier entries as
non-terminal leaves. `best_leaf` maximizes `root_sign * value` where
`root_sign = +1.0` if `current_player(root) == 0` else `-1.0`; ties keep first
encountered. `terminal_leaves` sorted by the same key, descending.

## Default evaluator (port of `beam_search.py:319-347`)

Mean of `rollouts_per_candidate` uniform random playouts to true terminal
(never > 16 plies, no depth cutoff). Per playout, mirror `MCTSEngine::simulate`
(`mcts.rs:182-210`): loop `check_winner` → `Player0Wins`=+1.0 /
`Player1Wins`=-1.0; if no legal moves, the current player is blocked and loses
(+1.0 if the blocked player is P1, -1.0 if P0); otherwise apply a uniformly
random legal move. Uses the engine's private `StdRng` (`seed` →
`StdRng::seed_from_u64`, else `from_entropy`) — reproducible, no global state.

## Constraints

- **No new dependencies** (`rand 0.8` already present). No changes to
  `mcts.rs`, `state.rs`, `symmetry.rs`, `moves.rs`, `game.rs`.
- Rust 2021, existing CI gates: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`.
- Tests live in `#[cfg(test)] mod tests` inside `beam_search.rs` (crate convention).
- Doc comments (`///`) on all public items; module-level `//!` overview.

## Test matrix (port of the 25 Python tests; see `test_beam_search.py` for exact scenarios)

| Concern | Python test(s) | Rust adaptation |
|---|---|---|
| Config defaults/custom/validation (×4 invalid) | `test_default_config` … `test_invalid_rollouts_per_candidate` | `new()` returns `Err` for each invalid field |
| Immediate win at depth 1 | `test_immediate_win_found` | fixture `(0,0,0),(1,1,1),(0,2,2)` → P1 to move; winning reply is shape 3 in row 0; `best_leaf` terminal, depth 1, value −1.0, PV = that move |
| Full-game reachability | `test_full_game_reachability` | empty board, width 4, seeded → `reached_terminal`; every terminal PV replays legally via `apply_move` chain to a terminal/blocked state |
| Symmetry dedup | `test_symmetry_dedup_depth_one` | empty board, `max_depth=1`: `candidates_generated==64`, post-dedup candidates == 3 |
| Memory bound | `test_memory_bound` | `nodes_inserted ≤ beam_width × max_depth`; compare width 2 vs 16 |
| Determinism | `test_determinism_same_seed`, `…different_seed_may_differ` | same seed ⇒ identical `BeamSearchResult` (derive `PartialEq` where needed or compare fields) |
| Pluggable evaluator + clamping | `test_pluggable_evaluator_is_used`, `test_evaluator_clamping` | closure with call counter (`Rc<Cell<u64>>`); evaluator returning 5.0 clamps to 1.0 |
| Adversarial pruning (mutation-killer) | `test_adversarial_perspective_p1_winning_reply`, `test_pruning_uses_mover_relative_score` | **must port both**; flipping the mover sign must fail the test |
| Root errors | `test_root_already_terminal_raises`, `test_root_no_legal_moves_raises` | `search` returns `Err` |
| Stalemate frontier entry | `test_stalemate_frontier_entry_marked_terminal` | blocked mover recorded as terminal loss |
| Result semantics | `test_beam_leaf_fields`, `test_best_leaf_prefers_root_player_perspective`, `test_best_leaf_none_only_when_no_leaves`, `test_get_statistics` | field assertions on `BeamLeaf` / stats |

(Shared-tree tests `test_shared_tree_*` are **not ported** — no tree in Rust.)

## Out of scope

- `CompactGameTree` port / sharing state with `MCTSEngine`.
- Parallel evaluation (rayon), persistence, opening-book integration, bin target.
- Any `mcts.rs` refactor to share the rollout (duplicate ~20 lines instead).
