# Cross-Engine Benchmark Port (Python → Rust) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the bug fixes and features from the Python worktree
`~/Code/quantik-ns/quantik/quantik-core-py/.claude/worktrees/cross-engine-benchmark-24/`
into `quantik-core-rust`, culminating in a `cross_engine_benchmark` binary whose
`run`/`report` subcommands reproduce the Python harness on the shared
`benchmarks/positions-v1.json` dataset — plus checkpoint/resume, compact
observation storage, and opening-book knowledge persistence that the Python
version lacks.

**Architecture:** Everything stays in the single `quantik-core` crate,
following the existing repo pattern. New engine modules (`evaluation`,
`minimax`, `beam_search`) sit beside `mcts`; the benchmark harness is a
`bench` module tree mirroring the Python `benchmarks/` package one file per
responsibility; the CLI is `src/bin/cross_engine_benchmark.rs`. Cross-language
portability rides on the already-matching 18-byte `pack`/`canonical_key`
format (VERSION=1, FLAG_CANON=2, `<BB8H>` little-endian) and on JSON artifacts
that are byte-compatible with the Python schema v1 (sha256 over sorted-key,
compact-separator JSON — serde_json's default BTreeMap ordering matches
Python's `sort_keys=True`).

**Tech Stack:** Rust 2021, existing deps (clap 4, rand 0.8, rusqlite 0.31,
serde/serde_json) plus `sha2` (checksums) and `chrono` (timestamps).

**Reference sources (authoritative spec — absolute paths):**
`PY=/Users/mauroberlanda/Code/quantik-ns/quantik/quantik-core-py/.claude/worktrees/cross-engine-benchmark-24`

- `$PY/src/quantik_core/evaluation.py` — feature vector + weights
- `$PY/src/quantik_core/minimax.py` — negamax/TT/dedup/ID semantics (docstrings explain every invariant)
- `$PY/src/quantik_core/mcts.py` — UCB perspective fix, time limit
- `$PY/src/quantik_core/beam_search.py` — beam engine incl. schedules/multiplicity
- `$PY/benchmarks/*.py` — dataset, reference, adapters, agreement, metrics, correctness, stability, head_to_head, bundle, report
- `$PY/examples/cross_engine_benchmark.py` — CLI flags & wiring
- `$PY/docs/BENCHMARKS.md` — methodology doc to adapt
- `$PY/benchmarks/positions-v1.json` — shared dataset artifact (copy into this repo)
- `$PY/CHANGELOG.md` Unreleased section — the feature/bugfix inventory being ported

## Progress Ledger (updated as tasks merge)

| Task | PR | Status |
|---|---|---|
| 1 evaluation module | #3 | MERGED |
| 2 minimax engine | #4 | MERGED |
| 3 MCTS fixes (UCB perspective, time limit, TT flag) | #5 | MERGED |
| 4 beam search engine | #6 | MERGED |
| 5 bench foundations (metrics, dataset, canonical JSON) | #7 | MERGED |
| 6 reference solver + adapters + preflight | #8 | MERGED |
| 7 aggregation + h2h + bundle + report + CLI | #9 | MERGED |
| 8 checkpoint/resume | #10 | MERGED |
| 9 opening-book persistence | #11 | IN REVIEW: PR #11 open |
| 10 changelog/docs + full benchmark run | — | TODO |

## Delegation Protocol (subagent-driven from Task 7 onward)

Implementation is delegated to **Sonnet** subagents; each finished PR is
reviewed by an **Opus** subagent before merge. Orchestrator merges.

**Implementation subagent contract** (one task per agent):
1. Work in `/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust` on the
   task's feature branch (branch from up-to-date `main` unless told the
   branch exists).
2. Follow this plan's task section EXACTLY; the Python sources under
   `$PY` (see header) are the authoritative semantics. Do not redesign.
3. Quality gates before committing — all must pass locally:
   `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets
   --all-features -- -D warnings`, `cargo test --workspace`,
   `cargo build --workspace --all-targets --release`. If Cargo.toml
   changed, run a plain `cargo build` first so Cargo.lock updates, and
   commit the lock file.
4. Keep debug-mode test runtime under ~30s total: anchor exact-solve tests
   on deep (10+ piece) positions, never shallow full solves (see Tasks 2/6
   for the established `random_position(seed, plies)` test helper pattern).
5. Commit message: conventional-commit style summary + short body + the
   two trailers used by prior commits (Co-Authored-By: Claude Fable 5
   <noreply@anthropic.com>, Claude-Session link — copy from `git log`).
6. Push branch, open PR with `gh pr create` (body ends with the standard
   generated-with footer — copy style from `gh pr view 8`). Report back:
   PR number, what was implemented, gate results, and any deviation from
   the plan with its reason.
7. NEVER merge. NEVER force-push. NEVER commit directly to main.

**Review subagent contract** (Opus, one per PR):
- Input: PR number. Review the full diff (`gh pr diff N`) against the plan
  task and the Python reference semantics; verify tests actually pin the
  ported invariants (perspective signs, dedup/multiplicity accounting,
  schema field names, checksum compatibility). Report findings as a list:
  CRITICAL (must fix before merge) / MINOR (note only), each with
  file:line and a concrete failure scenario. No findings = say so.
- Findings are applied by a follow-up Sonnet fix agent on the same branch.

**Orchestrator**: dispatch implementation → review → (fixes → re-review if
CRITICAL) → wait for CI green → `gh pr merge N --merge` → update the
Progress Ledger → next task.

## Global Constraints

- CI must pass per PR: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo test --workspace --all-targets --all-features --locked`, `cargo build --workspace --all-targets --release --locked`.
- One PR per task group below; branch from up-to-date `main`; merge with `gh pr merge --merge --delete-branch` after CI is green (user pre-authorized autonomous merge).
- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` + Claude-Session link.
- Binary state format stays 18 bytes `<BB8H>`, VERSION=1, FLAG_CANON=2 — never change.
- Dataset/bundle JSON schema_version stays 1 and must interoperate with the Python artifacts (golden checksum test against the copied `positions-v1.json`).
- Value conventions: terminal value +1.0 = P0 win, -1.0 = P1 win; "no legal moves ⇒ mover loses" everywhere (Python fix #27 already in Rust `game.rs`).
- Move key string format `player:shape:position` (e.g. `0:2:5`) exactly as `benchmarks/reference.py:move_key`.
- Autonomy: if API usage approaches limits, schedule a wakeup and resume; record progress in this file's checkboxes after each task.

---

### Task 1 (PR 1): Evaluation module

**Files:**
- Create: `crates/quantik-core/src/evaluation.rs`
- Modify: `crates/quantik-core/src/lib.rs` (add `pub mod evaluation;`)

**Interfaces (Produces):**
```rust
pub struct EvalConfig { pub weights: [f64; 6], pub win: f64 } // Default: [100.0,-100.0,20.0,3.0,2.0,0.0], win=10_000.0
pub const FEATURE_NAMES: [&str; 6]; // threat_own, threat_opp, threat_shared, mobility_diff, build_two, build_one
pub fn features(bb: &Bitboard, player: u8) -> [f64; 6];
pub fn evaluate(bb: &Bitboard, player: u8, cfg: &EvalConfig) -> f64; // dot product
pub fn count_legal_moves(bb: &Bitboard, player: u8) -> usize; // 0 when not player's turn
```

Port `$PY/src/quantik_core/evaluation.py` `features()` exactly: dead-line
skip (`len(present) < occupied`), 3-line threats with per-side completability
(`counts[side][missing_shape] < 2` and placement-legality on the empty cell),
signed shared-threat/build features (`sign = +1` if side-to-move == player
else `-1`), `mobility_diff = count_legal_moves(player) - count_legal_moves(1-player)`.
Note Rust's `generate_legal_moves` only generates for the current player, so
add a player-parameterized variant or reuse `is_move_legal` in a 4×16 loop.

- [x] Steps: write failing tests → implement → pass → commit `feat: add fitted-linear evaluation module`.
  Tests (translate from behavior, concrete): empty board features are all 0
  except mobility_diff 0; a board with a live 3-line completable only by the
  mover yields threat_own=1 for the mover and threat_opp=1 from the other
  perspective; evaluate(empty) == 0.0; `features` never mutates `bb` (Copy —
  compile-time, assert equality anyway); seeded weights dot product matches a
  hand-computed value on one midgame QFEN.

### Task 2 (PR 2): Minimax engine (exact solver)

**Files:**
- Create: `crates/quantik-core/src/minimax.rs`
- Modify: `crates/quantik-core/src/lib.rs`

**Interfaces (Produces):**
```rust
pub struct MinimaxConfig { pub max_depth: u32 /*=16*/, pub time_limit_s: Option<f64>,
    pub use_alpha_beta: bool /*=true*/, pub use_transposition_table: bool /*=true*/,
    pub dedup_children: bool /*=true*/, pub eval_config: EvalConfig, pub random_seed: Option<u64> }
pub struct MinimaxResult { pub best_move: Move, pub score: f64, pub depth_reached: u32,
    pub nodes: u64, pub pv: Vec<Move>, pub elapsed: f64 }
impl MinimaxEngine { pub fn new(config: MinimaxConfig) -> Self;
    pub fn search(&mut self, state: &State) -> Result<MinimaxResult, String>; // Err on no legal moves
    pub fn solve(&mut self, state: &State) -> Result<MinimaxResult, String>; } // max_depth=16, no time limit
```

Port `$PY/src/quantik_core/minimax.py` faithfully — its module docstring and
inline comments are the spec. Critical invariants:
- Negamax, terminal value `-(win - ply)` for the side to move at a lost node.
- Iterative deepening 1..=max_depth, PV hint floats previous best root move to front, keep deepest completed iteration; depth-1 always completes (check time only every 1024 nodes: `nodes & 0x3FF == 0`).
- Root children searched with FULL (-inf, inf) window; equal-value candidates collected; seeded RNG picks among ties, else first.
- TT keyed on `canonical_key()` `[u8;18]`, entry `(depth, value, Bound{Exact,Lower,Upper})`; bound classification against ORIGINAL window; Lower/Upper reuse only when `use_alpha_beta`.
- Sibling dedup by child canonical key, threading the computed key into the recursive TT probe.
- Child ordering: sort by `(shape, position)`, then stable-partition immediate winning replies first.

- [x] Steps: failing tests → implement → pass → commit `feat: add alpha-beta minimax engine with canonical-key transposition table`.
  Tests: (a) engine finds an immediate winning move (build a 3-line position
  where one move completes A,B,C,D — assert best_move completes it and score
  ≈ win-1); (b) `pv[0] == best_move` and PV length ≤ depth_reached;
  (c) alpha-beta result score equals plain minimax (`use_alpha_beta=false`)
  at depth 3 from a fixed midgame QFEN; (d) `solve` on a 12+-piece endgame
  position returns `depth_reached >= remaining plies` and a legal move;
  (e) `time_limit_s=0.001` from empty board still returns a depth-1 result;
  (f) deterministic without seed: two runs give identical best_move.

### Task 3 (PR 3): MCTS bugfixes + wall-clock limit

**Files:**
- Modify: `crates/quantik-core/src/mcts.rs`

Port the Python fixes to the Rust engine:
1. **UCB perspective bug (mcts.rs:121):** selection must use the win count of
   the player choosing at the PARENT (`current_player(&node.bb)`), not always
   `win_count_p0`. Same fix in `best_move()`: report win rate for the root's
   mover as `win_probability` (keep return contract `(Move, f64)` but the f64
   is the root mover's win rate — matches Python adapter usage).
2. **`time_limit_s: Option<f64>`** in `MCTSConfig`: deadline checked after each
   completed iteration; validate positive & finite in `new` (panic or Err —
   use `assert!` like existing style). Also expose `iterations_performed` as a
   real counter (root visit_count already equals it — keep but add counter).
3. **Canonical transposition option** `use_transposition_table: bool` (default
   true): maintain `HashMap<[u8;18], usize>`; in `expand`, if the child's
   canonical key already maps to a node, link it as child (dedup) instead of
   allocating; `false` always allocates fresh (the Python fix made the flag
   actually consulted). Node gains multiple parents — backpropagate along the
   selection path, not `parent` pointers: track the descent path in `select`
   and pass it to `backpropagate` (this also fixes correctness with dedup).

- [x] Steps: failing tests → fix → pass → commit `fix: MCTS UCB perspective, honor transposition flag, add wall-clock limit`.
  Tests: (a) *perspective regression*: position where P1 to move has an
  immediate winning reply — with seed + enough iterations the engine must pick
  it (fails on old p0-perspective code); (b) `time_limit_s: Some(0.05)`, huge
  max_iterations: search returns in < 1s wall and iterations < max;
  (c) same seed twice ⇒ same move; (d) `use_transposition_table=false` creates
  ≥ nodes than `true` from empty board with same seed/iterations.

### Task 4 (PR 4): Beam search engine

**Files:**
- Create: `crates/quantik-core/src/beam_search.rs`
- Modify: `crates/quantik-core/src/lib.rs`

**Interfaces (Produces):**
```rust
pub struct BeamSearchConfig { pub beam_width: usize /*=64*/, pub max_depth: u32 /*=16*/,
    pub rollouts_per_candidate: u32 /*=8*/, pub random_seed: Option<u64>,
    pub beam_schedule: Option<Vec<usize>>, pub rollout_schedule: Option<Vec<u32>>,
    pub time_limit_s: Option<f64> }
pub struct BeamLeaf { pub moves: Vec<Move>, pub value: f64, pub depth: u32,
    pub is_terminal: bool, pub multiplicity: u64 }
pub struct RankedRootMove { pub mv: Move, pub best_value: f64, pub mean_value: f64,
    pub win_probability: f64, pub leaf_count: usize, pub total_multiplicity: u64,
    pub has_terminal_win: bool }
pub struct BeamStats { pub candidates_generated: u64, pub candidates_deduped: u64,
    pub nodes_inserted: u64, pub nodes_pruned: u64, pub evaluations: u64, pub rollouts: u64 }
pub struct BeamSearchResult { pub best_leaf: Option<BeamLeaf>, pub terminal_leaves: Vec<BeamLeaf>,
    pub reached_terminal: bool, pub max_depth_reached: u32, pub stats: BeamStats,
    pub root_player: u8, pub frontier_leaves: Vec<BeamLeaf> }
impl BeamSearchResult { pub fn ranked_root_moves(&self, top_k: Option<usize>) -> Vec<RankedRootMove>; }
pub const UNIQUE_CANONICAL_STATES_PER_DEPTH: [(u32, u64); 8]; // (1,3)…(8,17_900_160)
impl BeamSearchEngine { pub fn new(config: BeamSearchConfig) -> Result<Self, String>;
    pub fn search(&mut self, state: &State) -> Result<BeamSearchResult, String>; }
```

Port `$PY/src/quantik_core/beam_search.py` semantics exactly, minus the
CompactGameTree coupling (no shared tree in Rust — stats `nodes_inserted`
counts frontier/terminal insertions, i.e. unique candidates kept + terminal
children recorded, matching the observable Python counters as closely as the
structure allows; document any divergence in the module doc):
- Level-by-level; candidates dedup by canonical key with **multiplicity
  accumulation** (first occurrence keeps PV/parent; weight adds).
- Terminals always recorded regardless of width; "mover has no legal moves ⇒
  other player wins" both at frontier-expansion (leaf depth = depth-1) and
  child level (leaf depth = depth).
- Scoring: value from P0 perspective in [-1,1] via mean of `rollouts` uniform
  playouts (rollout resolves fully, no depth cap); prune keeps top
  `beam_width` by `(mover==0 ? v : -v)` desc with insertion-index tiebreak.
- Schedules index `min(depth-1, len-1)`; time limit checked between levels,
  depth 1 always completes. Config validation mirrors Python ValueErrors.
- `ranked_root_moves`: group by first move, root-player perspective values,
  multiplicity-weighted mean, `win_probability=(mean+1)/2`, sort by
  best desc, mean desc, leaf_count desc, then (player,shape,position) asc.

- [x] Steps: failing tests → implement → pass → commit `feat: add beam search engine with schedules and multiplicity accounting`.
  Tests: (a) full-width search from empty board with max_depth=16 and
  beam_width 512 reaches `reached_terminal` (frontier empties) and every
  terminal leaf has value ±1; (b) tactical: a position with an immediate
  winning move yields `best_leaf.moves[0]` = that move and
  `ranked_root_moves()[0].has_terminal_win`; (c) beam_schedule `[3, 51, 4]`:
  level widths honored (`stats.nodes_pruned > 0` only where expected);
  (d) rollout_schedule `[1, 8]` makes `stats.rollouts` exactly
  `evaluations_at_depth1*1 + deeper_evaluations*8`; (e) same seed ⇒ identical
  result; (f) root with P1 to move: `root_player == 1` and ranked values are
  from P1's perspective (winning terminal for P1 ranks first);
  (g) invalid configs (width 0, empty schedule, max_depth 0/17, nonfinite
  time limit) return Err.

### Task 5 (PR 5): Benchmark foundations — metrics + dataset artifact I/O

**Files:**
- Create: `crates/quantik-core/src/bench/mod.rs` (`pub mod metrics; pub mod dataset;` …grows later)
- Create: `crates/quantik-core/src/bench/metrics.rs`
- Create: `crates/quantik-core/src/bench/dataset.rs`
- Create: `benchmarks/positions-v1.json` (copy verbatim from `$PY/benchmarks/positions-v1.json`)
- Modify: `crates/quantik-core/src/lib.rs` (`pub mod bench;`), `crates/quantik-core/Cargo.toml` (add `sha2 = "0.10"`), `.gitignore` (`benchmarks/results/`)

**Interfaces (Produces):**
```rust
// metrics.rs — exact ports of $PY/benchmarks/metrics.py
pub fn wilson_ci(hits: u64, n: u64) -> (f64, f64); // z=1.96
pub fn mean_std(xs: &[f64]) -> (f64, f64);         // sample std, 0 for n<2
pub fn percentile(xs: &[f64], p: f64) -> f64;      // linear interpolation
pub fn median(xs: &[f64]) -> f64;

// dataset.rs — schema_version 1, generator "benchmarks.dataset.generate/v1"
pub const PHASES: [(&str, (u32, u32)); 4]; // opening 0-4, early_mid 5-7, late_mid 8-11, endgame 12-16
pub fn phase_of(pieces: u32) -> &'static str;
pub fn checksum(payload: &serde_json::Value) -> String; // sha256 over sorted-key compact JSON, checksum field stripped
pub fn save(payload: &mut serde_json::Value, path: &Path) -> io::Result<String>; // injects checksum, pretty(indent 2, sorted)+\n
pub fn load(path: &Path) -> Result<serde_json::Value, String>; // verifies checksum
pub fn generate(requested: &BTreeMap<String, u32>, seed: u64) -> serde_json::Value;
```

Dataset generation ports `$PY/benchmarks/dataset.py`: random playouts to a
target ply within the phase range (`rng.gen_range(lo..=min(hi,15))`), reject
lines that hit a win or dead-end, global dedup by canonical key, per-position
payload `{id:"pNNNN", qfen, phase, pieces, side_to_move, legal_moves, reference:null}`,
attempts cap `want*500`. RNG streams differ from CPython — document that
regenerated datasets differ across languages but artifacts interoperate; the
committed artifact is the shared one.

**Checksum compatibility is the linchpin:** serde_json `Map` (default feature
set — NOT `preserve_order`) is a BTreeMap, so `serde_json::to_string(&stripped)`
produces sorted-key, `","`/`":"`-separated JSON like Python's
`json.dumps(sort_keys=True, separators=(",",":"))`. Floats: Python emits
shortest-roundtrip repr; serde_json (ryu) does too. Verify with the golden test.

- [x] Steps: failing tests → implement → pass → commit `feat: add benchmark metrics and shared dataset artifact I/O`.
  Tests: (a) `wilson_ci(8,10)` ≈ (0.4901, 0.9433) tol 1e-3 (compute reference
  with Python once); (b) `percentile(&[1,2,3,4], 95.0) == 3.85`; `median` of
  odd/even lists; `mean_std(&[1,2,3]) == (2.0, 1.0)`; (c) **golden**:
  `load("benchmarks/positions-v1.json")` succeeds — recomputed checksum equals
  the stored one; every position's qfen parses, is non-terminal, and
  `side_to_move`/`pieces`/`legal_moves` match recomputation; (d) save→load
  roundtrip on a tiny generated payload; corrupting one byte fails load;
  (e) `generate` with seed twice ⇒ identical JSON; phases respect bounds and
  canonical dedup.

### Task 6 (PR 6): Reference solver, adapters, correctness preflight

**Files:**
- Create: `crates/quantik-core/src/bench/reference.rs`
- Create: `crates/quantik-core/src/bench/adapters.rs`
- Create: `crates/quantik-core/src/bench/correctness.rs`
- Modify: `crates/quantik-core/src/bench/mod.rs`, `crates/quantik-core/Cargo.toml` (add `chrono = "0.4"` if not yet)

**Interfaces (Produces):**
```rust
// reference.rs — port of $PY/benchmarks/reference.py
pub fn move_key(mv: &Move) -> String;                    // "p:s:pos"
pub fn parse_move_key(key: &str) -> Result<(u8,u8,u8), String>;
pub fn solve_position(bb: &Bitboard, budget_s: f64) -> Option<serde_json::Value>;
pub fn augment_with_references(payload: &mut serde_json::Value, budget_s: f64); // skips "opening"

// adapters.rs — port of $PY/benchmarks/adapters.py
pub struct MoveObservation { pub engine: String, pub config_label: String, pub position_id: String,
    pub mv: String, pub wall_time_s: f64, pub cpu_time_s: f64, pub root_legal_moves: usize,
    pub exact: bool, pub seed: Option<u64>, pub nodes: Option<u64>, pub iterations: Option<u64>,
    pub depth_reached: Option<u32>, pub score: Option<f64>, pub peak_memory_bytes: Option<u64>,
    pub extra: BTreeMap<String, f64> } // impl to_json() matching Python asdict field names ("move" not "mv")
pub trait EngineAdapter { fn name(&self) -> &'static str; fn stochastic(&self) -> bool;
    fn config_label(&self) -> String;
    fn select_raw(&self, bb: &Bitboard, seed: Option<u64>) -> Result<(Move, RawMetrics), String>; }
pub fn select(adapter: &dyn EngineAdapter, bb: &Bitboard, position_id: &str, seed: Option<u64>)
    -> Result<(Move, MoveObservation), String>; // times, validates legality+terminality
pub struct MinimaxAdapter { pub max_depth: u32, pub time_limit_s: Option<f64> }
pub struct MCTSAdapter   { pub max_iterations: u32, pub max_depth: u32, pub exploration_weight: f64, pub time_limit_s: Option<f64> }
pub struct BeamAdapter   { pub beam_width: usize, pub max_depth: u32, pub time_limit_s: Option<f64> }
pub struct RandomAdapter;
pub fn fixed_time_adapters(time_limit_s: f64, beam_width: usize) -> Vec<Box<dyn EngineAdapter>>;
// labels identical to Python _label(): e.g. "minimax(d=16,t=1.0)", "mcts(it=10000000,d=16,t=1.0)", "beam(w=256,d=16,t=1.0)", "random"

// correctness.rs — port of $PY/benchmarks/correctness.py
pub fn run_preflight(adapters: &[Box<dyn EngineAdapter>], positions: &[serde_json::Value]) -> Vec<String>;
```

`solve_position` semantics: per root child, immediate win/no-reply ⇒ score
inf sentinel (use f64::INFINITY internally; JSON `value` is only ±1);
otherwise child solved by `MinimaxEngine::search` with `max_depth=16` and the
remaining budget as time limit — exact only if `depth_reached >= remaining
plies of child`; any cutoff ⇒ whole position unsolved (None). Reference JSON
fields exactly: `solved,no_cutoff,value,optimal_moves(sorted),pv,nodes,
solve_time_s(round 6),solver` (solver string:
`"MinimaxEngine(max_depth=16, budget_s={b}) quantik-core-rust {CARGO_PKG_VERSION}"`).
Minimax adapter exactness: `depth_reached >= 16 - pieces`. Beam adapter move
fallback: best_leaf's first move, else top ranked root move; score negated
for root_player 1. cpu_time via `std::time::Instant` twice is wrong — use
wall for both and document (or `libc::clock_gettime(CLOCK_PROCESS_CPUTIME_ID)`
via a tiny unsafe wrapper; prefer the latter, it's one call).
Preflight: every dataset position non-terminal; per adapter on first 3
positions: legal move for the right side, input not mutated (guaranteed by
`&Bitboard` — still assert), same seed ⇒ same move twice; minimax PV-head
consistency is validated inside its adapter like Python.

- [x] Steps: failing tests → implement → pass → commit `feat: add exact reference solver and uniform engine adapters with preflight`.
  Tests: (a) solve_position on a position with an immediate winning move:
  solved, value 1, that move in optimal_moves, pv[0]==its key; (b) on a
  losing-everywhere position (choose a late endgame where side to move loses):
  value -1 and every move listed only if scoring max; (c) budget 0.000001 on
  a 5-piece position ⇒ None; (d) each adapter returns legal moves and
  identical move for identical seed; labels match the exact strings above;
  (e) run_preflight on the golden dataset with cheap adapters
  (minimax d=2, mcts it=50, beam w=8 d=4, random) returns [].

### Task 7 (PR 7): Aggregation, head-to-head, bundle, report, CLI — parity milestone

**Files:**
- Create: `crates/quantik-core/src/bench/agreement.rs` (run_agreement, aggregate_agreement, aggregate_cost)
- Create: `crates/quantik-core/src/bench/stability.rs`
- Create: `crates/quantik-core/src/bench/head_to_head.rs`
- Create: `crates/quantik-core/src/bench/bundle.rs`
- Create: `crates/quantik-core/src/bench/report.rs`
- Create: `crates/quantik-core/src/bin/cross_engine_benchmark.rs`
- Create: `docs/BENCHMARKS.md` (adapt `$PY/docs/BENCHMARKS.md`, commands become `cargo run --release --bin cross_engine_benchmark -- …`)
- Modify: `crates/quantik-core/src/bench/mod.rs`

Port each Python module 1:1 (they are small, pure functions over row dicts —
use `serde_json::Value` rows to keep the JSON shape identical):
- `run_agreement`: rows = observation dict + `phase` + `hit` (null when no
  reference); stochastic adapters run all seeds, deterministic only seeds[0].
- `aggregate_agreement`: group (engine, config_label, phase), wilson CI.
- `aggregate_cost`: group (engine, config_label): median/p95 wall time,
  median nodes, max peak memory.
- `aggregate_stability`: modal-move consistency per position, per-seed
  agreement mean/std.
- `head_to_head`: `play_game` (mover = side already to move; winner by
  "cannot move / line completed ⇒ previous mover wins", returns (winner name,
  plies)); `run_head_to_head` both orientations per position×seed;
  `aggregate_head_to_head` with by_phase splits and `draws: 0`.
- `bundle`: schema_version 1, started_at `%Y-%m-%dT%H:%M:%S%z`, environment
  (quantik_core_version from CARGO_PKG_VERSION, git_sha via `git rev-parse
  HEAD`, rust_version instead of python_version — keep BOTH keys:
  `"python_version": null` omitted, add `"rust_version"`; report handles it),
  config (all CLI args + engine_seeds), dataset summary, observations,
  head_to_head, aggregates. save creates parent dirs, pretty sorted JSON + \n.
- `report::render_markdown`: exact table set and prose from
  `$PY/benchmarks/report.py` (environment line says `quantik-core-rust
  {version}, rust {rustc}` instead of python).
- CLI: clap derive, subcommands `dataset` / `run` / `report` with the exact
  flags and defaults of `$PY/examples/cross_engine_benchmark.py` (`--family
  fixed|native`, `--time-limit 1.0`, `--seeds 10`, `--seed-base 0`,
  `--minimax-depth 6`, `--minimax-time 0.2`, `--mcts-iterations 1500`,
  `--mcts-depth 16`, `--mcts-exploration 1.414`, `--beam-width 64`,
  `--beam-depth 16`, `--h2h-positions 8`, `--h2h-seeds 1`, `--skip-h2h`,
  `--output`; report `--input`, `--output` default `<input>.md`). H2H position
  pick: round-robin across sorted phases. Preflight failure ⇒ exit 1 with the
  failure list.

- [x] Steps: failing tests → implement → pass → commit `feat: add cross-engine benchmark aggregation, head-to-head, bundle, report, CLI`.
  NOTE (2026-07-12): implementation finished — build/fmt/clippy/test gates
  green, smoke CLI run verified end-to-end (252 observations, 24 games,
  5-section Markdown report), `docs/BENCHMARKS.md` written, fixed a real bug
  found during smoke testing (`rust_version` was rendering as an empty
  string because `CARGO_PKG_RUST_VERSION` is the crate's unset MSRV field,
  not the compiler in use — now shells out to `rustc --version` like
  `git_sha` shells out to `git rev-parse HEAD`). Cargo.lock is gitignored
  in this repo (never tracked) so it is not part of the commit, deviating
  from the literal task instruction to commit it.
  Tests: (a) agreement/stability/cost aggregations on handcrafted row fixtures
  reproduce hand-computed numbers (include a no-reference row ⇒ excluded);
  (b) play_game between RandomAdapter and RandomAdapter from a near-terminal
  position terminates and credits the correct winner; aggregate h2h counts
  sides correctly; (c) render_markdown on a minimal synthetic bundle contains
  all five `##` sections and a `| minimax` row; (d) **end-to-end smoke**
  (ignored-by-default `#[ignore]` if slow — but with 0.05s limits it's fast):
  `run` on the golden dataset with `--time-limit 0.05 --seeds 2
  --h2h-positions 2 --h2h-seeds 1` writes a bundle that `load`s, then
  `report` writes Markdown containing "Exact move agreement".
  Verify CLI equivalence manually: `cargo run --release --bin
  cross_engine_benchmark -- run --dataset benchmarks/positions-v1.json
  --family fixed --time-limit 1.0 --seeds 30 --h2h-positions 12 --h2h-seeds 10
  --output benchmarks/results/fixed-1s-seeds30.json` (full run reserved for
  after merge; smoke params during CI).

### Task 8 (PR 8): Checkpointing + compact observation storage (beyond Python)

**Files:**
- Create: `crates/quantik-core/src/bench/checkpoint.rs`
- Modify: `crates/quantik-core/src/bench/agreement.rs` (checkpoint hook in run_agreement), `head_to_head.rs` (same), `src/bin/cross_engine_benchmark.rs` (`--checkpoint <path>` and `--resume` flags on `run`)

Design (addresses "no checkpoints, verbose JSON, repeated trees"):
- Checkpoint file = JSON Lines: one line per completed observation row /
  h2h record, prefixed with a small header line
  `{"kind":"header","dataset_checksum":…,"config_fingerprint":…}` where
  config_fingerprint = sha256 of the canonical config JSON (excluding output
  paths). Appended after every row (`BufWriter` + flush per row) — crash-safe.
- `--resume`: on start, if checkpoint exists and header matches (checksum +
  fingerprint), previously completed (engine, config_label, position_id,
  seed) tuples and h2h (a, b, position_id, seed, orientation) tuples are
  skipped; rows are loaded into the run. Mismatched header ⇒ refuse with
  a clear error (never silently mix runs).
- Bundle gains `"resumed": bool`. Observations stay schema-identical.

- [x] Steps: failing tests → implement → pass → commit `feat: add crash-safe checkpoint/resume to benchmark runs`.
  NOTE (2026-07-12): implementation finished — `bench::checkpoint` (JSON
  Lines writer/loader, sha256 config fingerprint, truncated-tail
  tolerance), `--checkpoint`/`--resume` wired into the `run` CLI (on_row/
  on_record callbacks now return `Result` so a write failure aborts the
  run), `bundle["resumed"]` added, `docs/BENCHMARKS.md` "Checkpoint and
  resume" section written. fmt/clippy/test/release gates green (109
  workspace tests, ~2.6s). Smoke-verified manually: checkpoint file grew to
  277 lines (1 header + 252 observations + 24 h2h) during a
  `--time-limit 0.05 --seeds 2 --h2h-positions 2 --h2h-seeds 1` run; a
  `--resume` rerun after deleting only the output JSON reproduced 252
  observations / 24 games / 6 h2h aggregates with `"resumed": true` and
  left the checkpoint file unchanged (everything skipped); re-running
  without `--resume` against the same checkpoint path correctly refused
  (exit 1) instead of overwriting it.
  Tests: (a) run_agreement with a checkpoint writer, simulate interruption by
  running only 2 of 4 positions (truncate adapter list/position slice), then
  resume: completed tuples skipped, final row multiset equals an uninterrupted
  run (deterministic seeds); (b) mismatched fingerprint refuses resume;
  (c) checkpoint lines parse individually (JSONL, partial trailing line
  tolerated on read).

### Task 9 (PR 9): Opening-book knowledge persistence + cross-run reuse

**Files:**
- Create: `crates/quantik-core/src/bench/book_export.rs`
- Modify: `crates/quantik-core/src/bench/reference.rs` (optional book read-through), `src/bin/cross_engine_benchmark.rs` (`dataset --book <path>`, new `export-book` subcommand), `crates/quantik-core/src/opening_book.rs` (add `solved`/`game_value` columns via `ALTER TABLE`-safe migration: `solved INTEGER NOT NULL DEFAULT 0`, `game_value INTEGER` , plus `best_moves` reuse)

Design (addresses "does not persist knowledge in an opening book / portable
cross-language"):
- Every exactly-solved reference (value ±1 + complete optimal move set) is
  upserted into the existing SQLite book keyed by the 18-byte canonical key
  (already byte-identical to Python's): `evaluation = value as f64`,
  `best_moves` = optimal moves (shape,position), `is_terminal = Interior`,
  new `solved=1`, `game_value=±1`, `depth` = pieces placed.
- `solve_position` first probes the book (when `--book` given): a stored
  solved entry short-circuits the minimax solve — repeated dataset builds and
  reruns get faster instead of recomputing identical trees.
  **Caveat:** the book stores moves for the canonical representative;
  translating optimal moves back to the queried (non-canonical) orientation
  requires the symmetry transform. If `SymmetryHandler` doesn't expose the
  transform index, store optimal moves per-orientation is wrong — instead
  short-circuit ONLY when the queried state IS its canonical representative
  (`state.canonical_payload() == state.bb.to_le_bytes()`), else fall through
  to solving. Document this and leave move-translation as a follow-up.
- `export-book --input <bundle-or-dataset> --db <path>`: bulk-export all
  solved references from an artifact into the book. SQLite file remains
  readable by the Python `opening_book.py` (same schema family).

- [x] Steps: failing tests → implement → pass → commit `feat: persist solved references into the opening book and reuse them across runs`.
  NOTE (2026-07-12): implemented — idempotent ALTER TABLE migration
  (`solved`/`game_value`), `add_solved_position`, `bench::book_export`
  (export_references + representative-only lookup_reference),
  `solve_position_with_book`/`augment_with_references_with_book` (old fns
  delegate with None), `dataset --book` + `export-book` CLI, BENCHMARKS.md
  section. Review fix: the representative-only restriction now guards
  WRITES too (export, write-back, and add_solved_position itself) — a
  non-representative row under the canonical key would serve wrong-
  orientation moves to a lookup on the representative board. Golden
  export therefore inserts the representative solved subset (1 of 22),
  rerun idempotent; regression test pinned (fails on pre-fix commit).
  Gates green (122 tests, ~7s debug).
  Tests: (a) solving a position with `--book` writes a row; solving again
  hits the book (assert via nodes==0 marker or a probe counter) and returns
  an identical reference; (b) export-book from the golden dataset inserts
  == number of solved positions rows, idempotent on second run;
  (c) migration: opening a pre-existing DB without the new columns upgrades
  it and old tests still pass.

### Task 10 (final): Full benchmark execution + CHANGELOG + docs sync

**Files:**
- Create: `CHANGELOG.md` (mirror the ported Unreleased entries from `$PY/CHANGELOG.md`, adapted)
- Modify: `README.md` (benchmark section pointing to docs/BENCHMARKS.md)

- [ ] Run the real benchmark overnight:
  `cargo run --release --bin cross_engine_benchmark -- run --dataset benchmarks/positions-v1.json --family fixed --time-limit 1.0 --seeds 30 --h2h-positions 12 --h2h-seeds 10 --checkpoint benchmarks/results/fixed-1s-seeds30.ckpt --output benchmarks/results/fixed-1s-seeds30.json`
  then `… report --input benchmarks/results/fixed-1s-seeds30.json`; attach the
  Markdown to the final PR description (results dir is gitignored).
- [ ] Commit `docs: add changelog and benchmark documentation`, final PR, merge.

## Self-Review Notes

- Spec coverage: CHANGELOG Unreleased items map to Tasks 1–7 (beam engine +
  result fields + schedules + multiplicity → T4; benchmark harness + time
  limits → T5–T7; MCTS TT flag + root-parity + demo formatting fixes → T3 /
  N-A in Rust where the structure differs — Rust derives player from the
  board, so the parity bug can't exist; demo scripts aren't ported). User's
  extra asks (checkpoints, compact/resumable runs, tree-knowledge persistence,
  cross-language portability) → T8–T9. The two exact user commands → T7/T10.
- Known divergences, all documented in code: RNG streams (datasets regenerate
  differently; the committed artifact is shared), cpu_time source,
  nodes_inserted semantics without a shared CompactGameTree, environment keys
  rust_version vs python_version.
- Types are consistent: `Move{player,shape,position}: u8`, canonical key
  `[u8;18]`, move key string `p:s:pos`, observation field names match the
  Python `asdict` output exactly (JSON: `move`, not `mv`).
