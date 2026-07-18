//! Classical alpha-beta minimax (negamax formulation) for Quantik.
//!
//! Searches the exact game tree using `has_winning_line` for terminal
//! detection and falls back to [`crate::evaluation::evaluate`] once
//! `max_depth` is exhausted. With `max_depth = 16` ([`MinimaxEngine::solve`])
//! the search always reaches true terminal states — no Quantik game exceeds
//! 16 plies — so it acts as an exact solver, not just a heuristic engine.
//!
//! Negamax sign convention: `negamax` returns the value of a position from
//! the perspective of the side to move *at that node*. A caller negates a
//! child's value to fold it back into its own perspective.
//!
//! Terminal values use `win - ply` (not a flat `win`) so that a forced mate
//! found sooner scores strictly higher than one found deeper.
//!
//! `State::canonical_key()` collapses the D4 × S4 = 192 board symmetries
//! *without* swapping colors, so the negamax value (always relative to the
//! side to move, not a fixed color) is safe to cache/dedup by that key.
//! Only the value/bound is ever cached — never the move — since the key
//! alone doesn't preserve which concrete move produced a given child.
//!
//! Sibling dedup (`dedup_children`) and the transposition table
//! (`use_transposition_table`) both key off `canonical_key()`. Where dedup
//! and the TT are both active on the same call, the child key already
//! computed by dedup is threaded into the recursive call so the TT probe
//! does not recompute it.

use crate::bitboard::Bitboard;
use crate::evaluation::{evaluate, EvalConfig};
use crate::game::{current_player, has_winning_line};
use crate::moves::{apply_move, generate_legal_moves, Move};
use crate::search_telemetry::{
    EngineKind, PolicyMassKind, RootMoveStat, SearchEventCounters, SearchTelemetry,
};
use crate::state::State;
use rand::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

/// Transposition-table bound kind for a stored negamax value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

/// Configuration for [`MinimaxEngine`].
#[derive(Clone, Debug)]
pub struct MinimaxConfig {
    pub max_depth: u32,
    pub time_limit_s: Option<f64>,
    pub use_alpha_beta: bool,
    pub use_transposition_table: bool,
    pub dedup_children: bool,
    pub eval_config: EvalConfig,
    pub random_seed: Option<u64>,
}

impl Default for MinimaxConfig {
    fn default() -> Self {
        Self {
            max_depth: 16,
            time_limit_s: None,
            use_alpha_beta: true,
            use_transposition_table: true,
            dedup_children: true,
            eval_config: EvalConfig::default(),
            random_seed: None,
        }
    }
}

/// Result of a [`MinimaxEngine::search`] (or `.solve`) call.
#[derive(Clone, Debug)]
pub struct MinimaxResult {
    pub best_move: Move,
    pub score: f64,
    pub depth_reached: u32,
    pub nodes: u64,
    pub pv: Vec<Move>,
    pub elapsed: f64,
}

type TTEntry = (u32, f64, Bound);

/// A legal move paired with the bitboard it produces and — when dedup
/// computed one — that child's canonical key.
type ChildEntry = (Move, Bitboard, Option<[u8; 18]>);

/// Internal signal that the configured time limit was reached.
struct TimeUp;

fn move_sort_key(mv: &Move) -> (u8, u8) {
    (mv.shape, mv.position)
}

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

/// Alpha-beta negamax search engine over the exact Quantik game tree.
pub struct MinimaxEngine {
    pub config: MinimaxConfig,
    tt: HashMap<[u8; 18], TTEntry>,
    nodes: u64,
    deadline: Option<Instant>,
    rng: Option<StdRng>,
    pv_hint: Vec<Move>,
    counters: SearchEventCounters,
    /// Root moves paired with their negamax score, from the deepest
    /// completed `search_root` iteration. Empty until a search completes;
    /// used both as the telemetry root-move source and as the "has a
    /// search run" sentinel for [`Self::telemetry`].
    last_root_scored: Vec<(Move, f64)>,
    last_pv: Vec<Move>,
    last_elapsed_ms: u64,
    last_depth: u32,
    last_root_value: f64,
}

impl MinimaxEngine {
    pub fn new(config: MinimaxConfig) -> Self {
        let rng = config.random_seed.map(StdRng::seed_from_u64);
        Self {
            config,
            tt: HashMap::new(),
            nodes: 0,
            deadline: None,
            rng,
            pv_hint: Vec::new(),
            counters: SearchEventCounters::default(),
            last_root_scored: Vec::new(),
            last_pv: Vec::new(),
            last_elapsed_ms: 0,
            last_depth: 0,
            last_root_value: 0.0,
        }
    }

    /// Exact solve: [`Self::search`] with `max_depth = 16` and no time limit.
    ///
    /// Every Quantik game resolves (win or no-legal-moves) within 16 plies,
    /// so a depth-16 search from any reachable position always terminates on
    /// true terminal nodes rather than the heuristic eval cutoff.
    pub fn solve(&mut self, state: &State) -> Result<MinimaxResult, String> {
        let original = self.config.clone();
        self.config.max_depth = 16;
        self.config.time_limit_s = None;
        let result = self.search(state);
        self.config = original;
        result
    }

    /// Iterative-deepening alpha-beta negamax search from `state`.
    ///
    /// Deepens from depth 1 to `config.max_depth` (or until
    /// `config.time_limit_s` elapses), seeding each iteration's root move
    /// order with the previous iteration's principal variation. Returns the
    /// deepest iteration that completed before any time limit; the depth-1
    /// iteration always runs to completion.
    pub fn search(&mut self, state: &State) -> Result<MinimaxResult, String> {
        let start = Instant::now();
        self.nodes = 0;
        self.counters = SearchEventCounters::default();
        self.last_root_scored = Vec::new();
        self.last_pv = Vec::new();
        self.last_elapsed_ms = 0;
        self.last_depth = 0;
        self.last_root_value = 0.0;
        self.tt.clear();
        self.pv_hint.clear();
        self.deadline = self
            .config
            .time_limit_s
            .map(|s| start + std::time::Duration::from_secs_f64(s));

        let bb = state.bb;
        let root_moves = generate_legal_moves(&bb);
        if root_moves.is_empty() {
            return Err("Cannot search from a state with no legal moves.".into());
        }
        // The root's successor set is computed once here and reused across every
        // iterative-deepening pass, so count the root expansion exactly once
        // (not per depth). Interior nodes regenerate their moves on each visit
        // and are counted in `negamax`.
        self.counters.expanded_nodes += 1;

        let mut result: Option<MinimaxResult> = None;
        for depth in 1..=self.config.max_depth {
            match self.search_root(&bb, &root_moves, depth) {
                Ok((score, best_move, pv)) => {
                    self.pv_hint = pv.clone();
                    self.last_pv = pv.clone();
                    self.last_depth = depth;
                    self.last_root_value = score;
                    result = Some(MinimaxResult {
                        best_move,
                        score,
                        depth_reached: depth,
                        nodes: self.nodes,
                        pv,
                        elapsed: start.elapsed().as_secs_f64(),
                    });
                    if let Some(deadline) = self.deadline {
                        if Instant::now() >= deadline {
                            break;
                        }
                    }
                }
                Err(TimeUp) => break,
            }
        }

        // The depth-1 iteration is cheap enough that the first time check
        // (every 1024 nodes) cannot fire before it completes on any
        // reachable position; mirror the Python assertion.
        let mut result = result.expect("depth-1 iteration always completes");
        result.elapsed = start.elapsed().as_secs_f64();
        // Derive from the same (already-final) elapsed value written into
        // the returned result, so telemetry's elapsed_ms always matches
        // exactly what the caller sees on `result.elapsed`.
        self.last_elapsed_ms = (result.elapsed * 1000.0).round() as u64;
        Ok(result)
    }

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

    // ----- internals -----------------------------------------------------

    /// Sort root moves by the deterministic tie-break, then float the prior
    /// iteration's PV move (if any) to the front for better alpha-beta
    /// cutoffs on the next, deeper iteration.
    fn order_root_moves(&self, moves: &[Move]) -> Vec<Move> {
        let mut ordered: Vec<Move> = moves.to_vec();
        ordered.sort_by_key(move_sort_key);
        if let Some(pv_move) = self.pv_hint.first() {
            if let Some(i) = ordered.iter().position(|m| m == pv_move) {
                let mv = ordered.remove(i);
                ordered.insert(0, mv);
            }
        }
        ordered
    }

    /// Apply each move in `moves` (assumed pre-sorted), pairing it with the
    /// resulting bitboard and (when `dedup`) that child's canonical key.
    ///
    /// When `dedup`, siblings whose resulting state shares a
    /// `State::canonical_key()` collapse onto a single representative (the
    /// first survivor in `moves`' order, preserving the lowest
    /// `(shape, position)` tie-break).
    ///
    /// Counts `generated_nodes` for every move applied (before dedup filtering)
    /// and `canonical_dedup_hits` for each sibling collapsed by dedup.
    /// `expanded_nodes` (the "a state's successor set was computed" event) is
    /// counted at the move-generation sites instead -- once for the root moves
    /// in `search` and once per node right after `generate_legal_moves` in
    /// `negamax` -- so the no-legal-moves and depth-0 leaf nodes are counted too.
    fn children(&mut self, bb: &Bitboard, moves: &[Move], dedup: bool) -> Vec<ChildEntry> {
        self.counters.generated_nodes += moves.len() as u64;
        if !dedup {
            return moves
                .iter()
                .map(|&mv| (mv, apply_move(bb, &mv), None))
                .collect();
        }
        let mut seen: HashMap<[u8; 18], ()> = HashMap::new();
        let mut children = Vec::with_capacity(moves.len());
        for &mv in moves {
            let child_bb = apply_move(bb, &mv);
            let key = State::new(child_bb).canonical_key();
            if seen.insert(key, ()).is_some() {
                self.counters.canonical_dedup_hits += 1;
                continue;
            }
            children.push((mv, child_bb, Some(key)));
        }
        children
    }

    /// Search the root position to `depth`, returning `(score, best_move, pv)`.
    ///
    /// Every root child is searched with a FULL (-inf, +inf) window, so each
    /// returned value is EXACT rather than a fail-soft bound. This matters
    /// for the tie-break: a narrowed window could let an inferior sibling
    /// fail low onto a bound equal to `best_value`, pollute the equal-value
    /// candidate set, and — with `random_seed` set — be chosen. Each child's
    /// own subtree is still alpha-beta pruned inside `negamax`.
    fn search_root(
        &mut self,
        bb: &Bitboard,
        moves: &[Move],
        depth: u32,
    ) -> Result<(f64, Move, Vec<Move>), TimeUp> {
        let ordered = self.order_root_moves(moves);
        let dedup = self.config.dedup_children;
        let children = self.children(bb, &ordered, dedup);

        let mut best_value = f64::NEG_INFINITY;
        let mut scored: Vec<(Move, f64, Vec<Move>)> = Vec::with_capacity(children.len());

        for (mv, child_bb, child_key) in children {
            let mut child_pv = Vec::new();
            let value = -self.negamax(
                &child_bb,
                depth - 1,
                f64::NEG_INFINITY,
                f64::INFINITY,
                1,
                &mut child_pv,
                child_key,
            )?;
            if value > best_value {
                best_value = value;
            }
            scored.push((mv, value, child_pv));
        }

        let candidates: Vec<&(Move, f64, Vec<Move>)> =
            scored.iter().filter(|(_, v, _)| *v == best_value).collect();
        let (mv, _, child_pv) = match self.rng.as_mut() {
            Some(rng) => candidates[rng.gen_range(0..candidates.len())],
            None => candidates[0],
        };
        let mut pv = vec![*mv];
        pv.extend(child_pv.iter().copied());
        // Written at the end of the deepest COMPLETED iteration only: a
        // `search()` loop that breaks on TimeUp mid-iteration never reaches
        // this line for that iteration, so `last_root_scored` (and thus
        // `telemetry()`) always reflects the previous, fully-scored depth.
        self.last_root_scored = scored.iter().map(|(m, v, _)| (*m, *v)).collect();
        Ok((best_value, *mv, pv))
    }

    fn check_time(&self) -> Result<(), TimeUp> {
        if let Some(deadline) = self.deadline {
            if Instant::now() >= deadline {
                return Err(TimeUp);
            }
        }
        Ok(())
    }

    /// Negamax value of `bb` from the side-to-move's perspective.
    ///
    /// `pv_out` is filled in place with the principal variation from this
    /// node downward (empty if the node is terminal or a leaf).
    /// `precomputed_key` is this node's `canonical_key()` if the caller's
    /// sibling-dedup pass already computed it (else `None`, computed lazily
    /// here only if the TT is enabled).
    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        bb: &Bitboard,
        depth: u32,
        mut alpha: f64,
        mut beta: f64,
        ply: u32,
        pv_out: &mut Vec<Move>,
        precomputed_key: Option<[u8; 18]>,
    ) -> Result<f64, TimeUp> {
        self.nodes += 1;
        if self.deadline.is_some() && (self.nodes & 0x3FF) == 0 {
            self.check_time()?;
        }

        let win = self.config.eval_config.win;

        if has_winning_line(bb) {
            // The previous mover completed a line: the side to move here has
            // just lost. `ply` makes a sooner loss/win score more extremely
            // than a deeper one (shallower mates score higher).
            self.counters.terminal_hits += 1;
            return Ok(-(win - ply as f64));
        }

        let moves = generate_legal_moves(bb);
        // The successor set was just computed, so this node is expanded --
        // including the no-legal-moves case below (which is then ALSO terminal)
        // and the depth-0 leaf case, which returns before any child is built.
        self.counters.expanded_nodes += 1;
        if moves.is_empty() {
            // No legal moves: the side to move also loses.
            self.counters.terminal_hits += 1;
            return Ok(-(win - ply as f64));
        }

        if depth == 0 {
            let side = current_player(bb).unwrap_or(0);
            return Ok(evaluate(bb, side, &self.config.eval_config));
        }

        let mut tt_key: Option<[u8; 18]> = None;
        let orig_alpha = alpha;
        let orig_beta = beta;
        if self.config.use_transposition_table {
            let key = precomputed_key.unwrap_or_else(|| State::new(*bb).canonical_key());
            tt_key = Some(key);
            if let Some(&(stored_depth, stored_value, bound)) = self.tt.get(&key) {
                if stored_depth >= depth {
                    if bound == Bound::Exact {
                        self.counters.transposition_hits += 1;
                        return Ok(stored_value);
                    }
                    // LOWER/UPPER entries only narrow the window when
                    // alpha-beta is enabled: with `use_alpha_beta = false`
                    // the search contract is an exact, unpruned value, and
                    // reusing a bound would silently reintroduce pruning.
                    if self.config.use_alpha_beta {
                        match bound {
                            Bound::Lower => alpha = alpha.max(stored_value),
                            Bound::Upper => beta = beta.min(stored_value),
                            Bound::Exact => unreachable!(),
                        }
                        if alpha >= beta {
                            self.counters.transposition_hits += 1;
                            return Ok(stored_value);
                        }
                    }
                }
            }
        }

        let mut ordered = moves;
        ordered.sort_by_key(move_sort_key);
        let dedup = self.config.dedup_children;
        let mut children = self.children(bb, &ordered, dedup);
        // Move ordering: try immediate winning replies first — a move that
        // completes a line makes this node a forced win, so exploring it
        // first yields the earliest possible beta cutoff. Stable, so the
        // deterministic (shape, position) order is preserved among equals.
        children.sort_by_key(|(_, child_bb, _)| !has_winning_line(child_bb));

        let mut best_value = f64::NEG_INFINITY;
        let mut best_move: Option<Move> = None;
        let mut best_child_pv: Vec<Move> = Vec::new();

        for (mv, child_bb, child_key) in children {
            let mut child_pv = Vec::new();
            let value = -self.negamax(
                &child_bb,
                depth - 1,
                -beta,
                -alpha,
                ply + 1,
                &mut child_pv,
                child_key,
            )?;
            if value > best_value {
                best_value = value;
                best_move = Some(mv);
                best_child_pv = child_pv;
            }
            if self.config.use_alpha_beta {
                alpha = alpha.max(best_value);
                if alpha >= beta {
                    break;
                }
            }
        }

        if let Some(mv) = best_move {
            pv_out.push(mv);
            pv_out.extend(best_child_pv);
        }

        if let Some(key) = tt_key {
            // Classify against the ORIGINAL (pre-TT-narrowing) window, not
            // the possibly-tightened alpha/beta used for this search.
            let bound = if best_value <= orig_alpha {
                Bound::Upper
            } else if self.config.use_alpha_beta && best_value >= orig_beta {
                Bound::Lower
            } else {
                Bound::Exact
            };
            self.tt.insert(key, (depth, best_value, bound));
        }

        Ok(best_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A@0, b@1, C@2: whoever moves can win immediately with D (d) at 3.
    fn immediate_win_board() -> Bitboard {
        Bitboard::EMPTY
            .with_move(0, 0, 0)
            .with_move(1, 1, 1)
            .with_move(0, 2, 2)
    }

    #[test]
    fn finds_immediate_winning_move() {
        let bb = immediate_win_board();
        let mut engine = MinimaxEngine::new(MinimaxConfig {
            max_depth: 3,
            ..Default::default()
        });
        let result = engine.search(&State::new(bb)).unwrap();
        // p1 to move: d at position 3 completes row 0.
        assert_eq!(result.best_move, Move::new(1, 3, 3));
        // Child is terminal at ply 1: value is win - 1.
        assert_eq!(result.score, 10_000.0 - 1.0);
    }

    #[test]
    fn pv_head_matches_best_move() {
        let bb = immediate_win_board();
        let mut engine = MinimaxEngine::new(MinimaxConfig {
            max_depth: 4,
            ..Default::default()
        });
        let result = engine.search(&State::new(bb)).unwrap();
        assert_eq!(result.pv[0], result.best_move);
        assert!(result.pv.len() as u32 <= result.depth_reached);
    }

    #[test]
    fn alpha_beta_equals_plain_minimax() {
        let bb = Bitboard::EMPTY
            .with_move(0, 0, 0)
            .with_move(1, 3, 8)
            .with_move(0, 2, 2)
            .with_move(1, 3, 13)
            .with_move(0, 1, 1)
            .with_move(1, 0, 10);
        let mut pruned = MinimaxEngine::new(MinimaxConfig {
            max_depth: 3,
            use_alpha_beta: true,
            ..Default::default()
        });
        let mut plain = MinimaxEngine::new(MinimaxConfig {
            max_depth: 3,
            use_alpha_beta: false,
            ..Default::default()
        });
        let state = State::new(bb);
        let a = pruned.search(&state).unwrap();
        let b = plain.search(&state).unwrap();
        assert_eq!(a.score, b.score);
        assert_eq!(a.best_move, b.best_move);
        assert!(pruned.nodes <= plain.nodes);
    }

    /// Play `plies` deterministic pseudo-random legal moves from the empty
    /// board, retrying until the line hits no win/dead-end along the way.
    fn random_position(seed: u64, plies: usize) -> Bitboard {
        let mut rng = StdRng::seed_from_u64(seed);
        'attempt: loop {
            let mut bb = Bitboard::EMPTY;
            for _ in 0..plies {
                let moves = generate_legal_moves(&bb);
                if moves.is_empty() {
                    continue 'attempt;
                }
                bb = apply_move(&bb, &moves[rng.gen_range(0..moves.len())]);
                if has_winning_line(&bb) {
                    continue 'attempt;
                }
            }
            if generate_legal_moves(&bb).is_empty() {
                continue 'attempt;
            }
            return bb;
        }
    }

    #[test]
    fn solve_reaches_terminal_depth() {
        // 10 pieces on the board (≤ 6 plies remain); solve must complete all
        // 16 iterations with an exact terminal-range score.
        let bb = random_position(42, 10);
        let mut engine = MinimaxEngine::new(MinimaxConfig::default());
        let result = engine.solve(&State::new(bb)).unwrap();
        assert_eq!(result.depth_reached, 16);
        // Exact solve of a Quantik position is a forced win for one side:
        // score magnitude must be in the terminal range, not heuristic.
        assert!(result.score.abs() > 9_000.0, "score {}", result.score);
    }

    #[test]
    fn time_limit_returns_depth_one_result() {
        let mut engine = MinimaxEngine::new(MinimaxConfig {
            max_depth: 16,
            time_limit_s: Some(0.001),
            ..Default::default()
        });
        let result = engine.search(&State::empty()).unwrap();
        assert!(result.depth_reached >= 1);
        assert!(result.pv.first() == Some(&result.best_move));
    }

    #[test]
    fn deterministic_without_seed() {
        let bb = immediate_win_board();
        let mut e1 = MinimaxEngine::new(MinimaxConfig {
            max_depth: 2,
            ..Default::default()
        });
        let mut e2 = MinimaxEngine::new(MinimaxConfig {
            max_depth: 2,
            ..Default::default()
        });
        assert_eq!(
            e1.search(&State::new(bb)).unwrap().best_move,
            e2.search(&State::new(bb)).unwrap().best_move
        );
    }

    #[test]
    fn solve_prefers_faster_win() {
        // With an immediate win available, an exact solve of a late position
        // must take it: score win - 1. Deep start keeps the tree small.
        let bb = random_position(7, 11);
        let mut engine = MinimaxEngine::new(MinimaxConfig::default());
        let result = engine.solve(&State::new(bb)).unwrap();
        let winning: Vec<Move> = generate_legal_moves(&bb)
            .into_iter()
            .filter(|m| has_winning_line(&apply_move(&bb, m)))
            .collect();
        if winning.is_empty() {
            // No immediate win from this seed: still an exact result.
            assert!(result.score.abs() > 9_000.0);
        } else {
            assert!(winning.contains(&result.best_move));
            assert_eq!(result.score, 10_000.0 - 1.0);
        }
    }

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
            let q = stat
                .q_value
                .expect("minimax scores every searched root move");
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

    #[test]
    fn expanded_counted_at_move_generation() {
        // expanded_nodes is the "successor set was computed" event, so it fires
        // once per generate_legal_moves call: once for the root moves and once
        // per negamax node -- INCLUDING the depth-0 leaf children, which compute
        // their successor set before the leaf evaluation short-circuits. A
        // depth-1 search over K non-terminal root moves therefore yields
        // expanded_nodes == K + 1 and generated_nodes == K. Before the fix,
        // expanded was counted at the children() sites, so the depth-0 leaves
        // were never counted and expanded_nodes was just 1.
        let bb = Bitboard::EMPTY;
        let k = generate_legal_moves(&bb).len() as u64;
        let mut engine = MinimaxEngine::new(MinimaxConfig {
            max_depth: 1,
            dedup_children: false,
            use_transposition_table: false,
            ..Default::default()
        });
        engine.search(&State::new(bb)).unwrap();
        let t = engine.telemetry().unwrap();
        assert_eq!(t.counters.expanded_nodes, k + 1);
        assert_eq!(t.counters.generated_nodes, k);
    }

    #[test]
    fn no_legal_moves_node_is_expanded_and_terminal() {
        // A no-legal-moves node computed its (empty) successor set before being
        // ruled terminal, so it must be BOTH expanded and terminal (per the
        // normative counter semantics). A full single-shape board has no winning
        // line and no legal moves; negamax on it must bump both counters. Built
        // via the public `with_move` API (player 0, shape 0 on every cell)
        // rather than a raw plane literal, so the test does not depend on the
        // internal plane ordering.
        let no_moves = (0..16u8).fold(Bitboard::EMPTY, |b, pos| b.with_move(0, 0, pos));
        assert!(generate_legal_moves(&no_moves).is_empty());
        let mut engine = MinimaxEngine::new(MinimaxConfig {
            max_depth: 2,
            dedup_children: false,
            use_transposition_table: false,
            ..Default::default()
        });
        engine.counters = SearchEventCounters::default();
        engine.nodes = 0;
        engine.deadline = None;
        // deadline == None => negamax cannot return Err(TimeUp); discard the
        // Ok value (TimeUp is not Debug, so `.unwrap()` would not compile).
        let mut pv = Vec::new();
        let _ = engine.negamax(
            &no_moves,
            1,
            f64::NEG_INFINITY,
            f64::INFINITY,
            0,
            &mut pv,
            None,
        );
        assert_eq!(engine.counters.terminal_hits, 1);
        assert_eq!(engine.counters.expanded_nodes, 1);
    }
}
