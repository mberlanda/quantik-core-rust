use crate::bitboard::Bitboard;
use crate::game::{check_winner, current_player, WinStatus};
use crate::moves::{apply_move, generate_legal_moves, Move};
use crate::search_telemetry::{
    clamp_unproven, EngineKind, PolicyMassKind, RootMoveStat, SearchEventCounters, SearchTelemetry,
};
use crate::state::State;
use rand::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

pub struct MCTSConfig {
    pub exploration_weight: f64,
    pub max_iterations: u32,
    pub max_depth: u32,
    pub seed: Option<u64>,
    /// Optional wall-clock budget for `search`, in seconds. Checked after
    /// each completed iteration; `None` means the iteration count is the
    /// only stop condition.
    pub time_limit_s: Option<f64>,
    /// Merge children that reach an already-seen canonical state into the
    /// existing node instead of allocating a fresh one. `false` always
    /// allocates, so revisited canonical states get independent statistics.
    pub use_transposition_table: bool,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            exploration_weight: std::f64::consts::SQRT_2,
            max_iterations: 10_000,
            max_depth: 16,
            seed: None,
            time_limit_s: None,
            use_transposition_table: true,
        }
    }
}

struct MCTSNode {
    bb: Bitboard,
    children: Vec<usize>,
    mv: Option<Move>, // move that led here (first discovery)
    visit_count: u32,
    win_count_p0: u32,
    win_count_p1: u32,
    untried_moves: Vec<Move>,
    is_terminal: bool,
    terminal_value: f64, // +1 p0 win, -1 p1 win
}

pub struct MCTSEngine {
    config: MCTSConfig,
    nodes: Vec<MCTSNode>,
    transpositions: HashMap<[u8; 18], usize>,
    rng: StdRng,
    iterations_performed: u32,
    counters: SearchEventCounters,
    elapsed_ms: u64,
    max_depth_reached: u32,
}

impl MCTSEngine {
    pub fn new(config: MCTSConfig) -> Self {
        if let Some(limit) = config.time_limit_s {
            assert!(
                limit > 0.0 && limit.is_finite(),
                "time_limit_s must be positive and finite, got {limit}"
            );
        }
        let rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self {
            config,
            nodes: Vec::new(),
            transpositions: HashMap::new(),
            rng,
            iterations_performed: 0,
            counters: SearchEventCounters::default(),
            elapsed_ms: 0,
            max_depth_reached: 0,
        }
    }

    /// Run MCTS from the given bitboard and return
    /// `(best_move, win_probability_for_the_root_mover)`.
    pub fn search(&mut self, bb: &Bitboard) -> Option<(Move, f64)> {
        self.nodes.clear();
        self.transpositions.clear();
        self.iterations_performed = 0;
        self.counters = SearchEventCounters::default();
        self.elapsed_ms = 0;
        self.max_depth_reached = 0;

        let legal = generate_legal_moves(bb);
        if legal.is_empty() {
            return None;
        }

        let terminal = check_winner(bb);
        let is_terminal = terminal != WinStatus::NoWin;
        let terminal_value = match terminal {
            WinStatus::Player0Wins => 1.0,
            WinStatus::Player1Wins => -1.0,
            WinStatus::NoWin => 0.0,
        };

        self.nodes.push(MCTSNode {
            bb: *bb,
            children: Vec::new(),
            mv: None,
            visit_count: 0,
            win_count_p0: 0,
            win_count_p1: 0,
            untried_moves: legal,
            is_terminal,
            terminal_value,
        });
        self.counters.expanded_nodes += 1;
        if is_terminal {
            self.counters.terminal_hits += 1;
        }

        let deadline = self
            .config
            .time_limit_s
            .map(|s| Instant::now() + std::time::Duration::from_secs_f64(s));

        let started = Instant::now();
        for _ in 0..self.config.max_iterations {
            let mut path = self.select(0);
            let leaf = *path.last().expect("path always contains the root");
            let expanded = self.expand(leaf);
            if expanded != leaf {
                path.push(expanded);
            }
            self.max_depth_reached = self.max_depth_reached.max(path.len() as u32 - 1);
            let value = self.simulate(expanded);
            self.backpropagate(&path, value);
            self.iterations_performed += 1;

            if let Some(deadline) = deadline {
                if Instant::now() >= deadline {
                    break;
                }
            }
        }
        self.elapsed_ms = started.elapsed().as_millis() as u64;

        self.best_move(bb)
    }

    /// Visit-count distribution over the root's legal moves from the most
    /// recent `search()` call — the raw material for an AlphaZero-style
    /// soft policy target (`visits / total_visits` per move), as opposed
    /// to the single argmax move `search()` returns. Empty if `search()`
    /// returned `None` (no legal moves) or hasn't been called yet.
    ///
    /// **With `use_transposition_table` enabled (the default), this is NOT
    /// one entry per legal move.** Root moves that canonicalize to the same
    /// child state are merged onto one shared node, reported under a single
    /// arbitrary "first discovered" move — every other legal move that led
    /// there is silently absent, not just uncounted. This is worst exactly
    /// where self-play data collection needs it most: the empty board's 64
    /// legal first moves collapse to 3 entries (see
    /// `root_move_visits_default_config_collapses_symmetric_root_moves`
    /// below, and `docs/benchmarks/quantik-game-tree-census-2026-07-13.md`
    /// for how orbit size — and therefore collapse severity — shrinks with
    /// depth but is large at the shallow plies every game starts from). For
    /// a faithful per-legal-move policy target, run `search()` with
    /// `use_transposition_table: false`.
    pub fn root_move_visits(&self) -> Vec<(Move, u32)> {
        let Some(root) = self.nodes.first() else {
            return Vec::new();
        };
        root.children
            .iter()
            .map(|&child_idx| {
                let child = &self.nodes[child_idx];
                (
                    child.mv.expect("child node always has a move"),
                    child.visit_count,
                )
            })
            .collect()
    }

    /// Descend by UCB1 from `node_id`, returning the visited path
    /// (root..=leaf). Backpropagation follows this exact path — with
    /// transposition merging a node can have several parents, so parent
    /// pointers would be ambiguous.
    fn select(&self, node_id: usize) -> Vec<usize> {
        let mut path = vec![node_id];
        let mut current = node_id;
        loop {
            let node = &self.nodes[current];
            if node.is_terminal || !node.untried_moves.is_empty() || node.children.is_empty() {
                return path;
            }
            let parent_visits = node.visit_count as f64;
            let c = self.config.exploration_weight;
            // The win rate must be from the perspective of the player
            // choosing among this node's children — the side to move at
            // THIS node — not player 0. Using p0's count unconditionally
            // systematically preferred moves that were worse for the
            // player actually choosing.
            let mover = current_player(&node.bb).unwrap_or(0);

            let mut best_ucb = f64::NEG_INFINITY;
            let mut best_child = node.children[0];
            for &child_id in &node.children {
                let child = &self.nodes[child_id];
                if child.visit_count == 0 {
                    best_child = child_id;
                    break;
                }
                let child_visits = child.visit_count as f64;
                let wins = if mover == 0 {
                    child.win_count_p0 as f64
                } else {
                    child.win_count_p1 as f64
                };
                let win_rate = wins / child_visits;
                let ucb = win_rate + c * (parent_visits.ln() / child_visits).sqrt();
                if ucb > best_ucb {
                    best_ucb = ucb;
                    best_child = child_id;
                }
            }
            path.push(best_child);
            current = best_child;
        }
    }

    fn expand(&mut self, node_id: usize) -> usize {
        if self.nodes[node_id].is_terminal || self.nodes[node_id].untried_moves.is_empty() {
            return node_id;
        }

        let idx = self
            .rng
            .gen_range(0..self.nodes[node_id].untried_moves.len());
        let mv = self.nodes[node_id].untried_moves.swap_remove(idx);
        let parent_bb = self.nodes[node_id].bb;
        let new_bb = apply_move(&parent_bb, &mv);
        self.counters.generated_nodes += 1;

        if self.config.use_transposition_table {
            let key = State::new(new_bb).canonical_key();
            if let Some(&existing) = self.transpositions.get(&key) {
                if !self.nodes[node_id].children.contains(&existing) {
                    self.nodes[node_id].children.push(existing);
                }
                self.counters.transposition_hits += 1;
                return existing;
            }
        }

        let legal = generate_legal_moves(&new_bb);
        let terminal = check_winner(&new_bb);
        let is_terminal = terminal != WinStatus::NoWin || legal.is_empty();
        let terminal_value = match terminal {
            WinStatus::Player0Wins => 1.0,
            WinStatus::Player1Wins => -1.0,
            WinStatus::NoWin if legal.is_empty() => {
                // No legal moves: the player who cannot move loses
                if current_player(&new_bb) == Some(0) {
                    -1.0
                } else {
                    1.0
                }
            }
            WinStatus::NoWin => 0.0,
        };
        self.counters.expanded_nodes += 1;
        if is_terminal {
            self.counters.terminal_hits += 1;
        }

        let child_id = self.nodes.len();
        self.nodes.push(MCTSNode {
            bb: new_bb,
            children: Vec::new(),
            mv: Some(mv),
            visit_count: 0,
            win_count_p0: 0,
            win_count_p1: 0,
            untried_moves: legal,
            is_terminal,
            terminal_value,
        });
        if self.config.use_transposition_table {
            self.transpositions
                .insert(State::new(new_bb).canonical_key(), child_id);
        }

        self.nodes[node_id].children.push(child_id);
        child_id
    }

    fn simulate(&mut self, node_id: usize) -> f64 {
        let node = &self.nodes[node_id];
        if node.is_terminal {
            return node.terminal_value;
        }

        let mut current_bb = node.bb;
        let mut depth = 0u32;

        loop {
            if depth >= self.config.max_depth {
                return 0.0;
            }
            let w = check_winner(&current_bb);
            if w != WinStatus::NoWin {
                return match w {
                    WinStatus::Player0Wins => 1.0,
                    WinStatus::Player1Wins => -1.0,
                    WinStatus::NoWin => unreachable!(),
                };
            }
            let moves = generate_legal_moves(&current_bb);
            if moves.is_empty() {
                // No legal moves: the player who cannot move loses
                return if current_player(&current_bb) == Some(0) {
                    -1.0
                } else {
                    1.0
                };
            }
            let mv = moves[self.rng.gen_range(0..moves.len())];
            current_bb = apply_move(&current_bb, &mv);
            depth += 1;
        }
    }

    fn backpropagate(&mut self, path: &[usize], value: f64) {
        for &node_id in path.iter().rev() {
            let node = &mut self.nodes[node_id];
            node.visit_count += 1;
            if value > 0.0 {
                node.win_count_p0 += 1;
            } else if value < 0.0 {
                node.win_count_p1 += 1;
            }
        }
    }

    fn best_move(&self, root_bb: &Bitboard) -> Option<(Move, f64)> {
        let root = &self.nodes[0];
        if root.children.is_empty() {
            return None;
        }

        let mut best_visits = 0u32;
        let mut best_child = root.children[0];
        for &child_id in &root.children {
            let child = &self.nodes[child_id];
            if child.visit_count > best_visits {
                best_visits = child.visit_count;
                best_child = child_id;
            }
        }

        let child = &self.nodes[best_child];
        // Win probability from the perspective of the player who made the
        // choice at the root (the root's mover), matching the UCB fix.
        let mover = current_player(root_bb).unwrap_or(0);
        let win_rate = if child.visit_count > 0 {
            let wins = if mover == 0 {
                child.win_count_p0 as f64
            } else {
                child.win_count_p1 as f64
            };
            wins / child.visit_count as f64
        } else {
            0.5
        };

        child.mv.map(|mv| (mv, win_rate))
    }

    pub fn iterations_performed(&self) -> u32 {
        self.iterations_performed
    }

    pub fn nodes_created(&self) -> usize {
        self.nodes.len()
    }

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
        // A terminal child is a PROVEN result: its value is derived directly
        // from `terminal_value` (P0-perspective) rather than sampled, so it
        // is reported as exact ±1 from the root mover's perspective. A
        // non-terminal child's value is a rollout-sampled win rate and must
        // never collide with the proven-exclusive ±1.0, hence
        // `clamp_unproven`. See the `search_telemetry` module docs' value
        // invariant.
        let q_of = |node: &MCTSNode| -> Option<f64> {
            if node.is_terminal {
                let value = if mover == 0 {
                    node.terminal_value
                } else {
                    -node.terminal_value
                };
                return Some(value);
            }
            if node.visit_count == 0 {
                return None;
            }
            let wins = if mover == 0 {
                node.win_count_p0
            } else {
                node.win_count_p1
            };
            Some(clamp_unproven(
                2.0 * (wins as f64 / node.visit_count as f64) - 1.0,
            ))
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
        // Mirror `best_move`'s tie-break (first child with strictly more
        // visits) to find the same best child, so `root_value` can apply
        // the same proven/unproven rule as `q_of` above.
        let mut best_visits = 0u32;
        let mut best_child_id = root.children[0];
        for &child_id in &root.children {
            if self.nodes[child_id].visit_count > best_visits {
                best_visits = self.nodes[child_id].visit_count;
                best_child_id = child_id;
            }
        }
        let best_child = &self.nodes[best_child_id];
        let root_value = if best_child.is_terminal {
            if mover == 0 {
                best_child.terminal_value
            } else {
                -best_child.terminal_value
            }
        } else {
            clamp_unproven(2.0 * win_rate - 1.0)
        };
        Some(SearchTelemetry {
            engine_kind: EngineKind::Mcts,
            root_value,
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
                    (
                        std::cmp::Reverse(child.visit_count),
                        mv.shape * 16 + mv.position,
                    )
                });
            let Some(&best) = best else { break };
            pv.push(self.nodes[best].mv.expect("child node always has a move"));
            node_id = best;
        }
        pv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::has_winning_line;
    use crate::search_telemetry::UNPROVEN_VALUE_BOUND;

    #[test]
    fn mcts_returns_a_move() {
        let mut engine = MCTSEngine::new(MCTSConfig {
            max_iterations: 100,
            seed: Some(42),
            ..Default::default()
        });
        let result = engine.search(&Bitboard::EMPTY);
        assert!(result.is_some());
        let (mv, prob) = result.unwrap();
        assert_eq!(mv.player, 0);
        assert!(mv.shape < 4);
        assert!(mv.position < 16);
        assert!((0.0..=1.0).contains(&prob));
    }

    #[test]
    fn root_move_visits_covers_every_legal_move_and_sums_to_iterations() {
        let bb = Bitboard::EMPTY;
        let legal = generate_legal_moves(&bb);

        // Transposition merging is disabled for this test: on the empty
        // board, `search()`'s default `use_transposition_table: true`
        // canonicalizes away board/shape symmetry so aggressively that the
        // 64 legal first moves collapse into just 3 canonical tree nodes
        // (verified empirically), which would defeat the per-move
        // accounting this test is checking. `root_move_visits` itself does
        // not touch transposition behavior either way.
        let mut engine = MCTSEngine::new(MCTSConfig {
            max_iterations: 2000,
            seed: Some(7),
            use_transposition_table: false,
            ..Default::default()
        });
        let (best_move, _win_prob) = engine.search(&bb).expect("legal moves exist");

        let visits = engine.root_move_visits();

        // Every legal root move was expanded at least once (2000 iterations
        // against a 64-move branching factor is far more than one pass).
        assert_eq!(visits.len(), legal.len());
        let visited_moves: std::collections::HashSet<Move> =
            visits.iter().map(|(mv, _)| *mv).collect();
        for mv in &legal {
            assert!(
                visited_moves.contains(mv),
                "missing {mv:?} from root_move_visits"
            );
        }

        // Visit counts sum to the iterations actually performed (root gets
        // one visit per iteration via the selection pass starting there).
        let total_visits: u32 = visits.iter().map(|(_, v)| v).sum();
        assert_eq!(total_visits, 2000);

        // The move search() actually returned must be among the visited
        // moves, and must have the maximum visit count (search() picks by
        // visit count, not raw value).
        let best_visits = visits
            .iter()
            .find(|(mv, _)| *mv == best_move)
            .map(|(_, v)| *v)
            .unwrap();
        assert!(visits.iter().all(|(_, v)| *v <= best_visits));
    }

    #[test]
    fn root_move_visits_empty_before_search() {
        let engine = MCTSEngine::new(MCTSConfig::default());
        assert!(engine.root_move_visits().is_empty());
    }

    #[test]
    fn root_move_visits_default_config_collapses_symmetric_root_moves() {
        // Documents, deliberately, the exact limitation described in
        // `root_move_visits`'s doc comment: under the engine's actual
        // default (`use_transposition_table: true`, i.e. `..Default::
        // default()` with no override — what every real caller in this
        // crate uses), the empty board's 64 legal first moves canonicalize
        // onto just 3 shared tree nodes (matches the independently
        // cross-validated depth-1 canonical count in
        // docs/benchmarks/quantik-game-tree-census-2026-07-13.md: "3
        // canonical states, 64 raw boards"). A caller building a per-move
        // policy target from this output without disabling the
        // transposition table would silently drop 61 of 64 legal moves and
        // mislabel the rest — this test exists so that fact is asserted
        // and visible, not discovered later against real training data.
        let bb = Bitboard::EMPTY;
        let legal = generate_legal_moves(&bb);
        assert_eq!(legal.len(), 64);

        let mut engine = MCTSEngine::new(MCTSConfig {
            max_iterations: 2000,
            seed: Some(7),
            ..Default::default() // use_transposition_table: true, the real default
        });
        engine.search(&bb).expect("legal moves exist");

        let visits = engine.root_move_visits();
        assert_eq!(
            visits.len(),
            3,
            "expected the empty board's legal moves to collapse to 3 canonical \
             nodes under the default (transposition-table-enabled) config; if \
             this changes, root_move_visits's doc comment and Task 6 of \
             docs/superpowers/plans/2026-07-13-crates-io-packaging-and-ml-data-pipeline.md \
             need to be re-checked, not just this assertion"
        );

        let total_visits: u32 = visits.iter().map(|(_, v)| v).sum();
        assert_eq!(
            total_visits, 2000,
            "visit mass is preserved even though move identity is not"
        );
    }

    #[test]
    fn mcts_finds_winning_move() {
        let bb = Bitboard::EMPTY
            .with_move(0, 0, 0)
            .with_move(1, 1, 5)
            .with_move(0, 2, 2);
        let mut engine = MCTSEngine::new(MCTSConfig {
            max_iterations: 500,
            seed: Some(123),
            ..Default::default()
        });
        let result = engine.search(&bb);
        assert!(result.is_some());
    }

    #[test]
    fn mcts_no_moves_returns_none() {
        // A terminal (won) position: row 0 complete
        let bb = Bitboard::EMPTY
            .with_move(0, 0, 0)
            .with_move(1, 1, 1)
            .with_move(0, 2, 2)
            .with_move(1, 3, 3);
        let mut engine = MCTSEngine::new(MCTSConfig {
            max_iterations: 10,
            seed: Some(1),
            ..Default::default()
        });
        // The root is detected terminal, so it is never expanded: no
        // children exist and no best move can be reported.
        assert!(engine.search(&bb).is_none());
    }

    /// Regression for the UCB perspective bug: player 1 to move with an
    /// immediate winning reply must select it. With the old p0-perspective
    /// selection, p1's winning move was systematically starved.
    #[test]
    fn mcts_picks_immediate_win_for_player_1() {
        // A@0, b@1, C@2: p1 to move, d@3 completes row 0 and wins for p1.
        let bb = Bitboard::EMPTY
            .with_move(0, 0, 0)
            .with_move(1, 1, 1)
            .with_move(0, 2, 2);
        let mut engine = MCTSEngine::new(MCTSConfig {
            max_iterations: 3_000,
            seed: Some(7),
            ..Default::default()
        });
        let (mv, prob) = engine.search(&bb).unwrap();
        let after = apply_move(&bb, &mv);
        assert!(
            has_winning_line(&after),
            "p1 must play the immediate win, got {mv:?} (prob {prob})"
        );
        assert!(prob > 0.5, "win probability is for the root mover");
    }

    /// FIX 1 (value invariant): a root child that immediately wins the game
    /// is a PROVEN result and must report an exact `q_value` of `1.0` (and
    /// drive an exact `root_value` of `1.0`), while every other, merely
    /// sampled root move stays within `UNPROVEN_VALUE_BOUND` — proving the
    /// terminal-child conversion in `MCTSEngine::telemetry` end to end,
    /// rather than only unit-testing the arithmetic in isolation.
    #[test]
    fn telemetry_proven_terminal_child_is_exact_others_are_clamped() {
        // Same position as `mcts_picks_immediate_win_for_player_1`: A@0,
        // b@1, C@2, p1 to move, d@3 completes row 0 and wins for p1. No
        // other legal move at this position touches a line one move from
        // completion, so it is the only root move whose child is terminal.
        let bb = Bitboard::EMPTY
            .with_move(0, 0, 0)
            .with_move(1, 1, 1)
            .with_move(0, 2, 2);
        let winning_move = Move::new(1, 3, 3);
        let mut engine = MCTSEngine::new(MCTSConfig {
            max_iterations: 3_000,
            seed: Some(7),
            use_transposition_table: false,
            ..Default::default()
        });
        engine.search(&bb).unwrap();
        let t = engine.telemetry().unwrap();

        let winning_stat = t
            .root_moves
            .iter()
            .find(|s| s.mv == winning_move)
            .expect("the winning move is legal here and must be a root move");
        assert_eq!(
            winning_stat.q_value,
            Some(1.0),
            "a proven (terminal) root child must report exact q_value 1.0, got {:?}",
            winning_stat.q_value
        );

        for stat in &t.root_moves {
            if stat.mv == winning_move {
                continue;
            }
            if let Some(q) = stat.q_value {
                assert!(
                    q.abs() <= UNPROVEN_VALUE_BOUND,
                    "non-terminal root move {:?} must respect UNPROVEN_VALUE_BOUND, got q={q}",
                    stat.mv
                );
            }
        }

        assert_eq!(
            t.root_value, 1.0,
            "the terminal winning child is also the best (most-visited) \
             child here, so root_value must be the same exact proven 1.0"
        );
    }

    #[test]
    fn time_limit_stops_early() {
        let mut engine = MCTSEngine::new(MCTSConfig {
            max_iterations: u32::MAX,
            seed: Some(3),
            time_limit_s: Some(0.05),
            ..Default::default()
        });
        let start = Instant::now();
        let result = engine.search(&Bitboard::EMPTY);
        assert!(result.is_some());
        assert!(start.elapsed().as_secs_f64() < 1.0);
        assert!(engine.iterations_performed() < u32::MAX);
        assert!(engine.iterations_performed() > 0);
    }

    #[test]
    fn same_seed_same_move() {
        let bb = Bitboard::EMPTY.with_move(0, 0, 0);
        let run = |seed| {
            let mut engine = MCTSEngine::new(MCTSConfig {
                max_iterations: 300,
                seed: Some(seed),
                ..Default::default()
            });
            engine.search(&bb).unwrap().0
        };
        assert_eq!(run(11), run(11));
    }

    #[test]
    fn transposition_table_reduces_nodes() {
        let run = |use_tt| {
            let mut engine = MCTSEngine::new(MCTSConfig {
                max_iterations: 2_000,
                seed: Some(5),
                use_transposition_table: use_tt,
                ..Default::default()
            });
            engine.search(&Bitboard::EMPTY).unwrap();
            engine.nodes_created()
        };
        assert!(run(true) < run(false));
    }

    #[test]
    #[should_panic(expected = "time_limit_s must be positive")]
    fn invalid_time_limit_panics() {
        MCTSEngine::new(MCTSConfig {
            time_limit_s: Some(0.0),
            ..Default::default()
        });
    }

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
        // Mass lands only on legal root actions and every q is in range. No
        // Quantik game ends before ply 7 (see `rollout_terminals_are_
        // excluded_from_terminal_hits` below), so every depth-1 root child
        // here is necessarily non-terminal/unproven: q must respect the
        // tighter `UNPROVEN_VALUE_BOUND`, never the proven-exclusive ±1.0,
        // even though every rollout backing it may have agreed.
        let legal: std::collections::HashSet<u8> = generate_legal_moves(&Bitboard::EMPTY)
            .iter()
            .map(|m| m.shape * 16 + m.position)
            .collect();
        for stat in &t.root_moves {
            assert!(legal.contains(&stat.action_index));
            if let Some(q) = stat.q_value {
                assert!(
                    q.abs() <= UNPROVEN_VALUE_BOUND,
                    "depth-1 root child is unproven and must respect \
                     UNPROVEN_VALUE_BOUND, got q={q}"
                );
            }
        }
        assert!(t.root_value.abs() <= UNPROVEN_VALUE_BOUND);
        // PV starts with the best move and is a legal line from the root.
        assert_eq!(t.principal_variation.first(), Some(&best));
        let mut bb = Bitboard::EMPTY;
        for mv in &t.principal_variation {
            assert!(generate_legal_moves(&bb).contains(mv));
            bb = apply_move(&bb, mv);
        }
        // depth_reached must count the expanded leaf, not just the
        // selection path up to it (FIX 2): with 500 iterations against the
        // empty board's branching factor, the tree grows well past depth 1.
        assert!(
            t.depth_reached as usize >= t.principal_variation.len().min(1),
            "depth_reached ({}) must be at least as deep as the PV implies",
            t.depth_reached
        );
        assert!(
            t.depth_reached >= 2,
            "expanded leaf must count toward depth_reached, got {}",
            t.depth_reached
        );
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
}
