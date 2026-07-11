//! Parametrizable beam search for Quantik.
//!
//! Descends level-by-level from a root state, keeping only the top
//! `beam_width` non-terminal candidates per depth (breadth pruning) while
//! always discovering and recording every true terminal state encountered,
//! regardless of the beam width. Port of the Python
//! `quantik_core.beam_search` module (schedules, multiplicity accounting,
//! ranked root moves, wall-clock budget) — without the shared
//! `CompactGameTree`: this engine is standalone, and `nodes_inserted`
//! counts the terminal leaves recorded plus the survivors kept per level,
//! the closest observable analogue of the Python tree-insertion counter.

use crate::bitboard::Bitboard;
use crate::game::{check_winner, current_player, WinStatus};
use crate::moves::{apply_move, generate_legal_moves, Move};
use crate::symmetry::SymmetryHandler;
use rand::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

/// Evaluates a position from player 0's perspective; values are clamped
/// to `[-1, 1]` by the engine.
pub type Evaluator = Box<dyn Fn(&Bitboard) -> f64>;

/// Unique canonical states per depth (see `GAME_TREE_ANALYSIS.md` in the
/// Python repository). Useful for building an exhaustive-prefix
/// `beam_schedule` that keeps every legal line up to some depth before
/// switching to guided sampling.
pub const UNIQUE_CANONICAL_STATES_PER_DEPTH: [(u32, u64); 8] = [
    (1, 3),
    (2, 51),
    (3, 726),
    (4, 10_946),
    (5, 105_632),
    (6, 901_916),
    (7, 4_658_465),
    (8, 17_900_160),
];

/// Configuration for the beam search algorithm.
#[derive(Clone, Debug)]
pub struct BeamSearchConfig {
    /// Frontier nodes kept per depth (>= 1).
    pub beam_width: usize,
    /// Plies from root; 16 = full Quantik game. Must be 1..=16.
    pub max_depth: u32,
    /// Rollout budget for the default evaluator (>= 1).
    pub rollouts_per_candidate: u32,
    pub random_seed: Option<u64>,
    /// Depth-dependent beam width: width at depth d =
    /// `beam_schedule[min(d-1, len-1)]`, so the last entry extends to all
    /// deeper levels. `None` applies the flat `beam_width` everywhere.
    pub beam_schedule: Option<Vec<usize>>,
    /// Depth-dependent rollout budget for the BUILT-IN evaluator only — a
    /// custom evaluator keeps its plain signature and ignores this.
    /// Semantics mirror `beam_schedule`.
    pub rollout_schedule: Option<Vec<u32>>,
    /// Optional wall-clock budget for `search`, in seconds. Checked between
    /// depth levels (after each completed level), so depth 1 always
    /// completes and a wide level can overshoot the budget — callers that
    /// need honest numbers should measure actual elapsed time themselves.
    pub time_limit_s: Option<f64>,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 64,
            max_depth: 16,
            rollouts_per_candidate: 8,
            random_seed: None,
            beam_schedule: None,
            rollout_schedule: None,
            time_limit_s: None,
        }
    }
}

/// A single collected leaf: a principal variation and its value.
#[derive(Clone, Debug, PartialEq)]
pub struct BeamLeaf {
    /// Principal variation from the root.
    pub moves: Vec<Move>,
    /// P0 perspective; ±1.0 for terminal leaves.
    pub value: f64,
    pub depth: u32,
    pub is_terminal: bool,
    /// Number of raw (pre-canonicalization) move sequences this leaf stands
    /// in for, accumulated by summing parent multiplicities across every
    /// dedup hit on the path from the root. 1 for a leaf whose whole PV was
    /// never merged with a symmetric sibling.
    pub multiplicity: u64,
}

/// Aggregated beam-sampled statistics for one first move from the root.
///
/// These are optimistic, beam-sampled statistics computed over whichever
/// leaves this particular engine run happened to discover and keep — they
/// are **not** a minimax-proven guarantee. `win_probability` is a heuristic
/// rescaling of `mean_value` into `[0, 1]`, not a calibrated probability.
#[derive(Clone, Debug, PartialEq)]
pub struct RankedRootMove {
    pub mv: Move,
    /// Max leaf value via this move, root-player perspective.
    pub best_value: f64,
    /// Multiplicity-weighted mean, root-player perspective.
    pub mean_value: f64,
    /// Heuristic rescaling: `(mean_value + 1) / 2`.
    pub win_probability: f64,
    /// Number of collected leaves supporting this move.
    pub leaf_count: usize,
    /// Sum of multiplicity over supporting leaves.
    pub total_multiplicity: u64,
    /// A proven root-player-winning terminal exists via this move.
    pub has_terminal_win: bool,
}

/// Search statistics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeamStats {
    pub candidates_generated: u64,
    pub candidates_deduped: u64,
    pub nodes_inserted: u64,
    pub nodes_pruned: u64,
    pub evaluations: u64,
    pub rollouts: u64,
}

/// Result of a beam search run.
#[derive(Clone, Debug, PartialEq)]
pub struct BeamSearchResult {
    /// Best leaf for the ROOT player to move.
    pub best_leaf: Option<BeamLeaf>,
    /// All terminals discovered, best first (root-player perspective).
    pub terminal_leaves: Vec<BeamLeaf>,
    pub reached_terminal: bool,
    pub max_depth_reached: u32,
    pub stats: BeamStats,
    /// Player to move at the root.
    pub root_player: u8,
    /// Non-terminal leaves still live at `max_depth_reached`; empty once
    /// the search fully resolves (`reached_terminal` is true).
    pub frontier_leaves: Vec<BeamLeaf>,
}

impl BeamSearchResult {
    /// Aggregate every collected leaf by its first move from the root.
    ///
    /// Groups `terminal_leaves` and `frontier_leaves` by the first move of
    /// each leaf's principal variation and summarizes each group's value
    /// from the root player's perspective. See [`RankedRootMove`] for the
    /// caveat: these are beam-sampled statistics, not proven minimax values.
    pub fn ranked_root_moves(&self, top_k: Option<usize>) -> Vec<RankedRootMove> {
        let mut order: Vec<Move> = Vec::new();
        let mut groups: HashMap<(u8, u8, u8), Vec<&BeamLeaf>> = HashMap::new();

        for leaf in self
            .terminal_leaves
            .iter()
            .chain(self.frontier_leaves.iter())
        {
            let Some(first) = leaf.moves.first() else {
                continue;
            };
            let key = (first.player, first.shape, first.position);
            let entry = groups.entry(key).or_default();
            if entry.is_empty() {
                order.push(*first);
            }
            entry.push(leaf);
        }

        let root_perspective = |leaf: &BeamLeaf| -> f64 {
            if self.root_player == 0 {
                leaf.value
            } else {
                -leaf.value
            }
        };

        let mut ranked: Vec<RankedRootMove> = order
            .into_iter()
            .map(|mv| {
                let leaves = &groups[&(mv.player, mv.shape, mv.position)];
                let total_multiplicity: u64 = leaves.iter().map(|l| l.multiplicity).sum();
                let best_value = leaves
                    .iter()
                    .map(|l| root_perspective(l))
                    .fold(f64::NEG_INFINITY, f64::max);
                let mean_value = leaves
                    .iter()
                    .map(|l| root_perspective(l) * l.multiplicity as f64)
                    .sum::<f64>()
                    / total_multiplicity as f64;
                let has_terminal_win = leaves
                    .iter()
                    .any(|l| l.is_terminal && root_perspective(l) == 1.0);
                RankedRootMove {
                    mv,
                    best_value,
                    mean_value,
                    win_probability: (mean_value + 1.0) / 2.0,
                    leaf_count: leaves.len(),
                    total_multiplicity,
                    has_terminal_win,
                }
            })
            .collect();

        ranked.sort_by(|a, b| {
            b.best_value
                .total_cmp(&a.best_value)
                .then(b.mean_value.total_cmp(&a.mean_value))
                .then(b.leaf_count.cmp(&a.leaf_count))
                .then((a.mv.player, a.mv.shape, a.mv.position).cmp(&(
                    b.mv.player,
                    b.mv.shape,
                    b.mv.position,
                )))
        });

        if let Some(k) = top_k {
            ranked.truncate(k);
        }
        ranked
    }
}

/// Frontier entry: bitboard, move sequence from the root, evaluated value
/// (P0 perspective; 0.0 for the never-scored root) and accumulated
/// multiplicity.
struct FrontierEntry {
    bb: Bitboard,
    moves: Vec<Move>,
    multiplicity: u64,
}

/// Candidate keyed by canonical payload: the candidate's bitboard, move
/// sequence, the player who made the move leading to it, and accumulated
/// multiplicity.
struct Candidate {
    bb: Bitboard,
    moves: Vec<Move>,
    mover: u8,
    multiplicity: u64,
}

/// Level-by-level beam search over the Quantik game tree.
pub struct BeamSearchEngine {
    pub config: BeamSearchConfig,
    rng: StdRng,
    evaluator: Option<Evaluator>,
}

impl BeamSearchEngine {
    /// Initialize the engine, validating configuration.
    pub fn new(config: BeamSearchConfig) -> Result<Self, String> {
        if config.beam_width < 1 {
            return Err("beam_width must be >= 1".into());
        }
        if !(1..=16).contains(&config.max_depth) {
            return Err("max_depth must be between 1 and 16".into());
        }
        if config.rollouts_per_candidate < 1 {
            return Err("rollouts_per_candidate must be >= 1".into());
        }
        if let Some(schedule) = &config.beam_schedule {
            if schedule.is_empty() {
                return Err("beam_schedule must not be empty".into());
            }
            if schedule.iter().any(|&w| w < 1) {
                return Err("beam_schedule entries must all be >= 1".into());
            }
        }
        if let Some(schedule) = &config.rollout_schedule {
            if schedule.is_empty() {
                return Err("rollout_schedule must not be empty".into());
            }
            if schedule.iter().any(|&c| c < 1) {
                return Err("rollout_schedule entries must all be >= 1".into());
            }
        }
        if let Some(limit) = config.time_limit_s {
            if limit <= 0.0 || !limit.is_finite() {
                return Err(format!(
                    "time_limit_s must be positive and finite, got {limit}"
                ));
            }
        }
        let rng = match config.random_seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Ok(Self {
            config,
            rng,
            evaluator: None,
        })
    }

    /// Builder-style custom evaluator override (P0 perspective, `[-1, 1]`).
    pub fn with_evaluator(mut self, evaluator: Evaluator) -> Self {
        self.evaluator = Some(evaluator);
        self
    }

    /// Run beam search from `root`.
    pub fn search(&mut self, root: &Bitboard) -> Result<BeamSearchResult, String> {
        if check_winner(root) != WinStatus::NoWin {
            return Err("Cannot search from an already-terminal root state.".into());
        }
        let root_player =
            current_player(root).ok_or_else(|| "Inconsistent root piece counts.".to_string())?;
        if generate_legal_moves(root).is_empty() {
            return Err("Cannot search from a root state with no legal moves.".into());
        }

        let mut stats = BeamStats::default();
        let mut terminal_leaves: Vec<BeamLeaf> = Vec::new();
        let mut frontier: Vec<FrontierEntry> = vec![FrontierEntry {
            bb: *root,
            moves: Vec::new(),
            multiplicity: 1,
        }];
        // Values carried alongside the frontier (parallel vec keeps
        // FrontierEntry small); 0.0 for the never-scored root.
        let mut frontier_values: Vec<f64> = vec![0.0];
        let mut max_depth_reached = 0u32;

        let deadline = self
            .config
            .time_limit_s
            .map(|s| Instant::now() + std::time::Duration::from_secs_f64(s));

        for depth in 1..=self.config.max_depth {
            if frontier.is_empty() {
                break;
            }
            if depth > 1 {
                if let Some(deadline) = deadline {
                    if Instant::now() >= deadline {
                        break;
                    }
                }
            }

            let candidates =
                self.expand_frontier(&frontier, depth, &mut stats, &mut terminal_leaves);
            let beam_width = self.beam_width_for_depth(depth);
            let rollouts = self.rollouts_for_depth(depth);
            let (next_frontier, next_values) =
                self.score_and_prune(candidates, &mut stats, beam_width, rollouts);
            frontier = next_frontier;
            frontier_values = next_values;
            max_depth_reached = depth;
        }

        let root_perspective = |leaf: &BeamLeaf| -> f64 {
            if root_player == 0 {
                leaf.value
            } else {
                -leaf.value
            }
        };

        let frontier_leaves: Vec<BeamLeaf> = frontier
            .into_iter()
            .zip(frontier_values)
            .map(|(entry, value)| BeamLeaf {
                moves: entry.moves,
                value,
                depth: max_depth_reached,
                is_terminal: false,
                multiplicity: entry.multiplicity,
            })
            .collect();

        let reached_terminal = frontier_leaves.is_empty();
        let best_leaf = terminal_leaves
            .iter()
            .chain(frontier_leaves.iter())
            .max_by(|a, b| root_perspective(a).total_cmp(&root_perspective(b)))
            .cloned();
        terminal_leaves.sort_by(|a, b| root_perspective(b).total_cmp(&root_perspective(a)));

        Ok(BeamSearchResult {
            best_leaf,
            terminal_leaves,
            reached_terminal,
            max_depth_reached,
            stats,
            root_player,
            frontier_leaves,
        })
    }

    /// Resolve the beam width to use at a given depth (1-indexed).
    fn beam_width_for_depth(&self, depth: u32) -> usize {
        match &self.config.beam_schedule {
            None => self.config.beam_width,
            Some(schedule) => {
                let index = (depth as usize - 1).min(schedule.len() - 1);
                schedule[index]
            }
        }
    }

    /// Resolve the built-in evaluator's rollout count at a depth (1-indexed).
    fn rollouts_for_depth(&self, depth: u32) -> u32 {
        match &self.config.rollout_schedule {
            None => self.config.rollouts_per_candidate,
            Some(schedule) => {
                let index = (depth as usize - 1).min(schedule.len() - 1);
                schedule[index]
            }
        }
    }

    /// Expand every frontier entry, recording terminals and candidates.
    ///
    /// Every raw legal move contributes the parent's multiplicity to
    /// whatever it produces: its own terminal leaf, or — on a canonical
    /// dedup hit — accumulated into the existing candidate's multiplicity
    /// (the first-encountered move/parent is kept for the principal
    /// variation; only the weight accumulates).
    fn expand_frontier(
        &mut self,
        frontier: &[FrontierEntry],
        depth: u32,
        stats: &mut BeamStats,
        terminal_leaves: &mut Vec<BeamLeaf>,
    ) -> (Vec<Candidate>, HashMap<[u8; 16], usize>) {
        let mut candidates: Vec<Candidate> = Vec::new();
        let mut index_by_key: HashMap<[u8; 16], usize> = HashMap::new();

        for entry in frontier {
            let all_moves = generate_legal_moves(&entry.bb);
            let mover = current_player(&entry.bb).unwrap_or(0);

            if all_moves.is_empty() {
                // Mover has no legal moves: the other player wins.
                let value = if mover == 1 { 1.0 } else { -1.0 };
                terminal_leaves.push(BeamLeaf {
                    moves: entry.moves.clone(),
                    value,
                    depth: depth - 1,
                    is_terminal: true,
                    multiplicity: entry.multiplicity,
                });
                stats.nodes_inserted += 1;
                continue;
            }

            stats.candidates_generated += all_moves.len() as u64;

            for mv in all_moves {
                let new_bb = apply_move(&entry.bb, &mv);
                let winner = check_winner(&new_bb);

                if winner != WinStatus::NoWin {
                    let value = if winner == WinStatus::Player0Wins {
                        1.0
                    } else {
                        -1.0
                    };
                    let mut child_moves = entry.moves.clone();
                    child_moves.push(mv);
                    terminal_leaves.push(BeamLeaf {
                        moves: child_moves,
                        value,
                        depth,
                        is_terminal: true,
                        multiplicity: entry.multiplicity,
                    });
                    stats.nodes_inserted += 1;
                    continue;
                }

                let key = SymmetryHandler::canonical_payload(&new_bb);
                if let Some(&existing) = index_by_key.get(&key) {
                    stats.candidates_deduped += 1;
                    candidates[existing].multiplicity += entry.multiplicity;
                    continue;
                }
                let mut child_moves = entry.moves.clone();
                child_moves.push(mv);
                index_by_key.insert(key, candidates.len());
                candidates.push(Candidate {
                    bb: new_bb,
                    moves: child_moves,
                    mover: mv.player,
                    multiplicity: entry.multiplicity,
                });
            }
        }

        (candidates, index_by_key)
    }

    /// Evaluate candidates, keep the top `beam_width`, build the next
    /// frontier. Scoring and pruning are purely value-based; multiplicity
    /// is carried through unweighted.
    fn score_and_prune(
        &mut self,
        (candidates, _index): (Vec<Candidate>, HashMap<[u8; 16], usize>),
        stats: &mut BeamStats,
        beam_width: usize,
        rollouts: u32,
    ) -> (Vec<FrontierEntry>, Vec<f64>) {
        // (mover-relative score, insertion index, raw value)
        let mut scored: Vec<(f64, usize, f64)> = Vec::with_capacity(candidates.len());
        for (index, candidate) in candidates.iter().enumerate() {
            let raw_value = self.evaluate(&candidate.bb, rollouts, stats);
            stats.evaluations += 1;
            let score = if candidate.mover == 0 {
                raw_value
            } else {
                -raw_value
            };
            scored.push((score, index, raw_value));
        }

        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let kept = scored.len().min(beam_width);
        stats.nodes_pruned += (scored.len() - kept) as u64;
        scored.truncate(kept);

        // Extract survivors by index. Move candidates out via Option to
        // avoid cloning move vectors.
        let mut slots: Vec<Option<Candidate>> = candidates.into_iter().map(Some).collect();
        let mut next_frontier = Vec::with_capacity(kept);
        let mut next_values = Vec::with_capacity(kept);
        for (_, index, raw_value) in scored {
            let candidate = slots[index].take().expect("survivor extracted once");
            stats.nodes_inserted += 1;
            next_frontier.push(FrontierEntry {
                bb: candidate.bb,
                moves: candidate.moves,
                multiplicity: candidate.multiplicity,
            });
            next_values.push(raw_value);
        }
        (next_frontier, next_values)
    }

    /// Evaluate a state from player 0's perspective, clamped to `[-1, 1]`.
    ///
    /// A custom evaluator is called as-is (its cost model is its own, so
    /// `rollouts` is ignored and `stats.rollouts` stays untouched);
    /// otherwise the built-in evaluator runs `rollouts` playouts.
    fn evaluate(&mut self, bb: &Bitboard, rollouts: u32, stats: &mut BeamStats) -> f64 {
        let raw = match &self.evaluator {
            Some(evaluator) => evaluator(bb),
            None => {
                let mut total = 0.0;
                for _ in 0..rollouts {
                    total += self.rollout(bb);
                }
                stats.rollouts += rollouts as u64;
                total / rollouts as f64
            }
        };
        raw.clamp(-1.0, 1.0)
    }

    /// Play uniformly random legal moves until a terminal state.
    ///
    /// A Quantik playout always resolves within 16 plies, so no depth
    /// cutoff is required.
    fn rollout(&mut self, bb: &Bitboard) -> f64 {
        let mut current = *bb;
        loop {
            let winner = check_winner(&current);
            if winner != WinStatus::NoWin {
                return if winner == WinStatus::Player0Wins {
                    1.0
                } else {
                    -1.0
                };
            }
            let moves = generate_legal_moves(&current);
            if moves.is_empty() {
                return if current_player(&current) == Some(0) {
                    -1.0
                } else {
                    1.0
                };
            }
            let mv = moves[self.rng.gen_range(0..moves.len())];
            current = apply_move(&current, &mv);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::has_winning_line;
    use std::cell::Cell;
    use std::rc::Rc;

    fn engine(config: BeamSearchConfig) -> BeamSearchEngine {
        BeamSearchEngine::new(config).unwrap()
    }

    /// A@0, b@1, C@2: P1 to move; d@3 completes row 0 and wins for P1.
    fn immediate_win_board() -> Bitboard {
        Bitboard::EMPTY
            .with_move(0, 0, 0)
            .with_move(1, 1, 1)
            .with_move(0, 2, 2)
    }

    #[test]
    fn invalid_configs_are_rejected() {
        for config in [
            BeamSearchConfig {
                beam_width: 0,
                ..Default::default()
            },
            BeamSearchConfig {
                max_depth: 0,
                ..Default::default()
            },
            BeamSearchConfig {
                max_depth: 17,
                ..Default::default()
            },
            BeamSearchConfig {
                rollouts_per_candidate: 0,
                ..Default::default()
            },
            BeamSearchConfig {
                beam_schedule: Some(vec![]),
                ..Default::default()
            },
            BeamSearchConfig {
                beam_schedule: Some(vec![3, 0]),
                ..Default::default()
            },
            BeamSearchConfig {
                rollout_schedule: Some(vec![]),
                ..Default::default()
            },
            BeamSearchConfig {
                rollout_schedule: Some(vec![0]),
                ..Default::default()
            },
            BeamSearchConfig {
                time_limit_s: Some(0.0),
                ..Default::default()
            },
            BeamSearchConfig {
                time_limit_s: Some(f64::INFINITY),
                ..Default::default()
            },
        ] {
            assert!(BeamSearchEngine::new(config).is_err());
        }
    }

    #[test]
    fn immediate_win_found_and_ranked_first() {
        let bb = immediate_win_board();
        let mut engine = engine(BeamSearchConfig {
            beam_width: 8,
            max_depth: 2,
            rollouts_per_candidate: 1,
            random_seed: Some(1),
            ..Default::default()
        });
        let result = engine.search(&bb).unwrap();
        assert_eq!(result.root_player, 1);

        let best = result.best_leaf.as_ref().unwrap();
        assert!(best.is_terminal);
        assert_eq!(best.depth, 1);
        assert_eq!(best.value, -1.0, "P1 win is -1.0 in P0 perspective");
        let winning_move = best.moves[0];
        assert!(has_winning_line(&apply_move(&bb, &winning_move)));

        let ranked = result.ranked_root_moves(None);
        assert!(ranked[0].has_terminal_win);
        assert_eq!(ranked[0].mv, winning_move);
        assert_eq!(ranked[0].best_value, 1.0, "root-player perspective");
    }

    #[test]
    fn full_game_reachability_and_replayable_pvs() {
        let mut engine = engine(BeamSearchConfig {
            beam_width: 4,
            max_depth: 16,
            rollouts_per_candidate: 1,
            random_seed: Some(42),
            ..Default::default()
        });
        let result = engine.search(&Bitboard::EMPTY).unwrap();
        assert!(result.reached_terminal);
        assert!(result.frontier_leaves.is_empty());
        assert!(!result.terminal_leaves.is_empty());

        for leaf in &result.terminal_leaves {
            assert!(leaf.is_terminal);
            assert!(leaf.value == 1.0 || leaf.value == -1.0);
            // Replay the PV: every move legal, final state terminal.
            let mut bb = Bitboard::EMPTY;
            for mv in &leaf.moves {
                assert!(generate_legal_moves(&bb).contains(mv));
                bb = apply_move(&bb, mv);
            }
            assert!(has_winning_line(&bb) || generate_legal_moves(&bb).is_empty());
        }
    }

    #[test]
    fn symmetry_dedup_at_depth_one() {
        let mut engine = engine(BeamSearchConfig {
            beam_width: 64,
            max_depth: 1,
            rollouts_per_candidate: 1,
            random_seed: Some(0),
            ..Default::default()
        });
        let result = engine.search(&Bitboard::EMPTY).unwrap();
        assert_eq!(result.stats.candidates_generated, 64);
        // 64 raw moves collapse to 3 canonical states.
        assert_eq!(result.stats.candidates_deduped, 61);
        assert_eq!(result.frontier_leaves.len(), 3);
        // Path-count accounting: multiplicities cover all 64 raw moves.
        let total: u64 = result.frontier_leaves.iter().map(|l| l.multiplicity).sum();
        assert_eq!(total, 64);
    }

    #[test]
    fn beam_schedule_extends_last_entry() {
        let mut engine = engine(BeamSearchConfig {
            beam_schedule: Some(vec![3, 2]),
            max_depth: 4,
            rollouts_per_candidate: 1,
            random_seed: Some(9),
            ..Default::default()
        });
        let result = engine.search(&Bitboard::EMPTY).unwrap();
        // Depth 1 keeps 3 (all canonical states), depths 2..4 keep 2 each.
        assert!(result.frontier_leaves.len() <= 2);
        assert!(result.stats.nodes_pruned > 0);
    }

    #[test]
    fn rollout_schedule_counts_exactly() {
        let mut engine = engine(BeamSearchConfig {
            beam_width: 2,
            max_depth: 2,
            rollout_schedule: Some(vec![1, 8]),
            random_seed: Some(3),
            ..Default::default()
        });
        let result = engine.search(&Bitboard::EMPTY).unwrap();
        // Depth 1: 3 canonical candidates × 1 rollout; depth 2: evaluations
        // at 8 rollouts each.
        let depth1 = 3u64;
        let depth2 = result.stats.evaluations - depth1;
        assert_eq!(result.stats.rollouts, depth1 + depth2 * 8);
    }

    #[test]
    fn determinism_same_seed() {
        let run = || {
            let mut engine = engine(BeamSearchConfig {
                beam_width: 8,
                max_depth: 6,
                rollouts_per_candidate: 2,
                random_seed: Some(77),
                ..Default::default()
            });
            engine.search(&Bitboard::EMPTY).unwrap()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn custom_evaluator_used_and_clamped() {
        let calls = Rc::new(Cell::new(0u64));
        let calls_in = calls.clone();
        let mut engine = engine(BeamSearchConfig {
            beam_width: 4,
            max_depth: 2,
            random_seed: Some(5),
            ..Default::default()
        })
        .with_evaluator(Box::new(move |_bb| {
            calls_in.set(calls_in.get() + 1);
            5.0 // out of range: must clamp to 1.0
        }));
        let result = engine.search(&Bitboard::EMPTY).unwrap();
        assert!(calls.get() > 0);
        assert_eq!(result.stats.rollouts, 0, "custom evaluator: no rollouts");
        for leaf in &result.frontier_leaves {
            assert_eq!(leaf.value, 1.0);
        }
    }

    #[test]
    fn root_player_one_perspective() {
        let bb = immediate_win_board();
        let mut engine = engine(BeamSearchConfig {
            beam_width: 4,
            max_depth: 2,
            rollouts_per_candidate: 1,
            random_seed: Some(2),
            ..Default::default()
        });
        let result = engine.search(&bb).unwrap();
        assert_eq!(result.root_player, 1);
        // The P1-winning terminal (value -1.0) must rank first for P1.
        let first = &result.terminal_leaves[0];
        assert_eq!(first.value, -1.0);
    }

    #[test]
    fn root_errors() {
        let won = Bitboard::EMPTY
            .with_move(0, 0, 0)
            .with_move(1, 1, 1)
            .with_move(0, 2, 2)
            .with_move(1, 3, 3);
        let mut e = engine(BeamSearchConfig::default());
        assert!(e.search(&won).is_err());
    }

    #[test]
    fn memory_bound_holds() {
        let mut engine = engine(BeamSearchConfig {
            beam_width: 2,
            max_depth: 8,
            rollouts_per_candidate: 1,
            random_seed: Some(13),
            ..Default::default()
        });
        let result = engine.search(&Bitboard::EMPTY).unwrap();
        // Survivors per level ≤ beam_width; terminals are extra but each
        // level's kept frontier is bounded.
        assert!(result.frontier_leaves.len() <= 2);
    }

    #[test]
    fn time_limit_stops_between_levels() {
        let mut engine = engine(BeamSearchConfig {
            beam_width: 512,
            max_depth: 16,
            rollouts_per_candidate: 8,
            random_seed: Some(21),
            time_limit_s: Some(0.02),
            ..Default::default()
        });
        let start = Instant::now();
        let result = engine.search(&Bitboard::EMPTY).unwrap();
        // Depth 1 always completes; the wide levels stop early.
        assert!(result.max_depth_reached >= 1);
        assert!(start.elapsed().as_secs_f64() < 5.0);
    }
}
