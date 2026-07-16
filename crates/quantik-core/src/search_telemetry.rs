//! Shared types for search telemetry emitted by the MCTS, beam, and minimax
//! engines.
//!
//! These definitions are normative for every engine in this crate and for
//! the Python mirror in quantik-core-py.
//!
//! | Counter | Event (same in every engine) |
//! | --- | --- |
//! | `expanded_nodes` | A state's successor set was computed by the search. |
//! | `generated_nodes` | A successor state was constructed. |
//! | `transposition_hits` | A cached search result or subtree was reused via state-keyed lookup instead of being searched again. |
//! | `canonical_dedup_hits` | A generated state was merged with, or skipped in favor of, an already-present duplicate — without reusing any search result. |
//! | `terminal_hits` | A state was determined terminal during tree search. Rollout outcomes are excluded in every engine. |
//! | `tablebase_hits` | A value/policy result was obtained from an external probe artifact instead of search. Always 0 until such an artifact exists. |
//!
//! Counters are not mutually exclusive: a state whose enumeration finds zero
//! legal moves is both expanded and terminal. Identical semantics do not
//! imply comparable magnitudes: MCTS expands incrementally under an
//! iteration budget, minimax expands exhaustively to a depth, beam prunes by
//! design. `expanded_nodes` measures the same event everywhere, but
//! cross-engine workload comparison belongs to `elapsed_ms` only.
//!
//! ## Value invariant
//!
//! Every `root_value` and `RootMoveStat::q_value` lies in `[-1.0, 1.0]`,
//! positive is good for the root player, and exact `±1.0` is reserved for
//! **proven** results: MCTS terminal children (and a terminal best child's
//! `root_value`), beam terminal-win leaves (and a terminal best leaf's
//! `root_value`), and minimax mate scores. Every unproven (sampled or
//! heuristic) estimate is clamped to `[-UNPROVEN_VALUE_BOUND,
//! UNPROVEN_VALUE_BOUND]` via [`clamp_unproven`] so it can never be
//! mistaken for a proven result. One documented exception: beam has no way
//! to distinguish a terminal loss from a sampled loss once both collapse to
//! `best_value == -1.0` on a `RankedRootMove` (see [`clamp_unproven`]'s
//! call site in `beam_search.rs`), so a proven loss is conservatively
//! reported as `-UNPROVEN_VALUE_BOUND` rather than exactly `-1.0`.

use crate::moves::Move;

/// Largest magnitude an UNPROVEN (sampled or heuristic) value may take.
/// Exact ±1.0 is reserved for proven results (terminal nodes, mates); see
/// the module docs' value invariant.
pub const UNPROVEN_VALUE_BOUND: f64 = 1.0 - 1e-6;

/// Clamp a sampled/heuristic estimate into the open proven-exclusive range.
pub fn clamp_unproven(v: f64) -> f64 {
    v.clamp(-UNPROVEN_VALUE_BOUND, UNPROVEN_VALUE_BOUND)
}

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
    /// value estimate for this move (e.g. an unvisited MCTS child). Exact
    /// `±1.0` only for a proven result; see the module docs' value
    /// invariant and [`clamp_unproven`].
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
    /// See the module docs' value invariant and [`clamp_unproven`].
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
    fn clamp_unproven_never_reaches_exact_one() {
        assert_eq!(clamp_unproven(1.0), UNPROVEN_VALUE_BOUND);
        assert_eq!(clamp_unproven(-1.0), -UNPROVEN_VALUE_BOUND);
        assert_eq!(clamp_unproven(2.5), UNPROVEN_VALUE_BOUND);
        assert_eq!(clamp_unproven(-2.5), -UNPROVEN_VALUE_BOUND);
        assert_eq!(clamp_unproven(0.3), 0.3);
    }

    #[test]
    fn event_counters_default_to_zero() {
        let c = SearchEventCounters::default();
        assert_eq!(
            (
                c.expanded_nodes,
                c.generated_nodes,
                c.transposition_hits,
                c.canonical_dedup_hits,
                c.terminal_hits,
                c.tablebase_hits
            ),
            (0, 0, 0, 0, 0, 0)
        );
    }
}
