//! Uniform engine adapters with effective-work observations.
//!
//! Port of `benchmarks/adapters.py`: every adapter selects a move through
//! the same timed, validated path and reports a [`MoveObservation`] whose
//! JSON field names match the Python `dataclasses.asdict` output exactly,
//! so result bundles stay schema-compatible.

use crate::beam_search::{BeamSearchConfig, BeamSearchEngine};
use crate::bench::reference::move_key;
use crate::bitboard::Bitboard;
use crate::game::has_winning_line;
use crate::mcts::{MCTSConfig, MCTSEngine};
use crate::minimax::{MinimaxConfig, MinimaxEngine};
use crate::moves::{generate_legal_moves, Move};
use crate::state::State;
use rand::prelude::*;
use serde_json::{json, Map, Value};
use std::time::Instant;

/// Format a config label like the Python `_label` helper:
/// `name(k=v,k=v)` with `None` params omitted.
fn label(name: &str, params: &[(&str, Option<String>)]) -> String {
    let parts: Vec<String> = params
        .iter()
        .filter_map(|(key, value)| value.as_ref().map(|v| format!("{key}={v}")))
        .collect();
    if parts.is_empty() {
        name.to_string()
    } else {
        format!("{}({})", name, parts.join(","))
    }
}

/// Render a float parameter the way Python string-formats it in labels.
fn fmt_float(x: f64) -> String {
    crate::bench::canonical::python_float_repr(x)
}

/// Effective work measured for one move selection. Serializes with the
/// exact field names of the Python `MoveObservation` dataclass.
#[derive(Clone, Debug)]
pub struct MoveObservation {
    pub engine: &'static str,
    pub config_label: String,
    pub position_id: String,
    pub mv: String,
    pub wall_time_s: f64,
    pub cpu_time_s: f64,
    pub root_legal_moves: usize,
    pub exact: bool,
    pub seed: Option<u64>,
    pub nodes: Option<u64>,
    pub iterations: Option<u64>,
    pub depth_reached: Option<u32>,
    pub score: Option<f64>,
    pub peak_memory_bytes: Option<u64>,
    pub extra: Map<String, Value>,
}

impl MoveObservation {
    pub fn to_json(&self) -> Value {
        json!({
            "engine": self.engine,
            "config_label": self.config_label,
            "position_id": self.position_id,
            "move": self.mv,
            "wall_time_s": self.wall_time_s,
            "cpu_time_s": self.cpu_time_s,
            "root_legal_moves": self.root_legal_moves,
            "exact": self.exact,
            "seed": self.seed,
            "nodes": self.nodes,
            "iterations": self.iterations,
            "depth_reached": self.depth_reached,
            "score": self.score,
            "peak_memory_bytes": self.peak_memory_bytes,
            "extra": self.extra,
        })
    }
}

/// Engine-specific metrics returned by [`EngineAdapter::select_raw`].
#[derive(Clone, Debug, Default)]
pub struct RawMetrics {
    pub exact: bool,
    pub nodes: Option<u64>,
    pub iterations: Option<u64>,
    pub depth_reached: Option<u32>,
    pub score: Option<f64>,
    pub extra: Map<String, Value>,
}

/// Uniform interface over the benchmarked engines.
pub trait EngineAdapter {
    fn name(&self) -> &'static str;
    fn stochastic(&self) -> bool;
    fn config_label(&self) -> String;
    fn select_raw(&self, bb: &Bitboard, seed: Option<u64>) -> Result<(Move, RawMetrics), String>;
}

/// Process CPU time via `clock_gettime(CLOCK_PROCESS_CPUTIME_ID)`.
fn process_cpu_time_s() -> f64 {
    // SAFETY: clock_gettime with a valid clock id and out-pointer.
    unsafe {
        let mut ts: libc::timespec = std::mem::zeroed();
        if libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) == 0 {
            ts.tv_sec as f64 + ts.tv_nsec as f64 / 1e9
        } else {
            0.0
        }
    }
}

/// Time, validate, and record an engine call (port of
/// `EngineAdapter.select`). Rust's `&Bitboard` makes input mutation
/// impossible by construction; legality and terminality are still checked.
pub fn select(
    adapter: &dyn EngineAdapter,
    bb: &Bitboard,
    position_id: &str,
    seed: Option<u64>,
) -> Result<(Move, MoveObservation), String> {
    let legal = generate_legal_moves(bb);
    if has_winning_line(bb) || legal.is_empty() {
        return Err(format!(
            "{}: cannot select from a terminal state",
            adapter.name()
        ));
    }

    let wall0 = Instant::now();
    let cpu0 = process_cpu_time_s();
    let (mv, metrics) = adapter.select_raw(bb, seed)?;
    let wall_time_s = wall0.elapsed().as_secs_f64();
    let cpu_time_s = process_cpu_time_s() - cpu0;

    if !legal.contains(&mv) {
        return Err(format!("{}: returned illegal move {mv:?}", adapter.name()));
    }

    let observation = MoveObservation {
        engine: adapter.name(),
        config_label: adapter.config_label(),
        position_id: position_id.to_string(),
        mv: move_key(&mv),
        wall_time_s,
        cpu_time_s,
        root_legal_moves: legal.len(),
        exact: metrics.exact,
        seed,
        nodes: metrics.nodes,
        iterations: metrics.iterations,
        depth_reached: metrics.depth_reached,
        score: metrics.score,
        peak_memory_bytes: None,
        extra: metrics.extra,
    };
    Ok((mv, observation))
}

/// Alpha-beta iterative deepening adapter.
pub struct MinimaxAdapter {
    pub max_depth: u32,
    pub time_limit_s: Option<f64>,
}

impl EngineAdapter for MinimaxAdapter {
    fn name(&self) -> &'static str {
        "minimax"
    }
    fn stochastic(&self) -> bool {
        false
    }
    fn config_label(&self) -> String {
        label(
            "minimax",
            &[
                ("d", Some(self.max_depth.to_string())),
                ("t", self.time_limit_s.map(fmt_float)),
            ],
        )
    }
    fn select_raw(&self, bb: &Bitboard, _seed: Option<u64>) -> Result<(Move, RawMetrics), String> {
        let mut engine = MinimaxEngine::new(MinimaxConfig {
            max_depth: self.max_depth,
            time_limit_s: self.time_limit_s,
            ..Default::default()
        });
        let result = engine.search(&State::new(*bb))?;
        if result.pv.first() != Some(&result.best_move) {
            return Err("minimax: best_move inconsistent with reported PV".into());
        }
        let pieces = bb.player_piece_count(0) + bb.player_piece_count(1);
        Ok((
            result.best_move,
            RawMetrics {
                exact: result.depth_reached >= 16 - pieces,
                nodes: Some(result.nodes),
                depth_reached: Some(result.depth_reached),
                score: Some(result.score),
                ..Default::default()
            },
        ))
    }
}

/// Monte Carlo tree search adapter.
pub struct MCTSAdapter {
    pub max_iterations: u32,
    pub max_depth: u32,
    pub exploration_weight: f64,
    pub time_limit_s: Option<f64>,
}

impl EngineAdapter for MCTSAdapter {
    fn name(&self) -> &'static str {
        "mcts"
    }
    fn stochastic(&self) -> bool {
        true
    }
    fn config_label(&self) -> String {
        label(
            "mcts",
            &[
                ("it", Some(self.max_iterations.to_string())),
                ("d", Some(self.max_depth.to_string())),
                ("c", Some(fmt_float(self.exploration_weight))),
                ("t", self.time_limit_s.map(fmt_float)),
            ],
        )
    }
    fn select_raw(&self, bb: &Bitboard, seed: Option<u64>) -> Result<(Move, RawMetrics), String> {
        let mut engine = MCTSEngine::new(MCTSConfig {
            max_iterations: self.max_iterations,
            max_depth: self.max_depth,
            exploration_weight: self.exploration_weight,
            seed,
            time_limit_s: self.time_limit_s,
            ..Default::default()
        });
        let (mv, win_probability) = engine
            .search(bb)
            .ok_or_else(|| "mcts: no move returned".to_string())?;
        Ok((
            mv,
            RawMetrics {
                exact: false,
                iterations: Some(engine.iterations_performed() as u64),
                nodes: Some(engine.nodes_created() as u64),
                score: Some(win_probability),
                ..Default::default()
            },
        ))
    }
}

/// Beam search adapter.
pub struct BeamAdapter {
    pub beam_width: usize,
    pub max_depth: u32,
    pub time_limit_s: Option<f64>,
}

impl EngineAdapter for BeamAdapter {
    fn name(&self) -> &'static str {
        "beam"
    }
    fn stochastic(&self) -> bool {
        true
    }
    fn config_label(&self) -> String {
        label(
            "beam",
            &[
                ("w", Some(self.beam_width.to_string())),
                ("d", Some(self.max_depth.to_string())),
                ("t", self.time_limit_s.map(fmt_float)),
            ],
        )
    }
    fn select_raw(&self, bb: &Bitboard, seed: Option<u64>) -> Result<(Move, RawMetrics), String> {
        let mut engine = BeamSearchEngine::new(BeamSearchConfig {
            beam_width: self.beam_width,
            max_depth: self.max_depth,
            random_seed: seed,
            time_limit_s: self.time_limit_s,
            ..Default::default()
        })?;
        let result = engine.search(bb)?;

        let mv = match result.best_leaf.as_ref().and_then(|l| l.moves.first()) {
            Some(&mv) => mv,
            None => {
                let ranked = result.ranked_root_moves(None);
                ranked
                    .first()
                    .map(|r| r.mv)
                    .ok_or("beam: beam search produced no candidate moves")?
            }
        };

        let score = result.best_leaf.as_ref().map(|leaf| {
            if result.root_player == 1 {
                -leaf.value
            } else {
                leaf.value
            }
        });

        let mut extra = Map::new();
        extra.insert(
            "candidates_generated".into(),
            json!(result.stats.candidates_generated as f64),
        );
        extra.insert(
            "nodes_pruned".into(),
            json!(result.stats.nodes_pruned as f64),
        );
        extra.insert("rollouts".into(), json!(result.stats.rollouts as f64));

        Ok((
            mv,
            RawMetrics {
                exact: false,
                nodes: Some(result.stats.nodes_inserted),
                depth_reached: Some(result.max_depth_reached),
                score,
                extra,
                ..Default::default()
            },
        ))
    }
}

/// Uniform-random baseline adapter.
pub struct RandomAdapter;

impl EngineAdapter for RandomAdapter {
    fn name(&self) -> &'static str {
        "random"
    }
    fn stochastic(&self) -> bool {
        true
    }
    fn config_label(&self) -> String {
        "random".into()
    }
    fn select_raw(&self, bb: &Bitboard, seed: Option<u64>) -> Result<(Move, RawMetrics), String> {
        let moves = generate_legal_moves(bb);
        if moves.is_empty() {
            return Err("random: no legal moves".into());
        }
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Ok((moves[rng.gen_range(0..moves.len())], RawMetrics::default()))
    }
}

/// The fixed-time minimax, MCTS, and beam adapter family.
pub fn fixed_time_adapters(time_limit_s: f64, beam_width: usize) -> Vec<Box<dyn EngineAdapter>> {
    vec![
        Box::new(MinimaxAdapter {
            max_depth: 16,
            time_limit_s: Some(time_limit_s),
        }),
        Box::new(MCTSAdapter {
            max_iterations: 10_000_000,
            max_depth: 16,
            exploration_weight: std::f64::consts::SQRT_2,
            time_limit_s: Some(time_limit_s),
        }),
        Box::new(BeamAdapter {
            beam_width,
            max_depth: 16,
            time_limit_s: Some(time_limit_s),
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::apply_move;

    fn cheap_adapters() -> Vec<Box<dyn EngineAdapter>> {
        vec![
            Box::new(MinimaxAdapter {
                max_depth: 2,
                time_limit_s: None,
            }),
            Box::new(MCTSAdapter {
                max_iterations: 50,
                max_depth: 16,
                exploration_weight: std::f64::consts::SQRT_2,
                time_limit_s: None,
            }),
            Box::new(BeamAdapter {
                beam_width: 8,
                max_depth: 4,
                time_limit_s: None,
            }),
            Box::new(RandomAdapter),
        ]
    }

    #[test]
    fn adapters_return_legal_reproducible_moves() {
        let bb = Bitboard::EMPTY.with_move(0, 0, 5).with_move(1, 2, 10);
        let legal = generate_legal_moves(&bb);
        for adapter in cheap_adapters() {
            let (mv1, obs1) = select(adapter.as_ref(), &bb, "t1", Some(9)).unwrap();
            let (mv2, obs2) = select(adapter.as_ref(), &bb, "t1", Some(9)).unwrap();
            assert!(legal.contains(&mv1), "{}", adapter.name());
            assert_eq!(mv1, mv2, "{} not reproducible", adapter.name());
            assert_eq!(obs1.mv, obs2.mv);
            assert!(obs1.wall_time_s >= 0.0);
            assert_eq!(obs1.root_legal_moves, legal.len());
        }
    }

    #[test]
    fn labels_match_python_format() {
        let minimax = MinimaxAdapter {
            max_depth: 16,
            time_limit_s: Some(1.0),
        };
        assert_eq!(minimax.config_label(), "minimax(d=16,t=1.0)");
        let mcts = MCTSAdapter {
            max_iterations: 10_000_000,
            max_depth: 16,
            exploration_weight: 1.414,
            time_limit_s: Some(1.0),
        };
        assert_eq!(mcts.config_label(), "mcts(it=10000000,d=16,c=1.414,t=1.0)");
        let beam = BeamAdapter {
            beam_width: 256,
            max_depth: 16,
            time_limit_s: Some(1.0),
        };
        assert_eq!(beam.config_label(), "beam(w=256,d=16,t=1.0)");
        assert_eq!(RandomAdapter.config_label(), "random");
        let native = MinimaxAdapter {
            max_depth: 6,
            time_limit_s: None,
        };
        assert_eq!(native.config_label(), "minimax(d=6)");
    }

    #[test]
    fn select_rejects_terminal_state() {
        let won = Bitboard::EMPTY
            .with_move(0, 0, 0)
            .with_move(1, 1, 1)
            .with_move(0, 2, 2)
            .with_move(1, 3, 3);
        assert!(select(&RandomAdapter, &won, "t", Some(0)).is_err());
    }

    #[test]
    fn observation_json_field_names_match_python() {
        let bb = Bitboard::EMPTY;
        let (_, obs) = select(&RandomAdapter, &bb, "p0000", Some(1)).unwrap();
        let value = obs.to_json();
        for field in [
            "engine",
            "config_label",
            "position_id",
            "move",
            "wall_time_s",
            "cpu_time_s",
            "root_legal_moves",
            "exact",
            "seed",
            "nodes",
            "iterations",
            "depth_reached",
            "score",
            "peak_memory_bytes",
            "extra",
        ] {
            assert!(value.get(field).is_some(), "missing {field}");
        }
        assert_eq!(value["engine"], json!("random"));
        assert_eq!(value["nodes"], Value::Null);
    }

    #[test]
    fn minimax_exactness_flag() {
        // 12 pieces: depth 4 remains; max_depth 16 without a time limit
        // solves exactly, so exact must be true on a late position.
        let mut bb = Bitboard::EMPTY;
        let mut rng = StdRng::seed_from_u64(4);
        let mut placed = 0;
        while placed < 12 {
            let moves = generate_legal_moves(&bb);
            if moves.is_empty() {
                bb = Bitboard::EMPTY;
                placed = 0;
                continue;
            }
            let next = apply_move(&bb, &moves[rng.gen_range(0..moves.len())]);
            if has_winning_line(&next) {
                bb = Bitboard::EMPTY;
                placed = 0;
                continue;
            }
            bb = next;
            placed += 1;
        }
        if generate_legal_moves(&bb).is_empty() {
            return; // stalemate start: nothing to assert for this seed
        }
        let adapter = MinimaxAdapter {
            max_depth: 16,
            time_limit_s: None,
        };
        let (_, obs) = select(&adapter, &bb, "deep", None).unwrap();
        assert!(obs.exact);
        assert_eq!(obs.engine, "minimax");
    }
}
