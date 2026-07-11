//! Correctness preflight for benchmark inputs and adapters.
//!
//! Port of `benchmarks/correctness.py`: `run` refuses to benchmark until
//! these invariants pass — non-terminal dataset positions, legal moves for
//! the correct side, and seed reproducibility.

use crate::bench::adapters::{select, EngineAdapter};
use crate::bench::reference::parse_move_key;
use crate::game::has_winning_line;
use crate::moves::generate_legal_moves;
use crate::state::State;
use serde_json::Value;

fn probe_adapter(adapter: &dyn EngineAdapter, position: &Value, seed: u64) -> Vec<String> {
    let mut failures = Vec::new();
    let id = position["id"].as_str().unwrap_or("?");
    let bb = match State::from_qfen(position["qfen"].as_str().unwrap_or_default()) {
        Ok(state) => state.bb,
        Err(e) => return vec![format!("{} on {}: bad qfen: {e}", adapter.name(), id)],
    };

    let first = match select(adapter, &bb, id, Some(seed)) {
        Ok((_, observation)) => observation,
        Err(e) => return vec![format!("{} on {}: {e}", adapter.name(), id)],
    };

    match parse_move_key(&first.mv) {
        Ok((mover, _, _)) => {
            let side_to_move = position["side_to_move"].as_u64().unwrap_or(0) as u8;
            if mover != side_to_move {
                failures.push(format!(
                    "{} on {}: moved for player {mover}, but side to move is {side_to_move}",
                    adapter.name(),
                    id
                ));
            }
        }
        Err(e) => failures.push(format!("{} on {}: {e}", adapter.name(), id)),
    }

    match select(adapter, &bb, id, Some(seed)) {
        Ok((_, second)) => {
            if second.mv != first.mv {
                failures.push(format!(
                    "{} on {}: non-deterministic under identical settings and seed ({} vs {})",
                    adapter.name(),
                    id,
                    first.mv,
                    second.mv
                ));
            }
        }
        Err(e) => failures.push(format!(
            "{} on {} reproducibility check: {e}",
            adapter.name(),
            id
        )),
    }
    failures
}

/// Return human-readable invariant failures; an empty list means all good.
pub fn run_preflight(adapters: &[Box<dyn EngineAdapter>], positions: &[Value]) -> Vec<String> {
    let sample = 3;
    let seed = 0u64;
    let mut failures = Vec::new();

    for position in positions {
        let id = position["id"].as_str().unwrap_or("?");
        match State::from_qfen(position["qfen"].as_str().unwrap_or_default()) {
            Ok(state) => {
                let bb = state.bb;
                if has_winning_line(&bb) || generate_legal_moves(&bb).is_empty() {
                    failures.push(format!("dataset: position {id} is terminal"));
                }
            }
            Err(e) => failures.push(format!("dataset: position {id} bad qfen: {e}")),
        }
    }

    for adapter in adapters {
        for position in positions.iter().take(sample) {
            failures.extend(probe_adapter(adapter.as_ref(), position, seed));
        }
    }

    failures
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::adapters::{BeamAdapter, MCTSAdapter, MinimaxAdapter, RandomAdapter};
    use crate::bench::dataset;
    use std::path::Path;

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
    fn preflight_passes_on_golden_dataset() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/positions-v1.json");
        let payload = dataset::load(&path).unwrap();
        let positions = payload["positions"].as_array().unwrap();
        let failures = run_preflight(&cheap_adapters(), positions);
        assert!(failures.is_empty(), "{failures:#?}");
    }

    #[test]
    fn preflight_flags_terminal_position() {
        let position = serde_json::json!({
            "id": "bad1",
            // Row 0 completed: A b C d.
            "qfen": "AbCd/..../..../....",
            "side_to_move": 0,
        });
        let failures = run_preflight(&[], &[position]);
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("terminal"));
    }
}
