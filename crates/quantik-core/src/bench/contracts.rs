//! Contract-oriented projections for benchmark artifacts.
//!
//! The benchmark harness keeps its rich JSON bundle shape for reports and
//! checkpointing. This module projects those rows into the cross-repository
//! contracts used by training and analytics pipelines.

use crate::bench::canonical::canonical_json;
use crate::bench::reference::parse_move_key;
use crate::bitboard::Bitboard;
use crate::game::current_player;
use crate::moves::generate_legal_moves;
use crate::state::State;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

pub const CONTRACT_VERSION: &str = "1.0.0";
pub const OPENING_BOOK_SCHEMA: &str = "opening-book.v1";
pub const OBSERVATION_SCHEMA: &str = "observation.v1";
pub const GAME_RESULT_SCHEMA: &str = "game-result.v1";
pub const MODEL_CHECKPOINT_SCHEMA: &str = "model-checkpoint.v1";

pub fn action_index(shape: u8, position: u8) -> u8 {
    shape * 16 + position
}

pub fn legal_action_mask(bb: &Bitboard) -> u64 {
    generate_legal_moves(bb).into_iter().fold(0u64, |mask, mv| {
        mask | (1u64 << action_index(mv.shape, mv.position))
    })
}

pub fn canonical_key_hex(state: &State) -> String {
    state
        .canonical_key()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn position_lookup(dataset_payload: &Value) -> Result<BTreeMap<String, Value>, String> {
    let positions = dataset_payload["positions"]
        .as_array()
        .ok_or("dataset has no positions")?;
    let mut by_id = BTreeMap::new();
    for position in positions {
        let id = position["id"]
            .as_str()
            .ok_or("dataset position missing id")?
            .to_string();
        by_id.insert(id, position.clone());
    }
    Ok(by_id)
}

pub fn observation_v1_row(
    row_id: u64,
    run_id: &str,
    benchmark_row: &Value,
    position: &Value,
) -> Result<Value, String> {
    let qfen = position["qfen"]
        .as_str()
        .ok_or("dataset position missing qfen")?;
    let state = State::from_qfen(qfen)?;
    let bb = state.bb;
    let (_, shape, position_index) = parse_move_key(
        benchmark_row["move"]
            .as_str()
            .ok_or("benchmark observation missing move")?,
    )?;
    let selected_action = action_index(shape, position_index) as usize;
    let mut policy_visits = vec![0u32; 64];
    policy_visits[selected_action] = 1;
    let mut root_q_values = vec![0.0f64; 64];
    if let Some(score) = benchmark_row["score"].as_f64() {
        root_q_values[selected_action] = score;
    }

    let source_confidence = if benchmark_row["exact"].as_bool() == Some(true) {
        1.0
    } else {
        0.5
    };

    Ok(json!({
        "schema": OBSERVATION_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "row_id": row_id,
        "position_key": canonical_key_hex(&state),
        "ply": bb.player_piece_count(0) + bb.player_piece_count(1),
        "side_to_move": current_player(&bb).ok_or("inconsistent side to move")?,
        "bitboards": bb.planes,
        "qfen": qfen,
        "legal_action_mask": legal_action_mask(&bb),
        "engine_kind": benchmark_row["engine"].as_str().unwrap_or("unknown"),
        "engine_checkpoint": Value::Null,
        "engine_version": env!("CARGO_PKG_VERSION"),
        "search_depth": benchmark_row["depth_reached"],
        "rollouts": benchmark_row["iterations"],
        "beam_width": Value::Null,
        "node_budget": benchmark_row["nodes"],
        "time_budget_ms": Value::Null,
        "elapsed_ms": (benchmark_row["wall_time_s"].as_f64().unwrap_or(0.0) * 1000.0).round() as u64,
        "seed": benchmark_row["seed"],
        "policy_visits": policy_visits,
        "policy_priors": Value::Null,
        "root_q_values": root_q_values,
        "value": benchmark_row["score"].as_f64().unwrap_or(0.0),
        "value_source": if benchmark_row["exact"].as_bool() == Some(true) { "exact" } else { "heuristic" },
        "source_confidence": source_confidence,
        "principal_variation": Value::Null,
    }))
}

pub fn game_result_v1_row(
    row_id: u64,
    run_id: &str,
    started_at: &str,
    h2h_record: &Value,
    position: &Value,
) -> Result<Value, String> {
    let qfen = position["qfen"]
        .as_str()
        .ok_or("dataset position missing qfen")?;
    let state = State::from_qfen(qfen)?;
    let bb = state.bb;
    let side_to_move = current_player(&bb).ok_or("inconsistent side to move")?;
    let mover = h2h_record["mover"].as_str().ok_or("h2h missing mover")?;
    let responder = h2h_record["responder"]
        .as_str()
        .ok_or("h2h missing responder")?;
    let (p0_engine, p1_engine) = if side_to_move == 0 {
        (mover, responder)
    } else {
        (responder, mover)
    };
    let winner_engine = h2h_record["winner"].as_str().ok_or("h2h missing winner")?;
    let winner = if winner_engine == p0_engine { 0 } else { 1 };
    let move_action_indices = h2h_record["move_action_indices"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    Ok(json!({
        "schema": GAME_RESULT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "game_id": format!("{run_id}-{row_id:08}"),
        "started_at": started_at,
        "p0_engine_kind": p0_engine,
        "p0_engine_version": env!("CARGO_PKG_VERSION"),
        "p0_engine_checkpoint": Value::Null,
        "p1_engine_kind": p1_engine,
        "p1_engine_version": env!("CARGO_PKG_VERSION"),
        "p1_engine_checkpoint": Value::Null,
        "opening_book_id": Value::Null,
        "initial_position_key": canonical_key_hex(&state),
        "winner": winner,
        "plies": h2h_record["plies"],
        "terminal_reason": "win_condition_or_no_legal_moves",
        "seed": h2h_record["seed"],
        "time_budget_ms_per_move": Value::Null,
        "node_budget_per_move": Value::Null,
        "move_action_indices": move_action_indices,
        "position_keys": Value::Null,
        "hardware": Value::Null,
        "run_id": run_id,
    }))
}

pub fn export_observation_rows(
    bundle: &Value,
    dataset_payload: &Value,
    output: &Path,
) -> Result<usize, String> {
    let positions = position_lookup(dataset_payload)?;
    let run_id = bundle["started_at"].as_str().unwrap_or("benchmark-run");
    let rows = bundle["observations"]
        .as_array()
        .ok_or("bundle has no observations array")?;
    write_jsonl(
        output,
        rows.iter().enumerate().map(|(index, row)| {
            let position_id = row["position_id"]
                .as_str()
                .ok_or("observation missing position_id")?;
            let position = positions
                .get(position_id)
                .ok_or_else(|| format!("dataset missing position {position_id}"))?;
            observation_v1_row(index as u64, run_id, row, position)
        }),
    )
}

pub fn export_game_result_rows(
    bundle: &Value,
    dataset_payload: &Value,
    output: &Path,
) -> Result<usize, String> {
    let positions = position_lookup(dataset_payload)?;
    let run_id = bundle["started_at"].as_str().unwrap_or("benchmark-run");
    let started_at = bundle["started_at"].as_str().unwrap_or("unknown");
    let rows = bundle["head_to_head"]["records"]
        .as_array()
        .ok_or("bundle has no head_to_head.records array")?;
    write_jsonl(
        output,
        rows.iter().enumerate().map(|(index, row)| {
            let position_id = row["position_id"]
                .as_str()
                .ok_or("h2h record missing position_id")?;
            let position = positions
                .get(position_id)
                .ok_or_else(|| format!("dataset missing position {position_id}"))?;
            game_result_v1_row(index as u64, run_id, started_at, row, position)
        }),
    )
}

fn write_jsonl<I>(output: &Path, rows: I) -> Result<usize, String>
where
    I: Iterator<Item = Result<Value, String>>,
{
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {parent:?}: {e}"))?;
    }
    let mut file = std::fs::File::create(output).map_err(|e| format!("create {output:?}: {e}"))?;
    let mut count = 0usize;
    for row in rows {
        writeln!(file, "{}", canonical_json(&row?)).map_err(|e| format!("write: {e}"))?;
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset() -> Value {
        json!({
            "positions": [{
                "id": "p0000",
                "qfen": "..../..../..../....",
                "phase": "opening"
            }]
        })
    }

    #[test]
    fn legal_mask_empty_board_covers_all_actions() {
        assert_eq!(legal_action_mask(&Bitboard::EMPTY), u64::MAX);
    }

    #[test]
    fn observation_projection_has_contract_shape() {
        let positions = position_lookup(&dataset()).unwrap();
        let row = json!({
            "position_id": "p0000",
            "engine": "minimax",
            "move": "0:1:2",
            "wall_time_s": 0.25,
            "exact": true,
            "seed": null,
            "nodes": 10,
            "iterations": null,
            "depth_reached": 4,
            "score": 1.0
        });
        let projected = observation_v1_row(0, "run", &row, &positions["p0000"]).unwrap();
        assert_eq!(projected["schema"], OBSERVATION_SCHEMA);
        assert_eq!(projected["contract_version"], CONTRACT_VERSION);
        assert_eq!(projected["policy_visits"][18], json!(1));
        assert_eq!(projected["legal_action_mask"], json!(u64::MAX));
        assert_eq!(projected["value_source"], json!("exact"));
    }

    #[test]
    fn game_result_projection_maps_engines_to_players() {
        let positions = position_lookup(&dataset()).unwrap();
        let record = json!({
            "position_id": "p0000",
            "mover": "mcts",
            "responder": "minimax",
            "winner": "minimax",
            "plies": 3,
            "seed": 7,
            "move_action_indices": [0, 17, 2]
        });
        let projected = game_result_v1_row(
            0,
            "run",
            "2026-07-14T00:00:00+0200",
            &record,
            &positions["p0000"],
        )
        .unwrap();
        assert_eq!(projected["schema"], GAME_RESULT_SCHEMA);
        assert_eq!(projected["p0_engine_kind"], json!("mcts"));
        assert_eq!(projected["p1_engine_kind"], json!("minimax"));
        assert_eq!(projected["winner"], json!(1));
        assert_eq!(
            projected["move_action_indices"].as_array().unwrap().len(),
            3
        );
    }
}
