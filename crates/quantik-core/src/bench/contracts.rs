//! Contract-oriented projections for benchmark artifacts.
//!
//! The benchmark harness keeps its rich JSON bundle shape for reports and
//! checkpointing. This module projects those rows into the cross-repository
//! contracts used by training and analytics pipelines.

use crate::bench::canonical::canonical_json;
use crate::bench::reference::parse_move_key;
use crate::bitboard::Bitboard;
use crate::constants::{MAX_PIECES_PER_SHAPE, WIN_MASKS};
use crate::game::current_player;
use crate::moves::generate_legal_moves;
use crate::state::State;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

pub const CONTRACT_VERSION: &str = "1.1.0";
pub const MODEL_CHECKPOINT_CONTRACT_VERSION: &str = "1.1.0";
pub const OBSERVATION_CONTRACT_VERSION: &str = "1.0.0";
pub const GAME_RESULT_CONTRACT_VERSION: &str = "1.0.0";
pub const OPENING_BOOK_SCHEMA: &str = "opening-book.v1";
pub const OBSERVATION_SCHEMA: &str = "observation.v1";
pub const GAME_RESULT_SCHEMA: &str = "game-result.v1";
pub const MODEL_CHECKPOINT_SCHEMA: &str = "model-checkpoint.v1";

const SUPPORTED_MODEL_INPUT_CONTRACTS: &[&str] = &[
    "qfen.v1",
    "bitboard.v1",
    "action-index.v1",
    "selfplay.v1",
    "tensor-board.v1",
    "arrow-parquet-selfplay.v1",
    OPENING_BOOK_SCHEMA,
    "opening-book-summary.v1",
    OBSERVATION_SCHEMA,
    GAME_RESULT_SCHEMA,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCheckpointManifest {
    pub schema: String,
    pub contract_version: String,
    pub model_id: String,
    pub model_family: String,
    pub created_at: String,
    pub input_contracts: Vec<String>,
    pub output_contract: String,
    pub weights_format: String,
    pub weights_hash: String,
    pub size_bytes: u64,
    pub training_data_manifest: String,
    pub calibration_report: String,
    pub feature_hash: Option<String>,
    pub quantization: Option<String>,
    pub parameter_count: Option<u64>,
    pub architecture: Option<String>,
    pub legal_action_mask_required: Option<bool>,
    pub recommended_engine_order: Option<Vec<String>>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationRow {
    pub run_id: String,
    pub row_id: u64,
    pub position_key: String,
    pub ply: u64,
    pub side_to_move: u8,
    pub bitboards: Bitboard,
    pub legal_action_mask: u64,
    pub engine_kind: String,
    pub engine_version: String,
    pub policy_visits: Vec<u64>,
    pub value: f64,
    pub value_source: String,
    pub source_confidence: f64,
    pub qfen: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameResultRow {
    pub game_id: String,
    pub p0_engine_kind: String,
    pub p1_engine_kind: String,
    pub initial_position_key: String,
    pub winner: u8,
    pub plies: u64,
    pub terminal_reason: String,
    pub move_action_indices: Vec<u8>,
    pub run_id: Option<String>,
}

impl ModelCheckpointManifest {
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let value: Value =
            serde_json::from_str(text).map_err(|e| format!("parse model checkpoint: {e}"))?;
        Self::from_json_value(value)
    }

    pub fn from_json_value(value: Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or("model checkpoint manifest must be a JSON object")?;

        let manifest = Self {
            schema: required_string(object, "schema")?,
            contract_version: required_string(object, "contract_version")?,
            model_id: required_string(object, "model_id")?,
            model_family: required_string(object, "model_family")?,
            created_at: required_string(object, "created_at")?,
            input_contracts: required_string_list(object, "input_contracts")?,
            output_contract: required_string(object, "output_contract")?,
            weights_format: required_string(object, "weights_format")?,
            weights_hash: required_string(object, "weights_hash")?,
            size_bytes: required_u64(object, "size_bytes")?,
            training_data_manifest: required_string(object, "training_data_manifest")?,
            calibration_report: required_string(object, "calibration_report")?,
            feature_hash: optional_string(object, "feature_hash")?,
            quantization: optional_string(object, "quantization")?,
            parameter_count: optional_u64(object, "parameter_count")?,
            architecture: optional_string(object, "architecture")?,
            legal_action_mask_required: optional_bool(object, "legal_action_mask_required")?,
            recommended_engine_order: optional_string_list(object, "recommended_engine_order")?,
            notes: optional_string(object, "notes")?,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MODEL_CHECKPOINT_SCHEMA {
            return Err(format!(
                "schema must be {MODEL_CHECKPOINT_SCHEMA}, got {}",
                self.schema
            ));
        }
        if self.contract_version != MODEL_CHECKPOINT_CONTRACT_VERSION {
            return Err(format!(
                "contract_version must be {MODEL_CHECKPOINT_CONTRACT_VERSION}, got {}",
                self.contract_version
            ));
        }
        validate_non_empty("model_id", &self.model_id)?;
        validate_non_empty("model_family", &self.model_family)?;
        validate_non_empty("created_at", &self.created_at)?;
        if self.input_contracts.is_empty() {
            return Err("input_contracts must be non-empty".to_string());
        }
        for (index, input_contract) in self.input_contracts.iter().enumerate() {
            validate_non_empty(&format!("input_contracts[{index}]"), input_contract)?;
            if !is_supported_model_input_contract(input_contract) {
                return Err(format!("unsupported input contract: {input_contract}"));
            }
        }
        validate_non_empty("output_contract", &self.output_contract)?;
        validate_non_empty("weights_format", &self.weights_format)?;
        if !is_supported_weights_format(&self.weights_format) {
            return Err(format!(
                "weights_format {} is not supported",
                self.weights_format
            ));
        }
        validate_non_empty("weights_hash", &self.weights_hash)?;
        if self.size_bytes == 0 {
            return Err("size_bytes must be greater than 0".to_string());
        }
        validate_non_empty("training_data_manifest", &self.training_data_manifest)?;
        validate_non_empty("calibration_report", &self.calibration_report)?;

        if let Some(feature_hash) = &self.feature_hash {
            validate_non_empty("feature_hash", feature_hash)?;
        }
        if let Some(quantization) = &self.quantization {
            validate_non_empty("quantization", quantization)?;
        }
        if self.parameter_count == Some(0) {
            return Err("parameter_count must be greater than 0 when present".to_string());
        }
        if let Some(architecture) = &self.architecture {
            validate_non_empty("architecture", architecture)?;
        }
        if let Some(recommended_engine_order) = &self.recommended_engine_order {
            if recommended_engine_order.is_empty() {
                return Err("recommended_engine_order must be non-empty when present".to_string());
            }
            for (index, engine) in recommended_engine_order.iter().enumerate() {
                validate_non_empty(&format!("recommended_engine_order[{index}]"), engine)?;
            }
        }
        if let Some(notes) = &self.notes {
            validate_non_empty("notes", notes)?;
        }
        Ok(())
    }
}

pub fn parse_model_checkpoint_manifest(text: &str) -> Result<ModelCheckpointManifest, String> {
    ModelCheckpointManifest::from_json_str(text)
}

pub fn parse_observation_row(value: &Value) -> Result<ObservationRow, String> {
    let object = value
        .as_object()
        .ok_or("observation row must be a JSON object")?;
    validate_contract_shape(object, OBSERVATION_SCHEMA, OBSERVATION_CONTRACT_VERSION)?;

    let row_id = required_u64(object, "row_id")?;
    let ply = required_u64(object, "ply")?;
    if ply > u16::MAX as u64 {
        return Err("ply must fit in uint16".to_string());
    }
    let side_to_move = required_u8(object, "side_to_move")?;
    if side_to_move > 1 {
        return Err("side_to_move must be 0 or 1".to_string());
    }

    let bitboards = required_bitboards(object, "bitboards")?;
    let expected_side = validate_bitboard_state(&bitboards)?;
    if expected_side != side_to_move {
        return Err("side_to_move does not match bitboards".to_string());
    }

    let qfen = optional_string(object, "qfen")?;
    if let Some(qfen_text) = &qfen {
        let qfen_state = State::from_qfen(qfen_text)?;
        if qfen_state.bb != bitboards {
            return Err("qfen does not match bitboards".to_string());
        }
    }

    let legal_mask = required_u64(object, "legal_action_mask")?;
    let expected_mask = legal_action_mask(&bitboards);
    if legal_mask != expected_mask {
        return Err("legal_action_mask does not match bitboards".to_string());
    }

    let _elapsed_ms = required_u32(object, "elapsed_ms")?;
    let policy_visits = required_u64_list(object, "policy_visits", 64)?;
    if legal_mask != 0 && policy_visits.iter().sum::<u64>() == 0 {
        return Err("policy_visits must contain at least one visit".to_string());
    }
    for (index, visits) in policy_visits.iter().enumerate() {
        if *visits > 0 && ((legal_mask >> index) & 1) == 0 {
            return Err(format!("policy_visits[{index}] is not legal"));
        }
    }

    let source_confidence = required_f64(object, "source_confidence")?;
    if !(0.0..=1.0).contains(&source_confidence) {
        return Err("source_confidence must be in 0.0..1.0".to_string());
    }

    Ok(ObservationRow {
        run_id: required_string(object, "run_id")?,
        row_id,
        position_key: required_string(object, "position_key")?,
        ply,
        side_to_move,
        bitboards,
        legal_action_mask: legal_mask,
        engine_kind: required_string(object, "engine_kind")?,
        engine_version: required_string(object, "engine_version")?,
        policy_visits,
        value: required_f64(object, "value")?,
        value_source: required_string(object, "value_source")?,
        source_confidence,
        qfen,
    })
}

pub fn parse_game_result_row(value: &Value) -> Result<GameResultRow, String> {
    let object = value
        .as_object()
        .ok_or("game-result row must be a JSON object")?;
    validate_contract_shape(object, GAME_RESULT_SCHEMA, GAME_RESULT_CONTRACT_VERSION)?;

    let _started_at = required_string(object, "started_at")?;
    let _p0_engine_version = required_string(object, "p0_engine_version")?;
    let _p1_engine_version = required_string(object, "p1_engine_version")?;

    let winner = required_u8(object, "winner")?;
    if winner > 1 {
        return Err("winner must be 0 or 1".to_string());
    }
    let plies = required_u64(object, "plies")?;
    if plies > u16::MAX as u64 {
        return Err("plies must fit in uint16".to_string());
    }
    let move_action_indices = required_action_index_list(object, "move_action_indices")?;
    if plies != move_action_indices.len() as u64 {
        return Err("plies must match move_action_indices length".to_string());
    }

    Ok(GameResultRow {
        game_id: required_string(object, "game_id")?,
        p0_engine_kind: required_string(object, "p0_engine_kind")?,
        p1_engine_kind: required_string(object, "p1_engine_kind")?,
        initial_position_key: required_string(object, "initial_position_key")?,
        winner,
        plies,
        terminal_reason: required_string(object, "terminal_reason")?,
        move_action_indices,
        run_id: optional_string(object, "run_id")?,
    })
}

pub fn is_supported_weights_format(weights_format: &str) -> bool {
    matches!(
        weights_format,
        "safetensors" | "onnx" | "npz" | "custom-binary"
    )
}

pub fn is_supported_model_input_contract(input_contract: &str) -> bool {
    SUPPORTED_MODEL_INPUT_CONTRACTS.contains(&input_contract)
}

fn validate_non_empty(field: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{field} must be non-empty"));
    }
    Ok(())
}

fn required_string(object: &serde_json::Map<String, Value>, field: &str) -> Result<String, String> {
    let value = object
        .get(field)
        .ok_or_else(|| format!("{field} is required"))?;
    let text = value
        .as_str()
        .ok_or_else(|| format!("{field} must be a string"))?;
    validate_non_empty(field, text)?;
    Ok(text.to_string())
}

fn optional_string(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<String>, String> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let text = value
        .as_str()
        .ok_or_else(|| format!("{field} must be a string when present"))?;
    validate_non_empty(field, text)?;
    Ok(Some(text.to_string()))
}

fn required_string_list(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Vec<String>, String> {
    let value = object
        .get(field)
        .ok_or_else(|| format!("{field} is required"))?;
    let array = value
        .as_array()
        .ok_or_else(|| format!("{field} must be a list of strings"))?;
    string_list(field, array)
}

fn optional_string_list(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<Vec<String>>, String> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let array = value
        .as_array()
        .ok_or_else(|| format!("{field} must be a list of strings when present"))?;
    Ok(Some(string_list(field, array)?))
}

fn string_list(field: &str, array: &[Value]) -> Result<Vec<String>, String> {
    let mut strings = Vec::with_capacity(array.len());
    for (index, value) in array.iter().enumerate() {
        let item_field = format!("{field}[{index}]");
        let text = value
            .as_str()
            .ok_or_else(|| format!("{item_field} must be a string"))?;
        validate_non_empty(&item_field, text)?;
        strings.push(text.to_string());
    }
    Ok(strings)
}

fn required_u64(object: &serde_json::Map<String, Value>, field: &str) -> Result<u64, String> {
    let value = object
        .get(field)
        .ok_or_else(|| format!("{field} is required"))?;
    value
        .as_u64()
        .ok_or_else(|| format!("{field} must be an unsigned integer"))
}

fn required_u32(object: &serde_json::Map<String, Value>, field: &str) -> Result<u32, String> {
    let value = required_u64(object, field)?;
    u32::try_from(value).map_err(|_| format!("{field} must fit in uint32"))
}

fn optional_u64(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<u64>, String> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_u64()
        .ok_or_else(|| format!("{field} must be an unsigned integer when present"))
        .map(Some)
}

fn optional_bool(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<bool>, String> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_bool()
        .ok_or_else(|| format!("{field} must be a boolean when present"))
        .map(Some)
}

fn validate_contract_shape(
    object: &serde_json::Map<String, Value>,
    expected_schema: &str,
    expected_version: &str,
) -> Result<(), String> {
    let schema = required_string(object, "schema")?;
    if schema != expected_schema {
        return Err(format!("schema must be {expected_schema}, got {schema}"));
    }
    let contract_version = required_string(object, "contract_version")?;
    if contract_version != expected_version {
        return Err(format!(
            "contract_version must be {expected_version}, got {contract_version}"
        ));
    }
    Ok(())
}

fn required_u8(object: &serde_json::Map<String, Value>, field: &str) -> Result<u8, String> {
    let value = required_u64(object, field)?;
    u8::try_from(value).map_err(|_| format!("{field} must fit in uint8"))
}

fn required_f64(object: &serde_json::Map<String, Value>, field: &str) -> Result<f64, String> {
    let value = object
        .get(field)
        .ok_or_else(|| format!("{field} is required"))?;
    value
        .as_f64()
        .ok_or_else(|| format!("{field} must be numeric"))
}

fn required_u64_list(
    object: &serde_json::Map<String, Value>,
    field: &str,
    expected_len: usize,
) -> Result<Vec<u64>, String> {
    let value = object
        .get(field)
        .ok_or_else(|| format!("{field} is required"))?;
    let array = value
        .as_array()
        .ok_or_else(|| format!("{field} must be a list of unsigned integers"))?;
    if array.len() != expected_len {
        return Err(format!(
            "{field} must contain exactly {expected_len} unsigned integers"
        ));
    }
    array
        .iter()
        .enumerate()
        .map(|(index, item)| {
            item.as_u64()
                .ok_or_else(|| format!("{field}[{index}] must be an unsigned integer"))
        })
        .collect()
}

fn required_action_index_list(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Vec<u8>, String> {
    let value = object
        .get(field)
        .ok_or_else(|| format!("{field} is required"))?;
    let array = value
        .as_array()
        .ok_or_else(|| format!("{field} must be a list of action indices"))?;
    array
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let action = item
                .as_u64()
                .ok_or_else(|| format!("{field}[{index}] must be an unsigned integer"))?;
            if action >= 64 {
                return Err(format!("{field}[{index}] must be in 0..63"));
            }
            Ok(action as u8)
        })
        .collect()
}

fn required_bitboards(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Bitboard, String> {
    let value = object
        .get(field)
        .ok_or_else(|| format!("{field} is required"))?;
    let array = value
        .as_array()
        .ok_or_else(|| format!("{field} must contain exactly 8 uint16 planes"))?;
    if array.len() != 8 {
        return Err(format!("{field} must contain exactly 8 uint16 planes"));
    }
    let mut planes = [0u16; 8];
    for (index, item) in array.iter().enumerate() {
        let value = item
            .as_u64()
            .ok_or_else(|| format!("{field}[{index}] must be an unsigned integer"))?;
        if value > u16::MAX as u64 {
            return Err(format!("{field}[{index}] must be in 0..65535"));
        }
        planes[index] = value as u16;
    }
    Ok(Bitboard::new(planes))
}

fn validate_bitboard_state(bitboards: &Bitboard) -> Result<u8, String> {
    let mut occupied = 0u16;
    for (index, plane) in bitboards.planes.iter().enumerate() {
        if plane.count_ones() > MAX_PIECES_PER_SHAPE as u32 {
            return Err(format!("bitboards[{index}] exceeds max pieces per shape"));
        }
        if occupied & plane != 0 {
            return Err("bitboards contain overlapping pieces".to_string());
        }
        occupied |= plane;
    }

    for shape in 0..4 {
        let p0 = bitboards.planes[shape];
        let p1 = bitboards.planes[shape + 4];
        for &line in &WIN_MASKS {
            if (p0 & line != 0) && (p1 & line != 0) {
                return Err("bitboards contain illegal same-shape line conflict".to_string());
            }
        }
    }

    current_player(bitboards).ok_or_else(|| "side_to_move does not match bitboards".to_string())
}

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
        "contract_version": OBSERVATION_CONTRACT_VERSION,
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
        "contract_version": GAME_RESULT_CONTRACT_VERSION,
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
        assert_eq!(
            projected["contract_version"],
            json!(OBSERVATION_CONTRACT_VERSION)
        );
        assert_eq!(projected["policy_visits"][18], json!(1));
        assert_eq!(projected["legal_action_mask"], json!(u64::MAX));
        assert_eq!(projected["value_source"], json!("exact"));
    }

    #[test]
    fn observation_projection_round_trips_through_contract_parser() {
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
        let projected = observation_v1_row(7, "run", &row, &positions["p0000"]).unwrap();

        let parsed = parse_observation_row(&projected).unwrap();

        assert_eq!(parsed.run_id, "run");
        assert_eq!(parsed.row_id, 7);
        assert_eq!(parsed.side_to_move, 0);
        assert_eq!(parsed.policy_visits[18], 1);
        assert_eq!(parsed.legal_action_mask, u64::MAX);
    }

    #[test]
    fn observation_parser_rejects_drifted_or_inconsistent_rows() {
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
        let valid = observation_v1_row(0, "run", &row, &positions["p0000"]).unwrap();
        let cases = [
            ("bad version", json!({"contract_version": "1.1.0"})),
            ("bad schema", json!({"schema": "game-result.v1"})),
            ("bad mask", json!({"legal_action_mask": 0})),
            ("bad confidence", json!({"source_confidence": 1.5})),
        ];

        for (label, patch) in cases {
            let mut candidate = valid.clone();
            for (key, value) in patch.as_object().unwrap() {
                candidate[key] = value.clone();
            }

            let error = parse_observation_row(&candidate)
                .expect_err(label)
                .to_string();
            assert!(!error.is_empty(), "expected validation error for {label}");
        }
    }

    #[test]
    fn observation_parser_rejects_policy_on_illegal_action() {
        let bitboards = Bitboard::EMPTY.with_move(0, 0, 0);
        let mut policy_visits = vec![0u64; 64];
        policy_visits[0] = 1;
        let row = json!({
            "schema": OBSERVATION_SCHEMA,
            "contract_version": OBSERVATION_CONTRACT_VERSION,
            "run_id": "run",
            "row_id": 0,
            "position_key": "key",
            "ply": 1,
            "side_to_move": 1,
            "bitboards": bitboards.planes,
            "qfen": "A.../..../..../....",
            "legal_action_mask": legal_action_mask(&bitboards),
            "engine_kind": "minimax",
            "engine_version": "test",
            "elapsed_ms": 0,
            "policy_visits": policy_visits,
            "value": 0.0,
            "value_source": "heuristic",
            "source_confidence": 0.5
        });

        let error = parse_observation_row(&row).unwrap_err();

        assert!(error.contains("policy_visits[0] is not legal"), "{error}");
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
        assert_eq!(
            projected["contract_version"],
            json!(GAME_RESULT_CONTRACT_VERSION)
        );
        assert_eq!(projected["p0_engine_kind"], json!("mcts"));
        assert_eq!(projected["p1_engine_kind"], json!("minimax"));
        assert_eq!(projected["winner"], json!(1));
        assert_eq!(
            projected["move_action_indices"].as_array().unwrap().len(),
            3
        );
    }

    #[test]
    fn game_result_projection_round_trips_through_contract_parser() {
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
            2,
            "run",
            "2026-07-14T00:00:00+0200",
            &record,
            &positions["p0000"],
        )
        .unwrap();

        let parsed = parse_game_result_row(&projected).unwrap();

        assert_eq!(parsed.game_id, "run-00000002");
        assert_eq!(parsed.winner, 1);
        assert_eq!(parsed.plies, 3);
        assert_eq!(parsed.move_action_indices, vec![0, 17, 2]);
        assert_eq!(parsed.run_id.as_deref(), Some("run"));
    }

    #[test]
    fn game_result_parser_rejects_drifted_or_inconsistent_rows() {
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
        let valid = game_result_v1_row(
            0,
            "run",
            "2026-07-14T00:00:00+0200",
            &record,
            &positions["p0000"],
        )
        .unwrap();
        let cases = [
            ("bad version", json!({"contract_version": "1.1.0"})),
            ("bad schema", json!({"schema": "observation.v1"})),
            ("bad winner", json!({"winner": 2})),
            ("bad plies", json!({"plies": 4})),
            ("bad action", json!({"move_action_indices": [64]})),
        ];

        for (label, patch) in cases {
            let mut candidate = valid.clone();
            for (key, value) in patch.as_object().unwrap() {
                candidate[key] = value.clone();
            }

            let error = parse_game_result_row(&candidate)
                .expect_err(label)
                .to_string();
            assert!(!error.is_empty(), "expected validation error for {label}");
        }
    }

    fn valid_model_checkpoint_manifest() -> Value {
        json!({
            "schema": MODEL_CHECKPOINT_SCHEMA,
            "contract_version": MODEL_CHECKPOINT_CONTRACT_VERSION,
            "model_id": "quantik-policy-value-20260714",
            "model_family": "policy-value",
            "created_at": "2026-07-14T12:00:00Z",
            "input_contracts": ["observation.v1"],
            "output_contract": "policy-value.v1",
            "weights_format": "safetensors",
            "weights_hash": "sha256:0123456789abcdef",
            "size_bytes": 1024,
            "training_data_manifest": "training-data-20260714.json",
            "calibration_report": "calibration-20260714.json",
            "feature_hash": "sha256:abcdef0123456789",
            "quantization": "float32",
            "parameter_count": 123456,
            "architecture": "tiny-transformer",
            "legal_action_mask_required": true,
            "recommended_engine_order": ["rust", "python"],
            "notes": "fixture manifest"
        })
    }

    #[test]
    fn model_checkpoint_manifest_parses_and_validates_contract_shape() {
        let manifest =
            ModelCheckpointManifest::from_json_value(valid_model_checkpoint_manifest()).unwrap();

        assert_eq!(manifest.schema, MODEL_CHECKPOINT_SCHEMA);
        assert_eq!(manifest.contract_version, MODEL_CHECKPOINT_CONTRACT_VERSION);
        assert_eq!(manifest.input_contracts, vec!["observation.v1"]);
        assert_eq!(manifest.weights_format, "safetensors");
        assert_eq!(manifest.size_bytes, 1024);
        assert_eq!(manifest.parameter_count, Some(123456));
        assert_eq!(manifest.legal_action_mask_required, Some(true));
        assert_eq!(
            manifest.recommended_engine_order,
            Some(vec!["rust".to_string(), "python".to_string()])
        );
    }

    #[test]
    fn model_checkpoint_manifest_accepts_opening_book_input_contract() {
        let mut manifest = valid_model_checkpoint_manifest();
        manifest["input_contracts"] = json!(["opening-book.v1"]);

        let parsed = ModelCheckpointManifest::from_json_value(manifest).unwrap();

        assert_eq!(parsed.input_contracts, vec!["opening-book.v1"]);
    }

    #[test]
    fn model_checkpoint_manifest_accepts_opening_book_summary_input_contract() {
        let mut manifest = valid_model_checkpoint_manifest();
        manifest["input_contracts"] = json!(["opening-book-summary.v1"]);

        let parsed = ModelCheckpointManifest::from_json_value(manifest).unwrap();

        assert_eq!(parsed.input_contracts, vec!["opening-book-summary.v1"]);
    }

    #[test]
    fn model_checkpoint_manifest_rejects_invalid_required_fields() {
        let cases = [
            ("wrong schema", json!({"schema": "observation.v1"})),
            (
                "wrong contract version",
                json!({"contract_version": "1.0.0"}),
            ),
            ("empty model id", json!({"model_id": ""})),
            ("empty input contracts", json!({"input_contracts": []})),
            (
                "empty input contract value",
                json!({"input_contracts": [""]}),
            ),
            (
                "unsupported input contract",
                json!({"input_contracts": ["unknown.v1"]}),
            ),
            ("empty output contract", json!({"output_contract": ""})),
            (
                "unsupported weights format",
                json!({"weights_format": "pickle"}),
            ),
            ("empty weights hash", json!({"weights_hash": ""})),
            ("zero size", json!({"size_bytes": 0})),
            (
                "empty training data manifest",
                json!({"training_data_manifest": ""}),
            ),
            (
                "empty calibration report",
                json!({"calibration_report": ""}),
            ),
        ];

        for (label, patch) in cases {
            let mut manifest = valid_model_checkpoint_manifest();
            let target = manifest.as_object_mut().unwrap();
            for (key, value) in patch.as_object().unwrap() {
                target.insert(key.clone(), value.clone());
            }

            let error = ModelCheckpointManifest::from_json_value(manifest)
                .expect_err(label)
                .to_string();
            assert!(!error.is_empty(), "expected validation error for {label}");
        }
    }

    #[test]
    fn model_checkpoint_manifest_rejects_mistyped_optional_fields() {
        let cases = [
            ("feature_hash", json!(42)),
            ("quantization", json!(true)),
            ("parameter_count", json!("many")),
            ("architecture", json!(["tiny"])),
            ("legal_action_mask_required", json!("yes")),
            ("recommended_engine_order", json!(["rust", 7])),
            ("notes", json!(false)),
        ];

        for (field, value) in cases {
            let mut manifest = valid_model_checkpoint_manifest();
            manifest[field] = value;

            let error = ModelCheckpointManifest::from_json_value(manifest)
                .expect_err(field)
                .to_string();

            assert!(error.contains(field), "expected {field} in {error}");
        }
    }

    #[test]
    fn model_checkpoint_fixture_parses() {
        let manifest = parse_model_checkpoint_manifest(include_str!(
            "../../tests/fixtures/model-checkpoint-v1.json"
        ))
        .unwrap();

        assert_eq!(manifest.schema, MODEL_CHECKPOINT_SCHEMA);
        assert_eq!(manifest.contract_version, "1.1.0");
        assert_eq!(manifest.weights_format, "safetensors");
    }
}
