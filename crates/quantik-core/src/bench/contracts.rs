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
#[cfg(feature = "arrow-parquet")]
use std::collections::HashMap;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "arrow-parquet")]
use std::fs::File;
use std::io::Write;
use std::path::Path;
#[cfg(feature = "arrow-parquet")]
use std::sync::Arc;

pub const CONTRACT_VERSION: &str = "1.1.0";
pub const MODEL_CHECKPOINT_CONTRACT_VERSION: &str = "1.1.0";
pub const SELFPLAY_CONTRACT_VERSION: &str = "1.1.0";
pub const OBSERVATION_CONTRACT_VERSION: &str = "1.1.0";
pub const GAME_RESULT_CONTRACT_VERSION: &str = "1.1.0";
pub const SELFPLAY_SCHEMA: &str = "selfplay.v1";
pub const ARROW_PARQUET_SELFPLAY_SCHEMA: &str = "arrow-parquet-selfplay.v1";
pub const OPENING_BOOK_SCHEMA: &str = "opening-book.v1";
pub const OBSERVATION_SCHEMA: &str = "observation.v1";
pub const GAME_RESULT_SCHEMA: &str = "game-result.v1";
pub const MODEL_CHECKPOINT_SCHEMA: &str = "model-checkpoint.v1";

const SUPPORTED_MODEL_INPUT_CONTRACTS: &[&str] = &[
    "qfen.v1",
    "bitboard.v1",
    "action-index.v1",
    SELFPLAY_SCHEMA,
    "tensor-board.v1",
    ARROW_PARQUET_SELFPLAY_SCHEMA,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfPlayPolicyVisit {
    pub shape: u8,
    pub position: u8,
    pub visits: u32,
}

impl SelfPlayPolicyVisit {
    pub fn action_index(&self) -> usize {
        action_index(self.shape, self.position) as usize
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelfPlayRow {
    pub game_id: u64,
    pub ply: u64,
    pub qfen: String,
    pub side_to_move: u8,
    pub policy: Vec<SelfPlayPolicyVisit>,
    pub value: f64,
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

pub fn parse_selfplay_row(value: &Value) -> Result<SelfPlayRow, String> {
    let object = value
        .as_object()
        .ok_or("selfplay row must be a JSON object")?;
    validate_schema_with_optional_version(object, SELFPLAY_SCHEMA, SELFPLAY_CONTRACT_VERSION)?;

    let game_id = required_u64(object, "game_id")?;
    let ply = required_u64(object, "ply")?;
    let qfen = required_string(object, "qfen")?;
    let side_to_move = required_u8(object, "side_to_move")?;
    if side_to_move > 1 {
        return Err("side_to_move must be 0 or 1".to_string());
    }

    let state = State::from_qfen(&qfen)?;
    let expected_side =
        current_player(&state.bb).ok_or_else(|| "side_to_move does not match qfen".to_string())?;
    if expected_side != side_to_move {
        return Err("side_to_move does not match qfen".to_string());
    }

    let policy = required_selfplay_policy(object, "policy")?;
    validate_selfplay_policy_is_legal(&state.bb, &policy)?;

    let value = required_f64(object, "value")?;
    if value != -1.0 && value != 1.0 {
        return Err("value must be exactly -1.0 or 1.0".to_string());
    }

    Ok(SelfPlayRow {
        game_id,
        ply,
        qfen,
        side_to_move,
        policy,
        value,
    })
}

pub fn selfplay_dense_policy_visits(policy: &[SelfPlayPolicyVisit]) -> Result<[u32; 64], String> {
    if policy.is_empty() {
        return Err("policy must be non-empty".to_string());
    }
    let mut dense = [0u32; 64];
    for visit in policy {
        if visit.shape > 3 {
            return Err("policy shape must be in 0..3".to_string());
        }
        if visit.position > 15 {
            return Err("policy position must be in 0..15".to_string());
        }
        if visit.visits == 0 {
            return Err("policy visits must be positive".to_string());
        }
        let action = visit.action_index();
        dense[action] = dense[action]
            .checked_add(visit.visits)
            .ok_or_else(|| format!("policy visits overflow at action {action}"))?;
    }
    Ok(dense)
}

pub fn selfplay_arrow_parquet_record(row: &SelfPlayRow) -> Result<Value, String> {
    let ply = u16::try_from(row.ply)
        .map_err(|_| "ply must fit in uint16 for arrow-parquet-selfplay.v1".to_string())?;
    let state = State::from_qfen(&row.qfen)?;
    if row.side_to_move > 1 {
        return Err("side_to_move must be 0 or 1".to_string());
    }
    let expected_side =
        current_player(&state.bb).ok_or_else(|| "side_to_move does not match qfen".to_string())?;
    if expected_side != row.side_to_move {
        return Err("side_to_move does not match qfen".to_string());
    }
    validate_selfplay_policy_is_legal(&state.bb, &row.policy)?;
    let value = if row.value == 1.0 {
        1i8
    } else if row.value == -1.0 {
        -1i8
    } else {
        return Err("value must be exactly -1.0 or 1.0".to_string());
    };
    let policy_visits = selfplay_dense_policy_visits(&row.policy)?.to_vec();

    Ok(json!({
        "logical_schema": SELFPLAY_SCHEMA,
        "contract_version": SELFPLAY_CONTRACT_VERSION,
        "game_id": row.game_id,
        "ply": ply,
        "side_to_move": row.side_to_move,
        "bitboards": state.bb.planes,
        "policy_visits": policy_visits,
        "value": value,
        "qfen": row.qfen,
    }))
}

#[cfg(feature = "arrow-parquet")]
pub fn write_selfplay_arrow_parquet<P: AsRef<Path>>(
    path: P,
    rows: &[SelfPlayRow],
) -> Result<(), String> {
    use arrow_array::{
        ArrayRef, FixedSizeListArray, Int8Array, RecordBatch, StringArray, UInt16Array,
        UInt64Array, UInt8Array,
    };
    use arrow_schema::{DataType, Field};
    use parquet::arrow::arrow_writer::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;

    let schema = selfplay_arrow_parquet_schema();
    let mut logical_schema = Vec::with_capacity(rows.len());
    let mut contract_version = Vec::with_capacity(rows.len());
    let mut game_id = Vec::with_capacity(rows.len());
    let mut ply = Vec::with_capacity(rows.len());
    let mut side_to_move = Vec::with_capacity(rows.len());
    let mut bitboards = Vec::with_capacity(rows.len() * 8);
    let mut policy_visits = Vec::with_capacity(rows.len() * 64);
    let mut value = Vec::with_capacity(rows.len());
    let mut qfen = Vec::with_capacity(rows.len());

    for row in rows {
        let physical = selfplay_arrow_parquet_record(row)?;
        logical_schema.push(
            physical["logical_schema"]
                .as_str()
                .ok_or("logical_schema must be a string")?
                .to_string(),
        );
        contract_version.push(
            physical["contract_version"]
                .as_str()
                .ok_or("contract_version must be a string")?
                .to_string(),
        );
        game_id.push(row.game_id);
        ply.push(
            physical["ply"]
                .as_u64()
                .ok_or("ply must be an unsigned integer")? as u16,
        );
        side_to_move.push(row.side_to_move);
        bitboards.extend(
            physical["bitboards"]
                .as_array()
                .ok_or("bitboards must be an array")?
                .iter()
                .map(|plane| {
                    plane
                        .as_u64()
                        .ok_or_else(|| "bitboards entries must be unsigned integers".to_string())
                        .and_then(|plane| {
                            u16::try_from(plane)
                                .map_err(|_| "bitboards entries must fit in uint16".to_string())
                        })
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        policy_visits.extend(selfplay_dense_policy_visits(&row.policy)?);
        value.push(
            physical["value"]
                .as_i64()
                .ok_or("value must be a signed integer")? as i8,
        );
        qfen.push(row.qfen.clone());
    }

    let bitboards_array = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::UInt16, false)),
        8,
        Arc::new(UInt16Array::from(bitboards)),
        None,
    )
    .map_err(|e| format!("build bitboards column: {e}"))?;
    let policy_visits_array = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::UInt32, false)),
        64,
        Arc::new(arrow_array::UInt32Array::from(policy_visits)),
        None,
    )
    .map_err(|e| format!("build policy_visits column: {e}"))?;

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(logical_schema)) as ArrayRef,
            Arc::new(StringArray::from(contract_version)) as ArrayRef,
            Arc::new(UInt64Array::from(game_id)) as ArrayRef,
            Arc::new(UInt16Array::from(ply)) as ArrayRef,
            Arc::new(UInt8Array::from(side_to_move)) as ArrayRef,
            Arc::new(bitboards_array) as ArrayRef,
            Arc::new(policy_visits_array) as ArrayRef,
            Arc::new(Int8Array::from(value)) as ArrayRef,
            Arc::new(StringArray::from(qfen)) as ArrayRef,
        ],
    )
    .map_err(|e| format!("build selfplay record batch: {e}"))?;

    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_key_value_metadata(Some(selfplay_arrow_parquet_metadata()))
        .build();
    let file = File::create(path.as_ref())
        .map_err(|e| format!("create {}: {e}", path.as_ref().display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| format!("create selfplay parquet writer: {e}"))?;
    writer
        .write(&batch)
        .map_err(|e| format!("write selfplay parquet batch: {e}"))?;
    writer
        .close()
        .map_err(|e| format!("close selfplay parquet writer: {e}"))?;
    Ok(())
}

#[cfg(feature = "arrow-parquet")]
pub fn read_selfplay_arrow_parquet<P: AsRef<Path>>(path: P) -> Result<Vec<SelfPlayRow>, String> {
    use arrow_array::Array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let file =
        File::open(path.as_ref()).map_err(|e| format!("open {}: {e}", path.as_ref().display()))?;
    let metadata_reader = SerializedFileReader::new(
        file.try_clone()
            .map_err(|e| format!("clone {}: {e}", path.as_ref().display()))?,
    )
    .map_err(|e| format!("read parquet metadata: {e}"))?;
    validate_selfplay_arrow_parquet_metadata(
        metadata_reader
            .metadata()
            .file_metadata()
            .key_value_metadata(),
    )?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("open selfplay parquet reader: {e}"))?;
    validate_selfplay_arrow_schema(builder.schema().as_ref())?;
    let reader = builder
        .build()
        .map_err(|e| format!("build selfplay parquet reader: {e}"))?;
    let mut rows = Vec::new();

    for batch in reader {
        let batch = batch.map_err(|e| format!("read selfplay parquet batch: {e}"))?;
        validate_selfplay_arrow_schema(batch.schema().as_ref())?;
        let logical_schema = string_column(&batch, 0, "logical_schema")?;
        let contract_version = string_column(&batch, 1, "contract_version")?;
        let game_id = u64_column(&batch, 2, "game_id")?;
        let ply = u16_column(&batch, 3, "ply")?;
        let side_to_move = u8_column(&batch, 4, "side_to_move")?;
        let bitboards = fixed_list_column(&batch, 5, "bitboards", 8)?;
        let policy_visits = fixed_list_column(&batch, 6, "policy_visits", 64)?;
        let value = i8_column(&batch, 7, "value")?;
        let qfen = nullable_string_column(&batch, 8, "qfen")?;

        for row_index in 0..batch.num_rows() {
            let row_logical_schema = logical_schema.value(row_index);
            if row_logical_schema != SELFPLAY_SCHEMA {
                return Err(format!(
                    "logical_schema must be {SELFPLAY_SCHEMA}, got {row_logical_schema}"
                ));
            }
            let row_contract_version = contract_version.value(row_index);
            if row_contract_version != SELFPLAY_CONTRACT_VERSION {
                return Err(format!(
                    "contract_version must be {SELFPLAY_CONTRACT_VERSION}, got {row_contract_version}"
                ));
            }
            let side = side_to_move.value(row_index);
            if side > 1 {
                return Err("side_to_move must be 0 or 1".to_string());
            }
            let physical_value = value.value(row_index);
            let logical_value = match physical_value {
                -1 => -1.0,
                1 => 1.0,
                _ => return Err("value must be exactly -1 or 1".to_string()),
            };
            let physical_bitboards = bitboards_u16_at(bitboards, row_index, "bitboards")?;
            let bitboard = Bitboard::new(physical_bitboards);
            validate_bitboard_state(&bitboard)?;
            let qfen_text = if qfen.is_null(row_index) {
                State::new(bitboard).to_qfen()
            } else {
                let qfen_text = qfen.value(row_index);
                let qfen_state = State::from_qfen(qfen_text)?;
                if qfen_state.bb != bitboard {
                    return Err("qfen does not match bitboards".to_string());
                }
                qfen_text.to_string()
            };
            let dense_policy = policy_visits_u32_at(policy_visits, row_index, "policy_visits")?;
            let policy = dense_policy
                .iter()
                .enumerate()
                .filter(|(_, visits)| **visits > 0)
                .map(|(action, visits)| SelfPlayPolicyVisit {
                    shape: (action / 16) as u8,
                    position: (action % 16) as u8,
                    visits: *visits,
                })
                .collect::<Vec<_>>();
            if policy.is_empty() {
                return Err("policy_visits must contain at least one visit".to_string());
            }

            let logical = selfplay_v1_row(
                game_id.value(row_index),
                ply.value(row_index) as u64,
                &qfen_text,
                side,
                &policy,
                logical_value,
            )?;
            rows.push(parse_selfplay_row(&logical)?);
        }
    }

    Ok(rows)
}

pub fn selfplay_v1_row(
    game_id: u64,
    ply: u64,
    qfen: &str,
    side_to_move: u8,
    policy: &[SelfPlayPolicyVisit],
    value: f64,
) -> Result<Value, String> {
    let mut policy_json = Vec::with_capacity(policy.len());
    for visit in policy {
        policy_json.push(json!({
            "shape": visit.shape,
            "position": visit.position,
            "visits": visit.visits,
        }));
    }
    let record = json!({
        "schema": SELFPLAY_SCHEMA,
        "contract_version": SELFPLAY_CONTRACT_VERSION,
        "game_id": game_id,
        "ply": ply,
        "qfen": qfen,
        "side_to_move": side_to_move,
        "policy": policy_json,
        "value": value,
    });
    parse_selfplay_row(&record)?;
    Ok(record)
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

fn validate_schema_with_optional_version(
    object: &serde_json::Map<String, Value>,
    expected_schema: &str,
    expected_version: &str,
) -> Result<(), String> {
    let schema = required_string(object, "schema")?;
    if schema != expected_schema {
        return Err(format!("schema must be {expected_schema}, got {schema}"));
    }
    if let Some(contract_version) = optional_string(object, "contract_version")? {
        if contract_version != expected_version {
            return Err(format!(
                "contract_version must be {expected_version}, got {contract_version}"
            ));
        }
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

fn required_selfplay_policy(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Vec<SelfPlayPolicyVisit>, String> {
    let value = object
        .get(field)
        .ok_or_else(|| format!("{field} is required"))?;
    let array = value
        .as_array()
        .ok_or_else(|| format!("{field} must be a non-empty list"))?;
    if array.is_empty() {
        return Err(format!("{field} must be a non-empty list"));
    }

    let mut seen = BTreeSet::new();
    let mut visits = Vec::with_capacity(array.len());
    for (index, item) in array.iter().enumerate() {
        let item_object = item
            .as_object()
            .ok_or_else(|| format!("{field}[{index}] must be an object"))?;
        let shape = required_u8(item_object, "shape")?;
        if shape > 3 {
            return Err(format!("{field}[{index}].shape must be in 0..3"));
        }
        let position = required_u8(item_object, "position")?;
        if position > 15 {
            return Err(format!("{field}[{index}].position must be in 0..15"));
        }
        let visit_count = required_u32(item_object, "visits")?;
        if visit_count == 0 {
            return Err(format!("{field}[{index}].visits must be positive"));
        }
        if !seen.insert((shape, position)) {
            return Err(format!(
                "{field}[{index}] duplicates shape={shape}, position={position}"
            ));
        }
        visits.push(SelfPlayPolicyVisit {
            shape,
            position,
            visits: visit_count,
        });
    }
    Ok(visits)
}

fn validate_selfplay_policy_is_legal(
    bitboards: &Bitboard,
    policy: &[SelfPlayPolicyVisit],
) -> Result<(), String> {
    let legal_mask = legal_action_mask(bitboards);
    for visit in policy {
        let action = visit.action_index();
        if ((legal_mask >> action) & 1) == 0 {
            return Err(format!(
                "policy action is not legal for row state: shape={}, position={}",
                visit.shape, visit.position
            ));
        }
    }
    Ok(())
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

#[cfg(feature = "arrow-parquet")]
fn selfplay_arrow_parquet_schema() -> Arc<arrow_schema::Schema> {
    use arrow_schema::{DataType, Field, Schema};

    let metadata = HashMap::from([
        (
            "schema".to_string(),
            ARROW_PARQUET_SELFPLAY_SCHEMA.to_string(),
        ),
        ("logical_schema".to_string(), SELFPLAY_SCHEMA.to_string()),
        ("logical_contract".to_string(), SELFPLAY_SCHEMA.to_string()),
        (
            "contracts_release".to_string(),
            SELFPLAY_CONTRACT_VERSION.to_string(),
        ),
        (
            "contract_version".to_string(),
            SELFPLAY_CONTRACT_VERSION.to_string(),
        ),
    ]);

    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("logical_schema", DataType::Utf8, false),
            Field::new("contract_version", DataType::Utf8, false),
            Field::new("game_id", DataType::UInt64, false),
            Field::new("ply", DataType::UInt16, false),
            Field::new("side_to_move", DataType::UInt8, false),
            Field::new(
                "bitboards",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt16, false)), 8),
                false,
            ),
            Field::new(
                "policy_visits",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt32, false)), 64),
                false,
            ),
            Field::new("value", DataType::Int8, false),
            Field::new("qfen", DataType::Utf8, true),
        ],
        metadata,
    ))
}

#[cfg(feature = "arrow-parquet")]
fn selfplay_arrow_parquet_metadata() -> Vec<parquet::file::metadata::KeyValue> {
    vec![
        parquet::file::metadata::KeyValue {
            key: "schema".to_string(),
            value: Some(ARROW_PARQUET_SELFPLAY_SCHEMA.to_string()),
        },
        parquet::file::metadata::KeyValue {
            key: "logical_schema".to_string(),
            value: Some(SELFPLAY_SCHEMA.to_string()),
        },
        parquet::file::metadata::KeyValue {
            key: "logical_contract".to_string(),
            value: Some(SELFPLAY_SCHEMA.to_string()),
        },
        parquet::file::metadata::KeyValue {
            key: "contracts_release".to_string(),
            value: Some(SELFPLAY_CONTRACT_VERSION.to_string()),
        },
        parquet::file::metadata::KeyValue {
            key: "contract_version".to_string(),
            value: Some(SELFPLAY_CONTRACT_VERSION.to_string()),
        },
    ]
}

#[cfg(feature = "arrow-parquet")]
fn validate_selfplay_arrow_parquet_metadata(
    metadata: Option<&Vec<parquet::file::metadata::KeyValue>>,
) -> Result<(), String> {
    let metadata = metadata.ok_or("parquet metadata is required")?;
    let metadata = metadata
        .iter()
        .filter_map(|entry| {
            entry
                .value
                .as_ref()
                .map(|value| (entry.key.as_str(), value.as_str()))
        })
        .collect::<HashMap<_, _>>();

    validate_metadata_value_with_alias(
        &metadata,
        "schema",
        Some("physical_schema"),
        ARROW_PARQUET_SELFPLAY_SCHEMA,
    )?;
    validate_metadata_value_with_alias(
        &metadata,
        "logical_contract",
        Some("logical_schema"),
        SELFPLAY_SCHEMA,
    )?;
    validate_metadata_value_with_alias(
        &metadata,
        "contract_version",
        Some("contracts_release"),
        SELFPLAY_CONTRACT_VERSION,
    )?;
    Ok(())
}

#[cfg(feature = "arrow-parquet")]
fn validate_metadata_value_with_alias(
    metadata: &HashMap<&str, &str>,
    key: &str,
    alias: Option<&str>,
    expected: &str,
) -> Result<(), String> {
    let actual: &str = metadata
        .get(key)
        .copied()
        .or_else(|| alias.and_then(|alias| metadata.get(alias).copied()))
        .ok_or_else(|| {
            if let Some(alias) = alias {
                format!("parquet metadata {key} (or {alias}) is required")
            } else {
                format!("parquet metadata {key} is required")
            }
        })?;
    if actual != expected {
        return Err(format!(
            "parquet metadata {key} must be {expected}, got {actual}"
        ));
    }
    Ok(())
}

#[cfg(feature = "arrow-parquet")]
fn validate_selfplay_arrow_schema(schema: &arrow_schema::Schema) -> Result<(), String> {
    let expected = selfplay_arrow_parquet_schema();
    if schema.fields() != expected.fields() {
        return Err("parquet arrow schema does not match arrow-parquet-selfplay.v1".to_string());
    }
    Ok(())
}

#[cfg(feature = "arrow-parquet")]
fn string_column<'a>(
    batch: &'a arrow_array::RecordBatch,
    index: usize,
    field: &str,
) -> Result<&'a arrow_array::StringArray, String> {
    downcast_non_null_column(batch, index, field)
}

#[cfg(feature = "arrow-parquet")]
fn nullable_string_column<'a>(
    batch: &'a arrow_array::RecordBatch,
    index: usize,
    field: &str,
) -> Result<&'a arrow_array::StringArray, String> {
    downcast_column(batch, index, field)
}

#[cfg(feature = "arrow-parquet")]
fn u64_column<'a>(
    batch: &'a arrow_array::RecordBatch,
    index: usize,
    field: &str,
) -> Result<&'a arrow_array::UInt64Array, String> {
    downcast_non_null_column(batch, index, field)
}

#[cfg(feature = "arrow-parquet")]
fn u16_column<'a>(
    batch: &'a arrow_array::RecordBatch,
    index: usize,
    field: &str,
) -> Result<&'a arrow_array::UInt16Array, String> {
    downcast_non_null_column(batch, index, field)
}

#[cfg(feature = "arrow-parquet")]
fn u8_column<'a>(
    batch: &'a arrow_array::RecordBatch,
    index: usize,
    field: &str,
) -> Result<&'a arrow_array::UInt8Array, String> {
    downcast_non_null_column(batch, index, field)
}

#[cfg(feature = "arrow-parquet")]
fn i8_column<'a>(
    batch: &'a arrow_array::RecordBatch,
    index: usize,
    field: &str,
) -> Result<&'a arrow_array::Int8Array, String> {
    downcast_non_null_column(batch, index, field)
}

#[cfg(feature = "arrow-parquet")]
fn fixed_list_column<'a>(
    batch: &'a arrow_array::RecordBatch,
    index: usize,
    field: &str,
    expected_len: i32,
) -> Result<&'a arrow_array::FixedSizeListArray, String> {
    let array: &arrow_array::FixedSizeListArray = downcast_non_null_column(batch, index, field)?;
    if array.value_length() != expected_len {
        return Err(format!(
            "{field} must contain fixed-size lists of {expected_len}"
        ));
    }
    Ok(array)
}

#[cfg(feature = "arrow-parquet")]
fn downcast_non_null_column<'a, T: 'static>(
    batch: &'a arrow_array::RecordBatch,
    index: usize,
    field: &str,
) -> Result<&'a T, String> {
    use arrow_array::Array;

    let array = batch.column(index);
    if array.null_count() != 0 {
        return Err(format!("{field} must not contain nulls"));
    }
    downcast_column(batch, index, field)
}

#[cfg(feature = "arrow-parquet")]
fn downcast_column<'a, T: 'static>(
    batch: &'a arrow_array::RecordBatch,
    index: usize,
    field: &str,
) -> Result<&'a T, String> {
    let array = batch.column(index);
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| format!("{field} has unexpected arrow type"))
}

#[cfg(feature = "arrow-parquet")]
fn bitboards_u16_at(
    list: &arrow_array::FixedSizeListArray,
    row_index: usize,
    field: &str,
) -> Result<[u16; 8], String> {
    use arrow_array::Array;

    if list.is_null(row_index) {
        return Err(format!("{field} must not contain nulls"));
    }
    let values = list.value(row_index);
    let values = values
        .as_any()
        .downcast_ref::<arrow_array::UInt16Array>()
        .ok_or_else(|| format!("{field} child values have unexpected arrow type"))?;
    if values.len() != 8 {
        return Err(format!("{field} must contain exactly 8 uint16 planes"));
    }
    let mut bitboards = [0u16; 8];
    for (index, bitboard) in bitboards.iter_mut().enumerate() {
        *bitboard = values.value(index);
    }
    Ok(bitboards)
}

#[cfg(feature = "arrow-parquet")]
fn policy_visits_u32_at(
    list: &arrow_array::FixedSizeListArray,
    row_index: usize,
    field: &str,
) -> Result<[u32; 64], String> {
    use arrow_array::Array;

    if list.is_null(row_index) {
        return Err(format!("{field} must not contain nulls"));
    }
    let values = list.value(row_index);
    let values = values
        .as_any()
        .downcast_ref::<arrow_array::UInt32Array>()
        .ok_or_else(|| format!("{field} child values have unexpected arrow type"))?;
    if values.len() != 64 {
        return Err(format!("{field} must contain exactly 64 uint32 visits"));
    }
    let mut policy = [0u32; 64];
    for (index, visits) in policy.iter_mut().enumerate() {
        *visits = values.value(index);
    }
    Ok(policy)
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
            ("bad version", json!({"contract_version": "1.0.0"})),
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
            ("bad version", json!({"contract_version": "1.0.0"})),
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

    #[test]
    fn selfplay_row_parses_current_release_fixture_shape() {
        let row = json!({
            "schema": SELFPLAY_SCHEMA,
            "contract_version": SELFPLAY_CONTRACT_VERSION,
            "game_id": 0,
            "ply": 1,
            "qfen": "A.../..../..../....",
            "side_to_move": 1,
            "policy": [
                {"shape": 0, "position": 10, "visits": 2},
                {"shape": 1, "position": 1, "visits": 6}
            ],
            "value": -1.0
        });

        let parsed = parse_selfplay_row(&row).unwrap();

        assert_eq!(parsed.game_id, 0);
        assert_eq!(parsed.ply, 1);
        assert_eq!(parsed.side_to_move, 1);
        assert_eq!(parsed.policy[0].action_index(), 10);
        assert_eq!(parsed.policy[1].action_index(), 17);
        assert_eq!(parsed.value, -1.0);

        let dense = selfplay_dense_policy_visits(&parsed.policy).unwrap();
        assert_eq!(dense[10], 2);
        assert_eq!(dense[17], 6);
        assert_eq!(dense.iter().sum::<u32>(), 8);

        let physical = selfplay_arrow_parquet_record(&parsed).unwrap();
        assert_eq!(physical["logical_schema"], SELFPLAY_SCHEMA);
        assert_eq!(
            physical["contract_version"],
            json!(SELFPLAY_CONTRACT_VERSION)
        );
        assert_eq!(physical["ply"], json!(1u16));
        assert_eq!(physical["bitboards"], json!([1, 0, 0, 0, 0, 0, 0, 0]));
        assert_eq!(physical["policy_visits"][10], json!(2));
        assert_eq!(physical["policy_visits"][17], json!(6));
        assert_eq!(physical["value"], json!(-1));
    }

    #[test]
    fn selfplay_row_builder_emits_release_1_1_0_contract_json() {
        let row = selfplay_v1_row(
            7,
            0,
            "..../..../..../....",
            0,
            &[SelfPlayPolicyVisit {
                shape: 0,
                position: 0,
                visits: 3,
            }],
            1.0,
        )
        .unwrap();

        assert_eq!(row["schema"], SELFPLAY_SCHEMA);
        assert_eq!(row["contract_version"], SELFPLAY_CONTRACT_VERSION);
        assert_eq!(row["game_id"], json!(7));
        assert_eq!(row["policy"][0]["visits"], json!(3));
        parse_selfplay_row(&row).unwrap();
    }

    #[test]
    fn selfplay_parser_rejects_drifted_or_inconsistent_rows() {
        let valid = json!({
            "schema": SELFPLAY_SCHEMA,
            "contract_version": SELFPLAY_CONTRACT_VERSION,
            "game_id": 0,
            "ply": 0,
            "qfen": "..../..../..../....",
            "side_to_move": 0,
            "policy": [{"shape": 0, "position": 0, "visits": 1}],
            "value": 1.0
        });
        let cases = [
            ("bad schema", json!({"schema": "selfplay.v2"})),
            ("bad version", json!({"contract_version": "9.9.9"})),
            ("bad side", json!({"side_to_move": 1})),
            ("bad value", json!({"value": 0.0})),
            ("empty policy", json!({"policy": []})),
            (
                "illegal policy action",
                json!({
                    "qfen": "A.../..../..../....",
                    "side_to_move": 1,
                    "policy": [{"shape": 0, "position": 1, "visits": 1}]
                }),
            ),
            (
                "duplicate policy action",
                json!({
                    "policy": [
                        {"shape": 0, "position": 0, "visits": 1},
                        {"shape": 0, "position": 0, "visits": 2}
                    ]
                }),
            ),
        ];

        for (label, patch) in cases {
            let mut candidate = valid.clone();
            for (key, value) in patch.as_object().unwrap() {
                candidate[key] = value.clone();
            }

            let error = parse_selfplay_row(&candidate).expect_err(label).to_string();
            assert!(!error.is_empty(), "expected validation error for {label}");
        }
    }

    #[cfg(feature = "arrow-parquet")]
    #[test]
    fn selfplay_arrow_parquet_round_trips_fixture_shape_rows() {
        let rows = vec![
            parse_selfplay_row(&json!({
                "schema": SELFPLAY_SCHEMA,
                "contract_version": SELFPLAY_CONTRACT_VERSION,
                "game_id": 0,
                "ply": 1,
                "qfen": "A.../..../..../....",
                "side_to_move": 1,
                "policy": [
                    {"shape": 0, "position": 10, "visits": 2},
                    {"shape": 1, "position": 1, "visits": 6}
                ],
                "value": -1.0
            }))
            .unwrap(),
            parse_selfplay_row(&json!({
                "schema": SELFPLAY_SCHEMA,
                "contract_version": SELFPLAY_CONTRACT_VERSION,
                "game_id": 1,
                "ply": 0,
                "qfen": "..../..../..../....",
                "side_to_move": 0,
                "policy": [{"shape": 0, "position": 0, "visits": 3}],
                "value": 1.0
            }))
            .unwrap(),
        ];
        let path = std::env::temp_dir().join(format!(
            "quantik-selfplay-roundtrip-{}.parquet",
            std::process::id()
        ));

        write_selfplay_arrow_parquet(&path, &rows).unwrap();
        let round_tripped = read_selfplay_arrow_parquet(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(round_tripped, rows);
    }

    #[cfg(feature = "arrow-parquet")]
    #[test]
    fn selfplay_arrow_parquet_reader_rejects_drifted_metadata() {
        let path = temp_selfplay_parquet_path("bad-metadata");
        write_test_selfplay_parquet(&path, "selfplay.v2", 64, Some("A.../..../..../....")).unwrap();

        let error = read_selfplay_arrow_parquet(&path).expect_err("metadata drift must fail");
        std::fs::remove_file(&path).ok();

        assert!(
            error.contains("parquet metadata logical_contract must be selfplay.v1")
                && error.contains("selfplay.v2"),
            "unexpected error: {error}"
        );
    }

    #[cfg(feature = "arrow-parquet")]
    #[test]
    fn selfplay_arrow_parquet_reader_rejects_drifted_physical_shape() {
        let path = temp_selfplay_parquet_path("bad-policy-shape");
        write_test_selfplay_parquet(&path, SELFPLAY_SCHEMA, 63, Some("A.../..../..../...."))
            .unwrap();

        let error = read_selfplay_arrow_parquet(&path).expect_err("shape drift must fail");
        std::fs::remove_file(&path).ok();

        assert!(
            error.contains("schema") || error.contains("policy_visits"),
            "unexpected error: {error}"
        );
    }

    #[cfg(feature = "arrow-parquet")]
    #[test]
    fn selfplay_arrow_parquet_reader_accepts_nullable_qfen() {
        let path = temp_selfplay_parquet_path("null-qfen");
        write_test_selfplay_parquet(&path, SELFPLAY_SCHEMA, 64, None).unwrap();

        let rows = read_selfplay_arrow_parquet(&path).expect("nullable qfen should be supported");
        std::fs::remove_file(&path).ok();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].side_to_move, 1);
        assert_eq!(rows[0].qfen, "A.../..../..../....");
        let state = State::from_qfen(&rows[0].qfen).expect("derived qfen must be valid");
        assert_eq!(state.bb.planes, [1, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[cfg(feature = "arrow-parquet")]
    fn temp_selfplay_parquet_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "quantik-selfplay-{label}-{}.parquet",
            std::process::id()
        ))
    }

    #[cfg(feature = "arrow-parquet")]
    fn write_test_selfplay_parquet(
        path: &std::path::Path,
        logical_schema: &str,
        policy_len: i32,
        qfen: Option<&str>,
    ) -> Result<(), String> {
        use arrow_array::{
            ArrayRef, FixedSizeListArray, Int8Array, RecordBatch, StringArray, UInt16Array,
            UInt32Array, UInt64Array, UInt8Array,
        };
        use arrow_schema::{DataType, Field, Schema};
        use parquet::arrow::arrow_writer::ArrowWriter;
        use parquet::file::metadata::KeyValue;
        use parquet::file::properties::WriterProperties;
        use std::sync::Arc;

        let policy_values = (0..policy_len)
            .map(|index| match index {
                10 => 2,
                17 => 6,
                _ => 0,
            })
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(vec![
            Field::new("logical_schema", DataType::Utf8, false),
            Field::new("contract_version", DataType::Utf8, false),
            Field::new("game_id", DataType::UInt64, false),
            Field::new("ply", DataType::UInt16, false),
            Field::new("side_to_move", DataType::UInt8, false),
            Field::new(
                "bitboards",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt16, false)), 8),
                false,
            ),
            Field::new(
                "policy_visits",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::UInt32, false)),
                    policy_len,
                ),
                false,
            ),
            Field::new("value", DataType::Int8, false),
            Field::new("qfen", DataType::Utf8, true),
        ]));
        let bitboards = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::UInt16, false)),
            8,
            Arc::new(UInt16Array::from(vec![1, 0, 0, 0, 0, 0, 0, 0])),
            None,
        )
        .map_err(|e| format!("build bitboards: {e}"))?;
        let policy_visits = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::UInt32, false)),
            policy_len,
            Arc::new(UInt32Array::from(policy_values)),
            None,
        )
        .map_err(|e| format!("build policy_visits: {e}"))?;
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![SELFPLAY_SCHEMA])) as ArrayRef,
                Arc::new(StringArray::from(vec![SELFPLAY_CONTRACT_VERSION])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![0])) as ArrayRef,
                Arc::new(UInt16Array::from(vec![1])) as ArrayRef,
                Arc::new(UInt8Array::from(vec![1])) as ArrayRef,
                Arc::new(bitboards) as ArrayRef,
                Arc::new(policy_visits) as ArrayRef,
                Arc::new(Int8Array::from(vec![-1])) as ArrayRef,
                Arc::new(StringArray::from(vec![qfen])) as ArrayRef,
            ],
        )
        .map_err(|e| format!("build batch: {e}"))?;
        let props = WriterProperties::builder()
            .set_key_value_metadata(Some(vec![
                KeyValue {
                    key: "schema".to_string(),
                    value: Some(ARROW_PARQUET_SELFPLAY_SCHEMA.to_string()),
                },
                KeyValue {
                    key: "logical_schema".to_string(),
                    value: Some(logical_schema.to_string()),
                },
                KeyValue {
                    key: "logical_contract".to_string(),
                    value: Some(logical_schema.to_string()),
                },
                KeyValue {
                    key: "contracts_release".to_string(),
                    value: Some(SELFPLAY_CONTRACT_VERSION.to_string()),
                },
                KeyValue {
                    key: "contract_version".to_string(),
                    value: Some(SELFPLAY_CONTRACT_VERSION.to_string()),
                },
            ]))
            .build();
        let file = std::fs::File::create(path).map_err(|e| format!("create test parquet: {e}"))?;
        let mut writer =
            ArrowWriter::try_new(file, schema, Some(props)).map_err(|e| e.to_string())?;
        writer.write(&batch).map_err(|e| e.to_string())?;
        writer.close().map_err(|e| e.to_string())?;
        Ok(())
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
