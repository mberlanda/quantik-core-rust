use crate::bench::contracts::{action_index, canonical_key_hex};
use crate::game::{check_winner, current_player, WinStatus};
use crate::moves::{apply_move, generate_legal_moves, Move};
use crate::state::State;
use crate::symmetry::SymmetryHandler;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::fs;
use std::path::Path;

const REPORT_SCHEMA: &str = "api-portability-report.v1";
const FIXTURE_SCHEMA: &str = "api-portability-fixtures.v1";
const FIXTURE_PATH: &str = "fixtures/api-portability/game-state-v1.json";

#[derive(Debug, Deserialize)]
struct ContractsManifest {
    release_version: Option<String>,
    contracts: std::collections::BTreeMap<String, ContractEntry>,
}

#[derive(Debug, Deserialize)]
struct ContractEntry {
    id: String,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    contract_version: String,
    game_state_cases: Vec<GameStateCase>,
}

#[derive(Debug, Deserialize)]
struct GameStateCase {
    case_id: String,
    qfen: String,
    #[serde(default)]
    r#move: Option<MoveFixture>,
}

#[derive(Debug, Deserialize)]
struct MoveFixture {
    shape: u8,
    position: u8,
}

pub fn build_report(contracts_root: &Path) -> Result<Value, String> {
    let manifest = load_manifest(contracts_root)?;
    let fixture = load_fixture(contracts_root)?;
    let release = contracts_release(contracts_root, &manifest)?;
    validate_fixture_metadata(&fixture, &release)?;
    if fixture.game_state_cases.is_empty() {
        return Err(format!(
            "{} game_state_cases must be a non-empty array",
            FIXTURE_PATH
        ));
    }
    let mut cases: Vec<Value> = fixture
        .game_state_cases
        .iter()
        .map(project_case)
        .collect::<Result<_, _>>()?;
    cases.sort_by(|left, right| {
        left["case_id"]
            .as_str()
            .unwrap_or_default()
            .cmp(right["case_id"].as_str().unwrap_or_default())
    });

    Ok(json!({
        "schema": REPORT_SCHEMA,
        "contracts_release": release,
        "implementation": {
            "language": "rust",
            "package": env!("CARGO_PKG_NAME"),
            "version": env!("CARGO_PKG_VERSION"),
        },
        "contract_ids": {
            "qfen": contract_id(&manifest, "qfen")?,
            "bitboard": contract_id(&manifest, "bitboard")?,
            "action_index": contract_id(&manifest, "action_index")?,
        },
        "cases": cases,
    }))
}

pub fn write_report(contracts_root: &Path, output: &Path) -> Result<(), String> {
    let report = build_report(contracts_root)?;
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| format!("create output directory: {e}"))?;
        }
    }
    let text =
        serde_json::to_string_pretty(&report).map_err(|e| format!("serialize report: {e}"))?;
    fs::write(output, format!("{text}\n")).map_err(|e| format!("write report: {e}"))
}

fn load_manifest(contracts_root: &Path) -> Result<ContractsManifest, String> {
    let path = contracts_root.join("contracts.json");
    let text = fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn load_fixture(contracts_root: &Path) -> Result<Fixture, String> {
    let path = contracts_root.join(FIXTURE_PATH);
    let text = fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn validate_fixture_metadata(fixture: &Fixture, contracts_release: &str) -> Result<(), String> {
    if fixture.schema != FIXTURE_SCHEMA {
        return Err(format!(
            "{} schema {} must be {FIXTURE_SCHEMA}",
            FIXTURE_PATH, fixture.schema
        ));
    }
    if fixture.contract_version != contracts_release {
        return Err(format!(
            "{} contract_version {} does not match contracts release {contracts_release}",
            FIXTURE_PATH, fixture.contract_version
        ));
    }
    Ok(())
}

fn contracts_release(
    contracts_root: &Path,
    manifest: &ContractsManifest,
) -> Result<String, String> {
    let release = manifest
        .release_version
        .as_deref()
        .map(str::trim)
        .filter(|release| !release.is_empty())
        .ok_or_else(|| "contracts.json release_version is missing or empty".to_string())?;

    let version_path = contracts_root.join("VERSION");
    let version = fs::read_to_string(&version_path)
        .map_err(|e| format!("read {}: {e}", version_path.display()))?;
    let version = version.trim();
    if version.is_empty() {
        return Err(format!("{} is empty", version_path.display()));
    }
    if version != release {
        return Err(format!(
            "{} version {version} does not match contracts.json release_version {release}",
            version_path.display()
        ));
    }

    Ok(release.to_string())
}

fn contract_id(manifest: &ContractsManifest, key: &str) -> Result<String, String> {
    manifest
        .contracts
        .get(key)
        .map(|entry| entry.id.clone())
        .ok_or_else(|| format!("contracts.json missing {key} contract"))
}

fn project_case(case: &GameStateCase) -> Result<Value, String> {
    if case.case_id.is_empty() {
        return Err("game_state_case.case_id must be a non-empty string".to_string());
    }

    let state = State::from_qfen(&case.qfen)
        .map_err(|e| format!("case {} qfen parse failed: {e}", case.case_id))?;
    let qfen = state.to_qfen();
    let canonical = State::new(SymmetryHandler::find_canonical(&state.bb));
    let side_to_move = current_player(&state.bb)
        .ok_or_else(|| format!("case {} has invalid side-to-move parity", case.case_id))?;
    let legal_moves = generate_legal_moves(&state.bb);
    let winner = check_winner(&state.bb);
    let terminal = winner != WinStatus::NoWin || legal_moves.is_empty();
    let winner_label = if winner == WinStatus::NoWin && legal_moves.is_empty() {
        if side_to_move == 0 {
            "player1"
        } else {
            "player0"
        }
    } else {
        winner_label(winner)
    };

    let mut object = Map::new();
    object.insert("case_id".to_string(), json!(&case.case_id));
    object.insert("qfen".to_string(), json!(qfen));
    object.insert("bitboards".to_string(), json!(state.bb.planes));
    object.insert("side_to_move".to_string(), json!(side_to_move));
    object.insert("canonical_qfen".to_string(), json!(canonical.to_qfen()));
    object.insert(
        "canonical_key".to_string(),
        json!(canonical_key_hex(&state)),
    );
    object.insert("orbit_size".to_string(), json!(state.symmetry_count()));
    object.insert(
        "legal_action_mask".to_string(),
        json!(format!(
            "0x{:016x}",
            legal_action_mask_from_moves(&legal_moves)
        )),
    );
    object.insert(
        "legal_action_indices".to_string(),
        json!(legal_action_indices(&legal_moves)),
    );
    object.insert("terminal".to_string(), json!(terminal));
    object.insert("winner".to_string(), json!(winner_label));
    let move_fixture = case
        .r#move
        .as_ref()
        .ok_or_else(|| format!("case {} move must be present", case.case_id))?;
    object.insert(
        "move".to_string(),
        project_move(&state, side_to_move, &legal_moves, move_fixture)?,
    );
    Ok(Value::Object(object))
}

fn legal_action_indices(legal_moves: &[Move]) -> Vec<u8> {
    let mut indices: Vec<u8> = legal_moves
        .iter()
        .map(|mv| action_index(mv.shape, mv.position))
        .collect();
    indices.sort_unstable();
    indices
}

fn legal_action_mask_from_moves(legal_moves: &[Move]) -> u64 {
    legal_moves.iter().fold(0u64, |mask, mv| {
        mask | (1u64 << action_index(mv.shape, mv.position))
    })
}

fn project_move(
    state: &State,
    side_to_move: u8,
    legal_moves: &[Move],
    move_fixture: &MoveFixture,
) -> Result<Value, String> {
    if move_fixture.shape > 3 {
        return Err(format!(
            "move.shape must be between 0 and 3, got {}",
            move_fixture.shape
        ));
    }
    if move_fixture.position > 15 {
        return Err(format!(
            "move.position must be between 0 and 15, got {}",
            move_fixture.position
        ));
    }

    let index = action_index(move_fixture.shape, move_fixture.position);
    let is_legal = legal_moves
        .iter()
        .any(|mv| mv.shape == move_fixture.shape && mv.position == move_fixture.position);
    let mut object = Map::new();
    object.insert("shape".to_string(), json!(move_fixture.shape));
    object.insert("position".to_string(), json!(move_fixture.position));
    object.insert("action_index".to_string(), json!(index));
    object.insert("is_legal".to_string(), json!(is_legal));
    if is_legal {
        let mv = Move::new(side_to_move, move_fixture.shape, move_fixture.position);
        let after = State::new(apply_move(&state.bb, &mv));
        object.insert("after_qfen".to_string(), json!(after.to_qfen()));
    } else {
        object.insert("after_qfen".to_string(), Value::Null);
    }
    Ok(Value::Object(object))
}

fn winner_label(winner: WinStatus) -> &'static str {
    match winner {
        WinStatus::NoWin => "none",
        WinStatus::Player0Wins => "player0",
        WinStatus::Player1Wins => "player1",
    }
}
