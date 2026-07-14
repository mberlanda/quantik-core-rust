//! Shared, versioned position dataset for cross-engine benchmarks.
//!
//! Port of `benchmarks/dataset.py`. Artifacts are JSON with a sha256
//! checksum over the canonical (sorted-key, compact-separator) encoding of
//! the payload minus the `checksum` field — byte-compatible with the Python
//! implementation, so datasets interoperate across languages. RNG streams
//! differ from CPython, so `generate` with the same seed produces a
//! *different but equally valid* dataset; the committed artifact is the
//! shared one.

use crate::game::{current_player, has_winning_line};
use crate::moves::{apply_move, generate_legal_moves};
use crate::state::State;
use rand::prelude::*;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

use super::contracts::CONTRACT_VERSION;

pub const SCHEMA_VERSION: u64 = 1;
pub const GENERATOR: &str = "benchmarks.dataset.generate/v1";

/// Phase buckets by pieces placed (== plies from the empty board).
pub const PHASES: [(&str, (u32, u32)); 4] = [
    ("opening", (0, 4)),
    ("early_mid", (5, 7)),
    ("late_mid", (8, 11)),
    ("endgame", (12, 16)),
];

/// Return the phase bucket for a piece count.
pub fn phase_of(pieces: u32) -> Result<&'static str, String> {
    for (phase, (lo, hi)) in PHASES {
        if (lo..=hi).contains(&pieces) {
            return Ok(phase);
        }
    }
    Err(format!("no benchmark phase for {pieces} pieces"))
}

/// sha256 over canonical JSON excluding the `checksum` field.
///
/// The canonical encoding is byte-identical to Python's
/// `json.dumps(sort_keys=True, separators=(",", ":"))` — see
/// [`super::canonical::canonical_json`] for the float/string rules.
pub fn checksum(payload: &Value) -> String {
    let mut stripped = payload.as_object().cloned().unwrap_or_default();
    stripped.remove("checksum");
    let blob = super::canonical::canonical_json(&Value::Object(stripped));
    let mut hasher = Sha256::new();
    hasher.update(blob.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Write a checksum-bearing JSON dataset artifact and return its checksum.
pub fn save(payload: &Value, path: &Path) -> Result<String, String> {
    let digest = checksum(payload);
    let mut output = payload.as_object().cloned().unwrap_or_default();
    output.insert("checksum".into(), Value::String(digest.clone()));
    let text = serde_json::to_string_pretty(&Value::Object(output))
        .map_err(|e| format!("serialize: {e}"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {parent:?}: {e}"))?;
    }
    std::fs::write(path, text + "\n").map_err(|e| format!("write {path:?}: {e}"))?;
    Ok(digest)
}

/// Load a dataset artifact and verify its checksum.
pub fn load(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let payload: Value = serde_json::from_str(&text).map_err(|e| format!("parse: {e}"))?;
    let expected = payload
        .get("checksum")
        .and_then(Value::as_str)
        .map(str::to_string);
    let actual = checksum(&payload);
    if expected.as_deref() != Some(actual.as_str()) {
        return Err(format!(
            "dataset checksum mismatch: expected {expected:?}, actual {actual}"
        ));
    }
    Ok(payload)
}

/// Play `plies` random legal moves; `None` if the line hits a win or a
/// dead end (including a final state with no legal continuation).
fn random_position(rng: &mut StdRng, plies: u32) -> Option<crate::bitboard::Bitboard> {
    let mut bb = crate::bitboard::Bitboard::EMPTY;
    for _ in 0..plies {
        let moves = generate_legal_moves(&bb);
        if moves.is_empty() {
            return None;
        }
        bb = apply_move(&bb, &moves[rng.gen_range(0..moves.len())]);
        if has_winning_line(&bb) {
            return None;
        }
    }
    if generate_legal_moves(&bb).is_empty() {
        return None;
    }
    Some(bb)
}

fn position_payload(position_id: usize, bb: &crate::bitboard::Bitboard, phase: &str) -> Value {
    let pieces = bb.player_piece_count(0) + bb.player_piece_count(1);
    json!({
        "id": format!("p{position_id:04}"),
        "qfen": State::new(*bb).to_qfen(),
        "phase": phase,
        "pieces": pieces,
        "side_to_move": current_player(bb).expect("valid position"),
        "legal_moves": generate_legal_moves(bb).len(),
        "reference": Value::Null,
    })
}

/// Generate a deterministic benchmark dataset for requested phase counts.
pub fn generate(requested: &BTreeMap<String, u32>, seed: u64) -> Result<Value, String> {
    let known: Vec<&str> = PHASES.iter().map(|(name, _)| *name).collect();
    let unknown: Vec<&String> = requested
        .keys()
        .filter(|k| !known.contains(&k.as_str()))
        .collect();
    if !unknown.is_empty() {
        return Err(format!("unknown phase(s): {unknown:?}"));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut seen: std::collections::HashSet<[u8; 18]> = std::collections::HashSet::new();
    let mut positions: Vec<Value> = Vec::new();

    for (phase, (lo, hi)) in PHASES {
        let want = *requested.get(phase).unwrap_or(&0);
        let mut found = 0u32;
        let mut attempts = 0u32;
        let max_attempts = want * 500;

        while found < want && attempts < max_attempts {
            attempts += 1;
            let target_plies = rng.gen_range(lo..=hi.min(15));
            let Some(bb) = random_position(&mut rng, target_plies) else {
                continue;
            };
            let key = State::new(bb).canonical_key();
            if !seen.insert(key) {
                continue;
            }
            positions.push(position_payload(positions.len(), &bb, phase));
            found += 1;
        }
    }

    let requested_map: Map<String, Value> = requested
        .iter()
        .map(|(k, v)| (k.clone(), json!(v)))
        .collect();

    Ok(json!({
        "contract_version": CONTRACT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "seed": seed,
        "requested": requested_map,
        "positions": positions,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::check_winner;
    use crate::game::WinStatus;

    fn golden_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/positions-v1.json")
    }

    /// The linchpin of cross-language interop: the checksum recomputed by
    /// the Rust canonical-JSON encoder over the Python-generated artifact
    /// must equal the stored value (load() verifies it).
    #[test]
    fn golden_dataset_checksum_verifies() {
        let payload = load(&golden_path()).unwrap();
        assert_eq!(payload["schema_version"], json!(1));
        assert_eq!(payload["generator"], json!(GENERATOR));
        assert_eq!(payload["positions"].as_array().unwrap().len(), 36);
    }

    #[test]
    fn golden_dataset_positions_are_valid() {
        let payload = load(&golden_path()).unwrap();
        for position in payload["positions"].as_array().unwrap() {
            let qfen = position["qfen"].as_str().unwrap();
            let state = State::from_qfen(qfen).unwrap();
            let bb = state.bb;
            assert_eq!(check_winner(&bb), WinStatus::NoWin, "{qfen} is terminal");
            let legal = generate_legal_moves(&bb);
            assert!(!legal.is_empty(), "{qfen} has no legal moves");
            assert_eq!(
                position["legal_moves"].as_u64().unwrap() as usize,
                legal.len(),
                "{qfen} legal move count"
            );
            let pieces = bb.player_piece_count(0) + bb.player_piece_count(1);
            assert_eq!(position["pieces"].as_u64().unwrap() as u32, pieces);
            assert_eq!(
                position["side_to_move"].as_u64().unwrap() as u8,
                current_player(&bb).unwrap(),
                "{qfen} side to move"
            );
            assert_eq!(
                position["phase"].as_str().unwrap(),
                phase_of(pieces).unwrap()
            );
        }
    }

    #[test]
    fn save_load_roundtrip_and_corruption_detection() {
        let dir = std::env::temp_dir().join(format!("quantik-bench-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.json");

        let requested = BTreeMap::from([("opening".to_string(), 2u32)]);
        let payload = generate(&requested, 7).unwrap();
        let digest = save(&payload, &path).unwrap();
        let loaded = load(&path).unwrap();
        assert_eq!(loaded["checksum"].as_str().unwrap(), digest);

        // Corrupt one byte inside the positions array.
        let text = std::fs::read_to_string(&path).unwrap();
        let corrupted = text.replacen("\"pieces\": 3", "\"pieces\": 4", 1);
        let corrupted = if corrupted == text {
            text.replacen("\"pieces\": 4", "\"pieces\": 5", 1)
        } else {
            corrupted
        };
        std::fs::write(&path, corrupted).unwrap();
        assert!(load(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn generate_is_deterministic_and_deduped() {
        let requested = BTreeMap::from([
            ("opening".to_string(), 3u32),
            ("early_mid".to_string(), 2u32),
        ]);
        let a = generate(&requested, 123).unwrap();
        let b = generate(&requested, 123).unwrap();
        assert_eq!(a, b);

        let positions = a["positions"].as_array().unwrap();
        assert_eq!(positions.len(), 5);
        let mut keys = std::collections::HashSet::new();
        for position in positions {
            let state = State::from_qfen(position["qfen"].as_str().unwrap()).unwrap();
            assert!(keys.insert(state.canonical_key()), "duplicate canonical");
            let pieces = position["pieces"].as_u64().unwrap() as u32;
            let phase = position["phase"].as_str().unwrap();
            assert_eq!(phase_of(pieces).unwrap(), phase);
        }
    }

    #[test]
    fn unknown_phase_rejected() {
        let requested = BTreeMap::from([("blitz".to_string(), 1u32)]);
        assert!(generate(&requested, 1).is_err());
    }
}
