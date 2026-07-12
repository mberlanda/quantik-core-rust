//! Bulk export of solved benchmark references into the SQLite opening
//! book, and lookup of stored references to short-circuit repeated
//! solves.
//!
//! The book is keyed by the 18-byte canonical key
//! ([`crate::state::State::canonical_key`]), which is byte-identical to
//! the Python implementation's, so the same SQLite file is portable
//! across languages: a book built by `quantik-core-rust` can be read by
//! `opening_book.py` and vice versa.
//!
//! **Orientation caveat:** [`export_references`] writes every solved
//! dataset position regardless of whether that position is its own
//! canonical representative — its optimal moves are recorded in the
//! position's own orientation (as `(shape, position)` pairs), while the
//! row is keyed by the position's *canonical* key (computed inside
//! [`crate::opening_book::OpeningBookDatabase::add_solved_position`] via
//! symmetry reduction). Translating those moves back to an arbitrary
//! queried orientation would require tracking which symmetry transform
//! was applied, which the book does not currently store. Rather than risk
//! serving moves that are illegal in the queried orientation,
//! [`lookup_reference`] only ever returns a hit when the *query* itself is
//! already its own canonical representative
//! (`State::new(*bb).canonical_payload() == bb.to_le_bytes()`) — in that
//! case the stored moves are trivially in the right orientation whenever
//! the row was itself written from a canonical-orientation position, which
//! is the common case because the benchmark dataset deduplicates positions
//! by canonical key (at most one orientation is ever recorded per
//! canonical class within a single dataset). Full orientation tracking
//! (storing the symmetry transform index so moves can be translated to any
//! queried orientation) is left as a follow-up.

use crate::bitboard::Bitboard;
use crate::game::current_player;
use crate::opening_book::OpeningBookDatabase;
use crate::state::State;
use serde_json::{json, Value};

use super::reference::parse_move_key;

/// Upsert every solved reference in `dataset_payload` (a dataset or bundle
/// JSON artifact with a top-level `positions` array, each entry carrying
/// `qfen` and an optional `reference`) into `db`. Positions with a `null`
/// reference are skipped. Returns the number of positions inserted.
///
/// Idempotent: re-running against the same payload upserts the same rows
/// (`INSERT OR REPLACE` semantics via
/// [`crate::opening_book::OpeningBookDatabase::add_solved_position`]), so
/// the total row count in `positions` does not grow on a rerun.
pub fn export_references(dataset_payload: &Value, db: &OpeningBookDatabase) -> Result<u64, String> {
    let positions = dataset_payload
        .get("positions")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let mut count = 0u64;
    for position in &positions {
        let reference = &position["reference"];
        if reference.is_null() {
            continue;
        }

        let qfen = position["qfen"]
            .as_str()
            .ok_or("dataset position missing qfen")?;
        let state = State::from_qfen(qfen).map_err(|e| format!("parse qfen {qfen:?}: {e}"))?;

        let value = reference["value"]
            .as_i64()
            .ok_or_else(|| format!("reference for {qfen} missing integer value"))?
            as i32;

        let optimal_moves = reference["optimal_moves"]
            .as_array()
            .ok_or_else(|| format!("reference for {qfen} missing optimal_moves"))?
            .iter()
            .map(|v| {
                let key = v
                    .as_str()
                    .ok_or_else(|| format!("optimal_moves entry for {qfen} is not a string"))?;
                let (_, shape, pos) = parse_move_key(key)?;
                Ok((shape as i32, pos as i32))
            })
            .collect::<Result<Vec<(i32, i32)>, String>>()?;

        db.add_solved_position(&state, value, &optimal_moves)
            .map_err(|e| format!("add_solved_position for {qfen}: {e}"))?;
        count += 1;
    }

    Ok(count)
}

/// Probe the book for an exact reference at `bb`, reconstructing the JSON
/// shape produced by [`crate::bench::reference::solve_position`] (minus a
/// full principal variation — only the first optimal move is known).
///
/// Returns `None` unless (a) a row exists for `bb`'s canonical key with
/// `solved = true`, and (b) `bb` is its own canonical representative — see
/// the module doc for why the second condition is required.
pub fn lookup_reference(bb: &Bitboard, db: &OpeningBookDatabase) -> Option<Value> {
    let state = State::new(*bb);
    if state.canonical_payload() != bb.to_le_bytes() {
        return None;
    }

    let entry = db.get_position(&state).ok().flatten()?;
    if !entry.solved {
        return None;
    }
    let value = entry.game_value?;
    let player = current_player(bb)?;

    let optimal_moves: Vec<String> = entry
        .best_moves
        .iter()
        .map(|&(shape, position)| format!("{player}:{shape}:{position}"))
        .collect();
    if optimal_moves.is_empty() {
        return None;
    }

    Some(json!({
        "solved": true,
        "no_cutoff": true,
        "value": value,
        "optimal_moves": optimal_moves,
        "pv": [optimal_moves[0].clone()],
        "nodes": 0,
        "solve_time_s": 0.0,
        "solver": "opening-book",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::reference::{move_key, solve_position};
    use crate::game::has_winning_line;
    use crate::moves::{apply_move, generate_legal_moves};
    use crate::opening_book::OpeningBookConfig;
    use rand::prelude::*;

    fn temp_db_path(tag: &str) -> String {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir()
            .join(format!("quantik_book_export_{tag}_{id}.db"))
            .to_string_lossy()
            .to_string()
    }

    fn open_book(path: &str) -> OpeningBookDatabase {
        OpeningBookDatabase::open(&OpeningBookConfig {
            database_path: path.to_string(),
            ..Default::default()
        })
        .unwrap()
    }

    fn random_position(seed: u64, plies: usize) -> Bitboard {
        let mut rng = StdRng::seed_from_u64(seed);
        'attempt: loop {
            let mut bb = Bitboard::EMPTY;
            for _ in 0..plies {
                let moves = generate_legal_moves(&bb);
                if moves.is_empty() {
                    continue 'attempt;
                }
                bb = apply_move(&bb, &moves[rng.gen_range(0..moves.len())]);
                if has_winning_line(&bb) {
                    continue 'attempt;
                }
            }
            if generate_legal_moves(&bb).is_empty() {
                continue 'attempt;
            }
            return bb;
        }
    }

    /// Find a random reachable position that IS its own canonical
    /// representative (there is always at least one such board per
    /// equivalence class — the representative itself).
    fn canonical_representative_position(seed_start: u64, plies: usize) -> Bitboard {
        (seed_start..)
            .map(|seed| random_position(seed, plies))
            .find(|bb| State::new(*bb).canonical_payload() == bb.to_le_bytes())
            .expect("a canonical-representative position exists among random samples")
    }

    /// Find a random reachable position that is NOT its own canonical
    /// representative (the common case: most orientations aren't the
    /// distinguished representative of their symmetry class).
    fn non_canonical_position(seed_start: u64, plies: usize) -> Bitboard {
        (seed_start..)
            .map(|seed| random_position(seed, plies))
            .find(|bb| State::new(*bb).canonical_payload() != bb.to_le_bytes())
            .expect("a non-canonical position exists among random samples")
    }

    #[test]
    fn export_and_lookup_roundtrip() {
        let bb = canonical_representative_position(0, 11);
        let reference = solve_position(&bb, 60.0).unwrap();
        let qfen = State::new(bb).to_qfen();

        let dataset_payload = json!({
            "positions": [
                {"id": "p0000", "qfen": qfen, "phase": "endgame", "reference": reference},
            ],
        });

        let path = temp_db_path("roundtrip");
        let db = open_book(&path);
        let inserted = export_references(&dataset_payload, &db).unwrap();
        assert_eq!(inserted, 1);

        // Same orientation: lookup must succeed and match value/optimal_moves.
        let looked_up = lookup_reference(&bb, &db).unwrap();
        assert_eq!(looked_up["value"], reference["value"]);
        assert_eq!(looked_up["optimal_moves"], reference["optimal_moves"]);
        assert_eq!(looked_up["solver"], json!("opening-book"));
        assert_eq!(looked_up["nodes"], json!(0));

        // A non-canonical-representative symmetric variant must return None.
        let non_canon = non_canonical_position(1000, 11);
        assert!(lookup_reference(&non_canon, &db).is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn export_is_idempotent() {
        let bb = canonical_representative_position(2, 10);
        let reference = solve_position(&bb, 60.0).unwrap();
        let qfen = State::new(bb).to_qfen();
        let dataset_payload = json!({
            "positions": [
                {"id": "p0000", "qfen": qfen, "phase": "late_mid", "reference": reference},
            ],
        });

        let path = temp_db_path("idempotent");
        let db = open_book(&path);
        assert_eq!(export_references(&dataset_payload, &db).unwrap(), 1);
        assert_eq!(export_references(&dataset_payload, &db).unwrap(), 1);
        assert_eq!(db.total_positions().unwrap(), 1);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn positions_without_reference_are_skipped() {
        let dataset_payload = json!({
            "positions": [
                {"id": "p0000", "qfen": "..../..../..../....", "phase": "opening", "reference": Value::Null},
            ],
        });
        let path = temp_db_path("skip-null");
        let db = open_book(&path);
        assert_eq!(export_references(&dataset_payload, &db).unwrap(), 0);
        assert_eq!(db.total_positions().unwrap(), 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn lookup_returns_none_when_book_is_empty() {
        let bb = canonical_representative_position(5, 10);
        let path = temp_db_path("empty");
        let db = open_book(&path);
        assert!(lookup_reference(&bb, &db).is_none());
        std::fs::remove_file(&path).ok();
    }

    /// Smoke test for the `export-book` CLI path: the golden shared
    /// dataset carries exactly 22 solved references; exporting it inserts
    /// all 22, and a rerun is idempotent (still 22 rows total in the
    /// positions table, not 44).
    #[test]
    fn golden_dataset_export_inserts_all_solved_references_idempotently() {
        let golden = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../benchmarks/positions-v1.json");
        let payload = crate::bench::dataset::load(&golden).unwrap();

        let path = temp_db_path("golden");
        let db = open_book(&path);
        assert_eq!(export_references(&payload, &db).unwrap(), 22);
        assert_eq!(db.total_positions().unwrap(), 22);

        // Rerun: same 22 upserts, no row growth.
        assert_eq!(export_references(&payload, &db).unwrap(), 22);
        assert_eq!(db.total_positions().unwrap(), 22);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn optimal_moves_reconstruct_to_valid_move_keys() {
        let bb = canonical_representative_position(7, 11);
        let reference = solve_position(&bb, 60.0).unwrap();
        let qfen = State::new(bb).to_qfen();
        let dataset_payload = json!({
            "positions": [
                {"id": "p0000", "qfen": qfen, "phase": "endgame", "reference": reference},
            ],
        });

        let path = temp_db_path("valid-moves");
        let db = open_book(&path);
        export_references(&dataset_payload, &db).unwrap();
        let looked_up = lookup_reference(&bb, &db).unwrap();

        let legal: Vec<String> = generate_legal_moves(&bb).iter().map(move_key).collect();
        for mv in looked_up["optimal_moves"].as_array().unwrap() {
            let key = mv.as_str().unwrap();
            assert!(legal.contains(&key.to_string()), "{key} not legal");
        }

        std::fs::remove_file(&path).ok();
    }
}
