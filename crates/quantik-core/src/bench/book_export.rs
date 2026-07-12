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
//! **Orientation caveat — representative-only, on BOTH reads and writes:**
//! optimal moves are recorded as `(shape, position)` pairs in a specific
//! board orientation, but the row is keyed by the canonical key, which is
//! shared by up to eight symmetric orientations. The book does not (yet)
//! store which symmetry transform maps the stored orientation to an
//! arbitrary query, so moves cannot be translated across orientations.
//! Both directions are therefore restricted to boards that are their own
//! canonical representative
//! (`State::new(*bb).canonical_payload() == bb.to_le_bytes()`):
//!
//! - **Writes** ([`export_references`], and
//!   [`crate::opening_book::OpeningBookDatabase::add_solved_position`]
//!   itself as defense in depth) silently skip any solved position that
//!   is not its own canonical representative. Without this guard, a row
//!   written from a rotated orientation would later be served — with
//!   wrong, possibly illegal moves — to a query on the representative
//!   board, which passes the read-side check below.
//! - **Reads** ([`lookup_reference`]) only return a hit when the *query*
//!   is its own canonical representative; together with the write guard
//!   this means stored moves are always in exactly the orientation of any
//!   board they are served for.
//!
//! Full orientation tracking (storing the symmetry transform index so
//! moves can be translated to any queried orientation) is the documented
//! follow-up that would lift this restriction on both sides.

use crate::bitboard::Bitboard;
use crate::game::current_player;
use crate::opening_book::OpeningBookDatabase;
use crate::state::State;
use serde_json::{json, Value};

use super::reference::parse_move_key;

/// Upsert every eligible solved reference in `dataset_payload` (a dataset
/// or bundle JSON artifact with a top-level `positions` array, each entry
/// carrying `qfen` and an optional `reference`) into `db`. Positions with
/// a `null` reference are skipped, and — per the module-level orientation
/// caveat — so are solved positions that are **not their own canonical
/// representative** (silently: their optimal moves are in their own
/// orientation, which is the wrong orientation for the canonical key the
/// row would be stored under). Returns the number of positions actually
/// inserted.
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

        // Write-side orientation guard, symmetric with lookup_reference's
        // read-side guard: only canonical representatives may be stored.
        // (add_solved_position enforces this too, as defense in depth.)
        if state.canonical_payload() != state.bb.to_le_bytes() {
            continue;
        }

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

        let written = db
            .add_solved_position(&state, value, &optimal_moves)
            .map_err(|e| format!("add_solved_position for {qfen}: {e}"))?;
        if written {
            count += 1;
        }
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

    /// Smoke test for the `export-book` CLI path against the golden
    /// shared dataset (22 solved references). Only solved positions that
    /// are their own canonical representative are eligible for storage
    /// (the write-side orientation guard), so the expected insertion
    /// count is computed from the artifact rather than hardcoded, and the
    /// exported rows must be exactly that set. A rerun is idempotent (no
    /// row growth).
    #[test]
    fn golden_dataset_export_inserts_exactly_the_representative_solved_set() {
        let golden = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../benchmarks/positions-v1.json");
        let payload = crate::bench::dataset::load(&golden).unwrap();

        let positions = payload["positions"].as_array().unwrap();
        let (representative, skipped): (Vec<&Value>, Vec<&Value>) = positions
            .iter()
            .filter(|p| !p["reference"].is_null())
            .partition(|p| {
                let state = State::from_qfen(p["qfen"].as_str().unwrap()).unwrap();
                state.canonical_payload() == state.bb.to_le_bytes()
            });
        // The golden artifact must exercise both sides of the guard.
        assert!(!representative.is_empty(), "no representative solved refs");
        assert!(!skipped.is_empty(), "no non-representative solved refs");

        let expected = representative.len() as u64;
        let path = temp_db_path("golden");
        let db = open_book(&path);
        assert_eq!(export_references(&payload, &db).unwrap(), expected);
        assert_eq!(db.total_positions().unwrap() as u64, expected);

        // Exactly the representative set is present; the rest is absent.
        for p in &representative {
            let state = State::from_qfen(p["qfen"].as_str().unwrap()).unwrap();
            let entry = db.get_position(&state).unwrap().unwrap();
            assert!(entry.solved);
            assert_eq!(
                entry.game_value.unwrap() as i64,
                p["reference"]["value"].as_i64().unwrap()
            );
        }
        for p in &skipped {
            let state = State::from_qfen(p["qfen"].as_str().unwrap()).unwrap();
            assert!(
                db.get_position(&state).unwrap().is_none(),
                "non-representative {} must not be stored",
                p["id"]
            );
        }

        // Rerun: same upserts, no row growth.
        assert_eq!(export_references(&payload, &db).unwrap(), expected);
        assert_eq!(db.total_positions().unwrap() as u64, expected);

        std::fs::remove_file(&path).ok();
    }

    /// Regression test for the cross-orientation wrong-hit bug: storing a
    /// solved position that is NOT its own canonical representative, then
    /// querying its canonical representative (which passes the read-side
    /// guard), must return None — nothing may have been written, because
    /// the stored moves would be in the wrong orientation for that board.
    /// Before the write-side guard existed, this lookup returned a hit
    /// whose optimal_moves were wrong (possibly illegal) for the queried
    /// board.
    #[test]
    fn non_canonical_write_never_pollutes_the_representative_lookup() {
        let bb = non_canonical_position(0, 11);
        let reference = solve_position(&bb, 60.0).unwrap();
        let qfen = State::new(bb).to_qfen();
        let dataset_payload = json!({
            "positions": [
                {"id": "p0000", "qfen": qfen, "phase": "endgame", "reference": reference},
            ],
        });

        let path = temp_db_path("cross-orientation");
        let db = open_book(&path);

        // The write must be skipped entirely.
        assert_eq!(export_references(&dataset_payload, &db).unwrap(), 0);
        assert_eq!(db.total_positions().unwrap(), 0);

        // The canonical representative of the same class (idempotent:
        // it IS its own representative, so it passes the read guard) must
        // find nothing.
        let canon_bb = Bitboard::from_le_bytes(&State::new(bb).canonical_payload());
        assert_eq!(
            State::new(canon_bb).canonical_payload(),
            canon_bb.to_le_bytes(),
            "canonicalization is idempotent"
        );
        assert!(lookup_reference(&canon_bb, &db).is_none());

        // And so must the original orientation.
        assert!(lookup_reference(&bb, &db).is_none());

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
