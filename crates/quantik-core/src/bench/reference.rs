//! Exact game-theoretic references for benchmark positions.
//!
//! Port of `benchmarks/reference.py`: a position's reference is the game
//! value for the side to move plus the complete set of optimal moves,
//! produced by full-depth minimax and stored only when every child was
//! solved with no cutoff.

use crate::bench::book_export::lookup_reference;
use crate::bitboard::Bitboard;
use crate::game::has_winning_line;
use crate::minimax::{MinimaxConfig, MinimaxEngine};
use crate::moves::{apply_move, generate_legal_moves, Move};
use crate::opening_book::OpeningBookDatabase;
use crate::state::State;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::Instant;

/// Stable string identifier for a move: `player:shape:position`.
pub fn move_key(mv: &Move) -> String {
    format!("{}:{}:{}", mv.player, mv.shape, mv.position)
}

/// Parse a move key into `(player, shape, position)`.
pub fn parse_move_key(key: &str) -> Result<(u8, u8, u8), String> {
    let parts: Vec<&str> = key.split(':').collect();
    if parts.len() != 3 {
        return Err(format!("invalid move key {key:?}"));
    }
    let parse = |s: &str| -> Result<u8, String> {
        s.parse::<u8>()
            .map_err(|e| format!("move key {key:?}: {e}"))
    };
    Ok((parse(parts[0])?, parse(parts[1])?, parse(parts[2])?))
}

fn remaining_plies(bb: &Bitboard) -> u32 {
    16 - bb.player_piece_count(0) - bb.player_piece_count(1)
}

/// Solve one root child exactly within `remaining_budget` seconds.
/// Returns `(negated score, nodes, child pv keys)` or `None` on cutoff.
fn score_child(child_bb: &Bitboard, remaining_budget: f64) -> Option<(f64, u64, Vec<String>)> {
    let mut engine = MinimaxEngine::new(MinimaxConfig {
        max_depth: 16,
        time_limit_s: Some(remaining_budget),
        ..Default::default()
    });
    let result = engine.search(&State::new(*child_bb)).ok()?;
    if result.depth_reached < remaining_plies(child_bb) {
        return None;
    }
    Some((
        -result.score,
        result.nodes,
        result.pv.iter().map(move_key).collect(),
    ))
}

/// Return an exact reference for `bb`, or `None` on budget cutoff.
///
/// The reference is exact because Quantik never exceeds 16 plies: a
/// completed iterative-deepening depth at least equal to a child's
/// remaining plies proves the child was solved to true terminals.
pub fn solve_position(bb: &Bitboard, budget_s: f64) -> Option<Value> {
    let started = Instant::now();
    let legal_moves = generate_legal_moves(bb);
    if legal_moves.is_empty() {
        return None;
    }

    const IMMEDIATE_WIN: f64 = f64::INFINITY;
    let mut scored: BTreeMap<String, f64> = BTreeMap::new();
    let mut pvs: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut nodes = 0u64;

    for mv in &legal_moves {
        let key = move_key(mv);
        let child_bb = apply_move(bb, mv);

        if has_winning_line(&child_bb) || generate_legal_moves(&child_bb).is_empty() {
            scored.insert(key.clone(), IMMEDIATE_WIN);
            pvs.insert(key.clone(), vec![key]);
            continue;
        }

        let remaining_budget = budget_s - started.elapsed().as_secs_f64();
        if remaining_budget <= 0.0 {
            return None;
        }

        let (score, child_nodes, child_pv) = score_child(&child_bb, remaining_budget)?;
        scored.insert(key.clone(), score);
        let mut pv = vec![key.clone()];
        pv.extend(child_pv);
        pvs.insert(key, pv);
        nodes += child_nodes;
    }

    let best_score = scored.values().copied().fold(f64::NEG_INFINITY, f64::max);
    // BTreeMap iteration is already sorted by key.
    let optimal_moves: Vec<String> = scored
        .iter()
        .filter(|(_, &score)| score == best_score)
        .map(|(key, _)| key.clone())
        .collect();

    let solve_time_s = (started.elapsed().as_secs_f64() * 1e6).round() / 1e6;
    Some(json!({
        "solved": true,
        "no_cutoff": true,
        "value": if best_score > 0.0 { 1 } else { -1 },
        "optimal_moves": optimal_moves,
        "pv": pvs[&optimal_moves[0]],
        "nodes": nodes,
        "solve_time_s": solve_time_s,
        "solver": format!(
            "MinimaxEngine(max_depth=16, budget_s={}) quantik-core-rust {}",
            crate::bench::canonical::python_float_repr(budget_s),
            env!("CARGO_PKG_VERSION"),
        ),
    }))
}

/// Like [`solve_position`], but first probes `book` (when given) for a
/// stored solved reference at `bb`'s canonical key, short-circuiting the
/// minimax solve on a hit. On a fresh solve, the result is written back
/// through `book` (best-effort — a write failure is silently ignored, the
/// solve itself already succeeded and is returned regardless).
///
/// See [`crate::bench::book_export`] for the orientation caveat: both the
/// lookup and the write-back apply only when `bb` is its own canonical
/// representative — the stored moves are in that one orientation and
/// cannot be translated across symmetries yet.
pub fn solve_position_with_book(
    bb: &Bitboard,
    budget_s: f64,
    book: Option<&OpeningBookDatabase>,
) -> Option<Value> {
    if let Some(db) = book {
        if let Some(hit) = lookup_reference(bb, db) {
            return Some(hit);
        }
    }

    let reference = solve_position(bb, budget_s)?;

    // Write-side orientation guard, symmetric with lookup_reference's
    // read-side guard: only canonical representatives may be stored
    // (add_solved_position enforces this too, as defense in depth).
    if let Some(db) = book {
        if State::new(*bb).canonical_payload() == bb.to_le_bytes() {
            let value = reference["value"].as_i64().unwrap_or(0) as i32;
            if let Some(optimal_moves) = reference["optimal_moves"].as_array() {
                let parsed: Result<Vec<(i32, i32)>, String> = optimal_moves
                    .iter()
                    .map(|v| {
                        let key = v.as_str().ok_or("optimal_moves entry is not a string")?;
                        let (_, shape, pos) = parse_move_key(key)?;
                        Ok((shape as i32, pos as i32))
                    })
                    .collect();
                if let Ok(optimal_moves) = parsed {
                    let _ = db.add_solved_position(&State::new(*bb), value, &optimal_moves);
                }
            }
        }
    }

    Some(reference)
}

/// Fill reference fields in place; the `opening` phase is skipped (its
/// positions are too expensive to solve and never contribute to exact
/// move-agreement figures).
pub fn augment_with_references(payload: &mut Value, budget_s: f64) {
    augment_with_references_with_book(payload, budget_s, None);
}

/// Like [`augment_with_references`], but reads through (and writes back
/// into) `book` when given — see [`solve_position_with_book`].
pub fn augment_with_references_with_book(
    payload: &mut Value,
    budget_s: f64,
    book: Option<&OpeningBookDatabase>,
) {
    let Some(positions) = payload.get_mut("positions").and_then(Value::as_array_mut) else {
        return;
    };
    for position in positions {
        if position["phase"] == "opening" {
            position["reference"] = Value::Null;
            continue;
        }
        let bb = State::from_qfen(position["qfen"].as_str().unwrap_or_default())
            .map(|s| s.bb)
            .unwrap_or(Bitboard::EMPTY);
        position["reference"] =
            solve_position_with_book(&bb, budget_s, book).unwrap_or(Value::Null);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

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

    #[test]
    fn move_key_roundtrip() {
        let mv = Move::new(1, 3, 15);
        assert_eq!(move_key(&mv), "1:3:15");
        assert_eq!(parse_move_key("1:3:15").unwrap(), (1, 3, 15));
        assert!(parse_move_key("1:3").is_err());
        assert!(parse_move_key("a:b:c").is_err());
    }

    #[test]
    fn immediate_win_is_optimal() {
        // Find a deep (cheap to solve exactly) position where the side to
        // move has an immediate winning reply; the solver must value it +1
        // and list every immediate win among the optimal moves.
        let (bb, winning) = (0u64..)
            .find_map(|seed| {
                let bb = random_position(seed, 11);
                let winning: Vec<Move> = generate_legal_moves(&bb)
                    .into_iter()
                    .filter(|mv| has_winning_line(&apply_move(&bb, mv)))
                    .collect();
                (!winning.is_empty()).then_some((bb, winning))
            })
            .unwrap();

        let reference = solve_position(&bb, 60.0).unwrap();
        assert_eq!(reference["value"], json!(1));
        let optimal: Vec<&str> = reference["optimal_moves"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        for mv in &winning {
            assert!(optimal.contains(&move_key(mv).as_str()), "{optimal:?}");
        }
        assert_eq!(reference["pv"][0], json!(optimal[0]));
        assert_eq!(reference["solved"], json!(true));
        assert_eq!(reference["no_cutoff"], json!(true));
    }

    #[test]
    fn tiny_budget_returns_none() {
        let bb = random_position(3, 5);
        assert!(solve_position(&bb, 1e-9).is_none());
    }

    #[test]
    fn solved_reference_selected_moves_verify() {
        // Deep position: cheap exact solve; every optimal move must be legal.
        let bb = random_position(11, 10);
        let reference = solve_position(&bb, 60.0).unwrap();
        let legal: Vec<String> = generate_legal_moves(&bb).iter().map(move_key).collect();
        for mv in reference["optimal_moves"].as_array().unwrap() {
            assert!(legal.contains(&mv.as_str().unwrap().to_string()));
        }
        let value = reference["value"].as_i64().unwrap();
        assert!(value == 1 || value == -1);
    }

    fn temp_book_path(tag: &str) -> std::path::PathBuf {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("quantik_ref_book_{tag}_{id}.db"))
    }

    fn open_book(path: &std::path::Path) -> crate::opening_book::OpeningBookDatabase {
        crate::opening_book::OpeningBookDatabase::open(&crate::opening_book::OpeningBookConfig {
            database_path: path.to_string_lossy().to_string(),
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn solve_position_with_book_short_circuits_canonical_positions() {
        // A canonical-representative position: after the first solve
        // writes it back, a second lookup must hit the book (solver ==
        // "opening-book", nodes == 0) instead of re-solving.
        let bb = (0u64..)
            .map(|seed| random_position(seed, 11))
            .find(|bb| State::new(*bb).canonical_payload() == bb.to_le_bytes())
            .expect("a canonical-representative position exists among random samples");

        let path = temp_book_path("short-circuit");
        let db = open_book(&path);

        let first = solve_position_with_book(&bb, 60.0, Some(&db)).unwrap();
        assert_ne!(first["solver"], json!("opening-book"));

        let second = solve_position_with_book(&bb, 60.0, Some(&db)).unwrap();
        assert_eq!(second["solver"], json!("opening-book"));
        assert_eq!(second["nodes"], json!(0));
        assert_eq!(second["value"], first["value"]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn solve_position_with_book_never_short_circuits_non_canonical_positions() {
        // A non-canonical-orientation position: the book guard requires
        // the QUERY itself to be its own canonical representative, so
        // lookups here must always fall through to a fresh minimax solve,
        // even after a write-back.
        let bb = (0u64..)
            .map(|seed| random_position(seed, 11))
            .find(|bb| State::new(*bb).canonical_payload() != bb.to_le_bytes())
            .expect("a non-canonical position exists among random samples");

        let path = temp_book_path("no-short-circuit");
        let db = open_book(&path);

        let first = solve_position_with_book(&bb, 60.0, Some(&db)).unwrap();
        assert_ne!(first["solver"], json!("opening-book"));

        let second = solve_position_with_book(&bb, 60.0, Some(&db)).unwrap();
        assert_ne!(second["solver"], json!("opening-book"));

        std::fs::remove_file(&path).ok();
    }
}
