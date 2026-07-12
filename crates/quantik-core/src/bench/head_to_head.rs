//! Paired, side-balanced head-to-head games from shared positions.
//!
//! Port of `benchmarks/head_to_head.py`: for every sampled position and
//! seed, two games are played — engine A as the side already to move,
//! then engine B as the side to move. Results are attributed to the
//! actual engine/color mapping because sampled positions can have either
//! player to move.

use crate::bench::adapters::{select, EngineAdapter};
use crate::bench::metrics::wilson_ci;
use crate::game::{current_player, has_winning_line};
use crate::moves::{apply_move, generate_legal_moves};
use crate::state::State;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};

/// Play from `bb`; `mover` is the side already to move.
/// Returns `(winner adapter name, plies played)`.
pub fn play_game(
    mover: &dyn EngineAdapter,
    responder: &dyn EngineAdapter,
    bb: &crate::bitboard::Bitboard,
    seed: u64,
) -> Result<(String, u32), String> {
    let mut bb = *bb;
    let mut turn = current_player(&bb).ok_or("inconsistent position")?;
    let start_turn = turn;
    let engine_for = |player: u8| -> &dyn EngineAdapter {
        if player == start_turn {
            mover
        } else {
            responder
        }
    };
    let mut plies = 0u32;

    loop {
        if has_winning_line(&bb) || generate_legal_moves(&bb).is_empty() {
            // The previous mover ended the game (line or block): the side
            // to move now has lost.
            return Ok((engine_for(1 - turn).name().to_string(), plies));
        }
        let (mv, _) = select(engine_for(turn), &bb, "h2h", Some(seed))?;
        bb = apply_move(&bb, &mv);
        turn ^= 1;
        plies += 1;
    }
}

/// Identity of one head-to-head game — `(position_id, mover, responder,
/// seed)`, matching Python's `checkpoint.h2h_key` tuple order.
pub type H2hKey = (String, String, String, u64);

pub fn h2h_key(record: &Value) -> H2hKey {
    (
        record["position_id"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        record["mover"].as_str().unwrap_or_default().to_string(),
        record["responder"].as_str().unwrap_or_default().to_string(),
        record["seed"].as_u64().unwrap_or_default(),
    )
}

/// One (mover, responder, position, seed) unit of work, matching Python's
/// `_h2h_tasks` ordering: positions, then seeds, then the two orientations.
struct H2hTask<'a> {
    mover: &'a dyn EngineAdapter,
    responder: &'a dyn EngineAdapter,
    position: &'a Value,
    seed: u64,
}

fn build_h2h_tasks<'a>(
    adapter_a: &'a dyn EngineAdapter,
    adapter_b: &'a dyn EngineAdapter,
    positions: &'a [Value],
    seeds: &'a [u64],
    skip: &HashSet<H2hKey>,
) -> Vec<H2hTask<'a>> {
    let mut tasks = Vec::new();
    for position in positions {
        let position_id = position["id"].as_str().unwrap_or_default();
        for &seed in seeds {
            for (mover, responder) in [(adapter_a, adapter_b), (adapter_b, adapter_a)] {
                let key: H2hKey = (
                    position_id.to_string(),
                    mover.name().to_string(),
                    responder.name().to_string(),
                    seed,
                );
                if skip.contains(&key) {
                    continue;
                }
                tasks.push(H2hTask {
                    mover,
                    responder,
                    position,
                    seed,
                });
            }
        }
    }
    tasks
}

fn play_h2h_task(task: &H2hTask) -> Result<Value, String> {
    let bb = State::from_qfen(task.position["qfen"].as_str().unwrap_or_default())?.bb;
    let position_id = task.position["id"].as_str().unwrap_or_default();
    let (winner, plies) = play_game(task.mover, task.responder, &bb, task.seed)?;
    Ok(json!({
        "position_id": position_id,
        "phase": task.position["phase"],
        "mover": task.mover.name(),
        "responder": task.responder.name(),
        "winner": winner,
        "plies": plies,
        "seed": task.seed,
    }))
}

/// Play both engine orientations per position and seed.
///
/// `on_record` is invoked after each completed game, IN TASK ORDER
/// (checkpoint hook — it returns `Result` so a checkpoint write failure
/// aborts the run instead of being silently dropped); `skip` games are not
/// replayed (their records must already be accounted for by the caller,
/// e.g. loaded from a checkpoint).
///
/// `workers` follows the same contract as
/// [`super::agreement::run_agreement`]: `workers == 1` is serial;
/// `workers > 1` runs the task list on a sized rayon pool via
/// `par_iter().collect()` (order-preserving) and streams the results to
/// `on_record` in task order, so results are byte-identical to a
/// `workers == 1` run.
pub fn run_head_to_head(
    adapter_a: &dyn EngineAdapter,
    adapter_b: &dyn EngineAdapter,
    positions: &[Value],
    seeds: &[u64],
    skip: &HashSet<H2hKey>,
    workers: usize,
    mut on_record: impl FnMut(&Value) -> Result<(), String>,
) -> Result<Vec<Value>, String> {
    if workers < 1 {
        return Err("workers must be at least 1".into());
    }

    let tasks = build_h2h_tasks(adapter_a, adapter_b, positions, seeds, skip);
    if tasks.is_empty() {
        return Ok(Vec::new());
    }

    let mut records = Vec::with_capacity(tasks.len());
    if workers == 1 {
        for task in &tasks {
            let record = play_h2h_task(task)?;
            on_record(&record)?;
            records.push(record);
        }
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|e| format!("build worker pool: {e}"))?;
        let results: Vec<Result<Value, String>> =
            pool.install(|| tasks.par_iter().map(play_h2h_task).collect());
        for result in results {
            let record = result?;
            on_record(&record)?;
            records.push(record);
        }
    }
    Ok(records)
}

/// Aggregate totals, as-mover splits, and per-phase splits.
pub fn aggregate_head_to_head(records: &[Value], name_a: &str, name_b: &str) -> Value {
    let wins = |rows: &[&Value], name: &str| -> u64 {
        rows.iter()
            .filter(|row| row["winner"].as_str() == Some(name))
            .count() as u64
    };
    let all: Vec<&Value> = records.iter().collect();

    let mut by_phase: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for record in records {
        by_phase
            .entry(record["phase"].as_str().unwrap_or_default().to_string())
            .or_default()
            .push(record);
    }

    let games = records.len() as u64;
    let a_wins = wins(&all, name_a);
    let (ci_low, ci_high) = wilson_ci(a_wins, games);
    let paired: HashSet<(String, u64)> = records
        .iter()
        .map(|record| {
            (
                record["position_id"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                record["seed"].as_u64().unwrap_or_default(),
            )
        })
        .collect();

    let a_as_mover: Vec<&Value> = records
        .iter()
        .filter(|record| record["mover"].as_str() == Some(name_a))
        .collect();
    let b_as_mover: Vec<&Value> = records
        .iter()
        .filter(|record| record["mover"].as_str() == Some(name_b))
        .collect();

    json!({
        "engine_a": name_a,
        "engine_b": name_b,
        "games": games,
        "paired_positions": paired.len(),
        "a_wins": a_wins,
        "b_wins": wins(&all, name_b),
        "draws": 0,
        "a_win_rate": if games > 0 { a_wins as f64 / games as f64 } else { 0.0 },
        "a_win_rate_ci95": [ci_low, ci_high],
        "a_wins_as_mover": wins(&a_as_mover, name_a),
        "b_wins_as_mover": wins(&b_as_mover, name_b),
        "by_phase": by_phase
            .into_iter()
            .map(|(phase, rows)| {
                (phase, json!({
                    "games": rows.len(),
                    "a_wins": wins(&rows, name_a),
                    "b_wins": wins(&rows, name_b),
                }))
            })
            .collect::<serde_json::Map<String, Value>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::adapters::{MinimaxAdapter, RandomAdapter};
    use crate::bitboard::Bitboard;

    #[test]
    fn play_game_terminates_and_credits_winner() {
        // A@0, b@1, C@2: mover (P1) has an immediate win available; minimax
        // as mover must win in one ply.
        let bb = Bitboard::EMPTY
            .with_move(0, 0, 0)
            .with_move(1, 1, 1)
            .with_move(0, 2, 2);
        let minimax = MinimaxAdapter {
            max_depth: 3,
            time_limit_s: None,
        };
        let random = RandomAdapter;
        let (winner, plies) = play_game(&minimax, &random, &bb, 0).unwrap();
        assert_eq!(winner, "minimax");
        assert_eq!(plies, 1);
    }

    #[test]
    fn run_and_aggregate_head_to_head() {
        let positions = vec![json!({
            "id": "p0",
            "qfen": "..../..../..../....",
            "phase": "opening",
        })];
        let a = RandomAdapter;
        let minimax = MinimaxAdapter {
            max_depth: 1,
            time_limit_s: None,
        };
        let records = run_head_to_head(
            &a,
            &minimax,
            &positions,
            &[0, 1],
            &HashSet::new(),
            1,
            |_| Ok(()),
        )
        .unwrap();
        // 1 position × 2 seeds × 2 orientations.
        assert_eq!(records.len(), 4);
        for record in &records {
            let winner = record["winner"].as_str().unwrap();
            assert!(winner == "random" || winner == "minimax");
            assert!(record["plies"].as_u64().unwrap() >= 1);
        }

        let aggregate = aggregate_head_to_head(&records, "random", "minimax");
        assert_eq!(aggregate["games"], json!(4));
        assert_eq!(aggregate["paired_positions"], json!(2));
        assert_eq!(aggregate["draws"], json!(0));
        let total = aggregate["a_wins"].as_u64().unwrap() + aggregate["b_wins"].as_u64().unwrap();
        assert_eq!(total, 4, "every game has a winner");
        assert_eq!(aggregate["by_phase"]["opening"]["games"], json!(4));
    }

    #[test]
    fn workers_zero_rejected() {
        let a = RandomAdapter;
        let b = RandomAdapter;
        assert!(run_head_to_head(&a, &b, &[], &[0], &HashSet::new(), 0, |_| Ok(())).is_err());
    }

    /// workers=2 must produce EXACTLY the same records, in the same order,
    /// as workers=1.
    #[test]
    fn workers_two_matches_workers_one_exactly() {
        let positions = vec![
            json!({"id": "p0", "qfen": "..../..../..../....", "phase": "opening"}),
            json!({"id": "p1", "qfen": "..../..../..../....", "phase": "opening"}),
        ];
        let a = RandomAdapter;
        let minimax = MinimaxAdapter {
            max_depth: 1,
            time_limit_s: None,
        };
        let seeds = [0u64, 1u64];

        let serial = run_head_to_head(&a, &minimax, &positions, &seeds, &HashSet::new(), 1, |_| {
            Ok(())
        })
        .unwrap();
        let parallel =
            run_head_to_head(&a, &minimax, &positions, &seeds, &HashSet::new(), 2, |_| {
                Ok(())
            })
            .unwrap();

        assert_eq!(serial.len(), parallel.len());
        assert_eq!(serial, parallel, "workers=2 must match workers=1 exactly");
    }
}
