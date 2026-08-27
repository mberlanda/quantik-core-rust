//! Exact oracle exporter: solve QFEN positions and emit their true values.
//!
//! `MinimaxEngine::solve` is a depth-16 search, and no Quantik game exceeds
//! 16 plies, so it always terminates on true terminal nodes — it is an exact
//! solver, not a heuristic. This example turns that into a labelling tool:
//! read QFENs on stdin, write one JSON object per line with the root score,
//! every legal move's exact value, and the set of moves that preserve the
//! best achievable *outcome*.
//!
//! Consumers (notably `quantik-models-py`'s oracle probe) use it as ground
//! truth for "did the network pick a move that keeps the win?", which is a
//! sharper progress signal than win rates against another engine.
//!
//! ```sh
//! cargo run --release --example exact_oracle < positions.txt > oracle.jsonl
//! ```

use quantik_core::game::{current_player, has_winning_line};
use quantik_core::minimax::{MinimaxConfig, MinimaxEngine};
use quantik_core::moves::{apply_move, generate_legal_moves};
use quantik_core::state::State;
use rayon::prelude::*;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Score assigned to a position whose side to move has already lost.
const TERMINAL: f64 = -10_000.0;

fn solve_score(state: &State) -> f64 {
    if has_winning_line(&state.bb) || generate_legal_moves(&state.bb).is_empty() {
        return TERMINAL;
    }
    let mut engine = MinimaxEngine::new(MinimaxConfig::default());
    engine
        .solve(state)
        .map(|result| result.score)
        .unwrap_or(TERMINAL)
}

fn oracle_line(qfen: &str) -> Result<String, String> {
    let state = State::from_qfen(qfen)?;
    if current_player(&state.bb).is_none() {
        return Err(format!("inconsistent position: {qfen}"));
    }
    let moves = generate_legal_moves(&state.bb);
    if moves.is_empty() || has_winning_line(&state.bb) {
        return Err(format!("terminal position has no decision: {qfen}"));
    }

    // Value of each legal move from the mover's perspective: the negation of
    // the resulting position's value to its own mover.
    // Nested rayon: children in parallel too. Work-stealing keeps every core
    // busy whether the batch is one expensive opening or 60k cheap endgames.
    let mut scored: Vec<(u8, f64)> = moves
        .par_iter()
        .map(|mv| {
            let child = State::new(apply_move(&state.bb, mv));
            let action = mv.shape * 16 + mv.position;
            (action, -solve_score(&child))
        })
        .collect();
    scored.sort_by_key(|&(action, _)| action);

    let best = scored
        .iter()
        .map(|&(_, value)| value)
        .fold(f64::NEG_INFINITY, f64::max);
    let won = best > 0.0;
    // Outcome-optimal: keeps the best achievable result. A search whose
    // values are win/loss cannot rank mate distance, so this — not
    // "matches the single best move" — is the fair bar for an engine.
    let outcome_optimal: Vec<u8> = scored
        .iter()
        .filter(|&&(_, value)| (value > 0.0) == won)
        .map(|&(action, _)| action)
        .collect();
    let score_optimal: Vec<u8> = scored
        .iter()
        .filter(|&&(_, value)| (value - best).abs() < 1e-6)
        .map(|&(action, _)| action)
        .collect();

    let actions: Vec<String> = scored
        .iter()
        .map(|&(action, value)| format!("\"{action}\":{value}"))
        .collect();
    Ok(format!(
        "{{\"qfen\":\"{}\",\"score\":{},\"won\":{},\"outcome_optimal\":{:?},\"score_optimal\":{:?},\"action_values\":{{{}}}}}",
        qfen,
        best,
        won,
        outcome_optimal,
        score_optimal,
        actions.join(",")
    ))
}

fn main() {
    let stdin = io::stdin();
    let positions: Vec<String> = stdin
        .lock()
        .lines()
        .filter_map(|line| line.ok())
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();

    let total = positions.len();
    eprintln!("solving {total} positions");
    let done = AtomicUsize::new(0);
    // Parallelize across positions rather than across one position's
    // children: batch throughput is what matters here, and child-level
    // parallelism leaves most cores idle on cheap late-game positions.
    let lines: Vec<Result<String, String>> = positions
        .par_iter()
        .map(|qfen| {
            let result = oracle_line(qfen);
            let seen = done.fetch_add(1, Ordering::Relaxed) + 1;
            if seen % 2000 == 0 {
                eprintln!("  {seen}/{total}");
            }
            result
        })
        .collect();

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    for line in lines {
        match line {
            Ok(text) => writeln!(out, "{text}").expect("write"),
            Err(message) => eprintln!("skipped: {message}"),
        }
    }
}
