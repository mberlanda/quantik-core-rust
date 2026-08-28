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
//! cargo run --release --example exact_oracle -- --roots-only < positions.txt
//! ```
//!
//! `--roots-only` emits just each position's own exact value, skipping the
//! per-move breakdown. That is ~25x cheaper (one solve instead of one per
//! legal move) and is enough to reconstruct the moves by backward induction
//! when the *whole* level below is solved: a position's optimal moves are
//! exactly those leading to a child the opponent loses.
//!
//! Two operational properties matter for the million-position runs this is
//! built for:
//!
//! * **Results stream out.** Positions are solved in chunks and each chunk is
//!   written and flushed before the next starts, so interrupting the process
//!   keeps everything solved so far. With `--append-to`, re-running skips the
//!   QFENs already in the output, which makes a long solve resumable.
//! * **Thread count is bounded.** `--threads N` sizes the rayon pool. The
//!   default saturates every core, which is right for a dedicated batch run
//!   and wrong when anything else needs the machine — including a second copy
//!   of this tool.

use quantik_core::game::{current_player, has_winning_line};
use quantik_core::minimax::{MinimaxConfig, MinimaxEngine};
use quantik_core::moves::{apply_move, generate_legal_moves};
use quantik_core::state::State;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{self, BufRead, BufReader, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Positions solved between output flushes. Large enough that the write is
/// negligible, small enough that an interrupt loses seconds, not hours.
const CHUNK: usize = 2_000;

fn flag_value(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

/// QFENs already present in `path`, so a resumed run does not redo them.
fn already_solved(path: &str) -> HashSet<String> {
    let mut seen = HashSet::new();
    if let Ok(file) = std::fs::File::open(path) {
        for line in BufReader::new(file).lines().map_while(Result::ok) {
            if let Some(rest) = line.split("\"qfen\":\"").nth(1) {
                if let Some(qfen) = rest.split('"').next() {
                    seen.insert(qfen.to_string());
                }
            }
        }
    }
    seen
}

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

/// Just the root's own exact value — one solve, no per-move breakdown.
fn root_line(qfen: &str) -> Result<String, String> {
    let state = State::from_qfen(qfen)?;
    if current_player(&state.bb).is_none() {
        return Err(format!("inconsistent position: {qfen}"));
    }
    if has_winning_line(&state.bb) || generate_legal_moves(&state.bb).is_empty() {
        // Terminal: the side to move has already lost.
        return Ok(format!(
            "{{\"qfen\":\"{qfen}\",\"score\":{TERMINAL},\"won\":false}}"
        ));
    }
    let score = solve_score(&state);
    Ok(format!(
        "{{\"qfen\":\"{}\",\"score\":{},\"won\":{}}}",
        qfen,
        score,
        score > 0.0
    ))
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
    let roots_only = std::env::args().any(|arg| arg == "--roots-only");
    if let Some(threads) = flag_value("--threads").and_then(|v| v.parse::<usize>().ok()) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("thread pool");
        eprintln!("rayon pool: {threads} threads");
    }
    let append_to = flag_value("--append-to");

    let stdin = io::stdin();
    let mut positions: Vec<String> = stdin
        .lock()
        .lines()
        .map_while(Result::ok)
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();

    // Resume: drop anything the output file already holds.
    let mut sink: Box<dyn Write> = match &append_to {
        Some(path) => {
            let seen = already_solved(path);
            if !seen.is_empty() {
                let before = positions.len();
                positions.retain(|qfen| !seen.contains(qfen));
                eprintln!("resuming: {} of {before} already solved", seen.len());
            }
            Box::new(io::BufWriter::new(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .expect("open output"),
            ))
        }
        None => Box::new(io::BufWriter::new(io::stdout())),
    };

    let total = positions.len();
    eprintln!("solving {total} positions");
    let done = AtomicUsize::new(0);

    // Solve in chunks and flush each one: a million-position run must not lose
    // everything to an interrupt, and buffering it all costs memory for nothing.
    for chunk in positions.chunks(CHUNK) {
        let lines: Vec<Result<String, String>> = chunk
            .par_iter()
            .map(|qfen| {
                let result = if roots_only {
                    root_line(qfen)
                } else {
                    oracle_line(qfen)
                };
                done.fetch_add(1, Ordering::Relaxed);
                result
            })
            .collect();
        for line in lines {
            match line {
                Ok(text) => writeln!(sink, "{text}").expect("write"),
                Err(message) => eprintln!("skipped: {message}"),
            }
        }
        sink.flush().expect("flush");
        eprintln!("  {}/{}", done.load(Ordering::Relaxed), total);
    }
}
