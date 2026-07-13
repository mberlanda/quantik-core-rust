//! One-off survey: exactly solve every canonical, nonterminal depth-4
//! Quantik position and report the resulting value distribution.
//!
//! Enumerates by canonical-key BFS (matching the beam-search multiplicity
//! technique documented in docs/BEAM_SEARCH.md): each ply, every legal
//! child is folded onto its canonical representative, and multiplicity
//! (how many raw 4-ply move sequences land on that representative) is
//! summed. Terminal (already-won) children are excluded, matching the
//! "nonterminal" count in the Python worktree's depth4-canonical dataset.
//!
//! Solving reuses `bench::reference::solve_position_with_book` so results
//! land in a real, cross-language-portable SQLite opening book as a
//! byproduct.
//!
//! Usage:
//!   cargo run --release --example depth4_survey -- \
//!     --budget-s 5 --limit 0 --db benchmarks/results/depth4.db \
//!     --out benchmarks/results/depth4-survey.json

use quantik_core::bench::reference::solve_position_with_book;
use quantik_core::bitboard::Bitboard;
use quantik_core::game::has_winning_line;
use quantik_core::moves::{apply_move, generate_legal_moves};
use quantik_core::opening_book::{OpeningBookConfig, OpeningBookDatabase};
use quantik_core::state::State;
use quantik_core::symmetry::SymmetryHandler;
use std::collections::HashMap;
use std::time::Instant;

struct Args {
    budget_s: f64,
    limit: usize,
    sample: usize,
    seed: u64,
    db: String,
    out: String,
}

fn parse_args() -> Args {
    let mut budget_s = 5.0;
    let mut limit = 0usize;
    let mut sample = 0usize;
    let mut seed = 20260712u64;
    let mut db = "benchmarks/results/depth4.db".to_string();
    let mut out = "benchmarks/results/depth4-survey.json".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--budget-s" => budget_s = it.next().unwrap().parse().unwrap(),
            "--limit" => limit = it.next().unwrap().parse().unwrap(),
            "--sample" => sample = it.next().unwrap().parse().unwrap(),
            "--seed" => seed = it.next().unwrap().parse().unwrap(),
            "--db" => db = it.next().unwrap(),
            "--out" => out = it.next().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    Args {
        budget_s,
        limit,
        sample,
        seed,
        db,
        out,
    }
}

/// Canonical-key BFS to depth 4: fold every legal child onto its canonical
/// representative each ply, excluding wins, summing multiplicity from all
/// incoming (parent, move) pairs that collapse onto it.
fn enumerate_depth4() -> Vec<(Bitboard, u64)> {
    let mut frontier: HashMap<[u8; 16], (Bitboard, u64)> = HashMap::new();
    frontier.insert(Bitboard::EMPTY.to_le_bytes(), (Bitboard::EMPTY, 1));

    for _ply in 0..4 {
        let mut next: HashMap<[u8; 16], (Bitboard, u64)> = HashMap::new();
        for (bb, mult) in frontier.values() {
            for mv in generate_legal_moves(bb) {
                let child = apply_move(bb, &mv);
                if has_winning_line(&child) {
                    continue;
                }
                let canon = SymmetryHandler::find_canonical(&child);
                let key = canon.to_le_bytes();
                let entry = next.entry(key).or_insert((canon, 0));
                entry.1 += mult;
            }
        }
        frontier = next;
    }

    let mut states: Vec<(Bitboard, u64)> = frontier
        .into_values()
        .filter(|(bb, _)| !generate_legal_moves(bb).is_empty())
        .collect();
    states.sort_by_key(|(bb, _)| bb.to_le_bytes());
    states
}

fn main() {
    let args = parse_args();

    let started = Instant::now();
    let mut states = enumerate_depth4();
    println!(
        "enumerated {} canonical nonterminal depth-4 states in {:.2}s",
        states.len(),
        started.elapsed().as_secs_f64()
    );
    if args.sample > 0 && args.sample < states.len() {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(args.seed);
        states.shuffle(&mut rng);
        states.truncate(args.sample);
        states.sort_by_key(|(bb, _)| bb.to_le_bytes());
        println!(
            "sampled {} of {} states (seed={})",
            states.len(),
            10946,
            args.seed
        );
    } else if args.limit > 0 {
        states.truncate(args.limit);
        println!("limited to {} states for this run", states.len());
    }

    let book = OpeningBookDatabase::open(&OpeningBookConfig {
        database_path: args.db.clone(),
        ..Default::default()
    })
    .expect("open book");

    let mut rows: Vec<serde_json::Value> = Vec::with_capacity(states.len());
    let mut solved = 0usize;
    let mut cutoff = 0usize;
    let solve_started = Instant::now();

    for (i, (bb, mult)) in states.iter().enumerate() {
        let t0 = Instant::now();
        let reference = solve_position_with_book(bb, args.budget_s, Some(&book));
        let elapsed = t0.elapsed().as_secs_f64();
        println!(
            "  [{}/{}] {} solved={} elapsed={:.3}s",
            i + 1,
            states.len(),
            State::new(*bb).to_qfen(),
            reference.is_some(),
            elapsed
        );

        match &reference {
            Some(r) => {
                solved += 1;
                rows.push(serde_json::json!({
                    "qfen": State::new(*bb).to_qfen(),
                    "multiplicity": mult,
                    "value": r["value"],
                    "nodes": r["nodes"],
                    "solve_time_s": elapsed,
                    "solver": r["solver"],
                }));
            }
            None => {
                cutoff += 1;
                rows.push(serde_json::json!({
                    "qfen": State::new(*bb).to_qfen(),
                    "multiplicity": mult,
                    "value": null,
                    "nodes": null,
                    "solve_time_s": elapsed,
                    "solver": null,
                }));
            }
        }

        if (i + 1) % 25 == 0 || i + 1 == states.len() {
            println!(
                "progress: {}/{} solved={} cutoff={} elapsed_total={:.1}s",
                i + 1,
                states.len(),
                solved,
                cutoff,
                solve_started.elapsed().as_secs_f64()
            );
            // Flush partial results after every progress line so a killed
            // run still leaves a readable, if incomplete, artifact — the
            // SQLite book itself is already durable per-row regardless.
            let partial = serde_json::json!({
                "budget_s": args.budget_s,
                "total_canonical_depth4_states": states.len(),
                "processed": i + 1,
                "solved": solved,
                "cutoff": cutoff,
                "positions": rows,
            });
            std::fs::write(&args.out, serde_json::to_string_pretty(&partial).unwrap())
                .expect("write partial output");
        }
    }

    println!(
        "done: {} solved, {} cutoff, total solve wall time {:.1}s -> {}",
        solved,
        cutoff,
        solve_started.elapsed().as_secs_f64(),
        args.out
    );
}
