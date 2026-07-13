//! Self-play data exporter: plays full games with MCTSEngine on both
//! sides, and writes one JSONL training row per ply — position, the
//! visit-count policy target (Task 5's `root_move_visits`), and the
//! eventual game outcome from that ply's mover perspective. Quantik has
//! no draws (see docs/benchmarks/quantik-game-tree-census-2026-07-13.md),
//! so `value` is always exactly +1.0 or -1.0, never 0.0.
//!
//! Row schema (one per line, compact canonical JSON):
//! {
//!   "game_id": u64,
//!   "ply": u32,
//!   "qfen": string,
//!   "side_to_move": 0 | 1,
//!   "policy": [{"shape": u8, "position": u8, "visits": u32}, ...],
//!   "value": 1.0 | -1.0   // outcome for `side_to_move`, decided in hindsight
//! }
//!
//! Usage:
//!   cargo run --release --example selfplay_export -- \
//!     --games 100 --iterations 2000 --seed 20260713 \
//!     --out benchmarks/results/selfplay.jsonl

use quantik_core::bench::canonical::canonical_json;
use quantik_core::bitboard::Bitboard;
use quantik_core::game::{check_winner, current_player, has_winning_line, WinStatus};
use quantik_core::mcts::{MCTSConfig, MCTSEngine};
use quantik_core::moves::{apply_move, generate_legal_moves};
use quantik_core::state::State;
use serde_json::json;
use std::io::Write;

struct Args {
    games: u32,
    iterations: u32,
    seed: u64,
    out: String,
}

fn parse_args() -> Args {
    let mut games = 100u32;
    let mut iterations = 2000u32;
    let mut seed = 20260713u64;
    let mut out = "benchmarks/results/selfplay.jsonl".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--games" => games = it.next().unwrap().parse().unwrap(),
            "--iterations" => iterations = it.next().unwrap().parse().unwrap(),
            "--seed" => seed = it.next().unwrap().parse().unwrap(),
            "--out" => out = it.next().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    Args {
        games,
        iterations,
        seed,
        out,
    }
}

struct PendingRow {
    ply: u32,
    qfen: String,
    side_to_move: u8,
    policy: Vec<(u8, u8, u32)>, // (shape, position, visits)
}

/// Play one self-play game to completion, returning one pending row per
/// ply (value filled in afterward, once the winner is known).
fn play_game(seed: u64, iterations: u32) -> (Vec<PendingRow>, WinStatus) {
    let mut bb = Bitboard::EMPTY;
    let mut rows = Vec::new();
    let mut ply = 0u32;

    loop {
        if has_winning_line(&bb) {
            return (rows, check_winner(&bb));
        }
        let legal = generate_legal_moves(&bb);
        if legal.is_empty() {
            // No legal moves: the side to move loses (see Global
            // Constraints — this is a decisive result, never a draw).
            let loser = current_player(&bb).unwrap();
            let winner = if loser == 0 {
                WinStatus::Player1Wins
            } else {
                WinStatus::Player0Wins
            };
            return (rows, winner);
        }

        let side_to_move = current_player(&bb).unwrap();
        // use_transposition_table MUST be false here: with it on (the
        // engine's actual default), root moves that canonicalize to the
        // same child are merged onto one shared node and reported under a
        // single arbitrary move, silently dropping every other legal move
        // that led there — worst exactly at shallow plies (the empty
        // board's 64 legal moves collapse to 3), which every self-play
        // game passes through. Verified in mcts.rs's
        // root_move_visits_default_config_collapses_symmetric_root_moves
        // test — this is not a hypothetical concern, it was caught by
        // Opus review of the PR that added root_move_visits and is
        // exactly what this exporter must avoid to produce a faithful
        // per-legal-move policy target.
        let mut engine = MCTSEngine::new(MCTSConfig {
            max_iterations: iterations,
            seed: Some(seed.wrapping_add(ply as u64)),
            use_transposition_table: false,
            ..Default::default()
        });
        let (best_move, _) = engine.search(&bb).expect("legal moves exist");
        let policy: Vec<(u8, u8, u32)> = engine
            .root_move_visits()
            .into_iter()
            .map(|(mv, visits)| (mv.shape, mv.position, visits))
            .collect();

        rows.push(PendingRow {
            ply,
            qfen: State::new(bb).to_qfen(),
            side_to_move,
            policy,
        });

        bb = apply_move(&bb, &best_move);
        ply += 1;
    }
}

fn main() {
    let args = parse_args();
    if let Some(parent) = std::path::Path::new(&args.out).parent() {
        std::fs::create_dir_all(parent).expect("mkdir output dir");
    }
    let mut file = std::fs::File::create(&args.out).expect("create output file");

    for game_id in 0..args.games {
        let (rows, winner) = play_game(
            args.seed.wrapping_add(game_id as u64 * 1000),
            args.iterations,
        );
        assert_ne!(
            winner,
            WinStatus::NoWin,
            "game must resolve to a decisive winner"
        );

        for row in rows {
            let value = match (winner, row.side_to_move) {
                (WinStatus::Player0Wins, 0) => 1.0,
                (WinStatus::Player0Wins, 1) => -1.0,
                (WinStatus::Player1Wins, 0) => -1.0,
                (WinStatus::Player1Wins, 1) => 1.0,
                // current_player() (the only source of side_to_move) never
                // returns anything but 0 or 1; WinStatus::NoWin is asserted
                // impossible above. u8's full range still requires an
                // exhaustive catch-all for the match to compile.
                _ => unreachable!("side_to_move is always 0 or 1, winner is always decisive"),
            };
            let policy_json: Vec<_> = row
                .policy
                .iter()
                .map(|(shape, position, visits)| {
                    json!({"shape": shape, "position": position, "visits": visits})
                })
                .collect();
            let record = json!({
                "game_id": game_id,
                "ply": row.ply,
                "qfen": row.qfen,
                "side_to_move": row.side_to_move,
                "policy": policy_json,
                "value": value,
            });
            writeln!(file, "{}", canonical_json(&record)).expect("write row");
        }

        if (game_id + 1) % 10 == 0 || game_id + 1 == args.games {
            println!(
                "{}/{} games exported -> {}",
                game_id + 1,
                args.games,
                args.out
            );
        }
    }
}
