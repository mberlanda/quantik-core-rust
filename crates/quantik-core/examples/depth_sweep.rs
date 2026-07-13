//! Canonical-state census across every depth (ply count) of Quantik,
//! depth 1 through 16. Extends the depth-4 orbit-size survey: at each
//! depth, folds every legal child onto its canonical representative
//! (192-element symmetry group), tracks path-count multiplicity, and
//! reports the orbit-size distribution — same technique as
//! `depth4_orbit_histogram`, just carried through the whole game instead
//! of stopping at depth 4.
//!
//! Unlike a full raw-state enumeration (which blows up combinatorially —
//! ~232M raw transitions already at depth 5), this folds onto canonical
//! representatives at *every* level, so the frontier processed at each
//! step is bounded by the canonical state count, not the raw one.

use quantik_core::bitboard::Bitboard;
use quantik_core::game::{check_winner, has_winning_line, WinStatus};
use quantik_core::moves::{apply_move, generate_legal_moves};
use quantik_core::symmetry::SymmetryHandler;
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

fn main() {
    let mut frontier: HashMap<[u8; 16], (Bitboard, u64)> = HashMap::new();
    frontier.insert(Bitboard::EMPTY.to_le_bytes(), (Bitboard::EMPTY, 1));

    println!(
        "depth,canonical_states,raw_boards,p0_wins_raw,p1_wins_raw,with_legal_moves,mean_orbit,min_orbit,max_orbit,elapsed_s"
    );

    for depth in 1..=16u32 {
        let started = Instant::now();
        let mut next: HashMap<[u8; 16], (Bitboard, u64)> = HashMap::new();
        let mut p0_wins_raw: u64 = 0;
        let mut p1_wins_raw: u64 = 0;

        for (bb, mult) in frontier.values() {
            for mv in generate_legal_moves(bb) {
                let child = apply_move(bb, &mv);
                if has_winning_line(&child) {
                    match check_winner(&child) {
                        WinStatus::Player0Wins => p0_wins_raw += mult,
                        WinStatus::Player1Wins => p1_wins_raw += mult,
                        WinStatus::NoWin => unreachable!(),
                    }
                    continue;
                }
                let canon = SymmetryHandler::find_canonical(&child);
                let key = canon.to_le_bytes();
                let entry = next.entry(key).or_insert((canon, 0));
                entry.1 += mult;
            }
        }
        frontier = next;

        let canonical_states = frontier.len();
        let raw_boards: u64 = frontier.values().map(|(_, m)| *m).sum();
        let with_moves = frontier
            .values()
            .filter(|(bb, _)| !generate_legal_moves(bb).is_empty())
            .count();

        let mut orbit_sizes: Vec<usize> = frontier
            .values()
            .map(|(bb, _)| SymmetryHandler::orbit_size(bb))
            .collect();
        orbit_sizes.sort_unstable();
        let mean = if orbit_sizes.is_empty() {
            0.0
        } else {
            orbit_sizes.iter().sum::<usize>() as f64 / orbit_sizes.len() as f64
        };
        let min = orbit_sizes.first().copied().unwrap_or(0);
        let max = orbit_sizes.last().copied().unwrap_or(0);

        // Per-depth orbit-size histogram, written alongside the summary line.
        let mut hist: BTreeMap<usize, u64> = BTreeMap::new();
        for bb in frontier.values() {
            *hist.entry(SymmetryHandler::orbit_size(&bb.0)).or_insert(0) += 1;
        }
        let hist_str: Vec<String> = hist.iter().map(|(s, c)| format!("{s}:{c}")).collect();

        println!(
            "{depth},{canonical_states},{raw_boards},{p0_wins_raw},{p1_wins_raw},{with_moves},{mean:.2},{min},{max},{:.3}",
            started.elapsed().as_secs_f64()
        );
        eprintln!("  depth {depth} orbit histogram: {}", hist_str.join(" "));

        if canonical_states == 0 {
            eprintln!("frontier empty at depth {depth} — every line of play has ended by here");
            break;
        }
        if with_moves == 0 {
            eprintln!("all depth-{depth} states are dead ends (no legal moves) — game always over by depth {}", depth + 1);
        }
    }
}
