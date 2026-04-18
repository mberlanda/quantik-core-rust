use quantik_core::bitboard::Bitboard;
use quantik_core::constants::{FLAG_CANON, VERSION};
use quantik_core::game::has_winning_line;
use quantik_core::moves::{apply_move, generate_legal_moves};
use quantik_core::symmetry::SymmetryHandler;
use std::collections::HashSet;
use std::env;
use std::time::Instant;

fn canonical_key(bb: &Bitboard) -> [u8; 18] {
    let canon = SymmetryHandler::find_canonical(bb);
    let mut key = [0u8; 18];
    key[0] = VERSION;
    key[1] = FLAG_CANON;
    key[2..18].copy_from_slice(&canon.to_le_bytes());
    key
}

fn main() {
    let max_depth: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let root = Bitboard::EMPTY;
    let root_key = canonical_key(&root);

    let mut seen: HashSet<[u8; 18]> = HashSet::new();
    seen.insert(root_key);

    let mut frontier: Vec<(Bitboard, [u8; 18])> = vec![(root, root_key)];

    println!("Rust BFS benchmark (depth {})", max_depth);
    println!("{:>5} {:>12} {:>12} {:>10} {:>12}", "Depth", "Positions", "Frontier", "Time (s)", "Cumulative");

    let t_start = Instant::now();

    for depth in 1..=max_depth {
        let t_depth = Instant::now();
        let mut next_frontier: Vec<(Bitboard, [u8; 18])> = Vec::new();

        for &(bb, _parent_key) in &frontier {
            if has_winning_line(&bb) {
                continue;
            }
            let moves = generate_legal_moves(&bb);
            for m in &moves {
                let child_bb = apply_move(&bb, m);
                let child_key = canonical_key(&child_bb);
                if seen.insert(child_key) {
                    next_frontier.push((child_bb, child_key));
                }
            }
        }

        let elapsed_depth = t_depth.elapsed().as_secs_f64();
        let elapsed_total = t_start.elapsed().as_secs_f64();
        println!("{:>5} {:>12} {:>12} {:>10.3} {:>12.3}", depth, seen.len(), next_frontier.len(), elapsed_depth, elapsed_total);
        frontier = next_frontier;
    }

    let total = t_start.elapsed().as_secs_f64();
    println!("\nTotal: {} unique positions in {:.3}s", seen.len(), total);
}
