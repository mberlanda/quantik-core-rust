use quantik_core::bitboard::Bitboard;
use quantik_core::constants::{FLAG_CANON, VERSION};
use quantik_core::game::has_winning_line;
use quantik_core::moves::{apply_move, generate_legal_moves};
use quantik_core::symmetry::SymmetryHandler;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

fn canonical_key(bb: &Bitboard) -> [u8; 18] {
    let canon = SymmetryHandler::find_canonical(bb);
    let mut key = [0u8; 18];
    key[0] = VERSION;
    key[1] = FLAG_CANON;
    key[2..18].copy_from_slice(&canon.to_le_bytes());
    key
}

fn key_to_hex(key: &[u8; 18]) -> String {
    key.iter().map(|b| format!("{:02x}", b)).collect()
}

fn hex_to_key(hex: &str) -> Option<[u8; 18]> {
    if hex.len() != 36 { return None; }
    let mut key = [0u8; 18];
    for i in 0..18 {
        key[i] = u8::from_str_radix(&hex[i*2..i*2+2], 16).ok()?;
    }
    Some(key)
}

fn key_to_bb(key: &[u8; 18]) -> Bitboard {
    let mut planes = [0u16; 8];
    for i in 0..8 {
        planes[i] = u16::from_le_bytes([key[2 + i*2], key[2 + i*2 + 1]]);
    }
    Bitboard::new(planes)
}

const CHECKPOINT_DIR: &str = "bfs_checkpoints";
const CHECKPOINT_INTERVAL: usize = 100_000;

fn save_checkpoint(depth: usize, seen: &HashSet<[u8; 18]>, frontier: &[(Bitboard, [u8; 18])]) {
    fs::create_dir_all(CHECKPOINT_DIR).unwrap();

    let seen_path = format!("{}/seen_depth_{}.txt", CHECKPOINT_DIR, depth);
    let frontier_path = format!("{}/frontier_depth_{}.txt", CHECKPOINT_DIR, depth);
    let meta_path = format!("{}/meta.txt", CHECKPOINT_DIR);

    eprintln!("[checkpoint] Saving depth {} ({} seen, {} frontier)...", depth, seen.len(), frontier.len());

    let f = fs::File::create(&seen_path).unwrap();
    let mut w = BufWriter::new(f);
    for key in seen {
        writeln!(w, "{}", key_to_hex(key)).unwrap();
    }
    w.flush().unwrap();

    let f = fs::File::create(&frontier_path).unwrap();
    let mut w = BufWriter::new(f);
    for (_, key) in frontier {
        writeln!(w, "{}", key_to_hex(key)).unwrap();
    }
    w.flush().unwrap();

    fs::write(&meta_path, format!("{}\n{}\n{}\n", depth, seen.len(), frontier.len())).unwrap();
    eprintln!("[checkpoint] Saved.");
}

fn load_checkpoint() -> Option<(usize, HashSet<[u8; 18]>, Vec<(Bitboard, [u8; 18])>)> {
    let meta_path = format!("{}/meta.txt", CHECKPOINT_DIR);
    let meta = fs::read_to_string(&meta_path).ok()?;
    let lines: Vec<&str> = meta.trim().lines().collect();
    if lines.len() < 3 { return None; }
    let depth: usize = lines[0].parse().ok()?;

    eprintln!("[checkpoint] Loading from depth {}...", depth);

    let seen_path = format!("{}/seen_depth_{}.txt", CHECKPOINT_DIR, depth);
    let frontier_path = format!("{}/frontier_depth_{}.txt", CHECKPOINT_DIR, depth);

    let f = fs::File::open(&seen_path).ok()?;
    let reader = BufReader::new(f);
    let mut seen = HashSet::new();
    for line in reader.lines() {
        let hex = line.ok()?;
        seen.insert(hex_to_key(hex.trim())?);
    }

    let f = fs::File::open(&frontier_path).ok()?;
    let reader = BufReader::new(f);
    let mut frontier = Vec::new();
    for line in reader.lines() {
        let hex = line.ok()?;
        let key = hex_to_key(hex.trim())?;
        let bb = key_to_bb(&key);
        frontier.push((bb, key));
    }

    eprintln!("[checkpoint] Loaded: {} seen, {} frontier", seen.len(), frontier.len());
    Some((depth, seen, frontier))
}

fn main() {
    let max_depth: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let resume = env::args().any(|a| a == "--resume");

    let (start_depth, mut seen, mut frontier) = if resume {
        match load_checkpoint() {
            Some((d, s, f)) => (d + 1, s, f),
            None => {
                eprintln!("[checkpoint] No checkpoint found, starting fresh.");
                let root = Bitboard::EMPTY;
                let root_key = canonical_key(&root);
                let mut seen = HashSet::new();
                seen.insert(root_key);
                (1, seen, vec![(root, root_key)])
            }
        }
    } else {
        let root = Bitboard::EMPTY;
        let root_key = canonical_key(&root);
        let mut seen = HashSet::new();
        seen.insert(root_key);
        (1, seen, vec![(root, root_key)])
    };

    println!("Rust BFS benchmark (depth {}, starting from {})", max_depth, start_depth);
    println!("{:>5} {:>12} {:>12} {:>10} {:>12}", "Depth", "Positions", "Frontier", "Time (s)", "Cumulative");

    let t_start = Instant::now();

    for depth in start_depth..=max_depth {
        let t_depth = Instant::now();
        let mut next_frontier: Vec<(Bitboard, [u8; 18])> = Vec::new();
        let mut processed = 0usize;

        for &(bb, _) in &frontier {
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
            processed += 1;
            if processed % CHECKPOINT_INTERVAL == 0 {
                eprintln!("[depth {}] processed {}/{} frontier nodes, {} seen so far",
                    depth, processed, frontier.len(), seen.len());
            }
        }

        let elapsed_depth = t_depth.elapsed().as_secs_f64();
        let elapsed_total = t_start.elapsed().as_secs_f64();
        println!("{:>5} {:>12} {:>12} {:>10.3} {:>12.3}",
            depth, seen.len(), next_frontier.len(), elapsed_depth, elapsed_total);

        save_checkpoint(depth, &seen, &next_frontier);
        frontier = next_frontier;
    }

    let total = t_start.elapsed().as_secs_f64();
    println!("\nTotal: {} unique positions in {:.3}s", seen.len(), total);
}
