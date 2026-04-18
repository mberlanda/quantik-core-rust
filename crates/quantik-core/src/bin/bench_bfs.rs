use clap::Parser;
use quantik_core::bitboard::Bitboard;
use quantik_core::constants::{FLAG_CANON, VERSION};
use quantik_core::game::{current_player, has_winning_line};
use quantik_core::moves::{apply_move, generate_legal_moves, Move};
use quantik_core::symmetry::SymmetryHandler;
use rusqlite::{params, Connection};
use std::collections::HashSet;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "bench_bfs", about = "Quantik BFS/DFS position enumerator with SQLite storage")]
struct Cli {
    /// Maximum depth to explore
    depth: usize,

    /// SQLite database path
    #[arg(long, default_value = "quantik_bfs.db")]
    db: String,

    /// Use iterative-deepening DFS instead of BFS
    #[arg(long)]
    dfs: bool,

    /// Resume from existing database
    #[arg(long)]
    resume: bool,

    /// Stop after N total positions (dropout)
    #[arg(long)]
    max_positions: Option<usize>,

    /// Only print summary
    #[arg(long)]
    quiet: bool,
}

const PROGRESS_INTERVAL: usize = 10_000;

fn canonical_key(bb: &Bitboard) -> [u8; 18] {
    let canon = SymmetryHandler::find_canonical(bb);
    let mut key = [0u8; 18];
    key[0] = VERSION;
    key[1] = FLAG_CANON;
    key[2..18].copy_from_slice(&canon.to_le_bytes());
    key
}

fn key_to_bb(key: &[u8]) -> Bitboard {
    let mut buf = [0u8; 16];
    buf.copy_from_slice(&key[2..18]);
    Bitboard::from_le_bytes(&buf)
}

fn move_to_string(m: &Move) -> String {
    format!("P{}S{}P{}", m.player, m.shape, m.position)
}

fn determine_winner(bb: &Bitboard) -> Option<u8> {
    if has_winning_line(bb) {
        let p0 = bb.player_piece_count(0);
        let p1 = bb.player_piece_count(1);
        Some(if p0 > p1 { 0 } else { 1 })
    } else {
        let moves = generate_legal_moves(bb);
        if moves.is_empty() {
            current_player(bb)
                .map(|loser| 1 - loser)
        } else {
            None
        }
    }
}

fn is_terminal(bb: &Bitboard) -> bool {
    has_winning_line(bb) || generate_legal_moves(bb).is_empty()
}

// ── Database ─────────────────────────────────────────────────────────

struct BfsDb {
    conn: Connection,
}

impl BfsDb {
    fn open(path: &str) -> Self {
        let conn = Connection::open(path).expect("Failed to open database");
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA cache_size = -102400;",
        )
        .expect("Failed to set pragmas");

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS positions (
                canonical_key BLOB PRIMARY KEY,
                parent_key BLOB,
                parent_move TEXT,
                depth INTEGER NOT NULL,
                is_terminal INTEGER NOT NULL DEFAULT 0,
                winner INTEGER,
                symmetry_count INTEGER NOT NULL,
                status INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_depth ON positions(depth);
            CREATE INDEX IF NOT EXISTS idx_status ON positions(status);",
        )
        .expect("Failed to create schema");

        Self { conn }
    }

    fn insert_position(
        &self,
        key: &[u8; 18],
        parent_key: Option<&[u8; 18]>,
        parent_move: Option<&str>,
        depth: usize,
        terminal: bool,
        winner: Option<u8>,
        symmetry_count: usize,
        status: i32,
    ) {
        self.conn
            .execute(
                "INSERT OR IGNORE INTO positions
                 (canonical_key, parent_key, parent_move, depth, is_terminal, winner, symmetry_count, status)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    key.as_slice(),
                    parent_key.map(|k| k.as_slice()),
                    parent_move,
                    depth as i64,
                    terminal as i32,
                    winner.map(|w| w as i32),
                    symmetry_count as i64,
                    status,
                ],
            )
            .expect("Failed to insert position");
    }

    fn mark_expanded(&self, key: &[u8; 18]) {
        self.conn
            .execute(
                "UPDATE positions SET status = 1 WHERE canonical_key = ?1",
                params![key.as_slice()],
            )
            .expect("Failed to mark expanded");
    }

    fn mark_dropped_frontier(&self) {
        self.conn
            .execute("UPDATE positions SET status = 2 WHERE status = 0", [])
            .expect("Failed to mark dropped");
    }

    fn restore_dropped_to_frontier(&self) {
        self.conn
            .execute("UPDATE positions SET status = 0 WHERE status = 2", [])
            .expect("Failed to restore dropped");
    }

    fn load_seen(&self) -> HashSet<[u8; 18]> {
        let mut stmt = self
            .conn
            .prepare("SELECT canonical_key FROM positions")
            .expect("Failed to prepare seen query");
        let rows = stmt
            .query_map([], |row| {
                let blob: Vec<u8> = row.get(0)?;
                let mut key = [0u8; 18];
                key.copy_from_slice(&blob);
                Ok(key)
            })
            .expect("Failed to query seen");
        rows.filter_map(|r| r.ok()).collect()
    }

    fn load_frontier(&self) -> Vec<(Bitboard, [u8; 18], usize)> {
        let mut stmt = self
            .conn
            .prepare("SELECT canonical_key, depth FROM positions WHERE status IN (0, 2) ORDER BY depth")
            .expect("Failed to prepare frontier query");
        let rows = stmt
            .query_map([], |row| {
                let blob: Vec<u8> = row.get(0)?;
                let depth: i64 = row.get(1)?;
                let mut key = [0u8; 18];
                key.copy_from_slice(&blob);
                let bb = key_to_bb(&key);
                Ok((bb, key, depth as usize))
            })
            .expect("Failed to query frontier");
        rows.filter_map(|r| r.ok()).collect()
    }

    fn total_count(&self) -> usize {
        self.conn
            .query_row("SELECT COUNT(*) FROM positions", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap_or(0) as usize
    }

    fn begin_transaction(&self) {
        self.conn
            .execute_batch("BEGIN TRANSACTION")
            .expect("Failed to begin transaction");
    }

    fn commit_transaction(&self) {
        self.conn
            .execute_batch("COMMIT")
            .expect("Failed to commit transaction");
    }

    fn summary_by_depth(&self) -> Vec<(i64, i64, i64, i64)> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT depth, COUNT(*), SUM(is_terminal), SUM(symmetry_count)
                 FROM positions GROUP BY depth ORDER BY depth",
            )
            .expect("Failed to prepare summary");
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            })
            .expect("Failed to query summary");
        rows.filter_map(|r| r.ok()).collect()
    }
}

// ── BFS ──────────────────────────────────────────────────────────────

fn run_bfs(cli: &Cli) {
    let db = BfsDb::open(&cli.db);

    let (start_depth, mut seen, mut frontier) = if cli.resume {
        db.restore_dropped_to_frontier();
        let seen = db.load_seen();
        let raw_frontier = db.load_frontier();
        if seen.is_empty() {
            eprintln!("[resume] No existing data, starting fresh.");
            init_fresh(&db)
        } else {
            let min_depth = raw_frontier.iter().map(|f| f.2).min().unwrap_or(0);
            let frontier: Vec<(Bitboard, [u8; 18])> =
                raw_frontier.into_iter().map(|(bb, k, _)| (bb, k)).collect();
            eprintln!(
                "[resume] Loaded {} seen, {} frontier (depth {})",
                seen.len(),
                frontier.len(),
                min_depth
            );
            (min_depth, seen, frontier)
        }
    } else {
        init_fresh(&db)
    };

    if !cli.quiet {
        println!(
            "Rust BFS benchmark (max depth {}, starting from depth {})",
            cli.depth, start_depth
        );
        println!(
            "{:>5} {:>12} {:>12} {:>10} {:>12}",
            "Depth", "Positions", "Frontier", "Time (s)", "Pos/sec"
        );
    }

    let t_start = Instant::now();
    let mut total_positions = seen.len();

    for depth in start_depth..=cli.depth {
        if frontier.is_empty() {
            if !cli.quiet {
                println!("  No more frontier nodes at depth {}. Done.", depth);
            }
            break;
        }

        let t_depth = Instant::now();
        let mut next_frontier: Vec<(Bitboard, [u8; 18])> = Vec::new();
        let mut processed = 0usize;
        let mut batch_count = 0usize;

        db.begin_transaction();

        for &(bb, ref parent_key) in &frontier {
            if is_terminal(&bb) {
                db.mark_expanded(parent_key);
                continue;
            }

            let moves = generate_legal_moves(&bb);
            for m in &moves {
                let child_bb = apply_move(&bb, m);
                let child_key = canonical_key(&child_bb);

                if seen.insert(child_key) {
                    let terminal = is_terminal(&child_bb);
                    let winner = determine_winner(&child_bb);
                    let sym_count = SymmetryHandler::orbit_size(&child_bb);
                    let move_str = move_to_string(m);

                    db.insert_position(
                        &child_key,
                        Some(parent_key),
                        Some(&move_str),
                        depth,
                        terminal,
                        winner,
                        sym_count,
                        0,
                    );

                    if !terminal {
                        next_frontier.push((child_bb, child_key));
                    } else {
                        db.mark_expanded(&child_key);
                    }
                    total_positions += 1;
                    batch_count += 1;

                    if batch_count >= 50_000 {
                        db.commit_transaction();
                        db.begin_transaction();
                        batch_count = 0;
                    }

                    if let Some(max) = cli.max_positions {
                        if total_positions >= max {
                            db.commit_transaction();
                            db.mark_dropped_frontier();
                            if !cli.quiet {
                                println!(
                                    "  Dropout: reached {} positions (limit {})",
                                    total_positions, max
                                );
                            }
                            print_summary(&db, &t_start, cli.quiet);
                            return;
                        }
                    }
                }
            }

            db.mark_expanded(parent_key);
            processed += 1;

            if !cli.quiet && processed % PROGRESS_INTERVAL == 0 {
                let elapsed = t_depth.elapsed().as_secs_f64();
                let rate = if elapsed > 0.0 {
                    total_positions as f64 / elapsed
                } else {
                    0.0
                };
                eprintln!(
                    "  [depth {}] {}/{} expanded, {} total, {} frontier, {:.0} pos/sec, ~{} MB",
                    depth,
                    processed,
                    frontier.len(),
                    total_positions,
                    next_frontier.len(),
                    rate,
                    estimate_memory_mb(&seen, &next_frontier)
                );
            }
        }

        db.commit_transaction();

        let elapsed_depth = t_depth.elapsed().as_secs_f64();
        let rate = if elapsed_depth > 0.0 {
            total_positions as f64 / t_start.elapsed().as_secs_f64()
        } else {
            0.0
        };

        if !cli.quiet {
            println!(
                "{:>5} {:>12} {:>12} {:>10.3} {:>12.0}",
                depth,
                total_positions,
                next_frontier.len(),
                elapsed_depth,
                rate
            );
        }

        frontier = next_frontier;
    }

    print_summary(&db, &t_start, cli.quiet);
}

// ── DFS ──────────────────────────────────────────────────────────────

fn run_dfs(cli: &Cli) {
    let db = BfsDb::open(&cli.db);

    let (mut seen, stack) = if cli.resume {
        db.restore_dropped_to_frontier();
        let seen = db.load_seen();
        let raw_frontier = db.load_frontier();
        if seen.is_empty() {
            eprintln!("[resume] No existing data, starting fresh.");
            let (_, seen, frontier) = init_fresh(&db);
            let stack: Vec<(Bitboard, [u8; 18], usize)> =
                frontier.into_iter().map(|(bb, k)| (bb, k, 0)).collect();
            (seen, stack)
        } else {
            let stack: Vec<(Bitboard, [u8; 18], usize)> = raw_frontier;
            eprintln!(
                "[resume] Loaded {} seen, {} frontier",
                seen.len(),
                stack.len()
            );
            (seen, stack)
        }
    } else {
        let (_, seen, frontier) = init_fresh(&db);
        let stack: Vec<(Bitboard, [u8; 18], usize)> =
            frontier.into_iter().map(|(bb, k)| (bb, k, 0)).collect();
        (seen, stack)
    };

    if !cli.quiet {
        println!(
            "Rust DFS benchmark (max depth {}, {} starting nodes)",
            cli.depth,
            stack.len()
        );
    }

    let t_start = Instant::now();
    let mut total_positions = seen.len();
    let mut stack = stack;
    let mut processed = 0usize;
    let mut batch_count = 0usize;

    db.begin_transaction();

    while let Some((bb, parent_key, depth)) = stack.pop() {
        if depth >= cli.depth || is_terminal(&bb) {
            db.mark_expanded(&parent_key);
            continue;
        }

        let moves = generate_legal_moves(&bb);
        for m in &moves {
            let child_bb = apply_move(&bb, m);
            let child_key = canonical_key(&child_bb);

            if seen.insert(child_key) {
                let terminal = is_terminal(&child_bb);
                let winner = determine_winner(&child_bb);
                let sym_count = SymmetryHandler::orbit_size(&child_bb);
                let move_str = move_to_string(m);

                db.insert_position(
                    &child_key,
                    Some(&parent_key),
                    Some(&move_str),
                    depth + 1,
                    terminal,
                    winner,
                    sym_count,
                    0,
                );

                if !terminal && depth + 1 < cli.depth {
                    stack.push((child_bb, child_key, depth + 1));
                } else {
                    db.mark_expanded(&child_key);
                }

                total_positions += 1;
                batch_count += 1;

                if batch_count >= 50_000 {
                    db.commit_transaction();
                    db.begin_transaction();
                    batch_count = 0;
                }

                if let Some(max) = cli.max_positions {
                    if total_positions >= max {
                        db.commit_transaction();
                        db.mark_dropped_frontier();
                        if !cli.quiet {
                            println!(
                                "  Dropout: reached {} positions (limit {})",
                                total_positions, max
                            );
                        }
                        print_summary(&db, &t_start, cli.quiet);
                        return;
                    }
                }
            }
        }

        db.mark_expanded(&parent_key);
        processed += 1;

        if !cli.quiet && processed % PROGRESS_INTERVAL == 0 {
            let elapsed = t_start.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 {
                total_positions as f64 / elapsed
            } else {
                0.0
            };
            eprintln!(
                "  [dfs] {} expanded, {} total, {} stack, {:.0} pos/sec",
                processed, total_positions, stack.len(), rate
            );
        }
    }

    db.commit_transaction();
    print_summary(&db, &t_start, cli.quiet);
}

// ── Helpers ──────────────────────────────────────────────────────────

fn init_fresh(db: &BfsDb) -> (usize, HashSet<[u8; 18]>, Vec<(Bitboard, [u8; 18])>) {
    let root = Bitboard::EMPTY;
    let root_key = canonical_key(&root);
    let sym_count = SymmetryHandler::orbit_size(&root);

    db.insert_position(&root_key, None, None, 0, false, None, sym_count, 0);

    let mut seen = HashSet::new();
    seen.insert(root_key);
    (1, seen, vec![(root, root_key)])
}

fn estimate_memory_mb(seen: &HashSet<[u8; 18]>, frontier: &[(Bitboard, [u8; 18])]) -> usize {
    let seen_bytes = seen.len() * (18 + 8); // key + hash overhead
    let frontier_bytes = frontier.len() * (16 + 18 + 8); // Bitboard + key + Vec overhead
    (seen_bytes + frontier_bytes) / (1024 * 1024)
}

fn print_summary(db: &BfsDb, t_start: &Instant, quiet: bool) {
    let total = t_start.elapsed().as_secs_f64();
    let count = db.total_count();

    println!("\n--- Summary ---");
    println!("Total positions: {}", count);
    println!("Elapsed: {:.3}s", total);
    if total > 0.0 {
        println!("Throughput: {:.0} pos/sec", count as f64 / total);
    }

    if !quiet {
        let by_depth = db.summary_by_depth();
        if !by_depth.is_empty() {
            println!(
                "\n{:>5} {:>12} {:>10} {:>14}",
                "Depth", "Positions", "Terminal", "SymmetrySum"
            );
            for (d, cnt, term, sym) in &by_depth {
                println!("{:>5} {:>12} {:>10} {:>14}", d, cnt, term, sym);
            }
        }
    }
}

fn main() {
    let cli = Cli::parse();

    if cli.dfs {
        run_dfs(&cli);
    } else {
        run_bfs(&cli);
    }
}
