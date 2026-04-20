use clap::Parser;
use quantik_core::bitboard::Bitboard;
use quantik_core::constants::{FLAG_CANON, VERSION, WIN_MASKS};
use quantik_core::game::{current_player, has_winning_line};
use quantik_core::mcts::{MCTSConfig, MCTSEngine};
use quantik_core::moves::{apply_move, generate_legal_moves, Move};
use quantik_core::symmetry::SymmetryHandler;
use rusqlite::{params, Connection};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "book_builder",
    about = "Hybrid opening book builder: exhaustive BFS + selective DFS + MCTS deepening"
)]
struct Cli {
    /// SQLite database path
    #[arg(long, default_value = "quantik_book.db")]
    db: String,

    /// Depth for exhaustive BFS phase
    #[arg(long, default_value_t = 6)]
    exhaustive_depth: usize,

    /// Max depth for selective DFS phase
    #[arg(long, default_value_t = 10)]
    selective_depth: usize,

    /// Top K moves to expand in selective phase
    #[arg(long, default_value_t = 6)]
    top_k: usize,

    /// MCTS iterations for evaluation
    #[arg(long, default_value_t = 500)]
    mcts_iterations: u32,

    /// Stop after N total positions
    #[arg(long)]
    max_positions: Option<usize>,

    /// Resume from existing database
    #[arg(long)]
    resume: bool,

    /// Minimal output
    #[arg(long)]
    quiet: bool,
}

const STATUS_FRONTIER: i32 = 0;
const STATUS_EXPANDED: i32 = 1;
const STATUS_DROPPED: i32 = 2;
const STATUS_SOLVED: i32 = 3;

const BATCH_SIZE: usize = 10_000;
const PROGRESS_INTERVAL: usize = 10_000;

// ── Helpers ──────────────────────────────────────────────────────────

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

fn is_terminal(bb: &Bitboard) -> bool {
    has_winning_line(bb) || generate_legal_moves(bb).is_empty()
}

fn determine_winner(bb: &Bitboard) -> Option<u8> {
    if has_winning_line(bb) {
        let p0 = bb.player_piece_count(0);
        let p1 = bb.player_piece_count(1);
        Some(if p0 > p1 { 0 } else { 1 })
    } else if generate_legal_moves(bb).is_empty() {
        current_player(bb).map(|loser| 1 - loser)
    } else {
        None
    }
}

fn terminal_score(bb: &Bitboard) -> f64 {
    if has_winning_line(bb) {
        let p0 = bb.player_piece_count(0);
        let p1 = bb.player_piece_count(1);
        if p0 > p1 { 1.0 } else { -1.0 }
    } else {
        match current_player(bb) {
            Some(0) => -1.0,
            Some(1) => 1.0,
            _ => 0.0,
        }
    }
}

fn quick_evaluate(bb: &Bitboard) -> f64 {
    let mut score: f64 = 0.0;
    for &mask in WIN_MASKS.iter() {
        let mut p0_shapes = 0;
        let mut p1_shapes = 0;
        for s in 0..4 {
            if bb.planes[s] & mask != 0 {
                p0_shapes += 1;
            }
            if bb.planes[s + 4] & mask != 0 {
                p1_shapes += 1;
            }
        }
        if p1_shapes == 0 {
            score += match p0_shapes {
                3 => 0.3,
                2 => 0.1,
                _ => 0.0,
            };
        }
        if p0_shapes == 0 {
            score -= match p1_shapes {
                3 => 0.3,
                2 => 0.1,
                _ => 0.0,
            };
        }
    }
    score.clamp(-1.0, 1.0)
}

/// Rank moves by heuristic for the selective phase.
/// Returns moves sorted best-first (descending score).
fn rank_moves(bb: &Bitboard, moves: &[Move]) -> Vec<(Move, i32)> {
    let mut scored: Vec<(Move, i32)> = moves
        .iter()
        .map(|m| {
            let mut heuristic = 0i32;
            let child = apply_move(bb, m);

            if has_winning_line(&child) {
                heuristic += 1000;
                return (*m, heuristic);
            }

            // Check if this move blocks an opponent win:
            // simulate opponent moves on original board (without this move)
            // to see if any would win. This is expensive so we approximate:
            // if the opponent has 3-of-4 shapes on any line containing this position,
            // placing here blocks completion.
            let opponent = 1 - m.player;
            let pos_mask = 1u16 << m.position;
            for &mask in WIN_MASKS.iter() {
                if pos_mask & mask == 0 {
                    continue;
                }
                let mut opp_shapes_on_line = 0;
                for s in 0..4u8 {
                    let opp_plane = bb.planes[(opponent as usize) * 4 + s as usize];
                    if opp_plane & mask != 0 {
                        opp_shapes_on_line += 1;
                    }
                }
                if opp_shapes_on_line >= 3 {
                    heuristic += 500;
                }
            }

            // Central positions bonus
            match m.position {
                5 | 6 | 9 | 10 => heuristic += 10,
                0 | 3 | 12 | 15 => heuristic += 5,
                _ => {}
            }

            // Moves introducing a new shape to a win line
            let player_base = (m.player as usize) * 4;
            let shape_plane_before = bb.planes[player_base + m.shape as usize];
            for &mask in WIN_MASKS.iter() {
                if pos_mask & mask == 0 {
                    continue;
                }
                if shape_plane_before & mask == 0 {
                    heuristic += 20;
                }
            }

            (*m, heuristic)
        })
        .collect();

    scored.sort_by(|a, b| b.1.cmp(&a.1));
    scored
}

fn estimate_memory_mb(seen: &HashSet<[u8; 18]>, frontier: &[(Bitboard, [u8; 18])]) -> usize {
    let seen_bytes = seen.len() * (18 + 8);
    let frontier_bytes = frontier.len() * (16 + 18 + 8);
    (seen_bytes + frontier_bytes) / (1024 * 1024)
}

// ── Database ─────────────────────────────────────────────────────────

struct BookDb {
    conn: Connection,
}

impl BookDb {
    fn open(path: &str) -> Self {
        let conn = Connection::open(path).expect("Failed to open database");
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA cache_size = -204800;",
        )
        .expect("Failed to set pragmas");

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS positions (
                canonical_key BLOB PRIMARY KEY,
                depth INTEGER NOT NULL,
                is_terminal INTEGER NOT NULL DEFAULT 0,
                winner INTEGER,
                symmetry_count INTEGER NOT NULL,
                searched_depth INTEGER NOT NULL DEFAULT 0,
                score REAL,
                visits INTEGER NOT NULL DEFAULT 0,
                best_move TEXT,
                status INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS edges (
                parent_key BLOB NOT NULL,
                child_key BLOB NOT NULL,
                move TEXT NOT NULL,
                PRIMARY KEY (parent_key, child_key, move)
            );

            CREATE TABLE IF NOT EXISTS book (
                canonical_key BLOB PRIMARY KEY,
                moves_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_pos_depth ON positions(depth);
            CREATE INDEX IF NOT EXISTS idx_pos_status ON positions(status);
            CREATE INDEX IF NOT EXISTS idx_pos_score ON positions(score);
            CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges(parent_key);
            CREATE INDEX IF NOT EXISTS idx_edges_child ON edges(child_key);",
        )
        .expect("Failed to create schema");

        Self { conn }
    }

    fn begin(&self) {
        self.conn
            .execute_batch("BEGIN TRANSACTION")
            .expect("begin");
    }

    fn commit(&self) {
        self.conn.execute_batch("COMMIT").expect("commit");
    }

    fn insert_position(
        &self,
        key: &[u8; 18],
        depth: usize,
        terminal: bool,
        winner: Option<u8>,
        sym_count: usize,
        score: Option<f64>,
        status: i32,
    ) {
        self.conn
            .execute(
                "INSERT OR IGNORE INTO positions
                 (canonical_key, depth, is_terminal, winner, symmetry_count, score, status)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    key.as_slice(),
                    depth as i64,
                    terminal as i32,
                    winner.map(|w| w as i32),
                    sym_count as i64,
                    score,
                    status,
                ],
            )
            .expect("insert position");
    }

    fn insert_edge(&self, parent: &[u8; 18], child: &[u8; 18], move_str: &str) {
        self.conn
            .execute(
                "INSERT OR IGNORE INTO edges (parent_key, child_key, move)
                 VALUES (?1, ?2, ?3)",
                params![parent.as_slice(), child.as_slice(), move_str],
            )
            .expect("insert edge");
    }

    fn update_status(&self, key: &[u8; 18], status: i32) {
        self.conn
            .execute(
                "UPDATE positions SET status = ?2 WHERE canonical_key = ?1",
                params![key.as_slice(), status],
            )
            .expect("update status");
    }

    fn update_score(
        &self,
        key: &[u8; 18],
        score: f64,
        searched_depth: i32,
        best_move: Option<&str>,
        status: i32,
    ) {
        self.conn
            .execute(
                "UPDATE positions SET score = ?2, searched_depth = ?3, best_move = ?4, status = ?5
                 WHERE canonical_key = ?1",
                params![key.as_slice(), score, searched_depth, best_move, status],
            )
            .expect("update score");
    }

    fn update_mcts(&self, key: &[u8; 18], visits: i64, score: f64) {
        self.conn
            .execute(
                "UPDATE positions SET visits = visits + ?2, score = ?3 WHERE canonical_key = ?1",
                params![key.as_slice(), visits, score],
            )
            .expect("update mcts");
    }

    fn load_seen(&self) -> HashSet<[u8; 18]> {
        let mut stmt = self
            .conn
            .prepare("SELECT canonical_key FROM positions")
            .expect("prepare");
        let rows = stmt
            .query_map([], |row| {
                let blob: Vec<u8> = row.get(0)?;
                let mut key = [0u8; 18];
                key.copy_from_slice(&blob);
                Ok(key)
            })
            .expect("query");
        rows.filter_map(|r| r.ok()).collect()
    }

    fn load_frontier_for_bfs(&self) -> Vec<(Bitboard, [u8; 18], usize)> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT canonical_key, depth FROM positions
                 WHERE status = 0 AND is_terminal = 0 ORDER BY depth",
            )
            .expect("prepare");
        let rows = stmt
            .query_map([], |row| {
                let blob: Vec<u8> = row.get(0)?;
                let depth: i64 = row.get(1)?;
                let mut key = [0u8; 18];
                key.copy_from_slice(&blob);
                let bb = key_to_bb(&key);
                Ok((bb, key, depth as usize))
            })
            .expect("query");
        rows.filter_map(|r| r.ok()).collect()
    }

    fn load_selective_frontier(&self, exhaustive_depth: usize) -> Vec<([u8; 18], usize)> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT canonical_key, depth FROM positions
                 WHERE status IN (0, 1) AND is_terminal = 0 AND depth = ?1
                 ORDER BY depth",
            )
            .expect("prepare");
        let rows = stmt
            .query_map(params![exhaustive_depth as i64], |row| {
                let blob: Vec<u8> = row.get(0)?;
                let depth: i64 = row.get(1)?;
                let mut key = [0u8; 18];
                key.copy_from_slice(&blob);
                Ok((key, depth as usize))
            })
            .expect("query");
        rows.filter_map(|r| r.ok()).collect()
    }

    fn load_mcts_frontier(&self, limit: usize) -> Vec<([u8; 18], usize)> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT canonical_key, depth FROM positions
                 WHERE is_terminal = 0 AND status != ?1
                   AND (score IS NULL OR (score > -0.8 AND score < 0.8))
                 ORDER BY visits ASC, depth ASC LIMIT ?2",
            )
            .expect("prepare");
        let rows = stmt
            .query_map(params![STATUS_DROPPED, limit as i64], |row| {
                let blob: Vec<u8> = row.get(0)?;
                let depth: i64 = row.get(1)?;
                let mut key = [0u8; 18];
                key.copy_from_slice(&blob);
                Ok((key, depth as usize))
            })
            .expect("query");
        rows.filter_map(|r| r.ok()).collect()
    }

    fn total_count(&self) -> usize {
        self.conn
            .query_row("SELECT COUNT(*) FROM positions", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap_or(0) as usize
    }

    fn edge_count(&self) -> usize {
        self.conn
            .query_row("SELECT COUNT(*) FROM edges", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap_or(0) as usize
    }

    fn insert_book_entry(&self, key: &[u8; 18], moves_json: &str) {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO book (canonical_key, moves_json) VALUES (?1, ?2)",
                params![key.as_slice(), moves_json],
            )
            .expect("insert book entry");
    }

    fn load_exportable_positions(&self) -> Vec<[u8; 18]> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT canonical_key FROM positions
                 WHERE is_terminal = 0 AND searched_depth > 0",
            )
            .expect("prepare");
        let rows = stmt
            .query_map([], |row| {
                let blob: Vec<u8> = row.get(0)?;
                let mut key = [0u8; 18];
                key.copy_from_slice(&blob);
                Ok(key)
            })
            .expect("query");
        rows.filter_map(|r| r.ok()).collect()
    }

    fn load_child_moves_for_book(&self, parent_key: &[u8; 18]) -> Vec<BookMove> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT e.move, p.score, p.searched_depth
                 FROM edges e
                 JOIN positions p ON e.child_key = p.canonical_key
                 WHERE e.parent_key = ?1
                 ORDER BY p.score ASC",
            )
            .expect("prepare");

        stmt.query_map(params![parent_key.as_slice()], |row| {
            let move_str: String = row.get(0)?;
            let score: Option<f64> = row.get(1)?;
            let depth: i64 = row.get(2)?;
            Ok(BookMove {
                r#move: move_str,
                score: -(score.unwrap_or(0.0)),
                depth: depth as usize,
            })
        })
        .expect("query")
        .filter_map(|r| r.ok())
        .collect()
    }

    fn summary_by_depth(&self) -> Vec<(i64, i64, i64, i64)> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT depth, COUNT(*), SUM(is_terminal), SUM(symmetry_count)
                 FROM positions GROUP BY depth ORDER BY depth",
            )
            .expect("prepare");
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            })
            .expect("query");
        rows.filter_map(|r| r.ok()).collect()
    }

    fn book_count(&self) -> usize {
        self.conn
            .query_row("SELECT COUNT(*) FROM book", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap_or(0) as usize
    }
}

// ── Phase 1: Exhaustive BFS ─────────────────────────────────────────

fn run_exhaustive_bfs(db: &BookDb, cli: &Cli) -> (HashSet<[u8; 18]>, usize) {
    let (start_depth, mut seen, mut frontier) = if cli.resume {
        let seen = db.load_seen();
        let raw_frontier = db.load_frontier_for_bfs();
        if seen.is_empty() {
            init_fresh(db)
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
        init_fresh(db)
    };

    if !cli.quiet {
        eprintln!(
            "Phase 1: Exhaustive BFS (depth {}-{})",
            start_depth, cli.exhaustive_depth
        );
        eprintln!(
            "{:>5} {:>12} {:>12} {:>10} {:>12}",
            "Depth", "Positions", "Frontier", "Time (s)", "Pos/sec"
        );
    }

    let t_start = Instant::now();
    let mut total_positions = seen.len();

    for depth in start_depth..=cli.exhaustive_depth {
        if frontier.is_empty() {
            if !cli.quiet {
                eprintln!("  No more frontier nodes at depth {}. Done.", depth);
            }
            break;
        }

        let t_depth = Instant::now();
        let mut next_frontier: Vec<(Bitboard, [u8; 18])> = Vec::new();
        let mut processed = 0usize;
        let mut batch_count = 0usize;

        db.begin();

        for &(bb, ref parent_key) in &frontier {
            if is_terminal(&bb) {
                db.update_status(parent_key, STATUS_EXPANDED);
                continue;
            }

            let moves = generate_legal_moves(&bb);
            for m in &moves {
                let child_bb = apply_move(&bb, m);
                let child_key = canonical_key(&child_bb);
                let move_str = move_to_string(m);

                db.insert_edge(parent_key, &child_key, &move_str);

                if seen.insert(child_key) {
                    let terminal = is_terminal(&child_bb);
                    let winner = determine_winner(&child_bb);
                    let sym_count = SymmetryHandler::orbit_size(&child_bb);
                    let score = if terminal {
                        Some(terminal_score(&child_bb))
                    } else {
                        None
                    };
                    let status = if terminal {
                        STATUS_EXPANDED
                    } else {
                        STATUS_FRONTIER
                    };

                    db.insert_position(&child_key, depth, terminal, winner, sym_count, score, status);

                    if !terminal {
                        next_frontier.push((child_bb, child_key));
                    }
                    total_positions += 1;
                    batch_count += 1;

                    if batch_count >= BATCH_SIZE {
                        db.commit();
                        db.begin();
                        batch_count = 0;
                    }

                    if let Some(max) = cli.max_positions {
                        if total_positions >= max {
                            db.commit();
                            if !cli.quiet {
                                eprintln!(
                                    "  Dropout: reached {} positions (limit {})",
                                    total_positions, max
                                );
                            }
                            return (seen, total_positions);
                        }
                    }
                }
            }

            db.update_status(parent_key, STATUS_EXPANDED);
            processed += 1;

            if !cli.quiet && processed % PROGRESS_INTERVAL == 0 {
                let elapsed = t_depth.elapsed().as_secs_f64();
                let rate = if elapsed > 0.0 {
                    total_positions as f64 / elapsed
                } else {
                    0.0
                };
                eprint!(
                    "\r  [depth {}] {}/{} expanded, {} total, {} frontier, {:.0} pos/sec, ~{} MB",
                    depth,
                    processed,
                    frontier.len(),
                    total_positions,
                    next_frontier.len(),
                    rate,
                    estimate_memory_mb(&seen, &next_frontier)
                );
                let _ = std::io::stderr().flush();
            }
        }

        db.commit();

        let elapsed_depth = t_depth.elapsed().as_secs_f64();
        let rate = if elapsed_depth > 0.0 {
            total_positions as f64 / t_start.elapsed().as_secs_f64()
        } else {
            0.0
        };

        if !cli.quiet {
            eprintln!(
                "\r{:>5} {:>12} {:>12} {:>10.3} {:>12.0}",
                depth, total_positions, next_frontier.len(), elapsed_depth, rate
            );
        }

        frontier = next_frontier;
    }

    if !cli.quiet {
        eprintln!(
            "Phase 1 complete: {} positions, {} edges, {:.1}s",
            total_positions,
            db.edge_count(),
            t_start.elapsed().as_secs_f64()
        );
    }

    (seen, total_positions)
}

fn init_fresh(db: &BookDb) -> (usize, HashSet<[u8; 18]>, Vec<(Bitboard, [u8; 18])>) {
    let root = Bitboard::EMPTY;
    let root_key = canonical_key(&root);
    let sym_count = SymmetryHandler::orbit_size(&root);

    db.insert_position(&root_key, 0, false, None, sym_count, None, STATUS_FRONTIER);

    let mut seen = HashSet::new();
    seen.insert(root_key);
    (0, seen, vec![(root, root_key)])
}

// ── Phase 2: Selective Iterative-Deepening DFS ──────────────────────

struct TTEntry {
    score: f64,
    searched_depth: i32,
    best_move: Option<String>,
}

fn run_selective_ids(db: &BookDb, cli: &Cli) {
    if cli.selective_depth <= cli.exhaustive_depth {
        if !cli.quiet {
            eprintln!("Phase 2: Skipped (selective-depth <= exhaustive-depth)");
        }
        return;
    }

    let frontier = db.load_selective_frontier(cli.exhaustive_depth);
    if frontier.is_empty() {
        if !cli.quiet {
            eprintln!("Phase 2: No frontier positions to deepen.");
        }
        return;
    }

    if !cli.quiet {
        eprintln!(
            "Phase 2: Selective IDS (depth {}-{}, top-k={}, {} frontier positions)",
            cli.exhaustive_depth + 1,
            cli.selective_depth,
            cli.top_k,
            frontier.len()
        );
    }

    let t_start = Instant::now();
    let mut tt: HashMap<[u8; 18], TTEntry> = HashMap::new();
    let mut nodes_searched: u64 = 0;

    for depth_limit in (cli.exhaustive_depth + 1)..=cli.selective_depth {
        if !cli.quiet {
            eprint!("  IDS depth {}...", depth_limit);
            let _ = std::io::stderr().flush();
        }

        let t_iter = Instant::now();
        let mut iter_nodes: u64 = 0;

        for (key, depth_first_seen) in &frontier {
            let bb = key_to_bb(key);
            let remaining = depth_limit as i32 - *depth_first_seen as i32;
            if remaining <= 0 {
                continue;
            }
            negamax_dfs(
                &bb,
                remaining,
                &mut tt,
                &mut iter_nodes,
                cli.top_k,
                cli.exhaustive_depth,
            );
        }

        nodes_searched += iter_nodes;

        if !cli.quiet {
            eprintln!(
                " {} nodes, {:.1}s, TT size: {}",
                iter_nodes,
                t_iter.elapsed().as_secs_f64(),
                tt.len()
            );
        }
    }

    if !cli.quiet {
        eprintln!("  Flushing {} TT entries to database...", tt.len());
    }

    let max_remaining = cli.selective_depth as i32 - cli.exhaustive_depth as i32;
    db.begin();
    let mut batch = 0usize;
    for (key, entry) in &tt {
        let status = if entry.searched_depth >= max_remaining {
            STATUS_SOLVED
        } else {
            STATUS_EXPANDED
        };
        db.update_score(
            key,
            entry.score,
            entry.searched_depth,
            entry.best_move.as_deref(),
            status,
        );
        batch += 1;
        if batch >= BATCH_SIZE {
            db.commit();
            db.begin();
            batch = 0;
        }
    }
    db.commit();

    if !cli.quiet {
        eprintln!(
            "Phase 2 complete: {} total nodes searched, {:.1}s",
            nodes_searched,
            t_start.elapsed().as_secs_f64()
        );
    }
}

/// Negamax DFS with transposition table.
/// Score is from the perspective of the side to move: positive = good for mover.
fn negamax_dfs(
    bb: &Bitboard,
    remaining: i32,
    tt: &mut HashMap<[u8; 18], TTEntry>,
    nodes: &mut u64,
    top_k: usize,
    exhaustive_depth: usize,
) -> f64 {
    *nodes += 1;
    let key = canonical_key(bb);

    if let Some(entry) = tt.get(&key) {
        if entry.searched_depth >= remaining {
            return entry.score;
        }
    }

    if has_winning_line(bb) {
        // The last player to move completed a line — they win.
        // In negamax, the current side to move just lost.
        let score = -1.0;
        tt.insert(
            key,
            TTEntry {
                score,
                searched_depth: remaining,
                best_move: None,
            },
        );
        return score;
    }

    let moves = generate_legal_moves(bb);
    if moves.is_empty() {
        // Current player has no moves — they lose.
        let score = -1.0;
        tt.insert(
            key,
            TTEntry {
                score,
                searched_depth: remaining,
                best_move: None,
            },
        );
        return score;
    }

    if remaining <= 0 {
        let raw = quick_evaluate(bb);
        // Convert absolute score to relative: if P0 to move, positive raw = good
        let stm = current_player(bb).unwrap_or(0);
        let score = if stm == 0 { raw } else { -raw };
        tt.entry(key).or_insert(TTEntry {
            score,
            searched_depth: 0,
            best_move: None,
        });
        return score;
    }

    let ranked = rank_moves(bb, &moves);
    let depth = bb.player_piece_count(0) + bb.player_piece_count(1);
    let limit = if depth as usize > exhaustive_depth {
        top_k.min(ranked.len())
    } else {
        ranked.len()
    };
    let moves_to_expand = &ranked[..limit];

    let mut best_score = f64::NEG_INFINITY;
    let mut best_move: Option<String> = None;

    for (m, _heuristic) in moves_to_expand {
        let child = apply_move(bb, m);
        let child_score = -negamax_dfs(&child, remaining - 1, tt, nodes, top_k, exhaustive_depth);

        if child_score > best_score {
            best_score = child_score;
            best_move = Some(move_to_string(m));
        }
    }

    if best_score.is_infinite() {
        best_score = 0.0;
    }

    tt.insert(
        key,
        TTEntry {
            score: best_score,
            searched_depth: remaining,
            best_move,
        },
    );

    best_score
}

// ── Phase 3: MCTS Deepening ─────────────────────────────────────────

fn run_mcts_deepening(db: &BookDb, cli: &Cli) {
    if cli.mcts_iterations == 0 {
        if !cli.quiet {
            eprintln!("Phase 3: Skipped (mcts-iterations = 0)");
        }
        return;
    }

    let frontier = db.load_mcts_frontier(10_000);
    if frontier.is_empty() {
        if !cli.quiet {
            eprintln!("Phase 3: No uncertain positions for MCTS.");
        }
        return;
    }

    if !cli.quiet {
        eprintln!(
            "Phase 3: MCTS Deepening ({} iterations/position, {} positions)",
            cli.mcts_iterations,
            frontier.len()
        );
    }

    let t_start = Instant::now();
    let mut total_sims: u64 = 0;

    db.begin();
    let mut batch = 0usize;

    for (idx, (key, _depth)) in frontier.iter().enumerate() {
        let bb = key_to_bb(key);
        if is_terminal(&bb) {
            continue;
        }

        let mut engine = MCTSEngine::new(MCTSConfig {
            max_iterations: cli.mcts_iterations,
            ..Default::default()
        });

        let result = engine.search(&bb);
        let iterations = engine.iterations_performed() as u64;
        total_sims += iterations;

        if let Some((_best_mv, win_prob_p0)) = result {
            // Convert from P0 win probability [0,1] to score [-1,1]
            let score = win_prob_p0 * 2.0 - 1.0;
            db.update_mcts(key, iterations as i64, score);
        }

        batch += 1;
        if batch >= BATCH_SIZE {
            db.commit();
            db.begin();
            batch = 0;
        }

        if !cli.quiet && (idx + 1) % 100 == 0 {
            let elapsed = t_start.elapsed().as_secs_f64();
            eprint!(
                "\r  MCTS: {}/{} positions, {} sims, {:.0} sims/sec",
                idx + 1,
                frontier.len(),
                total_sims,
                total_sims as f64 / elapsed
            );
            let _ = std::io::stderr().flush();
        }
    }

    db.commit();

    if !cli.quiet {
        eprintln!(
            "\nPhase 3 complete: {} simulations, {:.1}s",
            total_sims,
            t_start.elapsed().as_secs_f64()
        );
    }
}

// ── Phase 4: Export Compact Book ─────────────────────────────────────

#[derive(Serialize)]
struct BookMove {
    r#move: String,
    score: f64,
    depth: usize,
}

fn export_compact_book(db: &BookDb, quiet: bool) {
    if !quiet {
        eprintln!("Phase 4: Exporting compact book...");
    }

    let keys = db.load_exportable_positions();
    if keys.is_empty() {
        if !quiet {
            eprintln!("  No exportable positions found.");
        }
        return;
    }

    db.begin();
    let mut batch = 0usize;
    let mut exported = 0usize;

    for key in &keys {
        let mut moves = db.load_child_moves_for_book(key);
        if moves.is_empty() {
            continue;
        }

        moves.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let json = serde_json::to_string(&moves).unwrap_or_default();
        db.insert_book_entry(key, &json);
        exported += 1;
        batch += 1;

        if batch >= BATCH_SIZE {
            db.commit();
            db.begin();
            batch = 0;
        }
    }

    db.commit();

    if !quiet {
        eprintln!("Phase 4 complete: {} book entries exported", exported);
    }
}

// ── Summary ──────────────────────────────────────────────────────────

fn print_summary(db: &BookDb, total_elapsed: f64, quiet: bool) {
    let count = db.total_count();
    let edges = db.edge_count();
    let book = db.book_count();

    println!("\n--- Book Builder Summary ---");
    println!("Total positions: {}", count);
    println!("Total edges:     {}", edges);
    println!("Book entries:    {}", book);
    println!("Elapsed:         {:.3}s", total_elapsed);
    if total_elapsed > 0.0 {
        println!("Throughput:      {:.0} pos/sec", count as f64 / total_elapsed);
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

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();

    let t_start = Instant::now();
    let db = BookDb::open(&cli.db);

    if !cli.quiet {
        eprintln!(
            "Book Builder: db={}, exhaustive_depth={}, selective_depth={}, top_k={}, mcts_iterations={}",
            cli.db, cli.exhaustive_depth, cli.selective_depth, cli.top_k, cli.mcts_iterations
        );
    }

    run_exhaustive_bfs(&db, &cli);
    run_selective_ids(&db, &cli);
    run_mcts_deepening(&db, &cli);
    export_compact_book(&db, cli.quiet);

    print_summary(&db, t_start.elapsed().as_secs_f64(), cli.quiet);
}
