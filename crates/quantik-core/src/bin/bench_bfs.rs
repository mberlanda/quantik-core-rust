use clap::Parser;
use quantik_core::bitboard::Bitboard;
use quantik_core::constants::{FLAG_CANON, VERSION};
use quantik_core::game::{current_player, has_winning_line};
use quantik_core::moves::{apply_move, generate_legal_moves, Move};
use quantik_core::symmetry::SymmetryHandler;
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "bench_bfs",
    about = "Quantik IDDFS opening-book builder with SQLite transposition table"
)]
struct Cli {
    /// Maximum depth to explore
    depth: usize,

    /// SQLite database path
    #[arg(long, default_value = "quantik_book.db")]
    db: String,

    /// Resume from existing database (uses searched_depth for transposition)
    #[arg(long)]
    resume: bool,

    /// Stop after N total positions (dropout)
    #[arg(long)]
    max_positions: Option<usize>,

    /// Depth for exhaustive IDDFS expansion (default: same as depth)
    #[arg(long)]
    exhaustive_depth: Option<usize>,

    /// SQLite transaction batch size
    #[arg(long, default_value = "50000")]
    batch_size: usize,

    /// Only print summary
    #[arg(long)]
    quiet: bool,
}

fn canonical_key(bb: &Bitboard) -> [u8; 18] {
    let canon = SymmetryHandler::find_canonical(bb);
    let mut key = [0u8; 18];
    key[0] = VERSION;
    key[1] = FLAG_CANON;
    key[2..18].copy_from_slice(&canon.to_le_bytes());
    key
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
            current_player(bb).map(|loser| 1 - loser)
        } else {
            None
        }
    }
}

fn is_terminal(bb: &Bitboard) -> bool {
    has_winning_line(bb) || generate_legal_moves(bb).is_empty()
}

// ── In-memory position cache ────────────────────────────────────────

struct PositionEntry {
    depth: usize,
    searched_depth: usize,
    is_terminal: bool,
    dirty: bool,
}

struct BookBuilder {
    conn: Connection,
    cache: HashMap<[u8; 18], PositionEntry>,
    edge_buffer: Vec<([u8; 18], [u8; 18], String)>,
    new_positions: Vec<([u8; 18], usize, bool, Option<u8>, usize)>,
    pending_searched_updates: Vec<([u8; 18], usize)>,
    pending_status_updates: Vec<[u8; 18]>,
    batch_size: usize,
    ops_since_commit: usize,
    total_edges_inserted: usize,
}

impl BookBuilder {
    fn open(path: &str, batch_size: usize) -> Self {
        let conn = Connection::open(path).expect("Failed to open database");
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA cache_size = -204800;
             PRAGMA temp_store = MEMORY;",
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
                status INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS edges (
                parent_key BLOB NOT NULL,
                child_key BLOB NOT NULL,
                move TEXT NOT NULL,
                PRIMARY KEY (parent_key, child_key)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_child ON edges(child_key);
            CREATE INDEX IF NOT EXISTS idx_pos_depth ON positions(depth);
            CREATE INDEX IF NOT EXISTS idx_pos_status ON positions(status);
            CREATE INDEX IF NOT EXISTS idx_pos_searched ON positions(searched_depth);",
        )
        .expect("Failed to create schema");

        Self {
            conn,
            cache: HashMap::new(),
            edge_buffer: Vec::new(),
            new_positions: Vec::new(),
            pending_searched_updates: Vec::new(),
            pending_status_updates: Vec::new(),
            batch_size,
            ops_since_commit: 0,
            total_edges_inserted: 0,
        }
    }

    fn load_cache_from_db(&mut self) {
        let mut stmt = self
            .conn
            .prepare("SELECT canonical_key, depth, searched_depth, is_terminal FROM positions")
            .expect("Failed to prepare cache load");
        let rows = stmt
            .query_map([], |row| {
                let blob: Vec<u8> = row.get(0)?;
                let depth: i64 = row.get(1)?;
                let searched_depth: i64 = row.get(2)?;
                let is_terminal: i32 = row.get(3)?;
                let mut key = [0u8; 18];
                key.copy_from_slice(&blob);
                Ok((key, depth as usize, searched_depth as usize, is_terminal != 0))
            })
            .expect("Failed to query cache");
        for r in rows.flatten() {
            self.cache.insert(
                r.0,
                PositionEntry {
                    depth: r.1,
                    searched_depth: r.2,
                    is_terminal: r.3,
                    dirty: false,
                },
            );
        }
    }

    fn total_edge_count(&self) -> usize {
        let db_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))
            .unwrap_or(0);
        db_count as usize
    }

    fn cache_position(
        &mut self,
        key: [u8; 18],
        depth: usize,
        terminal: bool,
        winner: Option<u8>,
        symmetry_count: usize,
    ) -> bool {
        if let Some(entry) = self.cache.get_mut(&key) {
            if depth < entry.depth {
                entry.depth = depth;
                entry.dirty = true;
            }
            return false;
        }
        self.cache.insert(
            key,
            PositionEntry {
                depth,
                searched_depth: 0,
                is_terminal: terminal,
                dirty: false,
            },
        );
        self.new_positions
            .push((key, depth, terminal, winner, symmetry_count));
        self.ops_since_commit += 1;
        true
    }

    fn cache_edge(&mut self, parent: [u8; 18], child: [u8; 18], move_str: String) {
        self.edge_buffer.push((parent, child, move_str));
        self.ops_since_commit += 1;
    }

    fn update_searched_depth(&mut self, key: &[u8; 18], new_searched: usize) {
        if let Some(entry) = self.cache.get_mut(key) {
            if new_searched > entry.searched_depth {
                entry.searched_depth = new_searched;
                self.pending_searched_updates.push((*key, new_searched));
                self.ops_since_commit += 1;
            }
        }
    }

    fn mark_expanded(&mut self, key: &[u8; 18]) {
        self.pending_status_updates.push(*key);
        self.ops_since_commit += 1;
    }

    fn should_flush(&self) -> bool {
        self.ops_since_commit >= self.batch_size
    }

    fn flush(&mut self) {
        self.conn
            .execute_batch("BEGIN TRANSACTION")
            .expect("Failed to begin transaction");

        {
            let mut stmt = self
                .conn
                .prepare_cached(
                    "INSERT OR IGNORE INTO positions
                     (canonical_key, depth, is_terminal, winner, symmetry_count, searched_depth, status)
                     VALUES (?1, ?2, ?3, ?4, ?5, 0, 0)",
                )
                .expect("Failed to prepare position insert");
            for &(ref key, depth, terminal, winner, sym) in &self.new_positions {
                stmt.execute(params![
                    key.as_slice(),
                    depth as i64,
                    terminal as i32,
                    winner.map(|w| w as i32),
                    sym as i64,
                ])
                .expect("Failed to insert position");
            }
        }

        {
            let mut stmt = self
                .conn
                .prepare_cached(
                    "INSERT OR IGNORE INTO edges (parent_key, child_key, move) VALUES (?1, ?2, ?3)",
                )
                .expect("Failed to prepare edge insert");
            for (parent, child, move_str) in &self.edge_buffer {
                stmt.execute(params![
                    parent.as_slice(),
                    child.as_slice(),
                    move_str.as_str(),
                ])
                .expect("Failed to insert edge");
            }
            self.total_edges_inserted += self.edge_buffer.len();
        }

        {
            let mut stmt = self
                .conn
                .prepare_cached(
                    "UPDATE positions SET searched_depth = MAX(searched_depth, ?2) WHERE canonical_key = ?1",
                )
                .expect("Failed to prepare searched_depth update");
            for (key, sd) in &self.pending_searched_updates {
                stmt.execute(params![key.as_slice(), *sd as i64])
                    .expect("Failed to update searched_depth");
            }
        }

        {
            let mut stmt = self
                .conn
                .prepare_cached(
                    "UPDATE positions SET status = 1 WHERE canonical_key = ?1 AND status != 1",
                )
                .expect("Failed to prepare status update");
            for key in &self.pending_status_updates {
                stmt.execute(params![key.as_slice()])
                    .expect("Failed to mark expanded");
            }
        }

        self.conn
            .execute_batch("COMMIT")
            .expect("Failed to commit transaction");

        self.new_positions.clear();
        self.edge_buffer.clear();
        self.pending_searched_updates.clear();
        self.pending_status_updates.clear();
        self.ops_since_commit = 0;
    }

    fn position_count(&self) -> usize {
        self.cache.len()
    }

    fn searched_depth_for(&self, key: &[u8; 18]) -> usize {
        self.cache.get(key).map_or(0, |e| e.searched_depth)
    }

    fn is_known_terminal(&self, key: &[u8; 18]) -> Option<bool> {
        self.cache.get(key).map(|e| e.is_terminal)
    }

    fn summary_by_depth(&self) -> Vec<(i64, i64, i64, i64, i64)> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT depth, COUNT(*), SUM(is_terminal), SUM(symmetry_count),
                        SUM(CASE WHEN searched_depth >= 1 THEN 1 ELSE 0 END)
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
                    row.get::<_, i64>(4)?,
                ))
            })
            .expect("Failed to query summary");
        rows.flatten().collect()
    }

    fn update_depth_if_shallower(&mut self, key: &[u8; 18], depth: usize) {
        if let Some(entry) = self.cache.get_mut(key) {
            if depth < entry.depth {
                entry.depth = depth;
                entry.dirty = true;
            }
        }
    }

    fn flush_depth_updates(&mut self) {
        let dirty: Vec<([u8; 18], usize)> = self
            .cache
            .iter()
            .filter(|(_, e)| e.dirty)
            .map(|(k, e)| (*k, e.depth))
            .collect();
        if dirty.is_empty() {
            return;
        }
        self.conn
            .execute_batch("BEGIN TRANSACTION")
            .expect("Failed to begin depth update txn");
        {
            let mut stmt = self
                .conn
                .prepare_cached(
                    "UPDATE positions SET depth = MIN(depth, ?2) WHERE canonical_key = ?1",
                )
                .expect("Failed to prepare depth update");
            for (key, depth) in &dirty {
                stmt.execute(params![key.as_slice(), *depth as i64])
                    .expect("Failed to update depth");
            }
        }
        self.conn
            .execute_batch("COMMIT")
            .expect("Failed to commit depth updates");
        for (key, _) in &dirty {
            if let Some(entry) = self.cache.get_mut(key) {
                entry.dirty = false;
            }
        }
    }
}

// ── IDDFS Core ──────────────────────────────────────────────────────

struct IddfsState {
    new_positions_this_iter: usize,
    dropout: bool,
    max_positions: Option<usize>,
}

fn iddfs(
    bb: Bitboard,
    depth: usize,
    depth_limit: usize,
    builder: &mut BookBuilder,
    state: &mut IddfsState,
) {
    if state.dropout {
        return;
    }

    let key = canonical_key(&bb);
    let remaining = depth_limit - depth;

    let already_searched = builder.searched_depth_for(&key);
    if already_searched >= remaining {
        return;
    }

    let terminal = match builder.is_known_terminal(&key) {
        Some(t) => t,
        None => {
            let t = is_terminal(&bb);
            let winner = if t { determine_winner(&bb) } else { None };
            let sym = SymmetryHandler::orbit_size(&bb);
            let is_new = builder.cache_position(key, depth, t, winner, sym);
            if is_new {
                state.new_positions_this_iter += 1;
                if let Some(max) = state.max_positions {
                    if builder.position_count() >= max {
                        state.dropout = true;
                        return;
                    }
                }
            }
            t
        }
    };

    builder.update_depth_if_shallower(&key, depth);

    if depth >= depth_limit || terminal {
        builder.update_searched_depth(&key, remaining);
        if builder.should_flush() {
            builder.flush();
        }
        return;
    }

    let moves = generate_legal_moves(&bb);
    for m in &moves {
        if state.dropout {
            return;
        }
        let child_bb = apply_move(&bb, m);
        let child_key = canonical_key(&child_bb);

        let child_is_known = builder.cache.contains_key(&child_key);
        if !child_is_known {
            let t = is_terminal(&child_bb);
            let winner = if t { determine_winner(&child_bb) } else { None };
            let sym = SymmetryHandler::orbit_size(&child_bb);
            let is_new = builder.cache_position(child_key, depth + 1, t, winner, sym);
            if is_new {
                state.new_positions_this_iter += 1;
                if let Some(max) = state.max_positions {
                    if builder.position_count() >= max {
                        state.dropout = true;
                    }
                }
            }
        } else {
            builder.update_depth_if_shallower(&child_key, depth + 1);
        }

        builder.cache_edge(key, child_key, move_to_string(m));

        if builder.should_flush() {
            builder.flush();
        }

        if !state.dropout {
            iddfs(child_bb, depth + 1, depth_limit, builder, state);
        }
    }

    builder.update_searched_depth(&key, remaining);
    builder.mark_expanded(&key);

    if builder.should_flush() {
        builder.flush();
    }
}

fn build_book(cli: &Cli) {
    let mut builder = BookBuilder::open(&cli.db, cli.batch_size);
    let exhaustive_depth = cli.exhaustive_depth.unwrap_or(cli.depth);
    let max_depth = cli.depth;

    if cli.resume {
        eprintln!("[resume] Loading existing positions from database...");
        builder.load_cache_from_db();
        eprintln!("[resume] Loaded {} positions into cache", builder.position_count());
    }

    let root = Bitboard::EMPTY;
    let root_key = canonical_key(&root);
    if !builder.cache.contains_key(&root_key) {
        let sym = SymmetryHandler::orbit_size(&root);
        builder.cache_position(root_key, 0, false, None, sym);
        builder.flush();
    }

    if !cli.quiet {
        println!(
            "IDDFS Book Builder (exhaustive depth {}, max depth {})",
            exhaustive_depth, max_depth
        );
        println!(
            "{:>5}  {:>12}  {:>10}  {:>12}  {:>10}",
            "Iter", "Positions", "New", "Edges", "Time (s)"
        );
    }

    let t_start = Instant::now();

    for depth_limit in 1..=max_depth {
        let t_iter = Instant::now();

        let mut state = IddfsState {
            new_positions_this_iter: 0,
            dropout: false,
            max_positions: cli.max_positions,
        };

        iddfs(root, 0, depth_limit, &mut builder, &mut state);

        builder.flush();
        builder.flush_depth_updates();

        let elapsed = t_iter.elapsed().as_secs_f64();
        let edge_count = builder.total_edge_count();

        if !cli.quiet {
            println!(
                "{:>5}  {:>12}  {:>10}  {:>12}  {:>10.3}",
                depth_limit,
                builder.position_count(),
                state.new_positions_this_iter,
                edge_count,
                elapsed,
            );
        }

        if state.dropout {
            if !cli.quiet {
                println!(
                    "  Dropout: reached {} positions (limit {})",
                    builder.position_count(),
                    cli.max_positions.unwrap_or(0)
                );
            }
            break;
        }

        if depth_limit >= exhaustive_depth && depth_limit < max_depth {
            if !cli.quiet {
                eprintln!(
                    "  [exhaustive phase complete at depth {}, switching to selective]",
                    exhaustive_depth
                );
            }
        }
    }

    print_summary(&builder, &t_start, cli.quiet);
}

fn print_summary(builder: &BookBuilder, t_start: &Instant, quiet: bool) {
    let total = t_start.elapsed().as_secs_f64();
    let count = builder.position_count();
    let edges = builder.total_edge_count();

    println!("\n--- Summary ---");
    println!("Total positions: {}", count);
    println!("Total edges: {}", edges);
    println!("Elapsed: {:.3}s", total);
    if total > 0.0 {
        println!("Throughput: {:.0} pos/sec", count as f64 / total);
    }

    if !quiet {
        let by_depth = builder.summary_by_depth();
        if !by_depth.is_empty() {
            println!(
                "\n{:>5}  {:>12}  {:>10}  {:>12}  {:>14}  {:>16}",
                "Depth", "Positions", "Terminal", "Edges", "SymmetrySum", "SearchedDepth>=1"
            );
            for (d, cnt, term, sym, searched) in &by_depth {
                let edge_count: i64 = builder
                    .conn
                    .query_row(
                        "SELECT COUNT(*) FROM edges WHERE parent_key IN (SELECT canonical_key FROM positions WHERE depth = ?1)",
                        params![d],
                        |row| row.get(0),
                    )
                    .unwrap_or(0);
                println!(
                    "{:>5}  {:>12}  {:>10}  {:>12}  {:>14}  {:>16}",
                    d, cnt, term, edge_count, sym, searched
                );
            }
        }
    }
}

fn main() {
    let cli = Cli::parse();
    build_book(&cli);
}
