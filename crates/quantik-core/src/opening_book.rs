use crate::state::State;
use rusqlite::{params, Connection, Result as SqlResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalStatus {
    Interior = 0,
    WinP0 = 1,
    WinP1 = 2,
    Stalemate = 3,
}

impl TerminalStatus {
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::WinP0,
            2 => Self::WinP1,
            3 => Self::Stalemate,
            _ => Self::Interior,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OpeningBookEntry {
    pub canonical_key: Vec<u8>,
    pub qfen: String,
    pub depth: i32,
    pub evaluation: f64,
    pub visit_count: i64,
    pub win_count_p0: i64,
    pub win_count_p1: i64,
    pub draw_count: i64,
    pub best_moves: Vec<(i32, i32)>, // (shape, position)
    pub is_terminal: TerminalStatus,
    pub symmetry_count: i32,
    /// Whether this position carries an exactly-solved game value (see
    /// [`OpeningBookDatabase::add_solved_position`]). `false`/default for
    /// rows written before the `solved`/`game_value` columns existed.
    pub solved: bool,
    /// Exact game value for the side to move (+1/-1), when `solved`.
    pub game_value: Option<i32>,
}

pub struct OpeningBookConfig {
    pub database_path: String,
    pub cache_size_mb: i32,
    pub enable_wal: bool,
}

impl Default for OpeningBookConfig {
    fn default() -> Self {
        Self {
            database_path: "quantik_opening_book.db".into(),
            cache_size_mb: 100,
            enable_wal: true,
        }
    }
}

pub struct OpeningBookDatabase {
    conn: Connection,
    db_path: String,
}

impl OpeningBookDatabase {
    pub fn open(config: &OpeningBookConfig) -> SqlResult<Self> {
        let conn = Connection::open(&config.database_path)?;

        conn.execute_batch(&format!(
            "PRAGMA cache_size = -{};",
            config.cache_size_mb * 1024
        ))?;
        if config.enable_wal {
            conn.execute_batch("PRAGMA journal_mode = WAL;")?;
        }
        conn.execute_batch("PRAGMA synchronous = NORMAL;")?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS positions (
                canonical_key BLOB PRIMARY KEY,
                qfen TEXT NOT NULL,
                depth INTEGER NOT NULL,
                evaluation REAL NOT NULL,
                visit_count INTEGER NOT NULL,
                win_count_p0 INTEGER NOT NULL,
                win_count_p1 INTEGER NOT NULL,
                draw_count INTEGER NOT NULL,
                is_terminal INTEGER NOT NULL DEFAULT 0,
                symmetry_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS best_moves (
                canonical_key BLOB NOT NULL,
                move_rank INTEGER NOT NULL,
                shape INTEGER NOT NULL,
                position INTEGER NOT NULL,
                FOREIGN KEY (canonical_key) REFERENCES positions(canonical_key),
                PRIMARY KEY (canonical_key, move_rank)
            );
            CREATE TABLE IF NOT EXISTS position_edges (
                parent_key BLOB NOT NULL,
                child_key  BLOB NOT NULL,
                PRIMARY KEY (parent_key, child_key),
                FOREIGN KEY (parent_key) REFERENCES positions(canonical_key),
                FOREIGN KEY (child_key)  REFERENCES positions(canonical_key)
            );
            CREATE INDEX IF NOT EXISTS idx_depth ON positions(depth);",
        )?;

        // Idempotent migration for pre-existing `positions` tables that
        // lack some of the columns above: SQLite has no `ADD COLUMN IF
        // NOT EXISTS`, so attempt each ALTER and swallow the "duplicate
        // column name" error it raises when the column is already there.
        // This covers two kinds of DBs:
        // - books created before the `solved`/`game_value` columns
        //   existed, and
        // - searched books built by `bench_bfs`, whose `positions` table
        //   has only structural search columns (no `qfen`, `evaluation`,
        //   `visit_count`, win/draw counters, `solved`, `game_value`).
        //   Migrated searched rows keep these defaults and are therefore
        //   never served as solved references (lookups require
        //   `solved = 1`); solved write-backs upsert over them.
        // `created_at` is deliberately not migrated: ALTER TABLE cannot
        // add a column with a non-constant default, and no query reads it.
        for stmt in [
            "ALTER TABLE positions ADD COLUMN qfen TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE positions ADD COLUMN evaluation REAL NOT NULL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN visit_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN win_count_p0 INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN win_count_p1 INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN draw_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN solved INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE positions ADD COLUMN game_value INTEGER",
        ] {
            if let Err(e) = conn.execute_batch(stmt) {
                if !e.to_string().contains("duplicate column name") {
                    return Err(e);
                }
            }
        }

        // These indexes reference columns that may only exist after the
        // migration above, so they cannot be created in the initial batch.
        //
        // The position_edges index must NOT be named `idx_edges_child`:
        // searched books already carry an index of that name on their
        // `edges` table, and SQLite index names are database-global, so
        // `CREATE INDEX IF NOT EXISTS` under the colliding name would
        // silently skip indexing position_edges. Books written before the
        // rename carry `idx_edges_child` on position_edges itself; drop
        // that one (and only that one) so the rename does not leave a
        // duplicate index behind.
        let legacy_edge_index: i64 = conn.query_row(
            "SELECT count(*) FROM sqlite_master
             WHERE type = 'index' AND name = 'idx_edges_child'
               AND tbl_name = 'position_edges'",
            [],
            |row| row.get(0),
        )?;
        if legacy_edge_index > 0 {
            conn.execute_batch("DROP INDEX idx_edges_child;")?;
        }
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_visit_count ON positions(visit_count DESC);
             CREATE INDEX IF NOT EXISTS idx_position_edges_child ON position_edges(child_key);",
        )?;

        Ok(Self {
            conn,
            db_path: config.database_path.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_position(
        &self,
        state: &State,
        evaluation: f64,
        visit_count: i64,
        win_count_p0: i64,
        win_count_p1: i64,
        draw_count: i64,
        best_moves: &[(i32, i32)],
        depth: i32,
        is_terminal: TerminalStatus,
        symmetry_count: i32,
    ) -> SqlResult<()> {
        let canonical_key = state.canonical_key().to_vec();
        let qfen = state.to_qfen();

        self.conn.execute(
            "INSERT OR REPLACE INTO positions
             (canonical_key, qfen, depth, evaluation, visit_count,
              win_count_p0, win_count_p1, draw_count, is_terminal, symmetry_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                canonical_key,
                qfen,
                depth,
                evaluation,
                visit_count,
                win_count_p0,
                win_count_p1,
                draw_count,
                is_terminal as i32,
                symmetry_count,
            ],
        )?;

        self.conn.execute(
            "DELETE FROM best_moves WHERE canonical_key = ?1",
            params![canonical_key],
        )?;

        for (rank, &(shape, position)) in best_moves.iter().take(5).enumerate() {
            self.conn.execute(
                "INSERT INTO best_moves (canonical_key, move_rank, shape, position)
                 VALUES (?1, ?2, ?3, ?4)",
                params![canonical_key, (rank + 1) as i32, shape, position],
            )?;
        }
        Ok(())
    }

    /// Upsert an exactly-solved position: `evaluation` is `game_value` as
    /// `f64`, `is_terminal` stays `Interior` (the position itself is not
    /// terminal — its *game value* is exactly known), `depth` is pieces
    /// placed (derived from `state.bb`), and `best_moves` records every
    /// optimal move (not just a top-5 slice, unlike [`Self::add_position`])
    /// as `(shape, position)` pairs in the given order.
    ///
    /// **Only canonical representatives are stored.** The optimal moves
    /// are expressed in `state`'s own board orientation, but the row is
    /// keyed by the canonical key shared by up to eight symmetric
    /// orientations; storing a non-representative orientation would let a
    /// later lookup on the *representative* serve moves that are wrong
    /// (possibly illegal) for that board. If `state` is not its own
    /// canonical representative (`canonical_payload() != bb.to_le_bytes()`),
    /// nothing is written and `Ok(false)` is returned; `Ok(true)` means
    /// the row was upserted. Translating moves across orientations via
    /// the symmetry transform is a documented follow-up that would lift
    /// this restriction.
    ///
    /// The position row and its best-move rows are written in one
    /// transaction, so a mid-way failure can never leave a solved row
    /// with partial `best_moves`.
    ///
    /// Visit/win/draw counters and `symmetry_count` are not meaningful for
    /// a solved reference and are initialized to `0` on first insert. When
    /// the row already exists, only the columns this API owns (`qfen`,
    /// `depth`, `evaluation`, `solved`, `game_value`) are updated: counters,
    /// terminal/symmetry fields, and any extra columns from other schemas
    /// (e.g. `bench_bfs` search metadata like `searched_depth`/`status`)
    /// are left untouched — a plain `INSERT OR REPLACE` would delete and
    /// re-insert the row, silently resetting them.
    pub fn add_solved_position(
        &self,
        state: &State,
        game_value: i32,
        optimal_moves: &[(i32, i32)],
    ) -> SqlResult<bool> {
        if state.canonical_payload() != state.bb.to_le_bytes() {
            return Ok(false);
        }

        let canonical_key = state.canonical_key().to_vec();
        let qfen = state.to_qfen();
        let depth = (state.bb.player_piece_count(0) + state.bb.player_piece_count(1)) as i32;
        let evaluation = game_value as f64;

        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO positions
             (canonical_key, qfen, depth, evaluation, visit_count,
              win_count_p0, win_count_p1, draw_count, is_terminal, symmetry_count,
              solved, game_value)
             VALUES (?1, ?2, ?3, ?4, 0, 0, 0, 0, ?5, 0, 1, ?6)
             ON CONFLICT(canonical_key) DO UPDATE SET
               qfen = excluded.qfen,
               depth = excluded.depth,
               evaluation = excluded.evaluation,
               solved = excluded.solved,
               game_value = excluded.game_value",
            params![
                canonical_key,
                qfen,
                depth,
                evaluation,
                TerminalStatus::Interior as i32,
                game_value,
            ],
        )?;

        tx.execute(
            "DELETE FROM best_moves WHERE canonical_key = ?1",
            params![canonical_key],
        )?;

        for (rank, &(shape, position)) in optimal_moves.iter().enumerate() {
            tx.execute(
                "INSERT INTO best_moves (canonical_key, move_rank, shape, position)
                 VALUES (?1, ?2, ?3, ?4)",
                params![canonical_key, (rank + 1) as i32, shape, position],
            )?;
        }
        tx.commit()?;
        Ok(true)
    }

    pub fn get_position(&self, state: &State) -> SqlResult<Option<OpeningBookEntry>> {
        let canonical_key = state.canonical_key().to_vec();

        let mut stmt = self.conn.prepare(
            "SELECT qfen, depth, evaluation, visit_count,
                    win_count_p0, win_count_p1, draw_count,
                    is_terminal, symmetry_count, solved, game_value
             FROM positions WHERE canonical_key = ?1",
        )?;

        let entry = stmt.query_row(params![canonical_key], |row| {
            Ok(OpeningBookEntry {
                canonical_key: canonical_key.clone(),
                qfen: row.get(0)?,
                depth: row.get(1)?,
                evaluation: row.get(2)?,
                visit_count: row.get(3)?,
                win_count_p0: row.get(4)?,
                win_count_p1: row.get(5)?,
                draw_count: row.get(6)?,
                best_moves: Vec::new(), // filled below
                is_terminal: TerminalStatus::from_i32(row.get(7)?),
                symmetry_count: row.get(8)?,
                solved: row.get::<_, i64>(9)? != 0,
                game_value: row.get(10)?,
            })
        });

        match entry {
            Ok(mut e) => {
                let mut mv_stmt = self.conn.prepare(
                    "SELECT shape, position FROM best_moves
                     WHERE canonical_key = ?1 ORDER BY move_rank",
                )?;
                let moves = mv_stmt.query_map(params![e.canonical_key], |row| {
                    Ok((row.get::<_, i32>(0)?, row.get::<_, i32>(1)?))
                })?;
                e.best_moves = moves.filter_map(|r| r.ok()).collect();
                Ok(Some(e))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn query_by_depth(&self, depth: i32, limit: i64) -> SqlResult<Vec<OpeningBookEntry>> {
        let mut stmt = self.conn.prepare(
            "SELECT canonical_key, qfen, depth, evaluation, visit_count,
                    win_count_p0, win_count_p1, draw_count,
                    is_terminal, symmetry_count, solved, game_value
             FROM positions WHERE depth = ?1
             ORDER BY visit_count DESC LIMIT ?2",
        )?;

        let entries: Vec<OpeningBookEntry> = stmt
            .query_map(params![depth, limit], |row| {
                Ok(OpeningBookEntry {
                    canonical_key: row.get(0)?,
                    qfen: row.get(1)?,
                    depth: row.get(2)?,
                    evaluation: row.get(3)?,
                    visit_count: row.get(4)?,
                    win_count_p0: row.get(5)?,
                    win_count_p1: row.get(6)?,
                    draw_count: row.get(7)?,
                    best_moves: Vec::new(),
                    is_terminal: TerminalStatus::from_i32(row.get(8)?),
                    symmetry_count: row.get(9)?,
                    solved: row.get::<_, i64>(10)? != 0,
                    game_value: row.get(11)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(entries)
    }

    // ── DAG edges ────────────────────────────────────────────────────

    pub fn add_edges(&self, edges: &[(&[u8], &[u8])]) -> SqlResult<()> {
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR IGNORE INTO position_edges (parent_key, child_key)
                 VALUES (?1, ?2)",
            )?;
            for &(parent, child) in edges {
                stmt.execute(params![parent, child])?;
            }
        }
        tx.commit()
    }

    pub fn get_children(&self, canonical_key: &[u8]) -> SqlResult<Vec<Vec<u8>>> {
        let mut stmt = self
            .conn
            .prepare("SELECT child_key FROM position_edges WHERE parent_key = ?1")?;
        let rows = stmt.query_map(params![canonical_key], |row| row.get(0))?;
        rows.collect()
    }

    pub fn get_parents(&self, canonical_key: &[u8]) -> SqlResult<Vec<Vec<u8>>> {
        let mut stmt = self
            .conn
            .prepare("SELECT parent_key FROM position_edges WHERE child_key = ?1")?;
        let rows = stmt.query_map(params![canonical_key], |row| row.get(0))?;
        rows.collect()
    }

    pub fn get_edge_count(&self) -> SqlResult<i64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM position_edges", [], |row| row.get(0))
    }

    // ── statistics ───────────────────────────────────────────────────

    pub fn total_positions(&self) -> SqlResult<i64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM positions", [], |row| row.get(0))
    }

    pub fn max_depth(&self) -> SqlResult<Option<i32>> {
        self.conn
            .query_row("SELECT MAX(depth) FROM positions", [], |row| row.get(0))
    }

    pub fn positions_by_depth(&self) -> SqlResult<Vec<(i32, i64)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT depth, COUNT(*) FROM positions GROUP BY depth ORDER BY depth")?;
        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect()
    }

    pub fn db_path(&self) -> &str {
        &self.db_path
    }

    pub fn file_size(&self) -> u64 {
        std::fs::metadata(&self.db_path)
            .map(|m| m.len())
            .unwrap_or(0)
    }
}

impl Drop for OpeningBookDatabase {
    fn drop(&mut self) {
        // Connection is dropped automatically
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_db_path() -> String {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("/tmp/quantik_test_{}.db", id)
    }

    #[test]
    fn open_and_add_position() {
        let path = temp_db_path();
        let config = OpeningBookConfig {
            database_path: path.clone(),
            ..Default::default()
        };
        let db = OpeningBookDatabase::open(&config).unwrap();

        let state = State::empty();
        db.add_position(
            &state,
            0.0,
            100,
            50,
            40,
            10,
            &[(0, 0), (1, 5)],
            0,
            TerminalStatus::Interior,
            1,
        )
        .unwrap();

        let entry = db.get_position(&state).unwrap().unwrap();
        assert_eq!(entry.visit_count, 100);
        assert_eq!(entry.best_moves.len(), 2);

        assert_eq!(db.total_positions().unwrap(), 1);

        fs::remove_file(&path).ok();
    }

    #[test]
    fn edge_operations() {
        let path = temp_db_path();
        let config = OpeningBookConfig {
            database_path: path.clone(),
            ..Default::default()
        };
        let db = OpeningBookDatabase::open(&config).unwrap();

        let s1 = State::empty();
        let s2 = State::from_qfen("A.../..../..../....").unwrap();

        db.add_position(&s1, 0.0, 1, 0, 0, 0, &[], 0, TerminalStatus::Interior, 1)
            .unwrap();
        db.add_position(&s2, 0.1, 1, 0, 0, 0, &[], 1, TerminalStatus::Interior, 1)
            .unwrap();

        let k1 = s1.canonical_key();
        let k2 = s2.canonical_key();
        db.add_edges(&[(&k1, &k2)]).unwrap();

        let children = db.get_children(&k1).unwrap();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0], k2.to_vec());

        let parents = db.get_parents(&k2).unwrap();
        assert_eq!(parents.len(), 1);

        assert_eq!(db.get_edge_count().unwrap(), 1);

        fs::remove_file(&path).ok();
    }

    #[test]
    fn query_by_depth_works() {
        let path = temp_db_path();
        let config = OpeningBookConfig {
            database_path: path.clone(),
            ..Default::default()
        };
        let db = OpeningBookDatabase::open(&config).unwrap();

        let state = State::empty();
        db.add_position(
            &state,
            0.0,
            10,
            5,
            3,
            2,
            &[],
            0,
            TerminalStatus::Interior,
            1,
        )
        .unwrap();

        let entries = db.query_by_depth(0, 10).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].depth, 0);

        let entries = db.query_by_depth(1, 10).unwrap();
        assert!(entries.is_empty());

        fs::remove_file(&path).ok();
    }

    #[test]
    fn add_solved_position_roundtrips() {
        let path = temp_db_path();
        let config = OpeningBookConfig {
            database_path: path.clone(),
            ..Default::default()
        };
        let db = OpeningBookDatabase::open(&config).unwrap();

        // The empty board is trivially its own canonical representative.
        let state = State::empty();
        let written = db
            .add_solved_position(&state, 1, &[(0, 0), (1, 5)])
            .unwrap();
        assert!(written);

        let entry = db.get_position(&state).unwrap().unwrap();
        assert!(entry.solved);
        assert_eq!(entry.game_value, Some(1));
        assert_eq!(entry.evaluation, 1.0);
        assert_eq!(entry.is_terminal, TerminalStatus::Interior);
        assert_eq!(entry.depth, 0);
        assert_eq!(entry.best_moves, vec![(0, 0), (1, 5)]);

        fs::remove_file(&path).ok();
    }

    /// Defense-in-depth guard: `add_solved_position` refuses (returns
    /// `Ok(false)`, writes nothing) when the state is not its own
    /// canonical representative — its moves would be stored in the wrong
    /// orientation for the canonical key.
    #[test]
    fn add_solved_position_skips_non_canonical_orientations() {
        let path = temp_db_path();
        let config = OpeningBookConfig {
            database_path: path.clone(),
            ..Default::default()
        };
        let db = OpeningBookDatabase::open(&config).unwrap();

        // Find a single-piece placement that is NOT its own canonical
        // representative (most of the 16 cells aren't).
        let state = (0..16)
            .map(|pos| {
                let mut qfen: Vec<char> = "..../..../..../....".chars().collect();
                let index = pos + pos / 4; // account for '/' separators
                qfen[index] = 'A';
                State::from_qfen(&qfen.into_iter().collect::<String>()).unwrap()
            })
            .find(|s| s.canonical_payload() != s.bb.to_le_bytes())
            .expect("some single-piece placement is non-canonical");

        let written = db.add_solved_position(&state, 1, &[(0, 0)]).unwrap();
        assert!(!written);
        assert_eq!(db.total_positions().unwrap(), 0);
        assert!(db.get_position(&state).unwrap().is_none());

        fs::remove_file(&path).ok();
    }

    /// Opening a pre-existing DB created with the OLD schema (no
    /// `solved`/`game_value` columns) must upgrade it in place: `open()`
    /// succeeds and `get_position` returns the migrated defaults
    /// (`solved: false`, `game_value: None`) for rows written before the
    /// migration existed.
    #[test]
    fn migration_upgrades_pre_existing_db() {
        let path = temp_db_path();

        // Build the OLD schema by hand (mirrors the CREATE TABLE that
        // predates the solved/game_value columns) and insert a row via raw
        // SQL, bypassing OpeningBookDatabase entirely.
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE positions (
                    canonical_key BLOB PRIMARY KEY,
                    qfen TEXT NOT NULL,
                    depth INTEGER NOT NULL,
                    evaluation REAL NOT NULL,
                    visit_count INTEGER NOT NULL,
                    win_count_p0 INTEGER NOT NULL,
                    win_count_p1 INTEGER NOT NULL,
                    draw_count INTEGER NOT NULL,
                    is_terminal INTEGER NOT NULL DEFAULT 0,
                    symmetry_count INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE best_moves (
                    canonical_key BLOB NOT NULL,
                    move_rank INTEGER NOT NULL,
                    shape INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (canonical_key, move_rank)
                );
                CREATE TABLE position_edges (
                    parent_key BLOB NOT NULL,
                    child_key  BLOB NOT NULL,
                    PRIMARY KEY (parent_key, child_key)
                );
                CREATE INDEX idx_edges_child ON position_edges(child_key);",
            )
            .unwrap();

            let state = State::empty();
            let canonical_key = state.canonical_key().to_vec();
            conn.execute(
                "INSERT INTO positions
                 (canonical_key, qfen, depth, evaluation, visit_count,
                  win_count_p0, win_count_p1, draw_count, is_terminal, symmetry_count)
                 VALUES (?1, ?2, 0, 0.5, 7, 3, 2, 1, 0, 1)",
                params![canonical_key, state.to_qfen()],
            )
            .unwrap();
        }

        // Opening via OpeningBookDatabase must succeed (idempotent ALTER
        // TABLE) and see the pre-existing row with migrated defaults.
        let config = OpeningBookConfig {
            database_path: path.clone(),
            ..Default::default()
        };
        let db = OpeningBookDatabase::open(&config).unwrap();
        let entry = db.get_position(&State::empty()).unwrap().unwrap();
        assert!(!entry.solved);
        assert_eq!(entry.game_value, None);
        assert_eq!(entry.visit_count, 7);

        // The legacy `idx_edges_child` index on position_edges is renamed:
        // dropped and re-created as `idx_position_edges_child`, with no
        // duplicate left behind.
        let index_names: Vec<String> = {
            let mut stmt = db
                .conn
                .prepare(
                    "SELECT name FROM sqlite_master
                     WHERE type = 'index' AND tbl_name = 'position_edges'
                       AND name NOT LIKE 'sqlite_autoindex%'",
                )
                .unwrap();
            let names = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect();
            names
        };
        assert_eq!(index_names, vec!["idx_position_edges_child".to_string()]);

        // Re-opening again (columns already present) must also succeed.
        OpeningBookDatabase::open(&config).unwrap();

        fs::remove_file(&path).ok();
    }

    /// Opening a searched book built by `bench_bfs` (positions table with
    /// only structural search columns — no `qfen`/`evaluation`/
    /// `visit_count`/win counters/`solved`) must upgrade it in place so
    /// the benchmark read-through path (`generate_positions.sh --book`)
    /// can open it: pre-existing searched rows read back with migrated
    /// defaults and are never served as solved references, and solved
    /// write-backs upsert over them.
    #[test]
    fn migration_upgrades_bench_bfs_searched_book() {
        let path = temp_db_path();

        // Build the bench_bfs schema by hand (mirrors the CREATE TABLE in
        // src/bin/bench_bfs.rs) and insert the empty-board row via raw
        // SQL, bypassing OpeningBookDatabase entirely.
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE positions (
                    canonical_key BLOB PRIMARY KEY,
                    depth INTEGER NOT NULL,
                    is_terminal INTEGER NOT NULL DEFAULT 0,
                    winner INTEGER,
                    symmetry_count INTEGER NOT NULL,
                    searched_depth INTEGER NOT NULL DEFAULT 0,
                    score REAL,
                    status INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE edges (
                    parent_key BLOB NOT NULL,
                    child_key BLOB NOT NULL,
                    move TEXT NOT NULL,
                    PRIMARY KEY (parent_key, child_key)
                );
                CREATE INDEX idx_edges_child ON edges(child_key);
                CREATE INDEX idx_pos_depth ON positions(depth);
                CREATE INDEX idx_pos_status ON positions(status);
                CREATE INDEX idx_pos_searched ON positions(searched_depth);",
            )
            .unwrap();

            let state = State::empty();
            conn.execute(
                "INSERT INTO positions
                 (canonical_key, depth, is_terminal, winner, symmetry_count,
                  searched_depth, score, status)
                 VALUES (?1, 0, 0, NULL, 1, 1, NULL, 1)",
                params![state.canonical_key().to_vec()],
            )
            .unwrap();
        }

        let config = OpeningBookConfig {
            database_path: path.clone(),
            ..Default::default()
        };
        let db = OpeningBookDatabase::open(&config).unwrap();

        // position_edges must get its own child_key index even though the
        // searched book already has an `idx_edges_child` index on the
        // unrelated `edges` table (SQLite index names are database-global,
        // so a colliding name would make CREATE INDEX IF NOT EXISTS a
        // silent no-op).
        let edge_indexes: i64 = db
            .conn
            .query_row(
                "SELECT count(*) FROM sqlite_master
                 WHERE type = 'index' AND tbl_name = 'position_edges'
                   AND name = 'idx_position_edges_child'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(edge_indexes, 1);
        // The searched book's own edges index must be left untouched.
        let bench_indexes: i64 = db
            .conn
            .query_row(
                "SELECT count(*) FROM sqlite_master
                 WHERE type = 'index' AND tbl_name = 'edges'
                   AND name = 'idx_edges_child'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(bench_indexes, 1);

        // The searched row reads back with migrated defaults and must not
        // look like a solved reference.
        let entry = db.get_position(&State::empty()).unwrap().unwrap();
        assert!(!entry.solved);
        assert_eq!(entry.game_value, None);
        assert_eq!(entry.visit_count, 0);
        assert_eq!(entry.qfen, "");
        assert!(entry.best_moves.is_empty());

        // Solved write-back upserts over the searched row.
        let written = db
            .add_solved_position(&State::empty(), 1, &[(0, 0)])
            .unwrap();
        assert!(written);
        let entry = db.get_position(&State::empty()).unwrap().unwrap();
        assert!(entry.solved);
        assert_eq!(entry.game_value, Some(1));
        assert_eq!(entry.best_moves, vec![(0, 0)]);

        // The write-back must not destroy bench_bfs search metadata: the
        // searched row was inserted with searched_depth = 1 and status = 1,
        // and those bench-owned columns must survive the solved upsert
        // (INSERT OR REPLACE would delete + re-insert the row and reset
        // them to their defaults).
        let (searched_depth, status): (i64, i64) = db
            .conn
            .query_row(
                "SELECT searched_depth, status FROM positions WHERE canonical_key = ?1",
                params![State::empty().canonical_key().to_vec()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(searched_depth, 1);
        assert_eq!(status, 1);

        // Re-opening again (columns already present) must also succeed.
        OpeningBookDatabase::open(&config).unwrap();

        fs::remove_file(&path).ok();
    }
}
