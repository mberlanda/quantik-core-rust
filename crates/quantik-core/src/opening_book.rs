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
            CREATE INDEX IF NOT EXISTS idx_depth ON positions(depth);
            CREATE INDEX IF NOT EXISTS idx_visit_count ON positions(visit_count DESC);
            CREATE INDEX IF NOT EXISTS idx_edges_child ON position_edges(child_key);",
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

    pub fn get_position(&self, state: &State) -> SqlResult<Option<OpeningBookEntry>> {
        let canonical_key = state.canonical_key().to_vec();

        let mut stmt = self.conn.prepare(
            "SELECT qfen, depth, evaluation, visit_count,
                    win_count_p0, win_count_p1, draw_count,
                    is_terminal, symmetry_count
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
                    is_terminal, symmetry_count
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
}
