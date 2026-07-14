use clap::{Parser, Subcommand};
use rusqlite::{params, Connection};

#[derive(Parser)]
#[command(
    name = "bench_bfs_inspect",
    about = "Inspect bench_bfs SQLite opening-book databases"
)]
struct Cli {
    /// SQLite database path produced by bench_bfs.
    #[arg(long)]
    db: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print storage and per-depth search statistics.
    Stats {
        /// Target depth to evaluate resumable work for. Defaults to max depth + 1.
        #[arg(long)]
        target_depth: Option<usize>,
    },
    /// Print the non-terminal frontier that still needs search for a target depth.
    Frontier {
        /// Target depth to evaluate resumable work for. Defaults to max depth + 1.
        #[arg(long)]
        target_depth: Option<usize>,

        /// Maximum sample rows to print.
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Print SQLite page/index/storage details.
    Storage,
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    let conn = Connection::open(&cli.db).map_err(|e| format!("open {}: {e}", cli.db))?;
    validate_schema(&conn)?;

    match cli.command {
        Command::Stats { target_depth } => print_stats(&conn, &cli.db, target_depth)?,
        Command::Frontier {
            target_depth,
            limit,
        } => print_frontier(&conn, target_depth, limit)?,
        Command::Storage => print_storage(&conn, &cli.db)?,
    }
    Ok(())
}

fn validate_schema(conn: &Connection) -> Result<(), String> {
    for table in ["positions", "edges"] {
        let exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
                params![table],
                |row| row.get(0),
            )
            .map_err(|e| format!("inspect schema: {e}"))?;
        if exists == 0 {
            return Err(format!(
                "missing `{table}` table; is this a bench_bfs database?"
            ));
        }
    }
    Ok(())
}

fn max_depth(conn: &Connection) -> Result<usize, String> {
    let depth: Option<i64> = conn
        .query_row("SELECT MAX(depth) FROM positions", [], |row| row.get(0))
        .map_err(|e| format!("read max depth: {e}"))?;
    Ok(depth.unwrap_or(0).max(0) as usize)
}

fn target_or_next(conn: &Connection, target_depth: Option<usize>) -> Result<usize, String> {
    Ok(target_depth.unwrap_or(max_depth(conn)? + 1))
}

fn file_size(path: &str) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn print_stats(conn: &Connection, path: &str, target_depth: Option<usize>) -> Result<(), String> {
    let target = target_or_next(conn, target_depth)?;
    let positions: i64 = conn
        .query_row("SELECT COUNT(*) FROM positions", [], |row| row.get(0))
        .map_err(|e| format!("count positions: {e}"))?;
    let edges: i64 = conn
        .query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))
        .map_err(|e| format!("count edges: {e}"))?;
    let terminal: i64 = conn
        .query_row(
            "SELECT COALESCE(SUM(is_terminal), 0) FROM positions",
            [],
            |row| row.get(0),
        )
        .map_err(|e| format!("count terminal: {e}"))?;
    let max_depth = max_depth(conn)?;
    let resumable = resumable_count(conn, target)?;

    println!("Database: {path}");
    println!("File size bytes: {}", file_size(path));
    println!("Positions: {positions}");
    println!("Terminal: {terminal}");
    println!("Edges: {edges}");
    println!("Max depth: {max_depth}");
    println!("Target depth: {target}");
    println!("Rows needing search for target depth: {resumable}");
    println!(
        "Resume command: scripts/generate_opening_book.sh search --depth {target} --db {path} --resume"
    );

    println!(
        "\n{:>5} {:>12} {:>10} {:>12} {:>14} {:>12} {:>12} {:>14}",
        "Depth",
        "Positions",
        "Terminal",
        "Edges",
        "SymmetrySum",
        "Searched>=1",
        "TermSearched",
        "NeedsTarget"
    );

    let mut stmt = conn
        .prepare(
            "SELECT p.depth,
                    COUNT(*) AS positions,
                    COALESCE(SUM(p.is_terminal), 0) AS terminal,
                    COALESCE(SUM(p.symmetry_count), 0) AS symmetry_sum,
                    COALESCE(SUM(CASE WHEN p.searched_depth >= 1 THEN 1 ELSE 0 END), 0) AS searched,
                    COALESCE(SUM(CASE WHEN p.is_terminal = 1 AND p.searched_depth >= 1 THEN 1 ELSE 0 END), 0) AS terminal_searched,
                    COALESCE(SUM(CASE
                        WHEN p.is_terminal = 0
                         AND p.depth < ?1
                         AND p.searched_depth < (?1 - p.depth)
                        THEN 1 ELSE 0 END), 0) AS needs_target,
                    (SELECT COUNT(*) FROM edges e
                     JOIN positions pp ON pp.canonical_key = e.parent_key
                     WHERE pp.depth = p.depth) AS edge_count
             FROM positions p
             GROUP BY p.depth
             ORDER BY p.depth",
        )
        .map_err(|e| format!("prepare depth summary: {e}"))?;
    let rows = stmt
        .query_map(params![target as i64], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, i64>(7)?,
            ))
        })
        .map_err(|e| format!("query depth summary: {e}"))?;
    for row in rows {
        let (depth, count, terminal, symmetry, searched, terminal_searched, needs, edges) =
            row.map_err(|e| format!("read depth summary: {e}"))?;
        println!(
            "{depth:>5} {count:>12} {terminal:>10} {edges:>12} {symmetry:>14} {searched:>12} {terminal_searched:>12} {needs:>14}"
        );
    }
    Ok(())
}

fn resumable_count(conn: &Connection, target_depth: usize) -> Result<i64, String> {
    conn.query_row(
        "SELECT COUNT(*) FROM positions
         WHERE is_terminal = 0
           AND depth < ?1
           AND searched_depth < (?1 - depth)",
        params![target_depth as i64],
        |row| row.get(0),
    )
    .map_err(|e| format!("count resumable rows: {e}"))
}

fn print_frontier(
    conn: &Connection,
    target_depth: Option<usize>,
    limit: usize,
) -> Result<(), String> {
    let target = target_or_next(conn, target_depth)?;
    let count = resumable_count(conn, target)?;
    println!("Target depth: {target}");
    println!("Rows needing search for target depth: {count}");
    println!(
        "\n{:>5} {:>14} {:>16} {:>36}",
        "Depth", "SearchedDepth", "RemainingNeeded", "CanonicalKeyHex"
    );

    let mut stmt = conn
        .prepare(
            "SELECT depth, searched_depth, hex(canonical_key)
             FROM positions
             WHERE is_terminal = 0
               AND depth < ?1
               AND searched_depth < (?1 - depth)
             ORDER BY depth DESC, searched_depth ASC, canonical_key
             LIMIT ?2",
        )
        .map_err(|e| format!("prepare frontier query: {e}"))?;
    let rows = stmt
        .query_map(params![target as i64, limit as i64], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(|e| format!("query frontier: {e}"))?;
    for row in rows {
        let (depth, searched, hex) = row.map_err(|e| format!("read frontier row: {e}"))?;
        let remaining = target as i64 - depth;
        println!("{depth:>5} {searched:>14} {remaining:>16} {hex:>36}");
    }
    Ok(())
}

fn print_storage(conn: &Connection, path: &str) -> Result<(), String> {
    let page_count: i64 = conn
        .query_row("PRAGMA page_count", [], |row| row.get(0))
        .map_err(|e| format!("read page_count: {e}"))?;
    let page_size: i64 = conn
        .query_row("PRAGMA page_size", [], |row| row.get(0))
        .map_err(|e| format!("read page_size: {e}"))?;
    let freelist_count: i64 = conn
        .query_row("PRAGMA freelist_count", [], |row| row.get(0))
        .map_err(|e| format!("read freelist_count: {e}"))?;
    println!("Database: {path}");
    println!("File size bytes: {}", file_size(path));
    println!("Page count: {page_count}");
    println!("Page size: {page_size}");
    println!("Freelist pages: {freelist_count}");
    println!("Logical page bytes: {}", page_count * page_size);

    println!("\nObjects:");
    let mut stmt = conn
        .prepare(
            "SELECT type, name
             FROM sqlite_master
             WHERE type IN ('table', 'index')
             ORDER BY type, name",
        )
        .map_err(|e| format!("prepare object query: {e}"))?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|e| format!("query objects: {e}"))?;
    for row in rows {
        let (kind, name) = row.map_err(|e| format!("read object row: {e}"))?;
        println!("  {kind:5} {name}");
    }
    Ok(())
}
