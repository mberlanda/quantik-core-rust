//! Build an `opening-probe.v1` file from an existing opening-book SQLite
//! database (the `positions` / `best_moves` layout written by
//! `OpeningBookDatabase`).
//!
//! Only exactly-solved rows (`solved = 1`, `game_value` of -1 or +1) are
//! projected; they are written with `status = exact`. `best_moves` already stores
//! the canonical representative's orientation (`opening_book.rs`,
//! `add_solved_position`), which is the frame the probe stores, so no transform is
//! applied here. The output is re-opened and fully verified before it is written.

use clap::Parser;
use quantik_core::constants::{FLAG_CANON, VERSION};
use quantik_core::opening_probe::{build_probe, ProbeBuildMeta, ProbeRecord, ProbeStatus};
use rusqlite::{Connection, OpenFlags};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "probe_builder",
    about = "Build an opening-probe.v1 file from an opening-book SQLite database"
)]
struct Cli {
    /// Source opening-book SQLite database (opened read-only)
    #[arg(long)]
    book: String,

    /// Output probe file
    #[arg(long)]
    out: String,

    /// `source_book.book_id`. Default: a SHA-256 over the projected rows, so it
    /// changes whenever a key, value or optimal move the probe carries changes.
    #[arg(long)]
    book_id: Option<String>,

    /// Contracts release recorded in the header (informational at runtime)
    #[arg(long, default_value = env!("CARGO_PKG_VERSION"))]
    contract_version: String,
}

fn run(cli: &Cli) -> Result<String, String> {
    let conn = Connection::open_with_flags(&cli.book, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("cannot open {}: {e}", cli.book))?;

    let mut moves: BTreeMap<Vec<u8>, u64> = BTreeMap::new();
    {
        let mut stmt = conn
            .prepare("SELECT canonical_key, shape, position FROM best_moves")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, Vec<u8>>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            })
            .map_err(|e| e.to_string())?;
        for row in rows {
            let (key, shape, position) = row.map_err(|e| e.to_string())?;
            if !(0..4).contains(&shape) || !(0..16).contains(&position) {
                return Err(format!(
                    "best_moves row out of range: ({shape}, {position})"
                ));
            }
            *moves.entry(key).or_default() |= 1u64 << (shape * 16 + position);
        }
    }

    let mut records = Vec::new();
    let mut skipped = 0usize;
    {
        let mut stmt = conn
            .prepare("SELECT canonical_key, solved, game_value FROM positions")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, Vec<u8>>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                ))
            })
            .map_err(|e| e.to_string())?;
        for row in rows {
            let (key, solved, value) = row.map_err(|e| e.to_string())?;
            if solved == 0 {
                skipped += 1;
                continue;
            }
            let Ok(key): Result<[u8; 18], _> = key.as_slice().try_into() else {
                return Err(format!("canonical_key of {} bytes, expected 18", key.len()));
            };
            if key[0] != VERSION || key[1] != FLAG_CANON {
                return Err("canonical_key does not start with 0x01 0x02".into());
            }
            let value = match value {
                Some(v @ (-1 | 1)) => v as i8,
                other => {
                    return Err(format!(
                        "solved row with game_value {other:?}, expected -1 or 1"
                    ))
                }
            };
            records.push(ProbeRecord {
                key,
                game_value: value,
                status: ProbeStatus::Exact,
                optimal_actions: moves.get(key.as_slice()).copied().unwrap_or(0),
            });
        }
    }

    let book_id = cli.book_id.clone().unwrap_or_else(|| {
        let mut sorted = records.clone();
        sorted.sort_by_key(|r| r.key);
        let mut h = Sha256::new();
        h.update(b"quantik-probe-source-rows-v1");
        for r in &sorted {
            h.update(r.to_bytes());
        }
        let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        format!("sha256:{hex}")
    });
    let meta = ProbeBuildMeta {
        book_id,
        generator: "probe_builder".into(),
        generator_version: env!("CARGO_PKG_VERSION").into(),
        contract_version: cli.contract_version.clone(),
        created_at: Some(chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()),
        // A miss inside the range means "the book had no such solved position"
        // only if no unsolved row was left out.
        coverage_complete: Some(skipped == 0),
    };
    let count = records.len();
    let bytes = build_probe(records, &meta).map_err(|e| e.to_string())?;
    std::fs::write(&cli.out, &bytes).map_err(|e| format!("cannot write {}: {e}", cli.out))?;
    Ok(format!(
        "wrote {} ({} records, {} bytes, {} unsolved rows skipped)",
        cli.out,
        count,
        bytes.len(),
        skipped
    ))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(msg) => {
            println!("{msg}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("probe_builder: {e}");
            ExitCode::FAILURE
        }
    }
}
