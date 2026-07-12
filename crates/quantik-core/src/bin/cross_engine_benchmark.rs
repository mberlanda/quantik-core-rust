//! Cross-engine benchmark CLI (port of `examples/cross_engine_benchmark.py`).
//!
//! Compares MinimaxEngine, MCTSEngine, BeamSearchEngine, and a random-mover
//! baseline on a shared, versioned, checksummed position dataset.
//! See `docs/BENCHMARKS.md`.

use clap::{Parser, Subcommand};
use quantik_core::bench::adapters::{
    fixed_time_adapters, BeamAdapter, EngineAdapter, MCTSAdapter, MinimaxAdapter, RandomAdapter,
};
use quantik_core::bench::agreement::{aggregate_agreement, aggregate_cost, run_agreement, RunKey};
use quantik_core::bench::book_export;
use quantik_core::bench::bundle::{make_bundle, save_bundle};
use quantik_core::bench::checkpoint::{self, CheckpointWriter};
use quantik_core::bench::correctness::run_preflight;
use quantik_core::bench::head_to_head::{aggregate_head_to_head, run_head_to_head, GameKey};
use quantik_core::bench::report::render_markdown;
use quantik_core::bench::stability::aggregate_stability;
use quantik_core::bench::{dataset, reference};
use quantik_core::opening_book::{OpeningBookConfig, OpeningBookDatabase};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "cross_engine_benchmark",
    about = "Reproducible cross-engine benchmark (docs/BENCHMARKS.md)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate the shared position artifact.
    Dataset {
        #[arg(long, default_value_t = 8)]
        opening: u32,
        #[arg(long = "early-mid", default_value_t = 8)]
        early_mid: u32,
        #[arg(long = "late-mid", default_value_t = 12)]
        late_mid: u32,
        #[arg(long, default_value_t = 8)]
        endgame: u32,
        #[arg(long, default_value_t = 20260711)]
        seed: u64,
        /// Max wall-clock seconds to exactly solve each position.
        #[arg(long = "solve-budget", default_value_t = 30.0)]
        solve_budget: f64,
        #[arg(long, default_value = "benchmarks/positions-v1.json")]
        output: PathBuf,
        /// Optional opening-book SQLite path: solved positions short-
        /// circuit repeated solves and are persisted for reuse across runs.
        #[arg(long)]
        book: Option<PathBuf>,
    },
    /// Run a benchmark family.
    Run {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long, value_parser = ["fixed", "native"], default_value = "fixed")]
        family: String,
        /// Fixed family: wall-clock budget per move, seconds.
        #[arg(long = "time-limit", default_value_t = 1.0)]
        time_limit: f64,
        #[arg(long, default_value_t = 10)]
        seeds: u64,
        #[arg(long = "seed-base", default_value_t = 0)]
        seed_base: u64,
        #[arg(long = "minimax-depth", default_value_t = 6)]
        minimax_depth: u32,
        #[arg(long = "minimax-time", default_value_t = 0.2)]
        minimax_time: f64,
        #[arg(long = "mcts-iterations", default_value_t = 1500)]
        mcts_iterations: u32,
        #[arg(long = "mcts-depth", default_value_t = 16)]
        mcts_depth: u32,
        #[arg(long = "mcts-exploration", default_value_t = 1.414)]
        mcts_exploration: f64,
        #[arg(long = "beam-width", default_value_t = 64)]
        beam_width: usize,
        #[arg(long = "beam-depth", default_value_t = 16)]
        beam_depth: u32,
        #[arg(long = "h2h-positions", default_value_t = 8)]
        h2h_positions: usize,
        #[arg(long = "h2h-seeds", default_value_t = 1)]
        h2h_seeds: u64,
        #[arg(long = "skip-h2h", default_value_t = false)]
        skip_h2h: bool,
        #[arg(long)]
        output: PathBuf,
        /// Crash-safe JSON Lines checkpoint: streams each completed
        /// observation/game as it finishes so an interrupted run can be
        /// resumed instead of restarted from scratch.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Resume from an existing --checkpoint file (its header must match
        /// this run's dataset and engine settings). Without this flag, an
        /// existing checkpoint file is refused rather than silently reused
        /// or overwritten.
        #[arg(long, default_value_t = false)]
        resume: bool,
    },
    /// Render a bundle to Markdown.
    Report {
        #[arg(long)]
        input: PathBuf,
        /// Default: <input>.md
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Bulk-export solved references from a dataset artifact into an
    /// opening-book SQLite database.
    ExportBook {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        db: PathBuf,
    },
}

struct RunArgs {
    family: String,
    time_limit: f64,
    minimax_depth: u32,
    minimax_time: f64,
    mcts_iterations: u32,
    mcts_depth: u32,
    mcts_exploration: f64,
    beam_width: usize,
    beam_depth: u32,
}

fn build_adapters(args: &RunArgs) -> Vec<Box<dyn EngineAdapter>> {
    let mut adapters: Vec<Box<dyn EngineAdapter>> = if args.family == "fixed" {
        fixed_time_adapters(args.time_limit, args.beam_width)
    } else {
        vec![
            Box::new(MinimaxAdapter {
                max_depth: args.minimax_depth,
                time_limit_s: Some(args.minimax_time),
            }),
            Box::new(MCTSAdapter {
                max_iterations: args.mcts_iterations,
                max_depth: args.mcts_depth,
                exploration_weight: args.mcts_exploration,
                time_limit_s: None,
            }),
            Box::new(BeamAdapter {
                beam_width: args.beam_width,
                max_depth: args.beam_depth,
                time_limit_s: None,
            }),
        ]
    };
    adapters.push(Box::new(RandomAdapter));
    adapters
}

/// Pick positions round-robin across phase buckets.
fn h2h_positions(payload: &Value, count: usize) -> Vec<Value> {
    let mut by_phase: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for position in payload["positions"].as_array().cloned().unwrap_or_default() {
        by_phase
            .entry(position["phase"].as_str().unwrap_or_default().to_string())
            .or_default()
            .push(position);
    }

    let mut picked = Vec::new();
    while picked.len() < count && by_phase.values().any(|v| !v.is_empty()) {
        for positions in by_phase.values_mut() {
            if !positions.is_empty() && picked.len() < count {
                picked.push(positions.remove(0));
            }
        }
    }
    picked
}

#[allow(clippy::too_many_arguments)]
fn cmd_run(
    dataset_path: &Path,
    args: RunArgs,
    seeds_count: u64,
    seed_base: u64,
    h2h_position_count: usize,
    h2h_seed_count: u64,
    skip_h2h: bool,
    output: &Path,
    checkpoint_path: Option<&Path>,
    resume: bool,
) -> Result<(), String> {
    let payload = dataset::load(dataset_path)?;

    let seeds: Vec<u64> = (0..seeds_count).map(|i| seed_base + i).collect();

    // Config is fixed by the CLI args alone (no run results feed back into
    // it), so it can be assembled up front and fingerprinted for the
    // checkpoint header before any engine work starts.
    let config = json!({
        "dataset": dataset_path.to_string_lossy(),
        "family": args.family,
        "time_limit": args.time_limit,
        "seeds": seeds_count,
        "seed_base": seed_base,
        "minimax_depth": args.minimax_depth,
        "minimax_time": args.minimax_time,
        "mcts_iterations": args.mcts_iterations,
        "mcts_depth": args.mcts_depth,
        "mcts_exploration": args.mcts_exploration,
        "beam_width": args.beam_width,
        "beam_depth": args.beam_depth,
        "h2h_positions": h2h_position_count,
        "h2h_seeds": h2h_seed_count,
        "skip_h2h": skip_h2h,
        "output": output.to_string_lossy(),
        "checkpoint": checkpoint_path.map(|p| p.to_string_lossy().to_string()),
        "engine_seeds": seeds,
    });
    let dataset_checksum = payload["checksum"].as_str().unwrap_or_default().to_string();
    let config_fingerprint = checkpoint::config_fingerprint(&config);

    // Checkpoint validation runs BEFORE preflight so user errors (an
    // existing checkpoint without --resume, a header mismatch, a missing
    // file with --resume) are refused immediately, not after expensive
    // engine work. The writer itself is only opened after preflight passes,
    // so an aborted preflight never leaves a header-only checkpoint behind.
    let mut loaded_rows: Vec<Value> = Vec::new();
    let mut loaded_records: Vec<Value> = Vec::new();
    let mut row_skip: HashSet<RunKey> = HashSet::new();
    let mut record_skip: HashSet<GameKey> = HashSet::new();
    let mut resumed = false;

    if resume && checkpoint_path.is_none() {
        eprintln!("warning: --resume has no effect without --checkpoint <path>; running fresh");
    }
    if let Some(path) = checkpoint_path {
        if resume {
            if !path.exists() {
                return Err(format!(
                    "--resume requires an existing checkpoint at {}; omit --resume to start a \
                     fresh run",
                    path.display()
                ));
            }
            let state = checkpoint::load_checkpoint(path, &dataset_checksum, &config_fingerprint)?;
            loaded_rows = state.rows;
            loaded_records = state.records;
            row_skip = state.row_skip;
            record_skip = state.record_skip;
            resumed = true;
        } else if path.exists() {
            return Err(format!(
                "checkpoint file already exists at {}; pass --resume to continue it, or delete \
                 it to start a fresh run",
                path.display()
            ));
        }
    }

    let adapters = build_adapters(&args);
    let positions = payload["positions"].as_array().cloned().unwrap_or_default();

    let failures = run_preflight(&adapters, &positions);
    if !failures.is_empty() {
        eprintln!("PREFLIGHT FAILED - benchmark aborted:");
        for failure in &failures {
            eprintln!("  - {failure}");
        }
        return Err("preflight failed".into());
    }

    let mut checkpoint_writer: Option<CheckpointWriter> = match (checkpoint_path, resume) {
        (Some(path), true) => Some(CheckpointWriter::resume(path)?),
        (Some(path), false) => Some(CheckpointWriter::create(
            path,
            &dataset_checksum,
            &config_fingerprint,
        )?),
        (None, _) => None,
    };

    let fresh_rows =
        run_agreement(
            &adapters,
            &payload,
            &seeds,
            &row_skip,
            |row| match checkpoint_writer.as_mut() {
                Some(writer) => writer.write_row(row),
                None => Ok(()),
            },
        )?;
    let mut rows = loaded_rows;
    rows.extend(fresh_rows);

    let mut h2h_records: Vec<Value> = Vec::new();
    let mut h2h_aggregates: Vec<Value> = Vec::new();
    if !skip_h2h {
        let sampled = h2h_positions(&payload, h2h_position_count);
        let h2h_seeds: Vec<u64> = (0..h2h_seed_count).map(|i| seed_base + i).collect();
        for i in 0..adapters.len() {
            for j in (i + 1)..adapters.len() {
                let name_i = adapters[i].name();
                let name_j = adapters[j].name();
                let fresh = run_head_to_head(
                    adapters[i].as_ref(),
                    adapters[j].as_ref(),
                    &sampled,
                    &h2h_seeds,
                    &record_skip,
                    |record| match checkpoint_writer.as_mut() {
                        Some(writer) => writer.write_record(record),
                        None => Ok(()),
                    },
                )?;
                // A checkpoint's h2h records span every pairing, streamed
                // in one flat file; attribute the loaded ones back to this
                // specific unordered pair by filtering on {mover,
                // responder}, mirroring how the per-pair aggregate is
                // accumulated below for freshly played games.
                let pair_records: Vec<Value> = loaded_records
                    .iter()
                    .filter(|record| {
                        let mover = record["mover"].as_str().unwrap_or_default();
                        let responder = record["responder"].as_str().unwrap_or_default();
                        (mover == name_i && responder == name_j)
                            || (mover == name_j && responder == name_i)
                    })
                    .cloned()
                    .chain(fresh.iter().cloned())
                    .collect();
                h2h_aggregates.push(aggregate_head_to_head(&pair_records, name_i, name_j));
                h2h_records.extend(fresh);
            }
        }
        h2h_records.extend(loaded_records);
    }

    let games = h2h_records.len();
    let mut bundle = make_bundle(
        config,
        &payload,
        rows.clone(),
        json!({"records": h2h_records, "aggregates": h2h_aggregates}),
        json!({
            "agreement": aggregate_agreement(&rows),
            "cost": aggregate_cost(&rows),
            "stability": aggregate_stability(&rows),
        }),
    );
    bundle["resumed"] = json!(resumed);
    save_bundle(&bundle, output)?;
    println!(
        "bundle: {} observations, {} games -> {}{}",
        rows.len(),
        games,
        output.display(),
        if resumed { " (resumed)" } else { "" }
    );
    Ok(())
}

fn open_book(path: &Path) -> Result<OpeningBookDatabase, String> {
    OpeningBookDatabase::open(&OpeningBookConfig {
        database_path: path.to_string_lossy().to_string(),
        ..Default::default()
    })
    .map_err(|e| format!("open opening book {path:?}: {e}"))
}

fn cmd_dataset(
    requested: BTreeMap<String, u32>,
    seed: u64,
    solve_budget: f64,
    output: &Path,
    book_path: Option<&Path>,
) -> Result<(), String> {
    let mut payload = dataset::generate(&requested, seed)?;
    let book = book_path.map(open_book).transpose()?;
    reference::augment_with_references_with_book(&mut payload, solve_budget, book.as_ref());
    let digest = dataset::save(&payload, output)?;

    let positions = payload["positions"].as_array().cloned().unwrap_or_default();
    let solved = positions
        .iter()
        .filter(|p| !p["reference"].is_null())
        .count();
    println!(
        "dataset: {} positions ({} with exact references) -> {}",
        positions.len(),
        solved,
        output.display()
    );
    println!("checksum: {digest}");
    for (phase, _) in dataset::PHASES {
        let phase_positions: Vec<&Value> = positions
            .iter()
            .filter(|p| p["phase"].as_str() == Some(phase))
            .collect();
        let phase_solved = phase_positions
            .iter()
            .filter(|p| !p["reference"].is_null())
            .count();
        println!(
            "  {phase:9}: {} positions, {phase_solved} solved",
            phase_positions.len()
        );
    }
    Ok(())
}

fn cmd_report(input: &Path, output: Option<PathBuf>) -> Result<(), String> {
    let text = std::fs::read_to_string(input).map_err(|e| format!("read {input:?}: {e}"))?;
    let bundle: Value = serde_json::from_str(&text).map_err(|e| format!("parse: {e}"))?;
    let output = output.unwrap_or_else(|| input.with_extension("md"));
    std::fs::write(&output, render_markdown(&bundle))
        .map_err(|e| format!("write {output:?}: {e}"))?;
    println!("report -> {}", output.display());
    Ok(())
}

fn cmd_export_book(input: &Path, db_path: &Path) -> Result<(), String> {
    let payload = dataset::load(input)?;
    let db = open_book(db_path)?;
    let inserted = book_export::export_references(&payload, &db)?;
    println!(
        "export-book: {inserted} solved references -> {}",
        db_path.display()
    );
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Dataset {
            opening,
            early_mid,
            late_mid,
            endgame,
            seed,
            solve_budget,
            output,
            book,
        } => {
            let requested = BTreeMap::from([
                ("opening".to_string(), opening),
                ("early_mid".to_string(), early_mid),
                ("late_mid".to_string(), late_mid),
                ("endgame".to_string(), endgame),
            ]);
            cmd_dataset(requested, seed, solve_budget, &output, book.as_deref())
        }
        Commands::Run {
            dataset,
            family,
            time_limit,
            seeds,
            seed_base,
            minimax_depth,
            minimax_time,
            mcts_iterations,
            mcts_depth,
            mcts_exploration,
            beam_width,
            beam_depth,
            h2h_positions,
            h2h_seeds,
            skip_h2h,
            output,
            checkpoint,
            resume,
        } => cmd_run(
            &dataset,
            RunArgs {
                family,
                time_limit,
                minimax_depth,
                minimax_time,
                mcts_iterations,
                mcts_depth,
                mcts_exploration,
                beam_width,
                beam_depth,
            },
            seeds,
            seed_base,
            h2h_positions,
            h2h_seeds,
            skip_h2h,
            &output,
            checkpoint.as_deref(),
            resume,
        ),
        Commands::Report { input, output } => cmd_report(&input, output),
        Commands::ExportBook { input, db } => cmd_export_book(&input, &db),
    };

    if let Err(message) = result {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}
