//! Cross-engine benchmark CLI (port of `examples/cross_engine_benchmark.py`).
//!
//! Compares MinimaxEngine, MCTSEngine, BeamSearchEngine, and a random-mover
//! baseline on a shared, versioned, checksummed position dataset.
//! See `docs/BENCHMARKS.md`.

use clap::{Parser, Subcommand};
use quantik_core::bench::adapters::{
    BeamAdapter, EngineAdapter, MCTSAdapter, MinimaxAdapter, RandomAdapter,
};
use quantik_core::bench::agreement::{
    aggregate_agreement, aggregate_cost, observation_key, run_agreement, ObservationKey,
};
use quantik_core::bench::book_export;
use quantik_core::bench::bundle::{make_bundle, save_bundle};
use quantik_core::bench::checkpoint::{
    append_jsonl, build_manifest, bundle_from_checkpoint, checkpoint_paths, key_set, load_jsonl,
    load_manifest, update_manifest_counts, validate_resume_manifest, write_manifest,
};
use quantik_core::bench::contracts::{export_game_result_rows, export_observation_rows};
use quantik_core::bench::correctness::run_preflight;
use quantik_core::bench::head_to_head::{
    aggregate_head_to_head, h2h_key, run_head_to_head, H2hKey,
};
use quantik_core::bench::report::render_markdown;
use quantik_core::bench::stability::aggregate_stability;
use quantik_core::bench::{dataset, reference};
use quantik_core::opening_book::{OpeningBookConfig, OpeningBookDatabase};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::io::Write as _;
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
        /// Comma-separated engines to include, in order. Supported:
        /// minimax (alias: minmax), mcts, beam, random.
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "minimax,mcts,beam,random"
        )]
        engines: Vec<String>,
        #[arg(long = "h2h-positions", default_value_t = 8)]
        h2h_positions: usize,
        #[arg(long = "h2h-seeds", default_value_t = 1)]
        h2h_seeds: u64,
        #[arg(long = "skip-h2h", default_value_t = false)]
        skip_h2h: bool,
        #[arg(long)]
        output: PathBuf,
        /// Crash-safe directory checkpoint (Python-compatible layout:
        /// `manifest.json` + `observations.jsonl` + `h2h.jsonl`): streams
        /// each completed observation/game as it finishes so an
        /// interrupted run can be resumed instead of restarted from
        /// scratch, and a `report --input <dir>` can render a partial
        /// state at any time.
        #[arg(long = "checkpoint-dir")]
        checkpoint_dir: Option<PathBuf>,
        /// Resume from an existing --checkpoint-dir (its manifest must
        /// match this run's dataset checksum and normalized config).
        /// Without this flag, --checkpoint-dir always starts fresh: the
        /// directory is created (or its observations.jsonl/h2h.jsonl
        /// truncated and manifest overwritten) unconditionally — matching
        /// the Python harness, there is no "refuse to clobber" guard here,
        /// so pass --resume whenever you mean to continue prior work.
        #[arg(long, default_value_t = false)]
        resume: bool,
        /// Update the checkpoint manifest, and print a progress line,
        /// every N completed observation rows / h2h games. 0 disables the
        /// periodic update (the manifest still updates once per phase).
        #[arg(long = "checkpoint-every", default_value_t = 1)]
        checkpoint_every: u64,
        /// Parallel worker threads for agreement observations and h2h
        /// games. Must be at least 1. Regardless of worker count, results
        /// are produced in the same task order and are byte-identical to
        /// `--workers 1`.
        #[arg(long, default_value_t = 1)]
        workers: usize,
    },
    /// Render a bundle to Markdown. `--input` may be a bundle JSON file or
    /// a `--checkpoint-dir` directory (rehydrated via
    /// `bundle_from_checkpoint`, so a partial/in-progress checkpoint
    /// reports fine).
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
    /// Project benchmark bundle/checkpoint observations to observation.v1 JSONL.
    ExportObservations {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Project benchmark bundle/checkpoint h2h games to game-result.v1 JSONL.
    ExportGames {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        output: PathBuf,
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
    engines: Vec<String>,
}

fn normalize_engine_name(name: &str) -> Option<&'static str> {
    match name.trim().to_ascii_lowercase().as_str() {
        "minimax" | "minmax" => Some("minimax"),
        "mcts" => Some("mcts"),
        "beam" | "beam_search" | "beam-search" => Some("beam"),
        "random" | "baseline" => Some("random"),
        _ => None,
    }
}

fn build_adapter(name: &str, args: &RunArgs) -> Box<dyn EngineAdapter> {
    match name {
        "minimax" if args.family == "fixed" => Box::new(MinimaxAdapter {
            max_depth: 16,
            time_limit_s: Some(args.time_limit),
        }),
        "mcts" if args.family == "fixed" => Box::new(MCTSAdapter {
            max_iterations: 10_000_000,
            max_depth: 16,
            exploration_weight: std::f64::consts::SQRT_2,
            time_limit_s: Some(args.time_limit),
        }),
        "beam" if args.family == "fixed" => Box::new(BeamAdapter {
            beam_width: args.beam_width,
            max_depth: 16,
            time_limit_s: Some(args.time_limit),
        }),
        "minimax" => Box::new(MinimaxAdapter {
            max_depth: args.minimax_depth,
            time_limit_s: Some(args.minimax_time),
        }),
        "mcts" => Box::new(MCTSAdapter {
            max_iterations: args.mcts_iterations,
            max_depth: args.mcts_depth,
            exploration_weight: args.mcts_exploration,
            time_limit_s: None,
        }),
        "beam" => Box::new(BeamAdapter {
            beam_width: args.beam_width,
            max_depth: args.beam_depth,
            time_limit_s: None,
        }),
        "random" => Box::new(RandomAdapter),
        _ => unreachable!("validated engine name"),
    }
}

fn build_adapters(args: &RunArgs) -> Result<Vec<Box<dyn EngineAdapter>>, String> {
    let requested = if args.engines.is_empty() {
        vec!["minimax", "mcts", "beam", "random"]
    } else {
        let mut normalized = Vec::new();
        for engine in &args.engines {
            let Some(name) = normalize_engine_name(engine) else {
                return Err(format!(
                    "unknown engine {engine:?}; supported engines: minimax, mcts, beam, random"
                ));
            };
            if normalized.contains(&name) {
                return Err(format!("duplicate engine {name:?}"));
            }
            normalized.push(name);
        }
        normalized
    };

    Ok(requested
        .into_iter()
        .map(|name| build_adapter(name, args))
        .collect())
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

/// Total observation rows a full (non-resumed) run would produce:
/// stochastic adapters run once per seed, deterministic ones once.
fn expected_observations(
    adapters: &[Box<dyn EngineAdapter>],
    positions: usize,
    seeds: usize,
) -> usize {
    let per_position: usize = adapters
        .iter()
        .map(|a| if a.stochastic() { seeds } else { 1 })
        .sum();
    per_position * positions
}

/// Total head-to-head records a full (non-resumed) run would produce:
/// every unordered adapter pair, both orientations, per position and seed.
fn expected_h2h_records(
    adapters: &[Box<dyn EngineAdapter>],
    positions: usize,
    seeds: usize,
) -> usize {
    let n = adapters.len();
    let pair_count = n.saturating_sub(1) * n / 2;
    pair_count * positions * seeds * 2
}

/// Print a progress line to stdout, flushed immediately (mirrors Python's
/// `print(..., flush=True)` — needed because stdout is fully buffered,
/// not line-buffered, when not attached to a terminal).
fn print_progress(message: &str) {
    println!("{message}");
    let _ = std::io::stdout().flush();
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
    checkpoint_dir: Option<&Path>,
    resume: bool,
    checkpoint_every: u64,
    workers: usize,
) -> Result<(), String> {
    if workers < 1 {
        return Err("workers must be at least 1".into());
    }

    let payload = dataset::load(dataset_path)?;
    let seeds: Vec<u64> = (0..seeds_count).map(|i| seed_base + i).collect();
    let adapters = build_adapters(&args)?;
    if !skip_h2h && adapters.len() < 2 {
        return Err("head-to-head requires at least two selected engines".into());
    }
    let engine_names: Vec<&str> = adapters.iter().map(|adapter| adapter.name()).collect();
    let positions = payload["positions"].as_array().cloned().unwrap_or_default();
    let sampled_h2h_positions = h2h_positions(&payload, h2h_position_count);
    let h2h_seeds: Vec<u64> = (0..h2h_seed_count).map(|i| seed_base + i).collect();

    // Config is fixed by the CLI args alone, so it can be assembled up
    // front and used both for the checkpoint manifest and the final
    // bundle. `checkpoint_dir`/`output`/`resume`/`workers` (and, were it
    // implemented, `skip_agreement`) are excluded from resume validation
    // by `checkpoint::normalize_run_config` — see docs/BENCHMARKS.md.
    let run_config = json!({
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
        "engines": engine_names,
        "h2h_positions": h2h_position_count,
        "h2h_seeds": h2h_seed_count,
        "skip_h2h": skip_h2h,
        "checkpoint_dir": checkpoint_dir.map(|p| p.to_string_lossy().to_string()),
        "resume": resume,
        "checkpoint_every": checkpoint_every,
        "workers": workers,
        "output": output.to_string_lossy(),
        "engine_seeds": seeds,
    });
    let dataset_checksum = payload["checksum"].as_str().map(str::to_string);

    let paths = checkpoint_dir.map(checkpoint_paths);

    // Checkpoint setup/validation runs BEFORE preflight so user errors (an
    // existing checkpoint without --resume, a missing/mismatched manifest)
    // are refused immediately, not after expensive engine work.
    if let Some(paths) = &paths {
        if resume {
            if !paths.manifest.exists() {
                return Err(format!(
                    "RESUME FAILED - checkpoint manifest not found: {}",
                    paths.manifest.display()
                ));
            }
            let manifest = load_manifest(&paths.manifest)?;
            let allow_skip_h2h_mismatch = if skip_h2h {
                let existing_records = load_jsonl(&paths.h2h)?;
                let expected_h2h =
                    expected_h2h_records(&adapters, sampled_h2h_positions.len(), h2h_seeds.len());
                if existing_records.len() != expected_h2h {
                    return Err(format!(
                        "RESUME FAILED - checkpoint h2h records incomplete: expected {expected_h2h}, found {}",
                        existing_records.len()
                    ));
                }
                true
            } else {
                false
            };
            validate_resume_manifest(
                &manifest,
                dataset_checksum.as_deref(),
                &run_config,
                allow_skip_h2h_mismatch,
            )
            .map_err(|e| format!("RESUME FAILED - {e}"))?;
        } else {
            std::fs::create_dir_all(&paths.root)
                .map_err(|e| format!("mkdir {:?}: {e}", paths.root))?;
            std::fs::write(&paths.observations, "")
                .map_err(|e| format!("truncate {:?}: {e}", paths.observations))?;
            std::fs::write(&paths.h2h, "").map_err(|e| format!("truncate {:?}: {e}", paths.h2h))?;
            let manifest = build_manifest(&run_config, &payload, "preflight", 0, 0);
            write_manifest(&paths.manifest, &manifest)?;
        }
    }

    print_progress(&format!(
        "preflight: checking {} adapters across {} sample positions",
        adapters.len(),
        positions.len().min(3)
    ));
    let failures = run_preflight(&adapters, &positions);
    if !failures.is_empty() {
        if let Some(paths) = &paths {
            if paths.manifest.exists() {
                let obs_count = load_jsonl(&paths.observations)?.len() as u64;
                let h2h_count = load_jsonl(&paths.h2h)?.len() as u64;
                update_manifest_counts(
                    &paths.manifest,
                    obs_count,
                    h2h_count,
                    Some("preflight_failed"),
                )?;
            }
        }
        eprintln!("PREFLIGHT FAILED - benchmark aborted:");
        for failure in &failures {
            eprintln!("  - {failure}");
        }
        return Err("preflight failed".into());
    }
    print_progress("preflight: passed");

    let result: Value = match &paths {
        None => {
            print_progress(&format!(
                "agreement: running {} positions with {} seed(s), workers={workers}",
                positions.len(),
                seeds.len()
            ));
            let rows = run_agreement(
                &adapters,
                &payload,
                &seeds,
                &HashSet::new(),
                workers,
                |_| Ok(()),
            )?;

            let mut h2h_records: Vec<Value> = Vec::new();
            let mut h2h_aggregates: Vec<Value> = Vec::new();
            if !skip_h2h {
                print_progress(&format!(
                    "h2h: running {} positions with {} seed(s), workers={workers}",
                    sampled_h2h_positions.len(),
                    h2h_seeds.len()
                ));
                for i in 0..adapters.len() {
                    for j in (i + 1)..adapters.len() {
                        let name_i = adapters[i].name();
                        let name_j = adapters[j].name();
                        let records = run_head_to_head(
                            adapters[i].as_ref(),
                            adapters[j].as_ref(),
                            &sampled_h2h_positions,
                            &h2h_seeds,
                            &HashSet::new(),
                            workers,
                            |_| Ok(()),
                        )?;
                        h2h_aggregates.push(aggregate_head_to_head(&records, name_i, name_j));
                        h2h_records.extend(records);
                    }
                }
            }

            make_bundle(
                run_config,
                &payload,
                rows.clone(),
                json!({"records": h2h_records, "aggregates": h2h_aggregates}),
                json!({
                    "agreement": aggregate_agreement(&rows),
                    "cost": aggregate_cost(&rows),
                    "stability": aggregate_stability(&rows),
                }),
            )
        }
        Some(paths) => {
            let existing_rows = load_jsonl(&paths.observations)?;
            if skip_h2h && !resume {
                std::fs::write(&paths.h2h, "")
                    .map_err(|e| format!("truncate {:?}: {e}", paths.h2h))?;
            }
            let existing_records = load_jsonl(&paths.h2h)?;

            let observation_skips: HashSet<ObservationKey> =
                key_set(&existing_rows, observation_key);
            let h2h_skips: HashSet<H2hKey> = key_set(&existing_records, h2h_key);

            update_manifest_counts(
                &paths.manifest,
                existing_rows.len() as u64,
                existing_records.len() as u64,
                Some("running"),
            )?;

            let mut completed_observations = existing_rows.len() as u64;
            let total_observations = expected_observations(&adapters, positions.len(), seeds.len());
            print_progress(&format!(
                "agreement: {completed_observations}/{total_observations} observations complete; \
                 workers={workers}; checkpoint {}",
                paths.observations.display()
            ));

            run_agreement(
                &adapters,
                &payload,
                &seeds,
                &observation_skips,
                workers,
                |row| {
                    append_jsonl(&paths.observations, row)?;
                    completed_observations += 1;
                    if checkpoint_every > 0
                        && completed_observations.is_multiple_of(checkpoint_every)
                    {
                        update_manifest_counts(
                            &paths.manifest,
                            completed_observations,
                            existing_records.len() as u64,
                            Some("running"),
                        )?;
                        print_progress(&format!(
                            "agreement: {completed_observations}/{total_observations} \
                             observations checkpointed"
                        ));
                    }
                    Ok(())
                },
            )?;

            let mut completed_h2h = existing_records.len() as u64;
            if !skip_h2h {
                let total_h2h =
                    expected_h2h_records(&adapters, sampled_h2h_positions.len(), h2h_seeds.len());
                print_progress(&format!(
                    "h2h: {completed_h2h}/{total_h2h} games complete; workers={workers}; \
                     checkpoint {}",
                    paths.h2h.display()
                ));
                for i in 0..adapters.len() {
                    for j in (i + 1)..adapters.len() {
                        run_head_to_head(
                            adapters[i].as_ref(),
                            adapters[j].as_ref(),
                            &sampled_h2h_positions,
                            &h2h_seeds,
                            &h2h_skips,
                            workers,
                            |record| {
                                append_jsonl(&paths.h2h, record)?;
                                completed_h2h += 1;
                                if checkpoint_every > 0
                                    && completed_h2h.is_multiple_of(checkpoint_every)
                                {
                                    update_manifest_counts(
                                        &paths.manifest,
                                        completed_observations,
                                        completed_h2h,
                                        Some("running"),
                                    )?;
                                    print_progress(&format!(
                                        "h2h: {completed_h2h}/{total_h2h} games checkpointed"
                                    ));
                                }
                                Ok(())
                            },
                        )?;
                    }
                }
            }

            update_manifest_counts(
                &paths.manifest,
                completed_observations,
                completed_h2h,
                Some("complete"),
            )?;
            bundle_from_checkpoint(&paths.root)?
        }
    };

    save_bundle(&result, output)?;
    println!(
        "bundle: {} observations, {} games -> {}",
        result["observations"].as_array().map_or(0, Vec::len),
        result["head_to_head"]["records"]
            .as_array()
            .map_or(0, Vec::len),
        output.display()
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
    let bundle: Value = if input.is_dir() {
        bundle_from_checkpoint(input)?
    } else {
        let text = std::fs::read_to_string(input).map_err(|e| format!("read {input:?}: {e}"))?;
        serde_json::from_str(&text).map_err(|e| format!("parse: {e}"))?
    };
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

fn load_bundle_input(input: &Path) -> Result<Value, String> {
    if input.is_dir() {
        bundle_from_checkpoint(input)
    } else {
        let text = std::fs::read_to_string(input).map_err(|e| format!("read {input:?}: {e}"))?;
        serde_json::from_str(&text).map_err(|e| format!("parse: {e}"))
    }
}

fn cmd_export_observations(input: &Path, dataset_path: &Path, output: &Path) -> Result<(), String> {
    let bundle = load_bundle_input(input)?;
    let dataset_payload = dataset::load(dataset_path)?;
    let count = export_observation_rows(&bundle, &dataset_payload, output)?;
    println!("export-observations: {count} rows -> {}", output.display());
    Ok(())
}

fn cmd_export_games(input: &Path, dataset_path: &Path, output: &Path) -> Result<(), String> {
    let bundle = load_bundle_input(input)?;
    let dataset_payload = dataset::load(dataset_path)?;
    let count = export_game_result_rows(&bundle, &dataset_payload, output)?;
    println!("export-games: {count} rows -> {}", output.display());
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
            engines,
            h2h_positions,
            h2h_seeds,
            skip_h2h,
            output,
            checkpoint_dir,
            resume,
            checkpoint_every,
            workers,
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
                engines,
            },
            seeds,
            seed_base,
            h2h_positions,
            h2h_seeds,
            skip_h2h,
            &output,
            checkpoint_dir.as_deref(),
            resume,
            checkpoint_every,
            workers,
        ),
        Commands::Report { input, output } => cmd_report(&input, output),
        Commands::ExportBook { input, db } => cmd_export_book(&input, &db),
        Commands::ExportObservations {
            input,
            dataset,
            output,
        } => cmd_export_observations(&input, &dataset, &output),
        Commands::ExportGames {
            input,
            dataset,
            output,
        } => cmd_export_games(&input, &dataset, &output),
    };

    if let Err(message) = result {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_selection_accepts_aliases_and_preserves_requested_order() {
        let adapters = build_adapters(&RunArgs {
            family: "native".to_string(),
            time_limit: 1.0,
            minimax_depth: 4,
            minimax_time: 0.1,
            mcts_iterations: 10,
            mcts_depth: 4,
            mcts_exploration: 1.414,
            beam_width: 8,
            beam_depth: 4,
            engines: vec!["mcts".to_string(), "minmax".to_string()],
        })
        .unwrap();

        let names: Vec<&str> = adapters.iter().map(|adapter| adapter.name()).collect();
        assert_eq!(names, vec!["mcts", "minimax"]);
    }

    #[test]
    fn engine_selection_rejects_unknown_engines() {
        let result = build_adapters(&RunArgs {
            family: "native".to_string(),
            time_limit: 1.0,
            minimax_depth: 4,
            minimax_time: 0.1,
            mcts_iterations: 10,
            mcts_depth: 4,
            mcts_exploration: 1.414,
            beam_width: 8,
            beam_depth: 4,
            engines: vec!["mcts".to_string(), "quantum".to_string()],
        });
        let err = match result {
            Ok(_) => panic!("unknown engine should be rejected"),
            Err(err) => err,
        };

        assert!(err.contains("unknown engine"), "{err}");
        assert!(err.contains("minimax"), "{err}");
        assert!(err.contains("mcts"), "{err}");
        assert!(err.contains("beam"), "{err}");
    }
}
