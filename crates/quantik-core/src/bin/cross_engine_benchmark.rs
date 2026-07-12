//! Cross-engine benchmark CLI (port of `examples/cross_engine_benchmark.py`).
//!
//! Compares MinimaxEngine, MCTSEngine, BeamSearchEngine, and a random-mover
//! baseline on a shared, versioned, checksummed position dataset.
//! See `docs/BENCHMARKS.md`.

use clap::{Parser, Subcommand};
use quantik_core::bench::adapters::{
    fixed_time_adapters, BeamAdapter, EngineAdapter, MCTSAdapter, MinimaxAdapter, RandomAdapter,
};
use quantik_core::bench::agreement::{aggregate_agreement, aggregate_cost, run_agreement};
use quantik_core::bench::bundle::{make_bundle, save_bundle};
use quantik_core::bench::correctness::run_preflight;
use quantik_core::bench::head_to_head::{aggregate_head_to_head, run_head_to_head};
use quantik_core::bench::report::render_markdown;
use quantik_core::bench::stability::aggregate_stability;
use quantik_core::bench::{dataset, reference};
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
    },
    /// Render a bundle to Markdown.
    Report {
        #[arg(long)]
        input: PathBuf,
        /// Default: <input>.md
        #[arg(long)]
        output: Option<PathBuf>,
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
) -> Result<(), String> {
    let payload = dataset::load(dataset_path)?;
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

    let seeds: Vec<u64> = (0..seeds_count).map(|i| seed_base + i).collect();
    let rows = run_agreement(&adapters, &payload, &seeds, &HashSet::new(), |_| {})?;

    let mut h2h_records: Vec<Value> = Vec::new();
    let mut h2h_aggregates: Vec<Value> = Vec::new();
    if !skip_h2h {
        let sampled = h2h_positions(&payload, h2h_position_count);
        let h2h_seeds: Vec<u64> = (0..h2h_seed_count).map(|i| seed_base + i).collect();
        for i in 0..adapters.len() {
            for j in (i + 1)..adapters.len() {
                let records = run_head_to_head(
                    adapters[i].as_ref(),
                    adapters[j].as_ref(),
                    &sampled,
                    &h2h_seeds,
                    &HashSet::new(),
                    |_| {},
                )?;
                h2h_aggregates.push(aggregate_head_to_head(
                    &records,
                    adapters[i].name(),
                    adapters[j].name(),
                ));
                h2h_records.extend(records);
            }
        }
    }

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
        "engine_seeds": seeds,
    });

    let games = h2h_records.len();
    let bundle = make_bundle(
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
    save_bundle(&bundle, output)?;
    println!(
        "bundle: {} observations, {} games -> {}",
        rows.len(),
        games,
        output.display()
    );
    Ok(())
}

fn cmd_dataset(
    requested: BTreeMap<String, u32>,
    seed: u64,
    solve_budget: f64,
    output: &Path,
) -> Result<(), String> {
    let mut payload = dataset::generate(&requested, seed)?;
    reference::augment_with_references(&mut payload, solve_budget);
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
        } => {
            let requested = BTreeMap::from([
                ("opening".to_string(), opening),
                ("early_mid".to_string(), early_mid),
                ("late_mid".to_string(), late_mid),
                ("endgame".to_string(), endgame),
            ]);
            cmd_dataset(requested, seed, solve_budget, &output)
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
        ),
        Commands::Report { input, output } => cmd_report(&input, output),
    };

    if let Err(message) = result {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}
