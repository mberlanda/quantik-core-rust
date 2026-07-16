//! Draft search-telemetry exporter: runs the MCTS, minimax, and beam
//! engines against a handful of fixed positions and writes one
//! `search-summary.v1-draft` JSONL row (see `bench::contracts::search_summary_row`)
//! per completed root search whose root identity was preserved. A row is
//! legitimately skipped (not an error) whenever canonical/transposition
//! merging collapsed distinct root moves onto shared statistics — see the
//! `search_telemetry` module's Root Identity docs.
//!
//! Every engine call is seeded so the output is reproducible run to run.
//!
//! Usage:
//!   cargo run -p quantik-core --example search_summary_export -- \
//!     --out search-summaries.jsonl

use quantik_core::beam_search::{BeamSearchConfig, BeamSearchEngine};
use quantik_core::bench::canonical::canonical_json;
use quantik_core::bench::contracts::{search_summary_row, SearchSummaryRunConfig};
use quantik_core::mcts::{MCTSConfig, MCTSEngine};
use quantik_core::minimax::{MinimaxConfig, MinimaxEngine};
use quantik_core::state::State;
use std::io::Write;

const SEED: u64 = 20260716;
const RUN_ID: &str = "search-summary-export";

/// Fixed positions to export telemetry for: the empty board plus two
/// known-valid mid-game positions reused from existing fixtures
/// (`qfen.rs`'s `mixed_position` QFEN test and
/// `tests/portability_report.rs`'s contract-shape fixture).
const POSITIONS: &[(&str, &str)] = &[
    ("empty", "..../..../..../...."),
    ("mid-6ply", "A.bC/..../d..B/...a"),
    ("mid-4ply", "Ab../..c./...D/...."),
];

struct Args {
    out: String,
}

fn parse_args() -> Args {
    let mut out = "search-summaries.jsonl".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--out" => out = it.next().expect("--out requires a path"),
            other => panic!("unknown flag {other}"),
        }
    }
    Args { out }
}

/// Write one row via `search_summary_row`, appending it to `file` on
/// `Ok(Some(_))`, printing a note and doing nothing on `Ok(None)` (skipped
/// per root-identity rules), and panicking on `Err` (a real contract
/// violation, not an expected skip).
#[allow(clippy::too_many_arguments)]
fn emit_row(
    file: &mut std::fs::File,
    row_id: &mut u64,
    rows_written: &mut usize,
    label: &str,
    engine_name: &str,
    qfen: &str,
    telemetry: &quantik_core::search_telemetry::SearchTelemetry,
    run_config: &SearchSummaryRunConfig,
) {
    match search_summary_row(*row_id, RUN_ID, qfen, telemetry, run_config) {
        Ok(Some(row)) => {
            writeln!(file, "{}", canonical_json(&row)).expect("write row");
            *row_id += 1;
            *rows_written += 1;
        }
        Ok(None) => {
            eprintln!("[{label}] {engine_name}: root identity not preserved, skipping row");
        }
        Err(e) => panic!("[{label}] {engine_name} row error: {e}"),
    }
}

fn main() {
    let args = parse_args();
    if let Some(parent) = std::path::Path::new(&args.out).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("mkdir output dir");
        }
    }
    let mut file = std::fs::File::create(&args.out).expect("create output file");

    let mut row_id = 0u64;
    let mut rows_written = 0usize;

    for (label, qfen) in POSITIONS {
        let bb = State::from_qfen(qfen)
            .expect("fixture qfen must be valid")
            .bb;

        // MCTS: use_transposition_table MUST be false, or symmetric root
        // moves collapse onto shared statistics and root_identity_preserved
        // goes false (see examples/selfplay_export.rs for the same
        // caveat, verified by mcts.rs's
        // root_move_visits_default_config_collapses_symmetric_root_moves).
        let mut mcts = MCTSEngine::new(MCTSConfig {
            max_iterations: 200,
            seed: Some(SEED),
            use_transposition_table: false,
            ..Default::default()
        });
        if mcts.search(&bb).is_some() {
            let telemetry = mcts.telemetry().expect("search just completed");
            let run_config = SearchSummaryRunConfig {
                config_label: "mcts-default",
                search_depth: None,
                rollouts: Some(200),
                beam_width: None,
                node_budget: None,
                time_budget_ms: None,
            };
            emit_row(
                &mut file,
                &mut row_id,
                &mut rows_written,
                label,
                "mcts",
                qfen,
                &telemetry,
                &run_config,
            );
        } else {
            eprintln!("[{label}] mcts: no legal moves, skipping");
        }

        // Minimax: dedup_children MUST be false, or symmetric siblings
        // merge onto shared statistics and root_identity_preserved goes
        // false.
        let mut minimax = MinimaxEngine::new(MinimaxConfig {
            max_depth: 4,
            dedup_children: false,
            random_seed: Some(SEED),
            ..Default::default()
        });
        match minimax.search(&State::new(bb)) {
            Ok(_) => {
                let telemetry = minimax.telemetry().expect("search just completed");
                let run_config = SearchSummaryRunConfig {
                    config_label: "minimax-depth4",
                    search_depth: Some(4),
                    rollouts: None,
                    beam_width: None,
                    node_budget: None,
                    time_budget_ms: None,
                };
                emit_row(
                    &mut file,
                    &mut row_id,
                    &mut rows_written,
                    label,
                    "minimax",
                    qfen,
                    &telemetry,
                    &run_config,
                );
            }
            Err(e) => eprintln!("[{label}] minimax: search failed ({e}), skipping"),
        }

        // Beam: default configuration plus a fixed seed. Any depth-1
        // canonical dedup makes root_identity_preserved false; that is a
        // legitimate, expected skip for this engine, not an error.
        let beam_config = BeamSearchConfig {
            random_seed: Some(SEED),
            ..Default::default()
        };
        let beam_width = beam_config.beam_width as u64;
        let rollouts_per_candidate = beam_config.rollouts_per_candidate as u64;
        let max_depth = beam_config.max_depth;
        let mut beam = BeamSearchEngine::new(beam_config).expect("valid beam config");
        match beam.search(&bb) {
            Ok(result) => {
                let telemetry = beam.telemetry(&result);
                let run_config = SearchSummaryRunConfig {
                    config_label: "beam-default",
                    search_depth: Some(max_depth),
                    rollouts: Some(rollouts_per_candidate),
                    beam_width: Some(beam_width),
                    node_budget: None,
                    time_budget_ms: None,
                };
                emit_row(
                    &mut file,
                    &mut row_id,
                    &mut rows_written,
                    label,
                    "beam",
                    qfen,
                    &telemetry,
                    &run_config,
                );
            }
            Err(e) => eprintln!("[{label}] beam: search failed ({e}), skipping"),
        }
    }

    println!("{rows_written} rows exported -> {}", args.out);
}
