use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate is under <repo>/crates/quantik-core")
        .to_path_buf()
}

fn run_script(script: &str, args: &[&str]) -> (bool, String) {
    let output = Command::new("bash")
        .arg(repo_root().join("scripts").join(script))
        .args(args)
        .output()
        .expect("script should be runnable by bash");
    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.success(), text)
}

fn run_bash(command: &str) -> (bool, String) {
    let output = Command::new("bash")
        .arg("--noprofile")
        .arg("--norc")
        .arg("-c")
        .arg(command)
        .current_dir(repo_root())
        .output()
        .expect("bash command should run");
    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.success(), text)
}

#[test]
fn benchmark_scripts_have_help() {
    for script in [
        "generate_positions.sh",
        "generate_opening_book.sh",
        "generate_observations.sh",
        "generate_h2h_stats.sh",
        "export_contract_rows.sh",
        "plan_runs.sh",
    ] {
        let (success, text) = run_script(script, &["--help"]);
        assert!(success, "{script} --help failed:\n{text}");
        assert!(text.contains("Usage:"), "{script} missing usage:\n{text}");
    }
}

#[test]
fn dry_run_contract_export_renders_projection_commands() {
    let (success, text) = run_script(
        "export_contract_rows.sh",
        &[
            "--input",
            "benchmarks/results/dev-ckpt",
            "--dataset",
            "benchmarks/positions-v1.json",
            "--observations-output",
            "benchmarks/results/observations-v1.jsonl",
            "--games-output",
            "benchmarks/results/games-v1.jsonl",
            "--dry-run",
        ],
    );

    assert!(success, "dry run failed:\n{text}");
    assert!(text.contains("export-observations"), "{text}");
    assert!(text.contains("export-games"), "{text}");
    assert!(
        text.contains("--dataset benchmarks/positions-v1.json"),
        "{text}"
    );
}

#[test]
fn plan_runs_calculates_h2h_position_seed_combinations() {
    let (success, text) = run_script(
        "plan_runs.sh",
        &[
            "h2h-games",
            "--games",
            "1000",
            "--engines",
            "mcts,minmax",
            "--positions",
            "50",
        ],
    );

    assert!(success, "plan failed:\n{text}");
    assert!(text.contains("engines=mcts,minimax"), "{text}");
    assert!(text.contains("h2h_positions=50"), "{text}");
    assert!(text.contains("h2h_seeds=10"), "{text}");
    assert!(text.contains("planned_games=1000"), "{text}");
}

#[test]
fn plan_runs_matrix_rejects_single_engine_sets() {
    let (success, text) = run_script(
        "plan_runs.sh",
        &[
            "matrix",
            "--games",
            "1000",
            "--engines",
            "mcts",
            "--positions",
            "50",
        ],
    );

    assert!(
        !success,
        "single-engine matrix planning unexpectedly passed:\n{text}"
    );
    assert!(
        text.contains("--engines must include at least two engines for h2h planning"),
        "{text}"
    );
    assert!(!text.contains("denominator must be positive"), "{text}");
}

#[test]
fn cross_engine_cmd_prints_one_shell_command_line() {
    let (success, text) = run_bash("source scripts/lib/bench_common.sh; cross_engine_cmd release");

    assert!(success, "cross_engine_cmd failed:\n{text}");
    assert_eq!(
        text.trim(),
        "cargo run --release --bin cross_engine_benchmark --"
    );
    assert_eq!(text.lines().count(), 1, "{text}");
}

#[test]
fn dry_run_observations_renders_filtered_engine_command() {
    let (success, text) = run_script(
        "generate_observations.sh",
        &[
            "--dataset",
            "benchmarks/positions-v1.json",
            "--output",
            "benchmarks/results/dev.json",
            "--checkpoint-dir",
            "benchmarks/results/dev-ckpt",
            "--engines",
            "mcts,minmax",
            "--dry-run",
        ],
    );

    assert!(success, "dry run failed:\n{text}");
    assert!(text.contains("cross_engine_benchmark"), "{text}");
    assert!(text.contains("--engines mcts\\,minimax"), "{text}");
    assert!(
        text.contains("--checkpoint-dir benchmarks/results/dev-ckpt"),
        "{text}"
    );
    assert!(text.contains("--skip-h2h"), "{text}");
}
