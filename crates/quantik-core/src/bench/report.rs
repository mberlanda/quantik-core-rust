//! Markdown report generation from benchmark result bundles.
//!
//! Port of `benchmarks/report.py`: same section set, same tables, same
//! interpretation guardrails.

use serde_json::Value;

fn fmt(value: &Value, decimals: usize) -> String {
    match value.as_f64() {
        Some(x) => format!("{x:.decimals$}"),
        None => "-".to_string(),
    }
}

fn fmt_int(value: &Value) -> String {
    match value.as_f64() {
        Some(x) => format!("{}", x.round() as i64),
        None => "-".to_string(),
    }
}

fn fmt_thousands(value: &Value) -> String {
    match value.as_f64() {
        Some(x) => {
            let raw = format!("{}", x.round() as i64);
            let mut out = String::new();
            let bytes = raw.as_bytes();
            let start = if bytes[0] == b'-' { 1 } else { 0 };
            let digits = &raw[start..];
            for (i, c) in digits.chars().enumerate() {
                if i > 0 && (digits.len() - i) % 3 == 0 {
                    out.push(',');
                }
                out.push(c);
            }
            format!("{}{}", &raw[..start], out)
        }
        None => "-".to_string(),
    }
}

fn table(headers: &[&str], rows: Vec<Vec<String>>) -> String {
    let mut lines = vec![format!("| {} |", headers.join(" | "))];
    lines.push(format!(
        "|{}|",
        headers
            .iter()
            .map(|_| " --- ")
            .collect::<Vec<_>>()
            .join("|")
    ));
    for row in rows {
        lines.push(format!("| {} |", row.join(" | ")));
    }
    lines.join("\n")
}

fn family_note(family: &str) -> &'static str {
    if family == "fixed" {
        "same wall-clock budget per move; fair practical-latency comparison"
    } else {
        "per-engine native settings; useful for scaling behavior, not fair head-to-head ranking"
    }
}

/// Render the required benchmark Markdown report tables.
pub fn render_markdown(bundle: &Value) -> String {
    let env = &bundle["environment"];
    let config = &bundle["config"];
    let aggregates = &bundle["aggregates"];
    let dataset = &bundle["dataset"];
    let family = config["family"].as_str().unwrap_or("?");
    let empty = Vec::new();

    let git_sha = env["git_sha"].as_str().unwrap_or("unknown");
    let short_sha = &git_sha[..git_sha.len().min(12)];

    let agreement_rows: Vec<Vec<String>> = aggregates["agreement"]
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .map(|row| {
            vec![
                row["engine"].as_str().unwrap_or("?").to_string(),
                format!("`{}`", row["config_label"].as_str().unwrap_or("?")),
                row["phase"].as_str().unwrap_or("?").to_string(),
                row["n"].to_string(),
                row["hits"].to_string(),
                fmt(&row["agreement"], 3),
                format!(
                    "[{}, {}]",
                    fmt(&row["ci95_low"], 3),
                    fmt(&row["ci95_high"], 3)
                ),
            ]
        })
        .collect();

    let cost_rows: Vec<Vec<String>> = aggregates["cost"]
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .map(|row| {
            vec![
                row["engine"].as_str().unwrap_or("?").to_string(),
                format!("`{}`", row["config_label"].as_str().unwrap_or("?")),
                row["n"].to_string(),
                fmt(&row["median_time_s"], 4),
                fmt(&row["p95_time_s"], 4),
                fmt_int(&row["median_nodes"]),
                fmt_thousands(&row["peak_memory_bytes"]),
            ]
        })
        .collect();

    let h2h_aggregates = bundle["head_to_head"]["aggregates"]
        .as_array()
        .unwrap_or(&empty);
    let h2h_rows: Vec<Vec<String>> = h2h_aggregates
        .iter()
        .map(|row| {
            vec![
                row["engine_a"].as_str().unwrap_or("?").to_string(),
                row["engine_b"].as_str().unwrap_or("?").to_string(),
                row["paired_positions"].to_string(),
                row["games"].to_string(),
                row["a_wins"].to_string(),
                row["b_wins"].to_string(),
                row["draws"].to_string(),
                format!(
                    "{} [{}, {}]",
                    fmt(&row["a_win_rate"], 3),
                    fmt(&row["a_win_rate_ci95"][0], 3),
                    fmt(&row["a_win_rate_ci95"][1], 3)
                ),
                row["a_wins_as_mover"].to_string(),
                row["b_wins_as_mover"].to_string(),
            ]
        })
        .collect();

    let h2h_phase_rows: Vec<Vec<String>> = h2h_aggregates
        .iter()
        .flat_map(|row| {
            let engine_a = row["engine_a"].as_str().unwrap_or("?").to_string();
            let engine_b = row["engine_b"].as_str().unwrap_or("?").to_string();
            row["by_phase"]
                .as_object()
                .map(|phases| {
                    phases
                        .iter()
                        .map(|(phase, split)| {
                            vec![
                                engine_a.clone(),
                                engine_b.clone(),
                                phase.clone(),
                                split["games"].to_string(),
                                split["a_wins"].to_string(),
                                split["b_wins"].to_string(),
                            ]
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        })
        .collect();

    let stability_rows: Vec<Vec<String>> = aggregates["stability"]
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .map(|row| {
            vec![
                row["engine"].as_str().unwrap_or("?").to_string(),
                format!("`{}`", row["config_label"].as_str().unwrap_or("?")),
                row["seeds"].to_string(),
                fmt(&row["move_consistency"], 3),
                fmt(&row["agreement_mean"], 3),
                fmt(&row["agreement_std"], 3),
            ]
        })
        .collect();

    let mut parts = vec![
        format!("# Cross-engine benchmark - `{short_sha}`"),
        String::new(),
        format!("- benchmark family: **{family}** ({})", family_note(family)),
        format!(
            "- dataset: `{}` - {} positions {}, generation seed {}",
            dataset["checksum"].as_str().unwrap_or("?"),
            dataset["positions"],
            dataset["phases"],
            dataset["seed"]
        ),
        format!("- engine seeds: `{}`", config["engine_seeds"]),
        format!(
            "- environment: quantik-core-rust {}, rust {}, {}, {} CPUs",
            env["quantik_core_version"].as_str().unwrap_or("?"),
            env["rust_version"].as_str().unwrap_or("?"),
            env["platform"].as_str().unwrap_or("?"),
            env["cpu_count"]
        ),
        format!(
            "- started: {}",
            bundle["started_at"].as_str().unwrap_or("?")
        ),
    ];

    if let Some(checkpoint) = bundle.get("checkpoint").filter(|c| !c.is_null()) {
        let counts = &checkpoint["counts"];
        parts.push(format!(
            "- checkpoint status: {}",
            checkpoint["status"].as_str().unwrap_or("unknown")
        ));
        parts.push(format!(
            "- checkpoint counts: observations {}, h2h_records {}",
            counts["observations"].as_u64().unwrap_or(0),
            counts["h2h_records"].as_u64().unwrap_or(0)
        ));
        if let Some(pairs) = checkpoint.get("h2h_pairs").filter(|p| !p.is_null()) {
            parts.push(format!("- checkpoint h2h pairs: {pairs}"));
        }
    }

    parts.extend([
        String::new(),
        "## Exact move agreement".into(),
        String::new(),
        "A hit means the selected move is in the complete optimal set proven \
         by the exact solver with no cutoff. Positions without exact \
         references are excluded. For stochastic engines, runs equal \
         positions times seeds."
            .into(),
        String::new(),
        table(
            &[
                "Engine",
                "Configuration",
                "Phase",
                "Runs",
                "Optimal selected",
                "Agreement",
                "95% CI",
            ],
            agreement_rows,
        ),
        String::new(),
        "## Computational cost".into(),
        String::new(),
        "Measured effective work per move, not just configured limits.".into(),
        String::new(),
        table(
            &[
                "Engine",
                "Configuration",
                "Moves",
                "Median time (s)",
                "p95 time (s)",
                "Median nodes",
                "Peak memory (bytes)",
            ],
            cost_rows,
        ),
        String::new(),
        "## Head-to-head (paired, side-balanced)".into(),
        String::new(),
        "Each position and seed is played twice, once with each engine as \
         the side to move. Wins are credited to the actual engine/color \
         mapping. Quantik cannot draw, so Draws is structurally 0."
            .into(),
        String::new(),
        table(
            &[
                "Engine A",
                "Engine B",
                "Paired positions",
                "Games",
                "A wins",
                "B wins",
                "Draws",
                "A win rate (95% CI)",
                "A wins as mover",
                "B wins as mover",
            ],
            h2h_rows,
        ),
        String::new(),
        "### Head-to-head by phase".into(),
        String::new(),
        table(
            &["Engine A", "Engine B", "Phase", "Games", "A wins", "B wins"],
            h2h_phase_rows,
        ),
        String::new(),
        "## Stability across seeds".into(),
        String::new(),
        "Move consistency is the average fraction of seeds choosing the modal \
         move per position. Agreement mean and std are computed per seed \
         first, then aggregated."
            .into(),
        String::new(),
        table(
            &[
                "Engine",
                "Configuration",
                "Seeds",
                "Move consistency",
                "Agreement mean",
                "Agreement std",
            ],
            stability_rows,
        ),
        String::new(),
        "## Interpretation guardrails".into(),
        String::new(),
        "- Minimax buys adversarial certainty when the remaining tree is \
         small enough; MCTS buys empirical confidence through repeated \
         sampling; beam search buys bounded, selectively deep exploration."
            .into(),
        "- No engine is universally superior unless the evidence spans \
         multiple phases, equivalent budgets, repeated seeds, and \
         statistically meaningful samples."
            .into(),
        "- Beam search honors its time limit only between depth levels; \
         compare measured times above, never configured caps."
            .into(),
        "- Algorithm-native tables explain scaling; only fixed-resource \
         tables support fair engine-vs-engine claims."
            .into(),
        String::new(),
    ]);
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn minimal_bundle() -> Value {
        json!({
            "started_at": "2026-07-12T01:00:00+0200",
            "environment": {
                "git_sha": "abcdef1234567890",
                "quantik_core_version": "0.1.0",
                "rust_version": "stable",
                "platform": "macos-aarch64",
                "cpu_count": 8,
            },
            "config": {"family": "fixed", "engine_seeds": [0, 1]},
            "dataset": {
                "checksum": "deadbeef",
                "positions": 2,
                "phases": {"opening": 2},
                "seed": 42,
            },
            "observations": [],
            "head_to_head": {
                "records": [],
                "aggregates": [{
                    "engine_a": "minimax", "engine_b": "random",
                    "paired_positions": 2, "games": 4,
                    "a_wins": 3, "b_wins": 1, "draws": 0,
                    "a_win_rate": 0.75, "a_win_rate_ci95": [0.3, 0.95],
                    "a_wins_as_mover": 2, "b_wins_as_mover": 1,
                    "by_phase": {"opening": {"games": 4, "a_wins": 3, "b_wins": 1}},
                }],
            },
            "aggregates": {
                "agreement": [{
                    "engine": "minimax", "config_label": "minimax(d=16,t=1.0)",
                    "phase": "endgame", "n": 10, "hits": 9,
                    "agreement": 0.9, "ci95_low": 0.596, "ci95_high": 0.982,
                }],
                "cost": [{
                    "engine": "minimax", "config_label": "minimax(d=16,t=1.0)",
                    "n": 10, "median_time_s": 0.01, "p95_time_s": 0.09,
                    "median_nodes": 1234.0, "peak_memory_bytes": null,
                }],
                "stability": [{
                    "engine": "mcts", "config_label": "mcts(it=100,d=16,c=1.414)",
                    "seeds": 2, "move_consistency": 0.8,
                    "agreement_mean": 0.7, "agreement_std": 0.1,
                }],
            },
        })
    }

    #[test]
    fn report_contains_all_sections_and_rows() {
        let markdown = render_markdown(&minimal_bundle());
        for section in [
            "# Cross-engine benchmark - `abcdef123456`",
            "## Exact move agreement",
            "## Computational cost",
            "## Head-to-head (paired, side-balanced)",
            "### Head-to-head by phase",
            "## Stability across seeds",
            "## Interpretation guardrails",
        ] {
            assert!(markdown.contains(section), "missing {section}");
        }
        assert!(markdown.contains(
            "| minimax | `minimax(d=16,t=1.0)` | endgame | 10 | 9 | 0.900 | [0.596, 0.982] |"
        ));
        assert!(markdown.contains("0.750 [0.300, 0.950]"));
        assert!(markdown.contains("benchmark family: **fixed**"));
        // Null memory renders as "-".
        assert!(markdown.contains("| 1234 | - |"));
    }

    #[test]
    fn checkpoint_block_renders_after_started_line_when_present() {
        let mut bundle = minimal_bundle();
        bundle["checkpoint"] = json!({
            "status": "running",
            "counts": {"observations": 5, "h2h_records": 2},
        });
        let markdown = render_markdown(&bundle);
        assert!(markdown.contains("- checkpoint status: running"));
        assert!(markdown.contains("- checkpoint counts: observations 5, h2h_records 2"));
        // No checkpoint block: report renders unaffected.
        let no_checkpoint = render_markdown(&minimal_bundle());
        assert!(!no_checkpoint.contains("checkpoint status"));
    }

    #[test]
    fn thousands_separator() {
        assert_eq!(fmt_thousands(&json!(1234567.0)), "1,234,567");
        assert_eq!(fmt_thousands(&json!(12.0)), "12");
        assert_eq!(fmt_thousands(&Value::Null), "-");
    }
}
