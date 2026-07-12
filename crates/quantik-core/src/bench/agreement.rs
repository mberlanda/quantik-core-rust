//! Agreement and cost aggregation for benchmark move selections.
//!
//! Port of `benchmarks/agreement.py`. Rows are JSON objects so the bundle
//! schema matches the Python harness.

use crate::bench::adapters::{select, EngineAdapter};
use crate::bench::metrics::{median, percentile, wilson_ci};
use crate::state::State;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};

/// Identity of one observation run: engine, config label, position, seed.
pub type RunKey = (String, String, String, Option<u64>);

/// The key an observation row would have (used for checkpoint resume).
pub fn row_key(row: &Value) -> RunKey {
    (
        row["engine"].as_str().unwrap_or_default().to_string(),
        row["config_label"].as_str().unwrap_or_default().to_string(),
        row["position_id"].as_str().unwrap_or_default().to_string(),
        row["seed"].as_u64(),
    )
}

/// Return one move-observation row per adapter, position, and seed run.
///
/// `on_row` is invoked after each completed row (checkpoint hook);
/// `skip` rows are not re-run (their rows must already be in `rows`).
pub fn run_agreement(
    adapters: &[Box<dyn EngineAdapter>],
    payload: &Value,
    seeds: &[u64],
    skip: &HashSet<RunKey>,
    mut on_row: impl FnMut(&Value),
) -> Result<Vec<Value>, String> {
    if seeds.is_empty() {
        return Err("seeds must be a non-empty ordered list".into());
    }

    let positions = payload["positions"]
        .as_array()
        .ok_or("payload has no positions")?;
    let mut rows: Vec<Value> = Vec::new();

    for position in positions {
        let qfen = position["qfen"].as_str().ok_or("position missing qfen")?;
        let bb = State::from_qfen(qfen)?.bb;
        let reference = position.get("reference").filter(|r| !r.is_null());
        let optimal_moves: Option<HashSet<&str>> = reference.map(|r| {
            r["optimal_moves"]
                .as_array()
                .map(|moves| moves.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default()
        });
        let position_id = position["id"].as_str().unwrap_or_default();

        for adapter in adapters {
            let adapter_seeds: &[u64] = if adapter.stochastic() {
                seeds
            } else {
                &seeds[..1]
            };
            for &seed in adapter_seeds {
                let key: RunKey = (
                    adapter.name().to_string(),
                    adapter.config_label(),
                    position_id.to_string(),
                    Some(seed),
                );
                if skip.contains(&key) {
                    continue;
                }
                let (_, observation) = select(adapter.as_ref(), &bb, position_id, Some(seed))?;
                let mut row = observation.to_json();
                row["phase"] = position["phase"].clone();
                row["hit"] = match &optimal_moves {
                    Some(optimal) => json!(optimal.contains(observation.mv.as_str())),
                    None => Value::Null,
                };
                on_row(&row);
                rows.push(row);
            }
        }
    }

    Ok(rows)
}

/// Aggregate exact-reference agreement by engine, config label, and phase.
pub fn aggregate_agreement(rows: &[Value]) -> Vec<Value> {
    let mut groups: BTreeMap<(String, String, String), Vec<&Value>> = BTreeMap::new();
    for row in rows {
        if row["hit"].is_null() {
            continue;
        }
        let key = (
            row["engine"].as_str().unwrap_or_default().to_string(),
            row["config_label"].as_str().unwrap_or_default().to_string(),
            row["phase"].as_str().unwrap_or_default().to_string(),
        );
        groups.entry(key).or_default().push(row);
    }

    groups
        .into_iter()
        .map(|((engine, config_label, phase), group)| {
            let n = group.len() as u64;
            let hits = group
                .iter()
                .filter(|row| row["hit"].as_bool() == Some(true))
                .count() as u64;
            let (ci95_low, ci95_high) = wilson_ci(hits, n);
            json!({
                "engine": engine,
                "config_label": config_label,
                "phase": phase,
                "n": n,
                "hits": hits,
                "agreement": hits as f64 / n as f64,
                "ci95_low": ci95_low,
                "ci95_high": ci95_high,
            })
        })
        .collect()
}

/// Aggregate measured selection cost by engine and config label.
pub fn aggregate_cost(rows: &[Value]) -> Vec<Value> {
    let mut groups: BTreeMap<(String, String), Vec<&Value>> = BTreeMap::new();
    for row in rows {
        let key = (
            row["engine"].as_str().unwrap_or_default().to_string(),
            row["config_label"].as_str().unwrap_or_default().to_string(),
        );
        groups.entry(key).or_default().push(row);
    }

    groups
        .into_iter()
        .map(|((engine, config_label), group)| {
            let wall_times: Vec<f64> = group
                .iter()
                .filter_map(|row| row["wall_time_s"].as_f64())
                .collect();
            let nodes: Vec<f64> = group
                .iter()
                .filter_map(|row| row["nodes"].as_u64())
                .map(|n| n as f64)
                .collect();
            let peak_memory: Vec<u64> = group
                .iter()
                .filter_map(|row| row["peak_memory_bytes"].as_u64())
                .collect();
            json!({
                "engine": engine,
                "config_label": config_label,
                "n": group.len(),
                "median_time_s": median(&wall_times),
                "p95_time_s": percentile(&wall_times, 95.0),
                "median_nodes": if nodes.is_empty() { Value::Null } else { json!(median(&nodes)) },
                "peak_memory_bytes": peak_memory.iter().max().map_or(Value::Null, |&m| json!(m)),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::adapters::RandomAdapter;

    fn fixture_rows() -> Vec<Value> {
        vec![
            json!({"engine": "e1", "config_label": "c", "phase": "endgame",
                   "position_id": "p1", "seed": 0, "move": "0:0:0", "hit": true,
                   "wall_time_s": 0.1, "nodes": 100, "peak_memory_bytes": null}),
            json!({"engine": "e1", "config_label": "c", "phase": "endgame",
                   "position_id": "p1", "seed": 1, "move": "0:0:1", "hit": false,
                   "wall_time_s": 0.3, "nodes": 300, "peak_memory_bytes": null}),
            json!({"engine": "e1", "config_label": "c", "phase": "opening",
                   "position_id": "p2", "seed": 0, "move": "0:0:0", "hit": null,
                   "wall_time_s": 0.2, "nodes": null, "peak_memory_bytes": 12}),
        ]
    }

    #[test]
    fn aggregate_agreement_excludes_null_hits() {
        let aggregates = aggregate_agreement(&fixture_rows());
        assert_eq!(aggregates.len(), 1);
        let row = &aggregates[0];
        assert_eq!(row["engine"], json!("e1"));
        assert_eq!(row["phase"], json!("endgame"));
        assert_eq!(row["n"], json!(2));
        assert_eq!(row["hits"], json!(1));
        assert_eq!(row["agreement"], json!(0.5));
        assert!(row["ci95_low"].as_f64().unwrap() < 0.5);
        assert!(row["ci95_high"].as_f64().unwrap() > 0.5);
    }

    #[test]
    fn aggregate_cost_medians() {
        let aggregates = aggregate_cost(&fixture_rows());
        assert_eq!(aggregates.len(), 1);
        let row = &aggregates[0];
        assert_eq!(row["n"], json!(3));
        assert_eq!(row["median_time_s"], json!(0.2));
        assert_eq!(row["median_nodes"], json!(200.0));
        assert_eq!(row["peak_memory_bytes"], json!(12));
    }

    #[test]
    fn run_agreement_produces_rows_with_hit_semantics() {
        let payload = json!({
            "positions": [
                {"id": "p0", "qfen": "..../..../..../....", "phase": "opening",
                 "reference": null},
            ]
        });
        let adapters: Vec<Box<dyn EngineAdapter>> = vec![Box::new(RandomAdapter)];
        let mut streamed = 0;
        let rows = run_agreement(&adapters, &payload, &[0, 1, 2], &HashSet::new(), |_| {
            streamed += 1
        })
        .unwrap();
        // Random is stochastic: 3 seeds × 1 position.
        assert_eq!(rows.len(), 3);
        assert_eq!(streamed, 3);
        for row in &rows {
            assert!(row["hit"].is_null(), "no reference: hit must be null");
            assert_eq!(row["phase"], json!("opening"));
        }
    }

    #[test]
    fn run_agreement_skip_set_prevents_reruns() {
        let payload = json!({
            "positions": [
                {"id": "p0", "qfen": "..../..../..../....", "phase": "opening",
                 "reference": null},
            ]
        });
        let adapters: Vec<Box<dyn EngineAdapter>> = vec![Box::new(RandomAdapter)];
        let mut skip = HashSet::new();
        skip.insert((
            "random".to_string(),
            "random".to_string(),
            "p0".to_string(),
            Some(0u64),
        ));
        let rows = run_agreement(&adapters, &payload, &[0, 1], &skip, |_| {}).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["seed"], json!(1));
    }

    #[test]
    fn empty_seeds_rejected() {
        let payload = json!({"positions": []});
        let adapters: Vec<Box<dyn EngineAdapter>> = vec![];
        assert!(run_agreement(&adapters, &payload, &[], &HashSet::new(), |_| {}).is_err());
    }
}
