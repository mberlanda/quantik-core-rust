//! Across-seed stability of benchmark move selections.
//!
//! Port of `benchmarks/stability.py`: aggregates the same raw rows
//! produced by [`super::agreement::run_agreement`]; engines are not
//! re-run here.

use crate::bench::metrics::mean_std;
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Aggregate move consistency and per-seed agreement by engine config.
pub fn aggregate_stability(rows: &[Value]) -> Vec<Value> {
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
            let mut seeds: Vec<Option<u64>> =
                group.iter().map(|row| row["seed"].as_u64()).collect();
            seeds.sort();
            seeds.dedup();

            // Move consistency: fraction of seeds choosing the modal move,
            // averaged over positions.
            let mut moves_by_position: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
            for row in &group {
                moves_by_position
                    .entry(row["position_id"].as_str().unwrap_or_default())
                    .or_default()
                    .push(row["move"].as_str().unwrap_or_default());
            }
            let consistency_values: Vec<f64> = moves_by_position
                .values()
                .map(|moves| {
                    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
                    for mv in moves {
                        *counts.entry(mv).or_default() += 1;
                    }
                    let modal = counts.values().copied().max().unwrap_or(0);
                    modal as f64 / moves.len() as f64
                })
                .collect();
            let (move_consistency, _) = mean_std(&consistency_values);

            // Per-seed exact-reference agreement, then mean/std across seeds.
            let mut per_seed_agreement: Vec<f64> = Vec::new();
            for &seed in &seeds {
                let solved: Vec<&&Value> = group
                    .iter()
                    .filter(|row| row["seed"].as_u64() == seed && !row["hit"].is_null())
                    .collect();
                if !solved.is_empty() {
                    let hits = solved
                        .iter()
                        .filter(|row| row["hit"].as_bool() == Some(true))
                        .count();
                    per_seed_agreement.push(hits as f64 / solved.len() as f64);
                }
            }
            let (agreement_mean, agreement_std) = mean_std(&per_seed_agreement);

            json!({
                "engine": engine,
                "config_label": config_label,
                "seeds": seeds.len(),
                "move_consistency": move_consistency,
                "agreement_mean": agreement_mean,
                "agreement_std": agreement_std,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stability_from_fixture_rows() {
        // Engine e: position p1 chose m1, m1, m2 across 3 seeds (consistency
        // 2/3); position p2 chose m1 across all seeds (consistency 1).
        // Hits per seed: seed0 2/2, seed1 1/2, seed2 0/2.
        let rows: Vec<Value> = vec![
            json!({"engine": "e", "config_label": "c", "position_id": "p1",
                   "seed": 0, "move": "m1", "hit": true}),
            json!({"engine": "e", "config_label": "c", "position_id": "p1",
                   "seed": 1, "move": "m1", "hit": true}),
            json!({"engine": "e", "config_label": "c", "position_id": "p1",
                   "seed": 2, "move": "m2", "hit": false}),
            json!({"engine": "e", "config_label": "c", "position_id": "p2",
                   "seed": 0, "move": "m1", "hit": true}),
            json!({"engine": "e", "config_label": "c", "position_id": "p2",
                   "seed": 1, "move": "m1", "hit": false}),
            json!({"engine": "e", "config_label": "c", "position_id": "p2",
                   "seed": 2, "move": "m1", "hit": false}),
        ];
        let aggregates = aggregate_stability(&rows);
        assert_eq!(aggregates.len(), 1);
        let row = &aggregates[0];
        assert_eq!(row["seeds"], json!(3));
        let consistency = row["move_consistency"].as_f64().unwrap();
        assert!((consistency - (2.0 / 3.0 + 1.0) / 2.0).abs() < 1e-12);
        let mean = row["agreement_mean"].as_f64().unwrap();
        assert!((mean - 0.5).abs() < 1e-12, "mean {mean}");
        assert!(row["agreement_std"].as_f64().unwrap() > 0.0);
    }
}
