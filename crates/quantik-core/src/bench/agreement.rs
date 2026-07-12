//! Agreement and cost aggregation for benchmark move selections.
//!
//! Port of `benchmarks/agreement.py`. Rows are JSON objects so the bundle
//! schema matches the Python harness.

use crate::bench::adapters::{select, EngineAdapter};
use crate::bench::metrics::{median, percentile, wilson_ci};
use crate::state::State;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};

/// Identity of one observation run — `(position_id, engine, config_label,
/// seed)`, matching Python's `checkpoint.observation_key` tuple order.
pub type ObservationKey = (String, String, String, Option<u64>);

/// The key an observation row would have (used for checkpoint resume).
pub fn observation_key(row: &Value) -> ObservationKey {
    (
        row["position_id"].as_str().unwrap_or_default().to_string(),
        row["engine"].as_str().unwrap_or_default().to_string(),
        row["config_label"].as_str().unwrap_or_default().to_string(),
        row["seed"].as_u64(),
    )
}

/// One (adapter, position, seed) unit of work, matching Python's
/// `_agreement_tasks` ordering: outer loop over positions, then adapters,
/// then that adapter's seeds.
struct AgreementTask<'a> {
    adapter: &'a dyn EngineAdapter,
    position: &'a Value,
    seed: u64,
}

fn build_agreement_tasks<'a>(
    adapters: &'a [Box<dyn EngineAdapter>],
    payload: &'a Value,
    seeds: &'a [u64],
    skip: &HashSet<ObservationKey>,
) -> Result<Vec<AgreementTask<'a>>, String> {
    let positions = payload["positions"]
        .as_array()
        .ok_or("payload has no positions")?;
    let mut tasks = Vec::new();
    for position in positions {
        let position_id = position["id"].as_str().unwrap_or_default();
        for adapter in adapters {
            let adapter_seeds: &[u64] = if adapter.stochastic() {
                seeds
            } else {
                &seeds[..1]
            };
            for &seed in adapter_seeds {
                let key: ObservationKey = (
                    position_id.to_string(),
                    adapter.name().to_string(),
                    adapter.config_label(),
                    Some(seed),
                );
                if skip.contains(&key) {
                    continue;
                }
                tasks.push(AgreementTask {
                    adapter: adapter.as_ref(),
                    position,
                    seed,
                });
            }
        }
    }
    Ok(tasks)
}

fn select_agreement_row(task: &AgreementTask) -> Result<Value, String> {
    let qfen = task.position["qfen"]
        .as_str()
        .ok_or("position missing qfen")?;
    let bb = State::from_qfen(qfen)?.bb;
    let reference = task.position.get("reference").filter(|r| !r.is_null());
    let optimal_moves: Option<HashSet<&str>> = reference.map(|r| {
        r["optimal_moves"]
            .as_array()
            .map(|moves| moves.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default()
    });
    let position_id = task.position["id"].as_str().unwrap_or_default();

    let (_, observation) = select(task.adapter, &bb, position_id, Some(task.seed))?;
    let mut row = observation.to_json();
    row["phase"] = task.position["phase"].clone();
    row["hit"] = match &optimal_moves {
        Some(optimal) => json!(optimal.contains(observation.mv.as_str())),
        None => Value::Null,
    };
    Ok(row)
}

/// Return one move-observation row per adapter, position, and seed run.
///
/// `on_row` is invoked after each completed row, IN TASK ORDER (checkpoint
/// hook — it returns `Result` so a checkpoint write failure aborts the run
/// instead of being silently dropped); `skip` rows are not re-run (their
/// rows must already be accounted for by the caller, e.g. loaded from a
/// checkpoint).
///
/// `workers == 1` runs the task list serially (identical to the original
/// single-threaded behavior). `workers > 1` builds a rayon pool sized to
/// `workers` and processes the ordered task list in sequential chunks of
/// `workers` tasks: each chunk runs in parallel via `par_iter().collect()`
/// (which preserves task order in the result `Vec`), and its rows are
/// streamed to `on_row` — in task order — before the next chunk starts.
/// Because chunks are sequential and each chunk preserves task order, the
/// produced rows and their order are byte-identical to a `workers == 1` run
/// regardless of worker count, while still delivering rows to the
/// checkpoint callback incrementally (at most one chunk of in-flight work
/// is lost on a crash, instead of the whole phase). An adapter error inside
/// a chunk fails that chunk (and the run) immediately, but rows from
/// previously completed chunks have already reached `on_row`.
pub fn run_agreement(
    adapters: &[Box<dyn EngineAdapter>],
    payload: &Value,
    seeds: &[u64],
    skip: &HashSet<ObservationKey>,
    workers: usize,
    mut on_row: impl FnMut(&Value) -> Result<(), String>,
) -> Result<Vec<Value>, String> {
    if seeds.is_empty() {
        return Err("seeds must be a non-empty ordered list".into());
    }
    if workers < 1 {
        return Err("workers must be at least 1".into());
    }

    let tasks = build_agreement_tasks(adapters, payload, seeds, skip)?;
    if tasks.is_empty() {
        return Ok(Vec::new());
    }

    let mut rows = Vec::with_capacity(tasks.len());
    if workers == 1 {
        for task in &tasks {
            let row = select_agreement_row(task)?;
            on_row(&row)?;
            rows.push(row);
        }
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|e| format!("build worker pool: {e}"))?;
        // Process the ordered task list in sequential chunks of `workers`
        // tasks: each chunk is computed in parallel, then its rows are
        // streamed to `on_row` in task order before the next chunk starts.
        // This keeps checkpoint durability incremental even under
        // parallelism — a crash loses at most one in-flight chunk instead
        // of the entire phase (see PR review finding M1).
        for chunk in tasks.chunks(workers) {
            let chunk_rows: Result<Vec<Value>, String> =
                pool.install(|| chunk.par_iter().map(select_agreement_row).collect());
            for row in chunk_rows? {
                on_row(&row)?;
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
    use crate::bench::adapters::{MinimaxAdapter, RandomAdapter, RawMetrics};
    use crate::bitboard::Bitboard;
    use crate::moves::{generate_legal_moves, Move};

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
        let rows = run_agreement(&adapters, &payload, &[0, 1, 2], &HashSet::new(), 1, |_| {
            streamed += 1;
            Ok(())
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
            "p0".to_string(),
            "random".to_string(),
            "random".to_string(),
            Some(0u64),
        ));
        let rows = run_agreement(&adapters, &payload, &[0, 1], &skip, 1, |_| Ok(())).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["seed"], json!(1));
    }

    #[test]
    fn empty_seeds_rejected() {
        let payload = json!({"positions": []});
        let adapters: Vec<Box<dyn EngineAdapter>> = vec![];
        assert!(run_agreement(&adapters, &payload, &[], &HashSet::new(), 1, |_| Ok(())).is_err());
    }

    #[test]
    fn workers_zero_rejected() {
        let payload = json!({"positions": []});
        let adapters: Vec<Box<dyn EngineAdapter>> = vec![];
        assert!(run_agreement(&adapters, &payload, &[0], &HashSet::new(), 0, |_| Ok(())).is_err());
    }

    /// workers=2 must produce EXACTLY the same rows, in the same order, as
    /// workers=1 — the deterministic-ordering contract from the plan.
    #[test]
    fn workers_two_matches_workers_one_exactly() {
        let payload = json!({
            "positions": [
                {"id": "p0", "qfen": "..../..../..../....", "phase": "opening", "reference": null},
                {"id": "p1", "qfen": "..../..../..../....", "phase": "opening",
                 "reference": {"optimal_moves": ["0:0:0"]}},
            ]
        });
        let adapters_1: Vec<Box<dyn EngineAdapter>> = vec![
            Box::new(RandomAdapter),
            Box::new(MinimaxAdapter {
                max_depth: 2,
                time_limit_s: Some(0.05),
            }),
        ];
        let adapters_2: Vec<Box<dyn EngineAdapter>> = vec![
            Box::new(RandomAdapter),
            Box::new(MinimaxAdapter {
                max_depth: 2,
                time_limit_s: Some(0.05),
            }),
        ];
        let seeds = [0u64, 1u64];

        let serial = run_agreement(
            &adapters_1,
            &payload,
            &seeds,
            &HashSet::new(),
            1,
            |_| Ok(()),
        )
        .unwrap();
        let parallel = run_agreement(
            &adapters_2,
            &payload,
            &seeds,
            &HashSet::new(),
            2,
            |_| Ok(()),
        )
        .unwrap();

        // Compare everything except measured timing (wall/cpu time are
        // real durations and will never be bit-identical across runs);
        // order and every other field — including move choice and hit —
        // must match exactly.
        let strip_timing = |rows: &[Value]| -> Vec<Value> {
            rows.iter()
                .map(|row| {
                    let mut row = row.clone();
                    row["wall_time_s"] = json!(null);
                    row["cpu_time_s"] = json!(null);
                    row
                })
                .collect()
        };
        assert_eq!(serial.len(), parallel.len());
        assert_eq!(
            strip_timing(&serial),
            strip_timing(&parallel),
            "workers=2 must match workers=1 exactly (ignoring measured timing)"
        );
    }

    /// Stochastic test adapter that counts every `select_raw` invocation in
    /// a shared atomic counter (so a test can observe how many tasks have
    /// actually been computed at a given point) and fails for one specific
    /// seed, so a regression test can force a task deep in the ordered
    /// task list to error deterministically.
    struct CountingFailAtSeedAdapter {
        computed: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        fail_seed: u64,
    }

    impl EngineAdapter for CountingFailAtSeedAdapter {
        fn name(&self) -> &'static str {
            "counting_failtest"
        }
        fn stochastic(&self) -> bool {
            true
        }
        fn config_label(&self) -> String {
            "counting_failtest".into()
        }
        fn select_raw(
            &self,
            bb: &Bitboard,
            seed: Option<u64>,
        ) -> Result<(Move, RawMetrics), String> {
            self.computed
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if seed == Some(self.fail_seed) {
                return Err("intentional test failure".into());
            }
            let moves = generate_legal_moves(bb);
            Ok((moves[0], RawMetrics::default()))
        }
    }

    /// Regression test for M1: under `workers > 1`, rows must reach
    /// `on_row` chunk by chunk — not only after the ENTIRE task list
    /// (including tasks the run never gets to) has finished computing.
    ///
    /// One position with a single stochastic adapter and 6 seeds produces
    /// 6 ordered tasks (seed 0..5); with `workers = 2` these split into 3
    /// sequential chunks of 2 tasks each: `[0,1]`, `[2,3]`, `[4,5]`. The
    /// adapter fails on seed 4 (the 5th task, in the last chunk), so the
    /// run must return `Err`, but the first two chunks (seeds 0..3, 4
    /// rows) must already have reached `on_row` beforehand — that's the
    /// checkpoint-durability guarantee (see PR review finding M1).
    ///
    /// The test also asserts a structural, non-racy fact that pins down
    /// *why* this holds, not just the end state: at the moment `on_row`
    /// fires for the very first row, the shared `computed` counter must
    /// read exactly 2 (only chunk 1's own two tasks have run). Against the
    /// OLD `pool.install(|| tasks.par_iter().collect())` implementation —
    /// which computes and collects results for the WHOLE task list inside
    /// one `pool.install` call before the first `on_row` is ever invoked —
    /// that counter would already read 6 (every task, including the
    /// seed-4 failure and the never-delivered seed-5 success) by the time
    /// `on_row` is first called, deterministically failing this assertion.
    /// This isn't a timing race: `pool.install(|| ...collect())` cannot
    /// return control to the caller until every item in the parallel
    /// iterator has been computed, so the counter value at first delivery
    /// is a structural property of which implementation is in use, not a
    /// coincidence of scheduling.
    #[test]
    fn workers_parallel_streams_completed_chunks_before_later_chunk_error() {
        let payload = json!({
            "positions": [
                {"id": "p0", "qfen": "..../..../..../....", "phase": "opening", "reference": null},
            ]
        });
        let computed = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let adapters: Vec<Box<dyn EngineAdapter>> = vec![Box::new(CountingFailAtSeedAdapter {
            computed: computed.clone(),
            fail_seed: 4,
        })];
        let seeds = [0u64, 1, 2, 3, 4, 5];

        let mut streamed_rows: Vec<Value> = Vec::new();
        let mut computed_at_first_delivery = None;
        let result = run_agreement(&adapters, &payload, &seeds, &HashSet::new(), 2, |row| {
            if computed_at_first_delivery.is_none() {
                computed_at_first_delivery =
                    Some(computed.load(std::sync::atomic::Ordering::SeqCst));
            }
            streamed_rows.push(row.clone());
            Ok(())
        });

        assert!(
            result.is_err(),
            "run must fail: task for seed 4 errors inside its chunk"
        );
        assert!(
            streamed_rows.len() >= 4,
            "rows from chunks completed before the failing chunk must already be \
             streamed to on_row; got {} rows",
            streamed_rows.len()
        );
        let streamed_seeds: Vec<u64> = streamed_rows
            .iter()
            .map(|row| row["seed"].as_u64().unwrap())
            .collect();
        assert_eq!(
            streamed_seeds,
            vec![0, 1, 2, 3],
            "exactly the two completed chunks (seeds 0..3) must have reached on_row, \
             in task order, before the failing chunk aborted the run"
        );
        assert_eq!(
            computed_at_first_delivery,
            Some(2),
            "on_row for the first row must fire after only its own chunk (2 tasks) \
             has been computed, not after the whole task list (this is what an \
             implementation that collects the entire task list before streaming \
             any row would violate)"
        );
    }
}
