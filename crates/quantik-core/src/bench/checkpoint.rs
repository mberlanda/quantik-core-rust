//! Directory-based resumable checkpoint storage for long benchmark runs.
//!
//! Port of `benchmarks/checkpoint.py`. A checkpoint is a directory:
//!
//! ```text
//! <checkpoint-dir>/
//!   manifest.json      pretty (indent 2), sorted-key JSON + trailing "\n",
//!                       written atomically (tmp file + rename)
//!   observations.jsonl one compact, sorted-key JSON row per completed
//!                       agreement observation, appended + flushed per line
//!   h2h.jsonl           one compact, sorted-key JSON row per completed
//!                       head-to-head game, appended + flushed per line
//! ```
//!
//! Both JSONL files use [`crate::bench::canonical::canonical_json`], the
//! same byte-exact encoder used for dataset/bundle artifacts, so a
//! checkpoint directory written by this crate loads and rehydrates
//! (`bundle_from_checkpoint`) correctly in Python's `benchmarks.checkpoint`
//! module and vice versa. `manifest.json` uses
//! [`crate::bench::canonical::canonical_json_pretty`], matching Python's
//! `json.dumps(manifest, indent=2, sort_keys=True) + "\n"`.
//!
//! **Resume validation is intra-language only.** `validate_resume_manifest`
//! compares the run's config dict against the manifest's stored config
//! dict, but the two languages' CLI argument dicts have different key sets
//! and shapes (e.g. `track_memory`/`skip_agreement` exist only in the
//! Python CLI) — a checkpoint directory started by one language's `run` is
//! not expected to `--resume` cleanly in the other. Loading and reporting
//! (`bundle_from_checkpoint`, `report --input <dir>`) work identically in
//! both languages regardless of which one wrote the directory.
//!
//! Unlike the single-file `.ckpt` format this module replaces (PR #10),
//! an invalid JSONL line is a hard error (file path + 1-based line
//! number) — there is no truncated-tail tolerance here, because
//! `manifest.json`'s atomic tmp+rename write is the integrity anchor: a
//! crash mid-write of an observation/h2h line can only ever leave a
//! trailing partial line (never a torn manifest), and callers that always
//! flush after `append_jsonl` should not see one in practice, but Python
//! chose to raise rather than silently drop, so this port matches that.

use crate::bench::agreement::{aggregate_agreement, aggregate_cost};
use crate::bench::bundle;
use crate::bench::canonical::{canonical_json, canonical_json_pretty};
use crate::bench::head_to_head;
use crate::bench::stability::aggregate_stability;
use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const MANIFEST: &str = "manifest.json";
pub const OBSERVATIONS: &str = "observations.jsonl";
pub const H2H_RECORDS: &str = "h2h.jsonl";

/// Config fields dropped from the resume signature: they legitimately
/// differ between the interrupted run and the resuming one (or, for
/// `checkpoint_dir`/`workers`, don't describe the *engine* configuration
/// at all). Matches Python's `checkpoint._RESUME_CONFIG_EXCLUDES` exactly
/// (five keys — the plan text lists four, but the Python source, which is
/// authoritative, has five).
const RESUME_CONFIG_EXCLUDES: [&str; 5] = [
    "checkpoint_dir",
    "output",
    "resume",
    "skip_agreement",
    "workers",
];

/// The paths that make up one checkpoint directory.
#[derive(Debug, Clone)]
pub struct CheckpointPaths {
    pub root: PathBuf,
    pub manifest: PathBuf,
    pub observations: PathBuf,
    pub h2h: PathBuf,
}

/// Resolve the manifest/observations/h2h file paths under `root`.
pub fn checkpoint_paths(root: &Path) -> CheckpointPaths {
    CheckpointPaths {
        root: root.to_path_buf(),
        manifest: root.join(MANIFEST),
        observations: root.join(OBSERVATIONS),
        h2h: root.join(H2H_RECORDS),
    }
}

/// Resolve a manifest target from either a checkpoint root directory or a
/// direct path to (or already ending in) `manifest.json` — mirrors
/// Python's `_manifest_path` so `write_manifest`/`load_manifest` accept
/// either form.
fn manifest_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        return path.join(MANIFEST);
    }
    let name_is_manifest = path.file_name().is_some_and(|n| n == MANIFEST);
    let has_json_suffix = path.extension().is_some_and(|e| e == "json");
    if !name_is_manifest && !has_json_suffix {
        return path.join(MANIFEST);
    }
    path.to_path_buf()
}

/// Append one JSON object as one compact, sorted-key JSONL row, flushing to
/// the OS immediately.
pub fn append_jsonl(path: &Path, row: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {parent:?}: {e}"))?;
        }
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("open {path:?}: {e}"))?;
    writeln!(file, "{}", canonical_json(row)).map_err(|e| format!("write {path:?}: {e}"))?;
    file.flush().map_err(|e| format!("flush {path:?}: {e}"))
}

/// Load a JSONL file; a missing file behaves like an empty stream. Blank
/// lines are skipped. An invalid line is a hard error naming the file and
/// its 1-based line number (no truncated-tail tolerance — see the module
/// doc).
pub fn load_jsonl(path: &Path) -> Result<Vec<Value>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(trimmed)
            .map_err(|e| format!("{}:{}: invalid checkpoint JSON: {e}", path.display(), i + 1))?;
        rows.push(value);
    }
    Ok(rows)
}

/// Build the set of stable resume keys already present in checkpoint rows.
pub fn key_set<K: Eq + std::hash::Hash>(
    rows: &[Value],
    key_func: impl Fn(&Value) -> K,
) -> HashSet<K> {
    rows.iter().map(key_func).collect()
}

/// Atomically write the checkpoint manifest (tmp file + rename), pretty
/// (indent 2), sorted-key JSON + trailing newline, matching Python's
/// `json.dumps(manifest, indent=2, sort_keys=True) + "\n"`.
pub fn write_manifest(path: &Path, manifest: &Value) -> Result<(), String> {
    let target = manifest_path(path);
    if let Some(parent) = target.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {parent:?}: {e}"))?;
        }
    }
    let file_name = target
        .file_name()
        .ok_or("manifest path has no file name")?
        .to_string_lossy()
        .to_string();
    let tmp = target.with_file_name(format!("{file_name}.tmp"));
    std::fs::write(&tmp, canonical_json_pretty(manifest))
        .map_err(|e| format!("write {tmp:?}: {e}"))?;
    std::fs::rename(&tmp, &target).map_err(|e| format!("rename {tmp:?} -> {target:?}: {e}"))
}

/// Load the checkpoint manifest, or an empty object if it doesn't exist
/// yet (mirrors Python's `load_manifest` returning `{}`).
pub fn load_manifest(path: &Path) -> Result<Value, String> {
    let target = manifest_path(path);
    if !target.exists() {
        return Ok(json!({}));
    }
    let text = std::fs::read_to_string(&target).map_err(|e| format!("read {target:?}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {target:?}: {e}"))
}

fn now_timestamp() -> String {
    chrono::Local::now()
        .format("%Y-%m-%dT%H:%M:%S%z")
        .to_string()
}

/// Update checkpoint progress counters in place (`counts`, optionally
/// `status`, always `updated_at`) and return the resulting manifest.
pub fn update_manifest_counts(
    path: &Path,
    observations: u64,
    h2h_records: u64,
    status: Option<&str>,
) -> Result<Value, String> {
    let target = manifest_path(path);
    let mut manifest = load_manifest(&target)?;
    manifest["counts"] = json!({"observations": observations, "h2h_records": h2h_records});
    if let Some(status) = status {
        manifest["status"] = json!(status);
    }
    manifest["updated_at"] = json!(now_timestamp());
    write_manifest(&target, &manifest)?;
    Ok(manifest)
}

/// Drop volatile fields that must not participate in resume validation.
pub fn normalize_run_config(config: &Value) -> Value {
    let mut map = config.as_object().cloned().unwrap_or_default();
    for key in RESUME_CONFIG_EXCLUDES {
        map.remove(key);
    }
    Value::Object(map)
}

fn config_signature(config: &Value, ignore_skip_h2h: bool) -> Value {
    let mut signature = normalize_run_config(config);
    if ignore_skip_h2h {
        if let Some(map) = signature.as_object_mut() {
            map.remove("skip_h2h");
        }
    }
    signature
}

/// Build a fresh checkpoint manifest (mirrors Python's `_build_manifest`).
pub fn build_manifest(
    config: &Value,
    dataset_payload: &Value,
    status: &str,
    observations: u64,
    h2h_records: u64,
) -> Value {
    let positions = dataset_payload["positions"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let mut phases: Map<String, Value> = Map::new();
    for position in &positions {
        let phase = position["phase"].as_str().unwrap_or_default().to_string();
        let count = phases.get(&phase).and_then(Value::as_u64).unwrap_or(0);
        phases.insert(phase, json!(count + 1));
    }
    json!({
        "schema_version": bundle::SCHEMA_VERSION,
        "started_at": now_timestamp(),
        "environment": bundle::collect_environment(),
        "config": config,
        "dataset": {
            "checksum": dataset_payload.get("checksum").cloned().unwrap_or(Value::Null),
            "generator": dataset_payload["generator"],
            "seed": dataset_payload["seed"],
            "schema_version": dataset_payload["schema_version"],
            "positions": positions.len(),
            "phases": phases,
        },
        "status": status,
        "counts": {"observations": observations, "h2h_records": h2h_records},
    })
}

/// Raise a clear error when a resume checkpoint does not match this run:
/// dataset checksum must be equal, and the normalized config signature
/// must be equal (per-key diff in the error message otherwise).
/// `allow_skip_h2h_mismatch` additionally drops `skip_h2h` from the
/// signature — the CLI only sets this when the checkpoint's existing h2h
/// records already equal the expected count, so a differing `--skip-h2h`
/// this run genuinely doesn't matter.
pub fn validate_resume_manifest(
    manifest: &Value,
    dataset_checksum: Option<&str>,
    config: &Value,
    allow_skip_h2h_mismatch: bool,
) -> Result<(), String> {
    let manifest_dataset = manifest.get("dataset").cloned().unwrap_or(json!({}));
    let actual_checksum = manifest_dataset.get("checksum").and_then(Value::as_str);
    if actual_checksum != dataset_checksum {
        return Err(format!(
            "checkpoint dataset checksum mismatch: expected {dataset_checksum:?}, found {actual_checksum:?}"
        ));
    }

    let expected_config = config_signature(config, allow_skip_h2h_mismatch);
    let empty = Value::Null;
    let actual_config = config_signature(
        manifest.get("config").unwrap_or(&empty),
        allow_skip_h2h_mismatch,
    );
    if actual_config != expected_config {
        let expected_map = expected_config.as_object().cloned().unwrap_or_default();
        let actual_map = actual_config.as_object().cloned().unwrap_or_default();
        let mut keys: BTreeSet<String> = expected_map.keys().cloned().collect();
        keys.extend(actual_map.keys().cloned());
        let mut diffs = Vec::new();
        for key in keys {
            let expected_value = expected_map.get(&key).cloned().unwrap_or(Value::Null);
            let actual_value = actual_map.get(&key).cloned().unwrap_or(Value::Null);
            if expected_value != actual_value {
                diffs.push(format!(
                    "{key}: expected {expected_value}, found {actual_value}"
                ));
            }
        }
        let detail = if diffs.is_empty() {
            "unknown difference".to_string()
        } else {
            diffs.join("; ")
        };
        return Err(format!("checkpoint config mismatch: {detail}"));
    }
    Ok(())
}

fn dataset_summary(manifest: &Value) -> Result<Value, String> {
    match manifest.get("dataset") {
        Some(d) if !d.is_null() => Ok(d.clone()),
        _ => Err("checkpoint manifest is missing dataset metadata".into()),
    }
}

/// Group head-to-head records by unordered engine pair, in first-seen
/// order, with each aggregate's `(engine_a, engine_b)` naming taken from
/// that pair's first record's `(mover, responder)` — mirrors Python's
/// `_head_to_head_aggregates`.
fn head_to_head_aggregates(records: &[Value]) -> Vec<Value> {
    let mut order: Vec<(String, String)> = Vec::new();
    let mut names: HashMap<(String, String), (String, String)> = HashMap::new();
    let mut groups: HashMap<(String, String), Vec<Value>> = HashMap::new();

    for record in records {
        let mover = record["mover"].as_str().unwrap_or_default().to_string();
        let responder = record["responder"].as_str().unwrap_or_default().to_string();
        let mut sorted_pair = [mover.clone(), responder.clone()];
        sorted_pair.sort();
        let pair_key = (sorted_pair[0].clone(), sorted_pair[1].clone());

        groups
            .entry(pair_key.clone())
            .or_insert_with(|| {
                order.push(pair_key.clone());
                names.insert(pair_key.clone(), (mover.clone(), responder.clone()));
                Vec::new()
            })
            .push(record.clone());
    }

    order
        .into_iter()
        .map(|pair_key| {
            let (name_a, name_b) = names[&pair_key].clone();
            head_to_head::aggregate_head_to_head(&groups[&pair_key], &name_a, &name_b)
        })
        .collect()
}

/// Rehydrate a checkpoint directory into the standard benchmark bundle,
/// including a partial state — an in-progress or preflight-failed
/// checkpoint rehydrates fine, with whatever rows/records/aggregates are
/// available so far. The bundle gains a `"checkpoint": {status, counts}`
/// block (and `h2h_pairs` when the manifest carries one).
pub fn bundle_from_checkpoint(root: &Path) -> Result<Value, String> {
    let paths = checkpoint_paths(root);
    let manifest = load_manifest(&paths.manifest)?;
    let observations = load_jsonl(&paths.observations)?;
    let records = load_jsonl(&paths.h2h)?;
    let dataset = dataset_summary(&manifest)?;

    let counts = manifest
        .get("counts")
        .cloned()
        .unwrap_or(json!({"observations": 0, "h2h_records": 0}));
    let mut checkpoint_info = json!({
        "status": manifest.get("status").and_then(Value::as_str).unwrap_or("unknown"),
        "counts": counts,
    });
    if let Some(pairs) = manifest.get("h2h_pairs") {
        checkpoint_info["h2h_pairs"] = pairs.clone();
    }

    let started_at = manifest
        .get("started_at")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(now_timestamp);
    let environment = manifest
        .get("environment")
        .cloned()
        .unwrap_or_else(bundle::collect_environment);
    let config = manifest.get("config").cloned().unwrap_or(json!({}));

    Ok(json!({
        "schema_version": bundle::SCHEMA_VERSION,
        "started_at": started_at,
        "environment": environment,
        "config": config,
        "dataset": dataset,
        "checkpoint": checkpoint_info,
        "observations": observations,
        "head_to_head": {
            "records": records,
            "aggregates": head_to_head_aggregates(&records),
        },
        "aggregates": {
            "agreement": aggregate_agreement(&observations),
            "cost": aggregate_cost(&observations),
            "stability": aggregate_stability(&observations),
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::adapters::{EngineAdapter, MinimaxAdapter, RandomAdapter};
    use crate::bench::agreement::{observation_key, run_agreement, ObservationKey};
    use crate::bench::head_to_head::h2h_key;
    use crate::bench::report;

    fn scratch_dir(tag: &str) -> PathBuf {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("quantik-ckpt-{}-{tag}-{id}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn jsonl_append_and_load_roundtrip() {
        let dir = scratch_dir("jsonl-roundtrip");
        let path = dir.join("rows.jsonl");
        append_jsonl(&path, &json!({"b": 2, "a": 1})).unwrap();
        append_jsonl(&path, &json!({"a": 3, "b": 4})).unwrap();

        assert_eq!(
            load_jsonl(&path).unwrap(),
            vec![json!({"a": 1, "b": 2}), json!({"a": 3, "b": 4})]
        );
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines, vec![r#"{"a":1,"b":2}"#, r#"{"a":3,"b":4}"#]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_jsonl_missing_file_returns_empty() {
        let dir = scratch_dir("missing");
        assert_eq!(
            load_jsonl(&dir.join("missing.jsonl")).unwrap(),
            Vec::<Value>::new()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_jsonl_bad_json_includes_file_and_line() {
        let dir = scratch_dir("bad-json");
        let path = dir.join("rows.jsonl");
        std::fs::write(&path, "{\"ok\":1}\n{\"bad\":\n").unwrap();
        let err = load_jsonl(&path).unwrap_err();
        assert!(err.contains("rows.jsonl:2"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_jsonl_skips_blank_lines() {
        let dir = scratch_dir("blank-lines");
        let path = dir.join("rows.jsonl");
        std::fs::write(&path, "{\"a\":1}\n\n   \n{\"a\":2}\n").unwrap();
        assert_eq!(
            load_jsonl(&path).unwrap(),
            vec![json!({"a": 1}), json!({"a": 2})]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn observation_and_h2h_keys_are_stable() {
        let observation = json!({
            "position_id": "p0008",
            "engine": "mcts",
            "config_label": "mcts(it=5000,d=16,c=1.414)",
            "seed": 7,
        });
        let h2h = json!({
            "position_id": "p0008",
            "mover": "beam",
            "responder": "mcts",
            "seed": 11,
        });
        assert_eq!(
            observation_key(&observation),
            (
                "p0008".to_string(),
                "mcts".to_string(),
                "mcts(it=5000,d=16,c=1.414)".to_string(),
                Some(7)
            )
        );
        assert_eq!(
            h2h_key(&h2h),
            (
                "p0008".to_string(),
                "beam".to_string(),
                "mcts".to_string(),
                11
            )
        );
        let set: HashSet<ObservationKey> =
            key_set(std::slice::from_ref(&observation), observation_key);
        assert!(set.contains(&(
            "p0008".to_string(),
            "mcts".to_string(),
            "mcts(it=5000,d=16,c=1.414)".to_string(),
            Some(7)
        )));
    }

    #[test]
    fn manifest_atomic_write_counts_and_status_transitions() {
        let dir = scratch_dir("manifest-lifecycle");
        let root = dir.join("checkpoint");
        let config = json!({"family": "native", "engine_seeds": [0, 1]});
        let dataset_payload = json!({
            "checksum": "abc123",
            "generator": "benchmarks.dataset.generate/v1",
            "seed": 20260711,
            "schema_version": 1,
            "positions": [{"phase": "late_mid"}],
        });
        let manifest = build_manifest(&config, &dataset_payload, "running", 0, 0);
        write_manifest(&root, &manifest).unwrap();
        assert!(root.join(MANIFEST).exists());
        // manifest.json.tmp must never be left behind after a successful write.
        assert!(!root.join("manifest.json.tmp").exists());

        let loaded = load_manifest(&root).unwrap();
        assert_eq!(loaded["status"], json!("running"));
        assert_eq!(loaded["dataset"]["checksum"], json!("abc123"));
        assert_eq!(loaded["dataset"]["phases"]["late_mid"], json!(1));

        let observation = json!({
            "engine": "minimax", "config_label": "minimax(d=16)",
            "position_id": "p0000", "move": "1:3:5", "seed": 0, "hit": true,
            "wall_time_s": 0.01, "nodes": 42, "peak_memory_bytes": Value::Null,
        });
        let h2h_record = json!({
            "position_id": "p0000", "phase": "late_mid",
            "mover": "minimax", "responder": "random",
            "winner": "minimax", "plies": 1, "seed": 0,
        });
        append_jsonl(&root.join(OBSERVATIONS), &observation).unwrap();
        append_jsonl(&root.join(H2H_RECORDS), &h2h_record).unwrap();

        let updated = update_manifest_counts(&root, 1, 1, Some("complete")).unwrap();
        assert_eq!(updated["status"], json!("complete"));
        assert_eq!(
            updated["counts"],
            json!({"observations": 1, "h2h_records": 1})
        );
        assert!(updated["updated_at"].is_string());

        let bundle_dict = bundle_from_checkpoint(&root).unwrap();
        assert_eq!(bundle_dict["schema_version"], json!(bundle::SCHEMA_VERSION));
        assert_eq!(bundle_dict["config"], config);
        assert_eq!(bundle_dict["observations"], json!([observation]));
        assert_eq!(bundle_dict["head_to_head"]["records"], json!([h2h_record]));
        assert_eq!(bundle_dict["aggregates"]["agreement"][0]["n"], json!(1));
        assert_eq!(
            bundle_dict["aggregates"]["cost"][0]["median_nodes"],
            json!(42.0)
        );
        assert_eq!(
            bundle_dict["head_to_head"]["aggregates"][0]["games"],
            json!(1)
        );
        assert_eq!(bundle_dict["checkpoint"]["status"], json!("complete"));
        assert_eq!(
            bundle_dict["checkpoint"]["counts"],
            json!({"observations": 1, "h2h_records": 1})
        );
        assert!(report::render_markdown(&bundle_dict).contains("checkpoint status: complete"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn running_checkpoint_bundle_and_report_show_partial_status() {
        let dir = scratch_dir("partial-status");
        let root = dir.join("checkpoint");
        let dataset_payload = json!({
            "schema_version": 1,
            "generator": "benchmarks.dataset.generate/v1",
            "seed": 20260711,
            "checksum": "abc123",
            "positions": [
                {"id": "p0000", "qfen": ".ba./..CC/DcbD/cA.A", "phase": "late_mid",
                 "pieces": 8, "side_to_move": 1, "legal_moves": 10, "reference": Value::Null},
            ],
        });
        let manifest = build_manifest(
            &json!({"family": "native", "engine_seeds": [0]}),
            &dataset_payload,
            "running",
            0,
            0,
        );
        write_manifest(&root, &manifest).unwrap();

        let bundle_dict = bundle_from_checkpoint(&root).unwrap();
        assert_eq!(bundle_dict["checkpoint"]["status"], json!("running"));
        assert_eq!(
            bundle_dict["checkpoint"]["counts"],
            json!({"observations": 0, "h2h_records": 0})
        );
        assert_eq!(bundle_dict["observations"], json!([]));
        assert!(report::render_markdown(&bundle_dict).contains("checkpoint status: running"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bundle_from_checkpoint_requires_dataset_metadata() {
        let dir = scratch_dir("no-dataset");
        let root = dir.join("checkpoint");
        write_manifest(&root, &json!({"status": "running"})).unwrap();
        let err = bundle_from_checkpoint(&root).unwrap_err();
        assert!(err.contains("dataset metadata"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resume_validation_checksum_mismatch() {
        let manifest = json!({"dataset": {"checksum": "a"}, "config": {}});
        let err = validate_resume_manifest(&manifest, Some("b"), &json!({}), false).unwrap_err();
        assert!(err.contains("checksum mismatch"), "{err}");
    }

    #[test]
    fn resume_validation_config_diff_message_names_the_key() {
        let manifest = json!({
            "dataset": {"checksum": "a"},
            "config": {"family": "fixed", "seeds": 10},
        });
        let err = validate_resume_manifest(
            &manifest,
            Some("a"),
            &json!({"family": "fixed", "seeds": 30}),
            false,
        )
        .unwrap_err();
        assert!(err.contains("seeds: expected 30, found 10"), "{err}");
    }

    #[test]
    fn resume_validation_excludes_volatile_fields() {
        let manifest = json!({
            "dataset": {"checksum": "a"},
            "config": {
                "family": "fixed", "checkpoint_dir": "old/dir",
                "output": "old.json", "resume": false, "workers": 1,
            },
        });
        // Only checkpoint_dir/output/resume/workers differ: excluded fields
        // must not trip the mismatch.
        validate_resume_manifest(
            &manifest,
            Some("a"),
            &json!({
                "family": "fixed", "checkpoint_dir": "new/dir",
                "output": "new.json", "resume": true, "workers": 4,
            }),
            false,
        )
        .unwrap();
    }

    #[test]
    fn resume_validation_skip_h2h_allowed_only_when_ignored() {
        let manifest = json!({
            "dataset": {"checksum": "a"},
            "config": {"family": "fixed", "skip_h2h": false},
        });
        let strict_config = json!({"family": "fixed", "skip_h2h": true});
        assert!(validate_resume_manifest(&manifest, Some("a"), &strict_config, false).is_err());
        validate_resume_manifest(&manifest, Some("a"), &strict_config, true).unwrap();
    }

    #[test]
    fn head_to_head_aggregates_group_unordered_pairs_first_seen_order() {
        let dir = scratch_dir("h2h-pairs");
        let root = dir.join("checkpoint");
        let dataset_payload = json!({
            "checksum": "c", "generator": "g/v1", "seed": 1, "schema_version": 1,
            "positions": [{"phase": "opening"}],
        });
        write_manifest(
            &root,
            &build_manifest(&json!({}), &dataset_payload, "running", 0, 0),
        )
        .unwrap();
        // beam-vs-mcts first, then minimax-vs-random; a later beam/mcts
        // record must still land in the FIRST group (name order from its
        // first record), not create a duplicate.
        append_jsonl(
            &root.join(H2H_RECORDS),
            &json!({"position_id": "p0", "phase": "opening", "mover": "beam",
                     "responder": "mcts", "winner": "beam", "plies": 3, "seed": 0}),
        )
        .unwrap();
        append_jsonl(
            &root.join(H2H_RECORDS),
            &json!({"position_id": "p0", "phase": "opening", "mover": "minimax",
                     "responder": "random", "winner": "minimax", "plies": 1, "seed": 0}),
        )
        .unwrap();
        append_jsonl(
            &root.join(H2H_RECORDS),
            &json!({"position_id": "p0", "phase": "opening", "mover": "mcts",
                     "responder": "beam", "winner": "mcts", "plies": 2, "seed": 1}),
        )
        .unwrap();

        let bundle_dict = bundle_from_checkpoint(&root).unwrap();
        let aggregates = bundle_dict["head_to_head"]["aggregates"]
            .as_array()
            .unwrap();
        assert_eq!(aggregates.len(), 2, "two distinct unordered pairs");
        assert_eq!(aggregates[0]["engine_a"], json!("beam"));
        assert_eq!(aggregates[0]["engine_b"], json!("mcts"));
        assert_eq!(
            aggregates[0]["games"],
            json!(2),
            "beam/mcts pair merges both orientations"
        );
        assert_eq!(aggregates[1]["engine_a"], json!("minimax"));
        assert_eq!(aggregates[1]["engine_b"], json!("random"));

        std::fs::remove_dir_all(&dir).ok();
    }

    fn cheap_adapters() -> Vec<Box<dyn EngineAdapter>> {
        vec![
            Box::new(RandomAdapter),
            Box::new(MinimaxAdapter {
                max_depth: 2,
                time_limit_s: Some(0.05),
            }),
        ]
    }

    fn two_position_payload() -> (Value, Value) {
        use crate::bitboard::Bitboard;
        use crate::state::State;
        let p0 = State::new(Bitboard::EMPTY).to_qfen();
        let p1 = State::new(Bitboard::EMPTY.with_move(0, 0, 0)).to_qfen();
        let full = json!({
            "positions": [
                {"id": "p0", "qfen": p0, "phase": "opening", "reference": Value::Null},
                {"id": "p1", "qfen": p1, "phase": "opening", "reference": Value::Null},
            ]
        });
        let truncated = json!({"positions": [full["positions"][0].clone()]});
        (full, truncated)
    }

    /// Adapted from the PR #10 resume test: interrupting a run after only
    /// the first position, then resuming over the full position list, must
    /// reproduce the same row multiset as an uninterrupted run (order need
    /// not match across the interrupted/resumed split, since resumed rows
    /// are the concatenation of loaded + freshly streamed).
    #[test]
    fn resume_after_interrupt_matches_uninterrupted_run() {
        let (full, truncated) = two_position_payload();
        let adapters = cheap_adapters();
        let seeds = [10u64, 11u64];
        let dir = scratch_dir("resume-interrupt");
        let root = dir.join("checkpoint");
        let obs_path = root.join(OBSERVATIONS);

        // "Interrupted" run: only the first position is processed.
        let interrupted_rows =
            run_agreement(&adapters, &truncated, &seeds, &HashSet::new(), 1, |row| {
                append_jsonl(&obs_path, row)
            })
            .unwrap();
        assert_eq!(
            interrupted_rows.len(),
            3,
            "1 position: random x2 + minimax x1"
        );

        // Resume: reload, seed the skip set, run over the full list.
        let loaded_rows = load_jsonl(&obs_path).unwrap();
        let skip: HashSet<ObservationKey> = key_set(&loaded_rows, observation_key);
        let fresh_rows = run_agreement(&adapters, &full, &seeds, &skip, 1, |row| {
            append_jsonl(&obs_path, row)
        })
        .unwrap();
        let mut resumed_rows = loaded_rows;
        resumed_rows.extend(fresh_rows);

        // From-scratch, uninterrupted run over the same full payload/seeds.
        let scratch_rows =
            run_agreement(&adapters, &full, &seeds, &HashSet::new(), 1, |_| Ok(())).unwrap();

        let fingerprint = |rows: &[Value]| -> BTreeSet<(ObservationKey, String)> {
            rows.iter()
                .map(|row| {
                    (
                        observation_key(row),
                        row["move"].as_str().unwrap_or_default().to_string(),
                    )
                })
                .collect()
        };
        assert_eq!(resumed_rows.len(), scratch_rows.len());
        assert_eq!(fingerprint(&resumed_rows), fingerprint(&scratch_rows));

        // Reloaded from disk, it must still match.
        let reloaded = load_jsonl(&obs_path).unwrap();
        assert_eq!(fingerprint(&reloaded), fingerprint(&scratch_rows));

        std::fs::remove_dir_all(&dir).ok();
    }
}
