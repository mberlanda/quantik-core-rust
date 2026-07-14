//! Reproducible benchmark result bundles (port of `benchmarks/bundle.py`).

use serde_json::{json, Map, Value};
use std::path::Path;
use std::process::Command;

use super::contracts::{CONTRACT_VERSION, GAME_RESULT_SCHEMA, OBSERVATION_SCHEMA};

pub const SCHEMA_VERSION: u64 = 1;

fn git_sha() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// Actual compiler version used at run time, e.g. `1.82.0` — mirrors the
/// Python `sys.version.split()[0]` bare-version convention.
/// `CARGO_PKG_RUST_VERSION` is the crate's declared MSRV, not the toolchain
/// actually compiling it, and is an empty string (not absent) when the
/// `rust-version` Cargo.toml field is unset, so it is not usable here.
fn rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| {
            String::from_utf8_lossy(&out.stdout)
                .split_whitespace()
                .nth(1)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "unknown".into())
}

/// Host and software fingerprint stored in result bundles. Mirrors the
/// Python `collect_environment`, with `rust_version` in place of
/// `python_version`.
pub fn collect_environment() -> Value {
    json!({
        "quantik_core_version": env!("CARGO_PKG_VERSION"),
        "git_sha": git_sha(),
        "rust_version": rustc_version(),
        "platform": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        "processor": std::env::consts::ARCH,
        "cpu_count": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        "total_memory_bytes": Value::Null,
    })
}

/// Assemble a JSON-serializable, self-describing benchmark result bundle.
pub fn make_bundle(
    config: Value,
    dataset_payload: &Value,
    observations: Vec<Value>,
    head_to_head: Value,
    aggregates: Value,
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

    // Local time with UTC offset, `%Y-%m-%dT%H:%M:%S%z` like Python.
    let started_at = chrono::Local::now()
        .format("%Y-%m-%dT%H:%M:%S%z")
        .to_string();

    json!({
        "contract_version": CONTRACT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "artifact_contracts": {
            "observations": OBSERVATION_SCHEMA,
            "head_to_head": GAME_RESULT_SCHEMA,
        },
        "environment": collect_environment(),
        "config": config,
        "dataset": {
            "checksum": dataset_payload.get("checksum").cloned().unwrap_or(Value::Null),
            "generator": dataset_payload["generator"],
            "seed": dataset_payload["seed"],
            "schema_version": dataset_payload["schema_version"],
            "positions": positions.len(),
            "phases": phases,
        },
        "observations": observations,
        "head_to_head": head_to_head,
        "aggregates": aggregates,
    })
}

/// Write a result bundle as JSON, creating parent directories as needed.
pub fn save_bundle(bundle: &Value, path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {parent:?}: {e}"))?;
    }
    let text =
        serde_json::to_string_pretty(bundle).map_err(|e| format!("serialize bundle: {e}"))?;
    std::fs::write(path, text + "\n").map_err(|e| format!("write {path:?}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_shape_and_save() {
        let dataset = json!({
            "checksum": "abc",
            "generator": "g/v1",
            "seed": 1,
            "schema_version": 1,
            "positions": [
                {"phase": "opening"}, {"phase": "opening"}, {"phase": "endgame"},
            ],
        });
        let bundle = make_bundle(
            json!({"family": "fixed"}),
            &dataset,
            vec![json!({"engine": "random"})],
            json!({"records": [], "aggregates": []}),
            json!({"agreement": [], "cost": [], "stability": []}),
        );
        assert_eq!(bundle["schema_version"], json!(1));
        assert_eq!(bundle["dataset"]["positions"], json!(3));
        assert_eq!(bundle["dataset"]["phases"]["opening"], json!(2));
        assert_eq!(bundle["config"]["family"], json!("fixed"));
        assert_eq!(bundle["observations"].as_array().unwrap().len(), 1);
        assert!(bundle["environment"]["quantik_core_version"].is_string());
        // Timestamp format sanity: 2026-07-12T01:23:45+0200.
        let ts = bundle["started_at"].as_str().unwrap();
        assert_eq!(ts.len(), 24, "{ts}");
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[10..11], "T");

        let dir = std::env::temp_dir().join(format!("quantik-bundle-{}", std::process::id()));
        let path = dir.join("nested").join("bundle.json");
        save_bundle(&bundle, &path).unwrap();
        let loaded: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(loaded["dataset"]["checksum"], json!("abc"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
