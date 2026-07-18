use quantik_core::bench::portability::build_report;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn test_contracts_root(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "quantik-contracts-portability-{name}-{}-{nanos}",
        std::process::id()
    ));
    let fixture_dir = root.join("fixtures/api-portability");
    fs::create_dir_all(&fixture_dir).expect("fixture dir should be created");
    fs::write(root.join("VERSION"), "1.2.0\n").expect("VERSION should be written");
    fs::write(
        root.join("contracts.json"),
        r#"{
  "release_version": "1.2.0",
  "contracts": {
    "qfen": {"id": "qfen.v1"},
    "bitboard": {"id": "bitboard.v1"},
    "action_index": {"id": "action-index.v1"}
  }
}
"#,
    )
    .expect("contracts.json should be written");
    fs::write(
        fixture_dir.join("game-state-v1.json"),
        r#"{
  "schema": "api-portability-fixtures.v1",
  "contract_version": "1.2.0",
  "game_state_cases": [
    {
      "case_id": "empty-board",
      "qfen": "..../..../..../....",
      "move": {"shape": 0, "position": 0}
    },
    {
      "case_id": "single-p0-corner",
      "qfen": "A.../..../..../....",
      "move": {"shape": 1, "position": 5}
    },
    {
      "case_id": "single-p0-occupied-corner",
      "qfen": "A.../..../..../....",
      "move": {"shape": 1, "position": 0}
    },
    {
      "case_id": "mixed-asymmetric",
      "qfen": "Ab../..c./...D/....",
      "move": {"shape": 2, "position": 12}
    },
    {
      "case_id": "stalemate-p1-blocked",
      "qfen": "A..C/bbd./CD.A/.adB",
      "move": {"shape": 0, "position": 0}
    }
  ]
}
"#,
    )
    .expect("api portability fixture should be written");
    root
}

fn temp_report_path() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "quantik-rust-portability-report-{}-{nanos}.json",
        std::process::id(),
    ))
}

#[test]
fn build_report_projects_contract_metadata_and_sorted_cases() {
    let contracts_root = test_contracts_root("metadata");
    let report = build_report(&contracts_root).expect("report should build from fixture");

    assert_eq!(report["schema"], "api-portability-report.v1");
    assert_eq!(report["contracts_release"], "1.2.0");
    assert_eq!(report["implementation"]["language"], "rust");
    assert_eq!(report["implementation"]["package"], "quantik-core");
    assert_eq!(
        report["implementation"]["version"],
        env!("CARGO_PKG_VERSION")
    );
    assert_eq!(report["contract_ids"]["qfen"], "qfen.v1");
    assert_eq!(report["contract_ids"]["bitboard"], "bitboard.v1");
    assert_eq!(report["contract_ids"]["action_index"], "action-index.v1");

    let case_ids: Vec<_> = report["cases"]
        .as_array()
        .expect("cases should be an array")
        .iter()
        .map(|case| case["case_id"].as_str().unwrap())
        .collect();
    assert_eq!(
        case_ids,
        vec![
            "empty-board",
            "mixed-asymmetric",
            "single-p0-corner",
            "single-p0-occupied-corner",
            "stalemate-p1-blocked"
        ]
    );

    let _ = fs::remove_dir_all(contracts_root);
}

#[test]
fn build_report_projects_game_state_case_semantics() {
    let contracts_root = test_contracts_root("semantics");
    let report = build_report(&contracts_root).expect("report should build from fixture");
    let cases = report["cases"].as_array().unwrap();
    let empty = cases
        .iter()
        .find(|case| case["case_id"] == "empty-board")
        .unwrap();

    assert_eq!(empty["qfen"], "..../..../..../....");
    assert_eq!(
        empty["bitboards"],
        Value::Array((0..8).map(|_| Value::from(0)).collect())
    );
    assert_eq!(empty["side_to_move"], 0);
    assert_eq!(empty["canonical_qfen"], "..../..../..../....");
    assert_eq!(
        empty["canonical_key"],
        "010200000000000000000000000000000000"
    );
    assert_eq!(empty["orbit_size"], 1);
    assert_eq!(empty["legal_action_mask"], "0xffffffffffffffff");
    assert_eq!(
        empty["legal_action_indices"],
        Value::Array((0..64).map(Value::from).collect())
    );
    assert_eq!(empty["terminal"], false);
    assert_eq!(empty["winner"], "none");
    assert_eq!(empty["move"]["shape"], 0);
    assert_eq!(empty["move"]["position"], 0);
    assert_eq!(empty["move"]["action_index"], 0);
    assert_eq!(empty["move"]["is_legal"], true);
    assert_eq!(empty["move"]["after_qfen"], "A.../..../..../....");

    let occupied = cases
        .iter()
        .find(|case| case["case_id"] == "single-p0-occupied-corner")
        .unwrap();
    assert_eq!(occupied["move"]["shape"], 1);
    assert_eq!(occupied["move"]["position"], 0);
    assert_eq!(occupied["move"]["action_index"], 16);
    assert_eq!(occupied["move"]["is_legal"], false);
    assert!(occupied["move"]["after_qfen"].is_null());

    let stalemate = cases
        .iter()
        .find(|case| case["case_id"] == "stalemate-p1-blocked")
        .unwrap();
    assert_eq!(stalemate["side_to_move"], 1);
    assert_eq!(stalemate["legal_action_indices"], Value::Array(vec![]));
    assert_eq!(stalemate["legal_action_mask"], "0x0000000000000000");
    assert_eq!(stalemate["terminal"], true);
    assert_eq!(stalemate["winner"], "player0");
    assert_eq!(stalemate["move"]["is_legal"], false);
    assert!(stalemate["move"]["after_qfen"].is_null());

    let _ = fs::remove_dir_all(contracts_root);
}

#[test]
fn build_report_rejects_empty_case_fixture() {
    let contracts_root = test_contracts_root("empty-cases");
    fs::write(
        contracts_root.join("fixtures/api-portability/game-state-v1.json"),
        r#"{
  "schema": "api-portability-fixtures.v1",
  "contract_version": "1.2.0",
  "game_state_cases": []
}
"#,
    )
    .expect("api portability fixture should be overwritten");

    let error = build_report(&contracts_root).expect_err("empty cases should fail");
    assert!(error.contains("game_state_cases must be a non-empty array"));

    let _ = fs::remove_dir_all(contracts_root);
}

#[test]
fn cli_writes_report_to_requested_output() {
    let contracts_root = test_contracts_root("cli");
    let output_path = temp_report_path();
    let _ = fs::remove_file(&output_path);

    let binary = std::env::var("CARGO_BIN_EXE_quantik-portability-report")
        .expect("quantik-portability-report bin should be built for tests");
    let output = Command::new(binary)
        .arg("--contracts-root")
        .arg(&contracts_root)
        .arg("--output")
        .arg(&output_path)
        .output()
        .expect("CLI should run");

    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "CLI failed:\n{text}");

    let report: Value =
        serde_json::from_str(&fs::read_to_string(&output_path).unwrap()).expect("valid JSON");
    assert_eq!(report["schema"], "api-portability-report.v1");
    assert_eq!(report["cases"].as_array().unwrap().len(), 5);

    let _ = fs::remove_file(output_path);
    let _ = fs::remove_dir_all(contracts_root);
}
