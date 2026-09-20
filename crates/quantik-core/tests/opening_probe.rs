//! Tests for `opening-probe.v1` (QW-004/W3).
//!
//! Two layers:
//!
//! * self-contained tests with hand-computed vectors from
//!   `docs/opening-probe-v1.md` (sections 3.5 and 3.6), which always run, so a
//!   standalone checkout still exercises the orientation logic; and
//! * fixture-driven tests over `quantik-core-contracts/fixtures/opening-probe/`,
//!   which skip with a message when that checkout is absent. Set
//!   `QUANTIK_CONTRACTS_DIR` to point at a contracts checkout elsewhere.

use quantik_core::bitboard::Bitboard;
use quantik_core::constants::{FLAG_CANON, VERSION};
use quantik_core::moves::is_move_legal;
use quantik_core::opening_book::{OpeningBookConfig, OpeningBookDatabase};
use quantik_core::opening_probe::{
    build_probe, encode_file, MissReason, ProbeBuildMeta, ProbeError, ProbeFile, ProbeLookup,
    ProbeRecord, ProbeStatus,
};
use quantik_core::qfen::bb_from_qfen;
use quantik_core::state::State;
use quantik_core::symmetry::SymmetryHandler;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

// ── helpers ─────────────────────────────────────────────────────────

fn bb(qfen: &str) -> Bitboard {
    bb_from_qfen(qfen).expect("valid qfen")
}

fn key_of(qfen: &str) -> [u8; 18] {
    State::new(bb(qfen)).canonical_key()
}

fn mask(actions: &[u8]) -> u64 {
    actions.iter().fold(0u64, |m, &a| m | 1u64 << a)
}

fn meta(book_id: &str) -> ProbeBuildMeta {
    ProbeBuildMeta {
        book_id: book_id.into(),
        generator: "test".into(),
        generator_version: "1".into(),
        contract_version: "1.3.0".into(),
        created_at: None,
        coverage_complete: Some(true),
    }
}

fn record(rep_qfen: &str, value: i8, actions: &[u8]) -> ProbeRecord {
    ProbeRecord {
        key: key_of(rep_qfen),
        game_value: value,
        status: ProbeStatus::Exact,
        optimal_actions: mask(actions),
    }
}

fn probe_of(records: Vec<ProbeRecord>) -> ProbeFile {
    ProbeFile::from_bytes(build_probe(records, &meta("book-a")).expect("build")).expect("open")
}

fn hit(file: &ProbeFile, qfen: &str) -> quantik_core::opening_probe::ProbeHit {
    match file.probe_detailed(&bb(qfen)).expect("no error") {
        ProbeLookup::Hit(h) => h,
        other => panic!("expected a hit for {qfen}, got {other:?}"),
    }
}

// ── symmetry primitive ──────────────────────────────────────────────

#[test]
fn find_canonical_with_transform_agrees_with_find_canonical() {
    for q in [
        "..../..../..../....",
        "A.../..../..../....",
        "..../.B../c.../A...",
        "B.../..../..c./....",
        "Ab../c.D./..../...A",
    ] {
        let b = bb(q);
        let (canon, t) = SymmetryHandler::find_canonical_with_transform(&b);
        assert_eq!(canon, SymmetryHandler::find_canonical(&b), "{q}");
        assert!(t < 192);
    }
}

#[test]
fn transform_matches_doc_examples_and_ties_take_lowest_index() {
    // Section 3.5: uniquely t* = 95.
    let (canon, t) = SymmetryHandler::find_canonical_with_transform(&bb("..../.B../c.../A..."));
    assert_eq!(t, 95);
    assert_eq!(canon, bb("..../..../.C../..bD"));
    // Section 3.6: minimisers {77, 91, 125, 139}; the contract fixes the lowest.
    let (_, t) = SymmetryHandler::find_canonical_with_transform(&bb("B.../..../..c./...."));
    assert_eq!(t, 77);
    // Identity: an already-canonical position with a large stabiliser.
    let (_, t) = SymmetryHandler::find_canonical_with_transform(&bb("..../..../..../D..."));
    assert_eq!(t, 0);
}

// ── self-contained hit / miss / orientation vectors ─────────────────

#[test]
fn hit_at_identity_transform() {
    let f = probe_of(vec![record("..../..../..../D...", 1, &[5, 21])]);
    let h = hit(&f, "..../..../..../D...");
    assert_eq!(h.transform_index, 0);
    assert_eq!(h.actions, vec![5, 21]);
    assert_eq!((h.game_value, h.status), (1, ProbeStatus::Exact));
}

#[test]
fn transformed_move_uses_the_inverse_transform() {
    // Section 3.5: stored 22 (B at 6) -> caller 42 (C at 10) via inverse(95) = 47.
    let f = probe_of(vec![record("..../..../.C../..bD", 1, &[22])]);
    let h = hit(&f, "..../.B../c.../A...");
    assert_eq!(h.transform_index, 95);
    assert_eq!(h.actions, vec![42]);
}

#[test]
fn wrong_direction_would_still_be_legal_and_is_not_returned() {
    // Stored 2 maps back to 59 (correct) or 52 (using t* itself, ALSO legal, so
    // the legality tripwire cannot catch that bug; only this assertion can).
    let caller = bb("..../.B../c.../A...");
    let f = probe_of(vec![record("..../..../.C../..bD", 1, &[2])]);
    let h = hit(&f, "..../.B../c.../A...");
    assert_eq!(h.actions, vec![59]);
    let wrong = SymmetryHandler::remap_action_index(2, 95);
    assert_eq!(wrong, 52);
    assert!(is_move_legal(&caller, 1, wrong / 16, wrong % 16));
    assert!(!h.actions.contains(&wrong));
}

#[test]
fn tie_returns_the_lowest_transform_result() {
    let f = probe_of(vec![record("..../..c./..../D...", 1, &[10])]);
    let h = hit(&f, "B.../..../..c./....");
    assert_eq!(h.transform_index, 77);
    assert_eq!(h.actions, vec![9]);
}

#[test]
fn bounded_unknown_is_a_hit_and_absent_is_a_miss() {
    let bounded = ProbeRecord {
        status: ProbeStatus::Bounded,
        game_value: 0,
        ..record("..../..../..../D...", 1, &[5, 21])
    };
    let f = probe_of(vec![bounded]);
    let h = hit(&f, "..../..../..../D...");
    assert_eq!((h.game_value, h.status), (0, ProbeStatus::Bounded));
    // Ply 1 is the whole range, and a centre piece is a different orbit from the
    // stored corner piece: a key-absent miss, distinct from the bounded hit above.
    assert_eq!(
        f.probe_detailed(&bb("..../.A../..../....")).unwrap(),
        ProbeLookup::Miss(MissReason::KeyAbsent)
    );
}

#[test]
fn ply_outside_coverage_is_a_miss_not_an_error() {
    let f = probe_of(vec![record("..../..../..../D...", 1, &[5])]);
    assert_eq!(
        f.probe_detailed(&bb("..../..../..../....")).unwrap(),
        ProbeLookup::Miss(MissReason::PlyOutsideCoverage)
    );
    assert_eq!(f.probe(&bb("..../..../..../....")).unwrap(), None);
}

#[test]
fn invalid_caller_position_is_an_error_not_a_miss() {
    let f = probe_of(vec![record("..../..../..../D...", 1, &[5])]);
    // Two player-0 pieces, none for player 1: turn balance invalid.
    let bad = Bitboard::new([0b01, 0b10, 0, 0, 0, 0, 0, 0]);
    let err = f.probe(&bad).unwrap_err();
    assert_eq!(err.kind(), "invalid caller position");
}

#[test]
fn corrupt_record_that_maps_to_an_illegal_move_trips_the_wire() {
    // Stored action 0 is A at position 0; the representative's own D sits there
    // only after transform, so map the caller to identity and pick an occupied
    // square: caller "..../..../..../D..." has D at 12, so A at 12 is action 12.
    let f = probe_of(vec![record("..../..../..../D...", 1, &[12])]);
    let err = f.probe(&bb("..../..../..../D...")).unwrap_err();
    assert_eq!(err.kind(), "illegal mapped-back action");
}

// ── fail-fast on mutated files ──────────────────────────────────────

fn good_bytes() -> Vec<u8> {
    build_probe(
        vec![
            record("..../..../..../D...", 1, &[5, 21]),
            record("..../..../.C../..bD", 1, &[22]),
        ],
        &meta("book-a"),
    )
    .unwrap()
}

fn kind_of(bytes: Vec<u8>) -> &'static str {
    ProbeFile::from_bytes(bytes).expect_err("must fail").kind()
}

#[test]
fn good_file_opens() {
    let f = ProbeFile::from_bytes(good_bytes()).unwrap();
    assert_eq!(f.len(), 2);
    assert_eq!(f.header().book_id, "book-a");
}

#[test]
fn truncated_and_short_files_fail() {
    let good = good_bytes();
    assert_eq!(kind_of(good[..good.len() - 1].to_vec()), "truncated");
    assert_eq!(kind_of(good[..5].to_vec()), "truncated");
    let mut long = good.clone();
    long.push(0);
    assert_eq!(kind_of(long), "truncated");
}

#[test]
fn bad_magic_and_future_format_major_fail() {
    let mut b = good_bytes();
    b[0] = b'X';
    assert_eq!(kind_of(b), "corrupt");
    let mut b = good_bytes();
    b[7] = 2;
    assert_eq!(kind_of(b), "incompatible version");
}

#[test]
fn flipped_body_bit_is_a_checksum_mismatch() {
    let mut b = good_bytes();
    let n = b.len();
    b[n - 1] ^= 0x01; // top byte of the last action mask: value stays valid
    assert_eq!(kind_of(b), "checksum mismatch");
}

#[test]
fn unsorted_table_with_a_valid_checksum_fails() {
    // Build the raw file by hand so the checksum covers the unsorted body.
    let mut recs = [
        record("..../..../..../D...", 1, &[5]),
        record("..../..../.C../..bD", 1, &[22]),
    ];
    recs.sort_by_key(|r| std::cmp::Reverse(r.key)); // descending
    let f = raw_file(&recs, |_| {});
    assert_eq!(kind_of(f), "unsorted or duplicate keys");
    let dup = [recs[0], recs[0]];
    assert_eq!(
        kind_of(raw_file(&dup, |_| {})),
        "unsorted or duplicate keys"
    );
}

/// A file with a correct checksum and counts over `recs` exactly as given
/// (unsorted allowed); `tweak` edits the header first.
fn raw_file(recs: &[ProbeRecord], tweak: impl FnOnce(&mut Value)) -> Vec<u8> {
    let good: Value = {
        let bytes = good_bytes();
        let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        serde_json::from_slice(&bytes[12..12 + len]).unwrap()
    };
    let mut header = good;
    let body: Vec<u8> = recs.iter().flat_map(|r| r.to_bytes()).collect();
    header["entry_count"] = (recs.len() as u64).into();
    header["body_sha256"] = hex(&Sha256::digest(&body)).into();
    let mut per_ply = std::collections::BTreeMap::new();
    for r in recs {
        let ply: u32 = r.key[2..].iter().map(|b| b.count_ones()).sum();
        *per_ply.entry(ply).or_insert(0u64) += 1;
    }
    header["per_ply"] = per_ply
        .iter()
        .map(|(p, n)| serde_json::json!({"ply": p, "entries": n}))
        .collect::<Vec<_>>()
        .into();
    header["ply_min"] = (*per_ply.keys().next().unwrap()).into();
    header["ply_max"] = (*per_ply.keys().last().unwrap()).into();
    tweak(&mut header);
    encode_file(&header, &body)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn unknown_optional_header_key_is_ignored_at_runtime() {
    let recs = [record("..../..../..../D...", 1, &[5])];
    let f = raw_file(&recs, |h| {
        h["some_future_optional_key"] = "x".into();
    });
    assert!(ProbeFile::from_bytes(f).is_ok());
}

#[test]
fn header_violations_are_corrupt_or_incompatible() {
    let recs = [record("..../..../..../D...", 1, &[5])];
    assert_eq!(
        kind_of(raw_file(&recs, |h| h["format_major"] = 2.into())),
        "incompatible version"
    );
    assert_eq!(
        kind_of(raw_file(&recs, |h| h["key_format"] = "canonical_key.v2".into())),
        "incompatible version"
    );
    assert_eq!(
        kind_of(raw_file(&recs, |h| h["schema"] = "opening-probe.v2".into())),
        "incompatible version"
    );
    assert_eq!(
        kind_of(raw_file(&recs, |h| h["record_size"] = 24.into())),
        "corrupt"
    );
    assert_eq!(
        kind_of(raw_file(&recs, |h| {
            h.as_object_mut().unwrap().remove("body_sha256");
        })),
        "corrupt"
    );
    assert_eq!(
        kind_of(raw_file(&recs, |h| h["per_ply"][0]["entries"] = 2.into())),
        "corrupt"
    );
    assert_eq!(
        kind_of(raw_file(&recs, |h| h["per_ply"][0]["ply"] = 2.into())),
        "corrupt"
    );
}

#[test]
fn record_field_violations_are_corrupt() {
    let base = record("..../..../..../D...", 1, &[5]);
    let mut r = base;
    r.key[1] = 0x03;
    assert_eq!(kind_of(raw_file(&[r], |_| {})), "corrupt");
    let mut r = base;
    r.key[0] = 0x02;
    assert_eq!(kind_of(raw_file(&[r], |_| {})), "corrupt");
    let mut r = base;
    r.game_value = 0; // exact with 0
    assert_eq!(kind_of(raw_file(&[r], |_| {})), "corrupt");
    let mut r = base;
    r.game_value = 2;
    assert_eq!(kind_of(raw_file(&[r], |_| {})), "corrupt");
    let mut r = base;
    r.optimal_actions = 0; // non-terminal
    assert_eq!(kind_of(raw_file(&[r], |_| {})), "corrupt");
    // status byte 3 cannot be built through ProbeRecord; patch the bytes.
    let mut bytes = raw_file(&[base], |_| {});
    let n = bytes.len();
    bytes[n - 28 + 19] = 3;
    // checksum now stale, but the record check runs first and names the fault.
    assert_eq!(kind_of(bytes), "corrupt");
}

#[test]
fn empty_actions_are_legal_on_a_terminal_position() {
    // AbCd in row 0 is a completed line: terminal, value -1 for the mover.
    let f = probe_of(vec![record("AbCd/..../..../....", -1, &[])]);
    let h = hit(&f, "AbCd/..../..../....");
    assert!(h.actions.is_empty());
}

#[test]
fn stale_check_is_opt_in() {
    let bytes = good_bytes();
    assert!(ProbeFile::from_bytes_expecting_book_id(bytes.clone(), "book-a").is_ok());
    let err = ProbeFile::from_bytes_expecting_book_id(bytes.clone(), "book-b").unwrap_err();
    assert_eq!(err.kind(), "stale");
    assert!(ProbeFile::from_bytes(bytes).is_ok());
}

#[test]
fn key_order_is_bytewise_not_numeric() {
    // Plane 0 words 0x0001 (bytes 01 00) and 0x0100 (bytes 00 01): bytewise the
    // second sorts first, numerically last. The reader has no canonicality check
    // on keys, so synthetic keys with a non-empty action set are enough.
    let mk = |plane0: u16| {
        let mut key = [0u8; 18];
        key[0] = VERSION;
        key[1] = FLAG_CANON;
        key[2..4].copy_from_slice(&plane0.to_le_bytes());
        ProbeRecord {
            key,
            game_value: 1,
            status: ProbeStatus::Exact,
            optimal_actions: 1,
        }
    };
    let (low_numeric, high_numeric) = (mk(0x0001), mk(0x0100));
    let f = probe_of(vec![low_numeric, high_numeric]);
    assert_eq!(f.record(0).unwrap().key, high_numeric.key, "bytewise order");
    assert_eq!(f.record(1).unwrap().key, low_numeric.key);
    let numeric_order = raw_file(&[low_numeric, high_numeric], |_| {});
    assert_eq!(kind_of(numeric_order), "unsorted or duplicate keys");
}

// ── builder end to end ──────────────────────────────────────────────

#[test]
fn probe_builder_projects_a_book_and_answers_in_the_callers_orientation() {
    let dir = std::env::temp_dir().join(format!("qw004-probe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("book.db");
    let out_path = dir.join("book.probe");
    let _ = std::fs::remove_file(&db_path);
    {
        let db = OpeningBookDatabase::open(&OpeningBookConfig {
            database_path: db_path.to_string_lossy().into(),
            cache_size_mb: 1,
            enable_wal: false,
        })
        .unwrap();
        // The book stores the representative's frame: B at position 6 = (1, 6).
        let rep = State::from_qfen("..../..../.C../..bD").unwrap();
        assert!(db.add_solved_position(&rep, 1, &[(1, 6)]).unwrap());
    }
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_probe_builder"))
        .args(["--book", db_path.to_str().unwrap()])
        .args(["--out", out_path.to_str().unwrap()])
        .output()
        .expect("probe_builder runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let f = ProbeFile::open(&out_path).unwrap();
    assert_eq!(f.len(), 1);
    assert!(f.header().book_id.starts_with("sha256:"));
    // The representative itself: stored frame, identity.
    let h = f.probe(&bb("..../..../.C../..bD")).unwrap().unwrap();
    assert_eq!((h.transform_index, h.actions), (0, vec![22]));
    // A rotated / relabelled caller: mapped back, not returned as stored.
    let h = f.probe(&bb("..../.B../c.../A...")).unwrap().unwrap();
    assert_eq!((h.transform_index, h.actions), (95, vec![42]));
    let _ = std::fs::remove_dir_all(&dir);
}

// ── contracts fixtures ──────────────────────────────────────────────

fn fixture_dir() -> Option<PathBuf> {
    let dir = match std::env::var("QUANTIK_CONTRACTS_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../quantik-core-contracts"),
    }
    .join("fixtures/opening-probe");
    if dir.is_dir() {
        Some(dir)
    } else if std::env::var("QUANTIK_REQUIRE_FIXTURES").is_ok() {
        panic!(
            "QUANTIK_REQUIRE_FIXTURES is set but {} is missing",
            dir.display()
        );
    } else {
        eprintln!(
            "SKIPPING opening-probe fixture tests: {} not found (set QUANTIK_CONTRACTS_DIR \
             or check out quantik-core-contracts next to this repository)",
            dir.display()
        );
        None
    }
}

struct Encoded {
    bytes: Vec<u8>,
    /// False when the row cannot be expressed in the binary format at all.
    encodable: bool,
}

/// Fixture row (decoded probe) to file bytes, leniently: invalid values that fit
/// a byte are written as-is so the reader, not the encoder, rejects them.
fn encode_row(row: &Value) -> Encoded {
    let mut body = Vec::new();
    let mut encodable = true;
    for r in row["records"].as_array().unwrap() {
        let key = r["key"].as_str().unwrap();
        let key: Vec<u8> = (0..key.len() / 2)
            .map(|i| u8::from_str_radix(&key[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        let status = match r["status"].as_str().unwrap() {
            "exact" => 1u8,
            "bounded" => 2,
            _ => 0xEE,
        };
        let mut m = 0u64;
        for a in r["optimal_actions"].as_array().unwrap() {
            let a = a.as_u64().unwrap();
            if a > 63 {
                encodable = false;
            } else {
                m |= 1 << a;
            }
        }
        body.extend_from_slice(&key);
        body.push(r["game_value"].as_i64().unwrap() as i8 as u8);
        body.push(status);
        body.extend_from_slice(&m.to_le_bytes());
    }
    let mut bytes = encode_file(&row["header"], &body);
    if let Some(n) = row.get("file_length").and_then(Value::as_u64) {
        bytes.resize(n as usize, 0);
    }
    Encoded { bytes, encodable }
}

/// Run one `probe_cases` entry; `Err(kind)` is a fail-fast error kind.
fn run_case(bytes: &[u8], case: &Value) -> Result<ProbeLookup, String> {
    let qfen = case["caller_qfen"].as_str().unwrap();
    let opened = match case.get("expected_book_id").and_then(Value::as_str) {
        Some(id) => ProbeFile::from_bytes_expecting_book_id(bytes.to_vec(), id),
        None => ProbeFile::from_bytes(bytes.to_vec()),
    };
    let file = opened.map_err(|e| e.kind().to_string())?;
    file.probe_detailed(&bb(qfen))
        .map_err(|e| e.kind().to_string())
}

#[test]
fn contracts_fixtures_valid_rows() {
    let Some(dir) = fixture_dir() else { return };
    let text = std::fs::read_to_string(dir.join("opening-probe-v1-synthetic.jsonl")).unwrap();
    let (mut rows, mut cases, mut hits, mut misses, mut errors, mut transformed) =
        (0, 0, 0, 0, 0, 0);
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let row: Value = serde_json::from_str(line).unwrap();
        let id = row["case_id"].as_str().unwrap();
        let enc = encode_row(&row);
        assert!(enc.encodable, "{id}");
        // Every valid row must open.
        ProbeFile::from_bytes(enc.bytes.clone()).unwrap_or_else(|e| panic!("{id}: {e}"));
        rows += 1;
        for case in row["probe_cases"].as_array().unwrap() {
            cases += 1;
            let cid = case["case_id"].as_str().unwrap();
            let exp = &case["expected"];
            let got = run_case(&enc.bytes, case);
            match exp["outcome"].as_str().unwrap() {
                "hit" => {
                    hits += 1;
                    let Ok(ProbeLookup::Hit(h)) = got else {
                        panic!("{id}/{cid}: expected a hit, got {got:?}")
                    };
                    let t = exp["transform_index"].as_u64().unwrap() as u8;
                    assert_eq!(h.transform_index, t, "{id}/{cid} transform_index");
                    if t != 0 {
                        transformed += 1;
                    }
                    assert_eq!(h.game_value as i64, exp["game_value"].as_i64().unwrap());
                    assert_eq!(h.status.as_str(), exp["status"].as_str().unwrap());
                    let actions: Vec<u8> = exp["actions"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|a| a.as_u64().unwrap() as u8)
                        .collect();
                    assert_eq!(h.actions, actions, "{id}/{cid} actions");
                    // Optional extras: recompute what the fixture asserts.
                    let stored = stored_actions(&row, qfen_key(case));
                    if let Some(w) = exp.get("wrong_direction_actions") {
                        let mut wrong: Vec<u8> = stored
                            .iter()
                            .map(|&a| SymmetryHandler::remap_action_index(a, t))
                            .collect();
                        wrong.sort_unstable();
                        let want: Vec<u8> = w
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|a| a.as_u64().unwrap() as u8)
                            .collect();
                        assert_eq!(wrong, want, "{id}/{cid} wrong_direction_actions");
                        let caller = bb(case["caller_qfen"].as_str().unwrap());
                        let side = if caller.player_piece_count(0) == caller.player_piece_count(1) {
                            0
                        } else {
                            1
                        };
                        let legal = wrong
                            .iter()
                            .all(|&a| is_move_legal(&caller, side, a / 16, a % 16));
                        assert_eq!(
                            legal,
                            exp["wrong_direction_legal"].as_bool().unwrap(),
                            "{id}/{cid} wrong_direction_legal"
                        );
                        // The contract's discriminating case: the wrong direction is
                        // legal and differs from the right answer.
                        if legal && wrong != h.actions {
                            assert_ne!(wrong, h.actions);
                        }
                    }
                    if let Some(m) = exp.get("minimiser_transform_indices") {
                        assert_eq!(
                            m[0].as_u64().unwrap() as u8,
                            t,
                            "{id}/{cid} lowest minimiser"
                        );
                    }
                }
                "miss" => {
                    misses += 1;
                    let reason = match exp["reason"].as_str().unwrap() {
                        "key_absent" => MissReason::KeyAbsent,
                        "ply_outside_coverage" => MissReason::PlyOutsideCoverage,
                        r => panic!("unknown miss reason {r}"),
                    };
                    assert_eq!(got, Ok(ProbeLookup::Miss(reason)), "{id}/{cid}");
                }
                "error" => {
                    errors += 1;
                    assert_eq!(
                        got.unwrap_err(),
                        exp["error"].as_str().unwrap(),
                        "{id}/{cid}"
                    );
                }
                o => panic!("unknown outcome {o}"),
            }
        }
    }
    eprintln!(
        "valid fixtures: {rows} rows, {cases} cases ({hits} hit, {misses} miss, {errors} error, \
         {transformed} with a non-identity transform)"
    );
    assert!(rows >= 5 && hits >= 5 && misses >= 2 && errors >= 1 && transformed >= 3);
}

/// The (only) record of a row whose key is the canonical key of the caller.
fn qfen_key(case: &Value) -> [u8; 18] {
    key_of(case["caller_qfen"].as_str().unwrap())
}

fn stored_actions(row: &Value, key: [u8; 18]) -> Vec<u8> {
    let want = hex(&key);
    row["records"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["key"].as_str().unwrap() == want)
        .map(|r| {
            r["optimal_actions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| a.as_u64().unwrap() as u8)
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn contracts_fixtures_invalid_rows_fail_fast() {
    let Some(dir) = fixture_dir() else { return };
    let doc: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("opening-probe-v1-invalid.json")).unwrap(),
    )
    .unwrap();
    let (mut checked, mut skipped) = (0, 0);
    for case in doc["cases"].as_array().unwrap() {
        let id = case["case_id"].as_str().unwrap();
        let want = case["expected_error"].as_str().unwrap();
        let row = &case["row"];
        let enc = encode_row(row);
        if !enc.encodable {
            // Action 64 does not exist in a 64-bit set; the binary format cannot
            // express this defect at all, which is the point of the fixed field.
            eprintln!("skip {id}: not expressible in the binary format");
            skipped += 1;
            continue;
        }
        if id == "corrupt-unknown-header-key" {
            // Fixture headers are closed; a runtime reader must still open it.
            ProbeFile::from_bytes(enc.bytes).expect("unknown optional key must not fail an open");
            checked += 1;
            continue;
        }
        let got = match ProbeFile::from_bytes(enc.bytes.clone()) {
            Err(e) => e.kind().to_string(),
            Ok(file) => {
                // Defect only visible on a lookup (the orientation tripwire).
                row["probe_cases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find_map(|c| {
                        file.probe(&bb(c["caller_qfen"].as_str().unwrap()))
                            .err()
                            .map(|e| e.kind().to_string())
                    })
                    .unwrap_or_else(|| "no error".into())
            }
        };
        assert_eq!(got, want, "{id}");
        checked += 1;
    }
    eprintln!("invalid fixtures: {checked} checked, {skipped} skipped");
    assert!(checked >= 20 && skipped <= 1);
}

#[test]
fn error_kinds_are_the_section_5_names() {
    let e = ProbeError::ChecksumMismatch;
    assert_eq!(e.kind(), "checksum mismatch");
    assert!(e.to_string().contains("checksum"));
}
