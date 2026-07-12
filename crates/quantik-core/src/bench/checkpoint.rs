//! Crash-safe checkpoint/resume for benchmark `run` (beyond the Python
//! harness, which has no checkpointing).
//!
//! The checkpoint file is JSON Lines: a header line identifying which
//! dataset and run configuration it belongs to, followed by one line per
//! completed observation row or head-to-head record. Every line is flushed
//! to the OS individually (crash-safe against process death — Ctrl-C,
//! panic, kill; not fsync'd against power loss), so a crash loses at most
//! the line in flight, and the loader tolerates a truncated tail —
//! everything written before it is intact and reloadable.
//!
//! ```text
//! {"kind":"header","dataset_checksum":"...","config_fingerprint":"..."}
//! {"kind":"observation","row":{...}}
//! {"kind":"h2h","record":{...}}
//! ```
//!
//! `config_fingerprint` is a sha256 over the canonical JSON of the run
//! config with `output`/`checkpoint` stripped, so resuming with the same
//! engine settings but a different `--output` (or `--checkpoint`) path still
//! matches. A checkpoint whose header doesn't match the current run's
//! dataset checksum or config fingerprint is refused outright — runs are
//! never silently mixed.

use crate::bench::agreement::{row_key, RunKey};
use crate::bench::canonical::canonical_json;
use crate::bench::head_to_head::{record_key, GameKey};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// sha256 (hex) of the canonical JSON of a run config, excluding the
/// `output` and `checkpoint` fields (so those may differ across a
/// checkpoint/resume pair without invalidating it).
pub fn config_fingerprint(config: &Value) -> String {
    let mut stripped = config.as_object().cloned().unwrap_or_default();
    stripped.remove("output");
    stripped.remove("checkpoint");
    let blob = canonical_json(&Value::Object(stripped));
    let mut hasher = Sha256::new();
    hasher.update(blob.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Rows/records recovered from a checkpoint, plus the skip sets derived
/// from them so a resumed run doesn't repeat completed work.
#[derive(Debug, Default)]
pub struct CheckpointState {
    pub rows: Vec<Value>,
    pub records: Vec<Value>,
    pub row_skip: HashSet<RunKey>,
    pub record_skip: HashSet<GameKey>,
}

/// Parsed checkpoint contents: the header, the (kind-tagged) body lines,
/// and the byte length of the file up to and including the last line that
/// parsed successfully. A truncated final line (partial write from a
/// crash) is excluded from both `entries` and `valid_len`.
///
/// `ends_with_newline` is false when the kept region's final line parsed
/// as valid JSON but its trailing `\n` was lost (a crash can persist the
/// bytes of a line without its newline); appending directly after such a
/// line would merge two records into one corrupt line, so
/// [`CheckpointWriter::resume`] must restore the newline first.
struct ParsedCheckpoint {
    header: Value,
    entries: Vec<Value>,
    valid_len: u64,
    ends_with_newline: bool,
}

fn parse_checkpoint_file(path: &Path) -> Result<ParsedCheckpoint, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let chunks: Vec<&str> = text.split_inclusive('\n').collect();
    let last_index = chunks.len().saturating_sub(1);

    let mut offset: u64 = 0;
    let mut valid_len: u64 = 0;
    let mut header: Option<Value> = None;
    let mut entries = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        offset += chunk.len() as u64;
        let line = chunk.trim_end_matches('\n');
        if line.trim().is_empty() {
            valid_len = offset;
            continue;
        }
        let parsed: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) if i == last_index => break, // tolerate a truncated tail line
            Err(e) => return Err(format!("corrupt checkpoint line {}: {e}", i + 1)),
        };
        if header.is_none() {
            if parsed["kind"] != json!("header") {
                return Err("checkpoint file is missing its header line".into());
            }
            header = Some(parsed);
        } else {
            entries.push(parsed);
        }
        valid_len = offset;
    }

    let header = header.ok_or("checkpoint file is empty or missing its header line")?;
    let ends_with_newline = valid_len == 0 || text.as_bytes()[valid_len as usize - 1] == b'\n';
    Ok(ParsedCheckpoint {
        header,
        entries,
        valid_len,
        ends_with_newline,
    })
}

/// Load and validate a checkpoint file, returning the completed
/// observation rows and head-to-head records plus their skip sets.
///
/// The header's `dataset_checksum` and `config_fingerprint` must match the
/// current run exactly; a mismatch is an error (never silently mixed
/// runs). A truncated trailing line (a crash mid-write of the last entry)
/// is tolerated and simply dropped.
pub fn load_checkpoint(
    path: &Path,
    expected_checksum: &str,
    expected_fingerprint: &str,
) -> Result<CheckpointState, String> {
    let parsed = parse_checkpoint_file(path)?;
    let checksum = parsed.header["dataset_checksum"].as_str().unwrap_or("");
    let fingerprint = parsed.header["config_fingerprint"].as_str().unwrap_or("");
    if checksum != expected_checksum {
        return Err(format!(
            "checkpoint dataset checksum mismatch: file has {checksum:?}, this run expects \
             {expected_checksum:?} (different --dataset?); refusing to mix runs"
        ));
    }
    if fingerprint != expected_fingerprint {
        return Err(format!(
            "checkpoint config fingerprint mismatch: file has {fingerprint:?}, this run expects \
             {expected_fingerprint:?} (engine settings changed?); refusing to mix runs"
        ));
    }

    let mut state = CheckpointState::default();
    for entry in parsed.entries {
        match entry["kind"].as_str() {
            Some("observation") => {
                let row = entry["row"].clone();
                state.row_skip.insert(row_key(&row));
                state.rows.push(row);
            }
            Some("h2h") => {
                let record = entry["record"].clone();
                state.record_skip.insert(record_key(&record));
                state.records.push(record);
            }
            other => return Err(format!("unknown checkpoint entry kind: {other:?}")),
        }
    }
    Ok(state)
}

/// Appends JSON Lines to a checkpoint file, flushing to the OS after every
/// line so a process crash loses at most the line currently being written
/// (not fsync'd, so power loss can lose more — the loader tolerates a
/// truncated tail either way).
#[derive(Debug)]
pub struct CheckpointWriter {
    file: BufWriter<File>,
}

impl CheckpointWriter {
    /// Start a brand-new checkpoint and write its header line. Refuses if
    /// the file already exists — callers must pass `--resume` or delete it
    /// first, so a fresh run never silently clobbers completed work.
    pub fn create(
        path: &Path,
        dataset_checksum: &str,
        config_fingerprint: &str,
    ) -> Result<Self, String> {
        if path.exists() {
            return Err(format!(
                "checkpoint file already exists at {}; pass --resume to continue it, or delete \
                 it to start a fresh run",
                path.display()
            ));
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {parent:?}: {e}"))?;
            }
        }
        let file = File::create(path).map_err(|e| format!("create {path:?}: {e}"))?;
        let mut writer = CheckpointWriter {
            file: BufWriter::new(file),
        };
        writer.write_line(&json!({
            "kind": "header",
            "dataset_checksum": dataset_checksum,
            "config_fingerprint": config_fingerprint,
        }))?;
        Ok(writer)
    }

    /// Reopen an already-validated checkpoint (see [`load_checkpoint`]) for
    /// appending. Any truncated trailing line left by a prior crash is
    /// dropped first, and if the last kept line is valid JSON that lost
    /// only its trailing newline, the newline is restored — so the file
    /// only ever holds complete, newline-terminated lines before new
    /// writes resume.
    pub fn resume(path: &Path) -> Result<Self, String> {
        let parsed = parse_checkpoint_file(path)?;
        {
            let truncator = std::fs::OpenOptions::new()
                .write(true)
                .open(path)
                .map_err(|e| format!("open {path:?}: {e}"))?;
            truncator
                .set_len(parsed.valid_len)
                .map_err(|e| format!("truncate {path:?}: {e}"))?;
        }
        let file = std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|e| format!("open {path:?}: {e}"))?;
        let mut writer = CheckpointWriter {
            file: BufWriter::new(file),
        };
        if !parsed.ends_with_newline {
            // A crash can persist a complete final line but drop its
            // newline; appending straight after it would merge two records
            // into one corrupt line and break the *next* resume.
            writer
                .file
                .write_all(b"\n")
                .and_then(|()| writer.file.flush())
                .map_err(|e| format!("repair {path:?}: {e}"))?;
        }
        Ok(writer)
    }

    fn write_line(&mut self, value: &Value) -> Result<(), String> {
        let line = serde_json::to_string(value).map_err(|e| format!("serialize: {e}"))?;
        writeln!(self.file, "{line}").map_err(|e| format!("write checkpoint: {e}"))?;
        self.file
            .flush()
            .map_err(|e| format!("flush checkpoint: {e}"))
    }

    /// Record one completed observation row (checkpoint hook for
    /// `run_agreement`'s `on_row` callback).
    pub fn write_row(&mut self, row: &Value) -> Result<(), String> {
        self.write_line(&json!({"kind": "observation", "row": row}))
    }

    /// Record one completed head-to-head record (checkpoint hook for
    /// `run_head_to_head`'s `on_record` callback).
    pub fn write_record(&mut self, record: &Value) -> Result<(), String> {
        self.write_line(&json!({"kind": "h2h", "record": record}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::adapters::{EngineAdapter, MinimaxAdapter, RandomAdapter};
    use crate::bench::agreement::{self, run_agreement};
    use crate::bitboard::Bitboard;
    use crate::state::State;
    use std::collections::BTreeSet;

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("quantik-ckpt-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn two_position_payload() -> (Value, Value) {
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

    fn cheap_adapters() -> Vec<Box<dyn EngineAdapter>> {
        vec![
            Box::new(RandomAdapter),
            Box::new(MinimaxAdapter {
                max_depth: 2,
                time_limit_s: Some(0.05),
            }),
        ]
    }

    /// Row identity (RunKey) + selected move, used to compare row
    /// *multisets* (spec: final ordering need not match, content must).
    fn row_fingerprints(rows: &[Value]) -> BTreeSet<(RunKey, String)> {
        rows.iter()
            .map(|row| {
                (
                    agreement::row_key(row),
                    row["move"].as_str().unwrap_or_default().to_string(),
                )
            })
            .collect()
    }

    #[test]
    fn interrupted_then_resumed_run_matches_uninterrupted_run() {
        let (full, truncated) = two_position_payload();
        let adapters = cheap_adapters();
        let seeds = [10u64, 11u64];

        let dir = scratch_dir("resume-match");
        let ckpt_path = dir.join("run.ckpt");
        let dataset_checksum = "dataset-checksum-abc";
        let config_fp = "config-fingerprint-xyz";

        // "Interrupted" run: only the first position is processed, streamed
        // to the checkpoint as it completes.
        {
            let mut writer =
                CheckpointWriter::create(&ckpt_path, dataset_checksum, config_fp).unwrap();
            run_agreement(&adapters, &truncated, &seeds, &HashSet::new(), |row| {
                writer.write_row(row)
            })
            .unwrap();
        }

        // Resume: load what's there, seed the skip set, run the full
        // position list, and append newly completed rows.
        let state = load_checkpoint(&ckpt_path, dataset_checksum, config_fp).unwrap();
        assert_eq!(
            state.rows.len(),
            3,
            "1 position: random x2 seeds + minimax x1 seed"
        );
        let mut writer = CheckpointWriter::resume(&ckpt_path).unwrap();
        let fresh_rows = run_agreement(&adapters, &full, &seeds, &state.row_skip, |row| {
            writer.write_row(row)
        })
        .unwrap();
        let mut resumed_rows = state.rows;
        resumed_rows.extend(fresh_rows);

        // From-scratch, uninterrupted run over the same full payload/seeds.
        let scratch_rows =
            run_agreement(&adapters, &full, &seeds, &HashSet::new(), |_| Ok(())).unwrap();

        assert_eq!(resumed_rows.len(), scratch_rows.len());
        assert_eq!(
            row_fingerprints(&resumed_rows),
            row_fingerprints(&scratch_rows)
        );

        // The checkpoint file itself, reloaded fresh, must also match.
        let reloaded = load_checkpoint(&ckpt_path, dataset_checksum, config_fp).unwrap();
        assert_eq!(reloaded.rows.len(), scratch_rows.len());
        assert_eq!(
            row_fingerprints(&reloaded.rows),
            row_fingerprints(&scratch_rows)
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn create_refuses_when_file_already_exists() {
        let dir = scratch_dir("exists");
        let path = dir.join("run.ckpt");
        CheckpointWriter::create(&path, "c", "f").unwrap();
        let err = CheckpointWriter::create(&path, "c", "f").unwrap_err();
        assert!(err.contains("--resume"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mismatched_dataset_checksum_refuses_resume() {
        let dir = scratch_dir("bad-checksum");
        let path = dir.join("run.ckpt");
        CheckpointWriter::create(&path, "checksum-a", "fingerprint-a").unwrap();
        let err = load_checkpoint(&path, "checksum-b", "fingerprint-a").unwrap_err();
        assert!(err.contains("checksum"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mismatched_config_fingerprint_refuses_resume() {
        let dir = scratch_dir("bad-fingerprint");
        let path = dir.join("run.ckpt");
        CheckpointWriter::create(&path, "checksum-a", "fingerprint-a").unwrap();
        let err = load_checkpoint(&path, "checksum-a", "fingerprint-b").unwrap_err();
        assert!(err.contains("fingerprint"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn truncated_trailing_line_is_tolerated() {
        let dir = scratch_dir("truncated");
        let path = dir.join("run.ckpt");
        {
            let mut writer = CheckpointWriter::create(&path, "c", "f").unwrap();
            writer
                .write_row(&json!({
                    "engine": "random", "config_label": "random",
                    "position_id": "p0", "seed": 0, "move": "0:0:0",
                }))
                .unwrap();
            writer
                .write_row(&json!({
                    "engine": "random", "config_label": "random",
                    "position_id": "p0", "seed": 1, "move": "0:0:1",
                }))
                .unwrap();
        }
        // Simulate a crash mid-write of the third line: partial JSON, no
        // trailing newline.
        {
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            write!(
                file,
                "{{\"kind\":\"observation\",\"row\":{{\"engine\":\"rand"
            )
            .unwrap();
        }

        let state = load_checkpoint(&path, "c", "f").unwrap();
        assert_eq!(
            state.rows.len(),
            2,
            "the truncated third line must be dropped"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resume_truncates_dangling_partial_line_before_appending() {
        let dir = scratch_dir("resume-truncate");
        let path = dir.join("run.ckpt");
        {
            let mut writer = CheckpointWriter::create(&path, "c", "f").unwrap();
            writer
                .write_row(&json!({
                    "engine": "random", "config_label": "random",
                    "position_id": "p0", "seed": 0, "move": "0:0:0",
                }))
                .unwrap();
        }
        {
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            write!(file, "{{\"kind\":\"observation\",\"row\":{{\"garbage").unwrap();
        }

        // Resuming must drop the dangling partial line, then append a
        // fresh, well-formed one after it — leaving the file entirely
        // parseable (no leftover garbage line in the middle).
        {
            let mut writer = CheckpointWriter::resume(&path).unwrap();
            writer
                .write_row(&json!({
                    "engine": "random", "config_label": "random",
                    "position_id": "p1", "seed": 0, "move": "0:0:1",
                }))
                .unwrap();
        }

        let state = load_checkpoint(&path, "c", "f").unwrap();
        assert_eq!(state.rows.len(), 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resume_repairs_valid_final_line_missing_only_its_newline() {
        let dir = scratch_dir("newline-lost");
        let path = dir.join("run.ckpt");
        {
            let mut writer = CheckpointWriter::create(&path, "c", "f").unwrap();
            writer
                .write_row(&json!({
                    "engine": "random", "config_label": "random",
                    "position_id": "p0", "seed": 0, "move": "0:0:0",
                }))
                .unwrap();
        }
        // Simulate a crash that persisted the full final line but dropped
        // only its trailing newline: chop exactly one byte off the file.
        {
            let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            let len = file.metadata().unwrap().len();
            assert_eq!(
                std::fs::read(&path).unwrap().last(),
                Some(&b'\n'),
                "precondition: file ends with a newline"
            );
            file.set_len(len - 1).unwrap();
        }

        // The newline-less final line is still valid JSON and must be kept.
        let state = load_checkpoint(&path, "c", "f").unwrap();
        assert_eq!(state.rows.len(), 1);

        // Resume must restore the newline before appending, or the next
        // record would merge into the previous line and corrupt the file.
        {
            let mut writer = CheckpointWriter::resume(&path).unwrap();
            writer
                .write_row(&json!({
                    "engine": "random", "config_label": "random",
                    "position_id": "p1", "seed": 0, "move": "0:0:1",
                }))
                .unwrap();
        }

        let state = load_checkpoint(&path, "c", "f").unwrap();
        assert_eq!(state.rows.len(), 2, "both records survive the repair");
        assert_eq!(state.rows[0]["position_id"], json!("p0"));
        assert_eq!(state.rows[1]["position_id"], json!("p1"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_fingerprint_ignores_output_and_checkpoint_paths() {
        let a = json!({"family": "fixed", "seeds": 10, "output": "a.json", "checkpoint": "a.ckpt"});
        let b = json!({"family": "fixed", "seeds": 10, "output": "b.json", "checkpoint": "b.ckpt"});
        assert_eq!(config_fingerprint(&a), config_fingerprint(&b));

        let c = json!({"family": "fixed", "seeds": 30, "output": "a.json", "checkpoint": "a.ckpt"});
        assert_ne!(config_fingerprint(&a), config_fingerprint(&c));
    }
}
