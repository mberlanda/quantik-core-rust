//! `opening-probe.v1`: a read-only, engine-facing opening lookup.
//!
//! The normative specification is `docs/opening-probe-v1.md` in
//! `quantik-core-contracts`; this module implements it and does not
//! reinterpret it. Section numbers below refer to that paper.
//!
//! A probe file is a JSON metadata header followed by a table of fixed-width
//! 28-byte records sorted by their 18-byte `canonical_key.v1` (bytewise, not
//! numeric `u16`, order). A hit answers for the canonical *representative's*
//! orientation, so every stored action is mapped back into the caller's
//! orientation with `inverse_transform_index(t*)` (section 3). Skipping that
//! step returns a legal-looking wrong move, which is the failure this contract
//! exists to prevent.
//!
//! Errors are fail-fast (section 5): a missing key or a ply outside coverage is
//! an ordinary miss, everything else is a [`ProbeError`], and the library never
//! degrades corruption into a miss.

use crate::bitboard::Bitboard;
use crate::constants::{FLAG_CANON, VERSION};
use crate::game::has_winning_line;
use crate::moves::{generate_legal_moves, is_move_legal};
use crate::symmetry::SymmetryHandler;
use crate::validation::{validate_bitboard_state, InvalidStateReason};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

/// File magic: ASCII `QPROBE`, `0x00`, format major `1`.
pub const MAGIC: [u8; 8] = *b"QPROBE\x00\x01";
/// Size in bytes of one record.
pub const RECORD_SIZE: usize = 28;
/// Size in bytes of a `canonical_key.v1`.
pub const KEY_SIZE: usize = 18;
/// Value of the header `schema` field.
pub const SCHEMA: &str = "opening-probe.v1";
/// Value of the header `key_format` field.
pub const KEY_FORMAT: &str = "canonical_key.v1";

/// The 18-byte `canonical_key.v1`.
pub type ProbeKey = [u8; KEY_SIZE];

/// Record `status` byte. `unsolved`, `inferred` and `tablebase` book statuses
/// are never written into a probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeStatus {
    Exact = 1,
    Bounded = 2,
}

impl ProbeStatus {
    fn from_byte(b: u8) -> Option<Self> {
        match b {
            1 => Some(Self::Exact),
            2 => Some(Self::Bounded),
            _ => None,
        }
    }

    /// The fixture / contract spelling.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Bounded => "bounded",
        }
    }
}

/// One record, with `optimal_actions` in the representative's orientation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbeRecord {
    pub key: ProbeKey,
    /// Side-to-move perspective, in `{-1, 0, 1}`. `0` is "unknown", never a draw.
    pub game_value: i8,
    pub status: ProbeStatus,
    /// Bit `i` set means action `i` (`shape * 16 + position`) is optimal.
    pub optimal_actions: u64,
}

impl ProbeRecord {
    /// The 28-byte on-disk form.
    pub fn to_bytes(&self) -> [u8; RECORD_SIZE] {
        let mut out = [0u8; RECORD_SIZE];
        out[..KEY_SIZE].copy_from_slice(&self.key);
        out[18] = self.game_value as u8;
        out[19] = self.status as u8;
        out[20..28].copy_from_slice(&self.optimal_actions.to_le_bytes());
        out
    }
}

/// Why a probe could not be opened or used. Names follow section 5.
#[derive(Debug)]
pub enum ProbeError {
    /// The file could not be read.
    Io(std::io::Error),
    /// Bad magic, bad or missing header field, bad record, bad `per_ply`.
    Corrupt(String),
    /// The file length disagrees with the layout the header declares.
    Truncated { expected: u64, actual: u64 },
    /// `schema`, `format_major` or `key_format` is not one this reader knows.
    IncompatibleVersion(String),
    /// `body_sha256` differs from the record region.
    ChecksumMismatch,
    /// Keys are not strictly ascending bytewise.
    UnsortedOrDuplicateKeys(String),
    /// The caller's position fails engine validation. Not a miss.
    InvalidCallerPosition(InvalidStateReason),
    /// A mapped-back action is not a legal move of the caller's position: a
    /// wrong transform or a corrupt record. The orientation tripwire.
    IllegalMappedAction { action: u8, planes: [u16; 8] },
    /// The opt-in stale check: `source_book.book_id` is not the expected one.
    Stale { expected: String, found: String },
}

impl ProbeError {
    /// The section 5 kind name, as used by the contracts fixtures.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Io(_) => "io",
            Self::Corrupt(_) => "corrupt",
            Self::Truncated { .. } => "truncated",
            Self::IncompatibleVersion(_) => "incompatible version",
            Self::ChecksumMismatch => "checksum mismatch",
            Self::UnsortedOrDuplicateKeys(_) => "unsorted or duplicate keys",
            Self::InvalidCallerPosition(_) => "invalid caller position",
            Self::IllegalMappedAction { .. } => "illegal mapped-back action",
            Self::Stale { .. } => "stale",
        }
    }
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Corrupt(m) => write!(f, "corrupt: {m}"),
            Self::Truncated { expected, actual } => write!(
                f,
                "truncated: header declares {expected} bytes, file has {actual}"
            ),
            Self::IncompatibleVersion(m) => write!(f, "incompatible version: {m}"),
            Self::ChecksumMismatch => {
                write!(f, "checksum mismatch: body_sha256 differs from the records")
            }
            Self::UnsortedOrDuplicateKeys(m) => write!(f, "unsorted or duplicate keys: {m}"),
            Self::InvalidCallerPosition(r) => write!(f, "invalid caller position: {r}"),
            Self::IllegalMappedAction { action, planes } => write!(
                f,
                "illegal mapped-back action {action} for caller planes {planes:?}"
            ),
            Self::Stale { expected, found } => write!(
                f,
                "stale: expected source book {expected:?}, probe was built from {found:?}"
            ),
        }
    }
}

impl std::error::Error for ProbeError {}

impl From<std::io::Error> for ProbeError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

fn corrupt<T>(msg: impl Into<String>) -> Result<T, ProbeError> {
    Err(ProbeError::Corrupt(msg.into()))
}

/// Why a valid probe had no answer. Both are ordinary, not errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissReason {
    KeyAbsent,
    PlyOutsideCoverage,
}

/// A hit, already in the **caller's** orientation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeHit {
    pub game_value: i8,
    pub status: ProbeStatus,
    /// Optimal actions of the caller's position, ascending, each verified legal.
    pub actions: Vec<u8>,
    /// `t*`: the caller-to-representative transform (lowest index on ties).
    pub transform_index: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProbeLookup {
    Hit(ProbeHit),
    Miss(MissReason),
}

/// Header fields the reader needs after open. Unknown optional keys in the
/// file are ignored (section 4).
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeHeader {
    pub entry_count: u64,
    pub ply_min: u32,
    pub ply_max: u32,
    /// `(ply, entries)`.
    pub per_ply: Vec<(u32, u64)>,
    pub coverage_complete: Option<bool>,
    pub book_id: String,
    pub generator: String,
    pub generator_version: String,
    pub contract_version: String,
}

/// An opened, fully verified probe.
#[derive(Debug)]
pub struct ProbeFile {
    data: Vec<u8>,
    body_start: usize,
    header: ProbeHeader,
}

fn padded_body_start(metadata_len: u64) -> u64 {
    (12 + metadata_len).div_ceil(8) * 8
}

fn ply_of_key(key: &[u8]) -> u32 {
    key[2..].iter().map(|b| b.count_ones()).sum()
}

fn bitboard_of_key(key: &[u8]) -> Bitboard {
    let mut buf = [0u8; 16];
    buf.copy_from_slice(&key[2..18]);
    Bitboard::from_le_bytes(&buf)
}

fn req<'a>(obj: &'a Map<String, Value>, key: &str) -> Result<&'a Value, ProbeError> {
    obj.get(key)
        .ok_or_else(|| ProbeError::Corrupt(format!("header missing mandatory key {key:?}")))
}

fn req_str(obj: &Map<String, Value>, key: &str) -> Result<String, ProbeError> {
    match req(obj, key)? {
        Value::String(s) if !s.is_empty() => Ok(s.clone()),
        _ => corrupt(format!("header {key} must be a non-empty string")),
    }
}

fn req_uint(obj: &Map<String, Value>, key: &str) -> Result<u64, ProbeError> {
    req(obj, key)?
        .as_u64()
        .ok_or_else(|| ProbeError::Corrupt(format!("header {key} must be a non-negative integer")))
}

impl ProbeFile {
    /// Open and fully verify a probe file. There is no unverified mode: the
    /// checksum and sortedness passes are O(n) reads and always run
    /// (`verify = false` was left to this implementation to size, and is not
    /// offered).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ProbeError> {
        Self::from_bytes(std::fs::read(path)?)
    }

    /// Like [`open`](Self::open), plus the opt-in stale check: fail with
    /// [`ProbeError::Stale`] unless `source_book.book_id` equals `expected_book_id`.
    pub fn open_expecting_book_id(
        path: impl AsRef<Path>,
        expected_book_id: &str,
    ) -> Result<Self, ProbeError> {
        Self::from_bytes_expecting_book_id(std::fs::read(path)?, expected_book_id)
    }

    pub fn from_bytes_expecting_book_id(
        data: Vec<u8>,
        expected_book_id: &str,
    ) -> Result<Self, ProbeError> {
        let file = Self::from_bytes(data)?;
        if file.header.book_id != expected_book_id {
            return Err(ProbeError::Stale {
                expected: expected_book_id.to_string(),
                found: file.header.book_id.clone(),
            });
        }
        Ok(file)
    }

    /// Verify and adopt the bytes of a probe file (section 5, open-time checks).
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, ProbeError> {
        if data.len() < 12 {
            return Err(ProbeError::Truncated {
                expected: 12,
                actual: data.len() as u64,
            });
        }
        if data[..7] != MAGIC[..7] {
            return corrupt("bad magic");
        }
        if data[7] != MAGIC[7] {
            return Err(ProbeError::IncompatibleVersion(format!(
                "format major {} (this reader reads {})",
                data[7], MAGIC[7]
            )));
        }
        let metadata_len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as u64;
        let metadata_end = 12 + metadata_len;
        if metadata_end > data.len() as u64 {
            return Err(ProbeError::Truncated {
                expected: padded_body_start(metadata_len),
                actual: data.len() as u64,
            });
        }
        let metadata: Value = serde_json::from_slice(&data[12..metadata_end as usize])
            .map_err(|e| ProbeError::Corrupt(format!("metadata is not valid JSON: {e}")))?;
        let Value::Object(obj) = metadata else {
            return corrupt("metadata must be a JSON object");
        };

        // Version gate first: never partially read an incompatible file.
        if obj.get("schema").and_then(Value::as_str) != Some(SCHEMA)
            || obj.get("format_major").and_then(Value::as_u64) != Some(1)
            || obj.get("key_format").and_then(Value::as_str) != Some(KEY_FORMAT)
        {
            return Err(ProbeError::IncompatibleVersion(
                "schema, format_major and key_format must be opening-probe.v1, 1, canonical_key.v1"
                    .into(),
            ));
        }
        let header = Self::parse_header(&obj)?;

        let body_start = padded_body_start(metadata_len);
        let expected = header
            .entry_count
            .checked_mul(RECORD_SIZE as u64)
            .and_then(|b| b.checked_add(body_start))
            .ok_or_else(|| ProbeError::Corrupt("entry_count overflows the file layout".into()))?;
        if expected != data.len() as u64 {
            return Err(ProbeError::Truncated {
                expected,
                actual: data.len() as u64,
            });
        }
        let body_start = body_start as usize;
        let body = &data[body_start..];

        // Records: corrupt-field checks.
        for (index, rec) in body.as_chunks::<RECORD_SIZE>().0.iter().enumerate() {
            Self::check_record(index, rec)?;
        }
        let sha = Sha256::digest(body);
        let hex: String = sha.iter().map(|b| format!("{b:02x}")).collect();
        let declared = req_str(&obj, "body_sha256")?;
        if hex != declared {
            return Err(ProbeError::ChecksumMismatch);
        }
        // Strictly ascending, bytewise (slice comparison is lexicographic).
        let mut counts: BTreeMap<u32, u64> = BTreeMap::new();
        let mut prev: Option<&[u8]> = None;
        for rec in body.as_chunks::<RECORD_SIZE>().0 {
            let key = &rec[..KEY_SIZE];
            if let Some(p) = prev {
                if p >= key {
                    return Err(ProbeError::UnsortedOrDuplicateKeys(
                        "keys must be strictly ascending bytewise".into(),
                    ));
                }
            }
            prev = Some(key);
            *counts.entry(ply_of_key(key)).or_default() += 1;
        }
        let declared_counts: BTreeMap<u32, u64> = header.per_ply.iter().copied().collect();
        if declared_counts.len() != header.per_ply.len() || declared_counts != counts {
            return corrupt(format!(
                "per_ply {:?} does not match the records' plies {:?}",
                header.per_ply, counts
            ));
        }
        if counts
            .keys()
            .any(|&p| p < header.ply_min || p > header.ply_max)
        {
            return corrupt("a record lies outside ply_min..ply_max");
        }
        Ok(Self {
            data,
            body_start,
            header,
        })
    }

    fn parse_header(obj: &Map<String, Value>) -> Result<ProbeHeader, ProbeError> {
        for key in [
            "schema",
            "format_major",
            "key_format",
            "record_size",
            "entry_count",
            "ply_min",
            "ply_max",
            "per_ply",
            "source_book",
            "generator",
            "generator_version",
            "contract_version",
            "body_sha256",
        ] {
            req(obj, key)?;
        }
        if req_uint(obj, "record_size")? != RECORD_SIZE as u64 {
            return corrupt("record_size must be 28");
        }
        let book_id = match req(obj, "source_book")? {
            Value::Object(sb) => req_str(sb, "book_id").map_err(|_| {
                ProbeError::Corrupt("source_book must have a non-empty book_id".into())
            })?,
            _ => return corrupt("source_book must be an object"),
        };
        let generator = req_str(obj, "generator")?;
        let generator_version = req_str(obj, "generator_version")?;
        let contract_version = req_str(obj, "contract_version")?;
        let entry_count = req_uint(obj, "entry_count")?;
        let ply_min = req_uint(obj, "ply_min")?;
        let ply_max = req_uint(obj, "ply_max")?;
        if ply_min > ply_max || ply_max > 16 {
            return corrupt("ply_min must be <= ply_max <= 16");
        }
        let Value::Array(rows) = req(obj, "per_ply")? else {
            return corrupt("per_ply must be a list");
        };
        let mut per_ply = Vec::with_capacity(rows.len());
        let mut total = 0u64;
        for row in rows {
            let parsed = row.as_object().and_then(|o| {
                if o.len() != 2 {
                    return None;
                }
                Some((o.get("ply")?.as_u64()?, o.get("entries")?.as_u64()?))
            });
            let Some((ply, entries)) = parsed else {
                return corrupt("per_ply must be a list of {ply, entries} non-negative integers");
            };
            if ply > 16 {
                return corrupt("per_ply ply must be at most 16");
            }
            total = total
                .checked_add(entries)
                .ok_or_else(|| ProbeError::Corrupt("per_ply entries overflow".into()))?;
            per_ply.push((ply as u32, entries));
        }
        if total != entry_count {
            return corrupt("per_ply does not sum to entry_count");
        }
        let sha = req_str(obj, "body_sha256")?;
        if sha.len() != 64 || !sha.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            return corrupt("body_sha256 must be 64 lowercase hex characters");
        }
        let coverage_complete = match obj.get("coverage_complete") {
            None => None,
            Some(Value::Bool(b)) => Some(*b),
            Some(_) => return corrupt("coverage_complete must be a boolean"),
        };
        if let Some(v) = obj.get("created_at") {
            if !v.is_string() {
                return corrupt("created_at must be a string");
            }
        }
        Ok(ProbeHeader {
            entry_count,
            ply_min: ply_min as u32,
            ply_max: ply_max as u32,
            per_ply,
            coverage_complete,
            book_id,
            generator,
            generator_version,
            contract_version,
        })
    }

    fn check_record(index: usize, rec: &[u8]) -> Result<(), ProbeError> {
        let where_ = format!("record {index}");
        if rec[0] != VERSION || rec[1] != FLAG_CANON {
            return corrupt(format!(
                "{where_}: key must start with 0x01 0x02 (canonical_key.v1)"
            ));
        }
        let value = rec[18] as i8;
        if !(-1..=1).contains(&value) {
            return corrupt(format!("{where_}: game_value must be -1, 0 or 1"));
        }
        let Some(status) = ProbeStatus::from_byte(rec[19]) else {
            return corrupt(format!("{where_}: status must be 1 (exact) or 2 (bounded)"));
        };
        if status == ProbeStatus::Exact && value == 0 {
            return corrupt(format!("{where_}: exact requires game_value -1 or 1"));
        }
        let mask = u64::from_le_bytes(rec[20..28].try_into().expect("8 bytes"));
        if mask == 0 {
            let bb = bitboard_of_key(rec);
            if !(has_winning_line(&bb) || generate_legal_moves(&bb).is_empty()) {
                return corrupt(format!(
                    "{where_}: empty optimal_actions on a non-terminal position"
                ));
            }
        }
        Ok(())
    }

    pub fn header(&self) -> &ProbeHeader {
        &self.header
    }

    pub fn len(&self) -> usize {
        self.header.entry_count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.header.entry_count == 0
    }

    /// The `index`-th record in file order (ascending key order).
    pub fn record(&self, index: usize) -> Option<ProbeRecord> {
        if index >= self.len() {
            return None;
        }
        Some(self.decode(index))
    }

    fn raw(&self, index: usize) -> &[u8] {
        let start = self.body_start + index * RECORD_SIZE;
        &self.data[start..start + RECORD_SIZE]
    }

    /// Only called on records that passed `check_record`.
    fn decode(&self, index: usize) -> ProbeRecord {
        let rec = self.raw(index);
        let mut key = [0u8; KEY_SIZE];
        key.copy_from_slice(&rec[..KEY_SIZE]);
        ProbeRecord {
            key,
            game_value: rec[18] as i8,
            status: ProbeStatus::from_byte(rec[19]).expect("checked at open"),
            optimal_actions: u64::from_le_bytes(rec[20..28].try_into().expect("8 bytes")),
        }
    }

    /// Binary search on the bytewise key order.
    fn find(&self, key: &ProbeKey) -> Option<usize> {
        let (mut lo, mut hi) = (0usize, self.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.raw(mid)[..KEY_SIZE].cmp(&key[..]) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// Probe the caller's position (section 3.2). A miss is `Ok(None)`;
    /// everything else that goes wrong is an `Err`.
    pub fn probe(&self, bb: &Bitboard) -> Result<Option<ProbeHit>, ProbeError> {
        Ok(match self.probe_detailed(bb)? {
            ProbeLookup::Hit(hit) => Some(hit),
            ProbeLookup::Miss(_) => None,
        })
    }

    /// Like [`probe`](Self::probe), but says why a miss was a miss.
    pub fn probe_detailed(&self, bb: &Bitboard) -> Result<ProbeLookup, ProbeError> {
        let side = validate_bitboard_state(bb).map_err(ProbeError::InvalidCallerPosition)?;
        let ply = bb.player_piece_count(0) + bb.player_piece_count(1);
        if ply < self.header.ply_min || ply > self.header.ply_max {
            return Ok(ProbeLookup::Miss(MissReason::PlyOutsideCoverage));
        }

        // t*: caller -> representative, lowest index among the minimisers.
        let (canon, t_star) = SymmetryHandler::find_canonical_with_transform(bb);
        let mut key = [0u8; KEY_SIZE];
        key[0] = VERSION;
        key[1] = FLAG_CANON;
        key[2..].copy_from_slice(&canon.to_le_bytes());
        let Some(index) = self.find(&key) else {
            return Ok(ProbeLookup::Miss(MissReason::KeyAbsent));
        };
        let record = self.decode(index);

        // Stored actions are in the representative's frame; go back to the
        // caller's with the INVERSE of t*, bit by bit. Using t* itself is the bug
        // this contract exists to prevent (wrong for rotate90/270 and for
        // shape permutations of order > 2).
        let back = SymmetryHandler::inverse_transform_index(t_star);
        let mut actions = Vec::with_capacity(record.optimal_actions.count_ones() as usize);
        for bit in 0..64u8 {
            if record.optimal_actions >> bit & 1 == 0 {
                continue;
            }
            let action = SymmetryHandler::remap_action_index(bit, back);
            // Mandatory orientation tripwire, in release builds too (section 5).
            if !is_move_legal(bb, side, action / 16, action % 16) {
                return Err(ProbeError::IllegalMappedAction {
                    action,
                    planes: bb.planes,
                });
            }
            actions.push(action);
        }
        actions.sort_unstable();
        Ok(ProbeLookup::Hit(ProbeHit {
            game_value: record.game_value,
            status: record.status,
            actions,
            transform_index: t_star,
        }))
    }
}

/// Metadata a producer supplies; the rest of the header is derived.
#[derive(Clone, Debug)]
pub struct ProbeBuildMeta {
    /// `source_book.book_id`.
    pub book_id: String,
    pub generator: String,
    pub generator_version: String,
    pub contract_version: String,
    /// RFC 3339; omitted from the header when `None`.
    pub created_at: Option<String>,
    /// Omitted from the header when `None`.
    pub coverage_complete: Option<bool>,
}

/// Recursively re-insert object keys in sorted order.
fn sort_keys(v: &Value) -> Value {
    match v {
        Value::Object(m) => {
            let sorted: BTreeMap<&String, Value> =
                m.iter().map(|(k, v)| (k, sort_keys(v))).collect();
            let mut out = Map::new();
            for (k, v) in sorted {
                out.insert(k.clone(), v);
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(sort_keys).collect()),
        other => other.clone(),
    }
}

/// Lay a file out: magic, `metadata_len`, compact key-sorted JSON, zero padding
/// to a multiple of 8 bytes, then `body`. No validation: use [`build_probe`]
/// to produce a valid file; this is the raw layout step.
pub fn encode_file(header: &Value, body: &[u8]) -> Vec<u8> {
    let metadata = serde_json::to_vec(&sort_keys(header)).expect("a Value always serialises");
    let start = padded_body_start(metadata.len() as u64) as usize;
    let mut out = Vec::with_capacity(start + body.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
    out.extend_from_slice(&metadata);
    out.resize(start, 0);
    out.extend_from_slice(body);
    out
}

/// Build a probe file from records: sorts them bytewise, derives `per_ply`,
/// the ply range and the checksum, and then re-opens its own output so a
/// producer bug fails here rather than at a consumer.
pub fn build_probe(
    mut records: Vec<ProbeRecord>,
    meta: &ProbeBuildMeta,
) -> Result<Vec<u8>, ProbeError> {
    records.sort_by_key(|r| r.key);
    if let Some(w) = records.windows(2).find(|w| w[0].key == w[1].key) {
        return Err(ProbeError::UnsortedOrDuplicateKeys(format!(
            "duplicate key {}",
            w[0].key
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        )));
    }
    let mut counts: BTreeMap<u32, u64> = BTreeMap::new();
    let mut body = Vec::with_capacity(records.len() * RECORD_SIZE);
    for r in &records {
        *counts.entry(ply_of_key(&r.key)).or_default() += 1;
        body.extend_from_slice(&r.to_bytes());
    }
    let ply_min = counts.keys().next().copied().unwrap_or(0);
    let ply_max = counts.keys().next_back().copied().unwrap_or(0);
    let sha: String = Sha256::digest(&body)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    let mut h = Map::new();
    h.insert("schema".into(), SCHEMA.into());
    h.insert("format_major".into(), 1.into());
    h.insert("key_format".into(), KEY_FORMAT.into());
    h.insert("record_size".into(), (RECORD_SIZE as u64).into());
    h.insert("entry_count".into(), (records.len() as u64).into());
    h.insert("ply_min".into(), ply_min.into());
    h.insert("ply_max".into(), ply_max.into());
    h.insert(
        "per_ply".into(),
        Value::Array(
            counts
                .iter()
                .map(|(p, n)| serde_json::json!({"ply": p, "entries": n}))
                .collect(),
        ),
    );
    h.insert(
        "source_book".into(),
        serde_json::json!({"book_id": meta.book_id}),
    );
    h.insert("generator".into(), meta.generator.clone().into());
    h.insert(
        "generator_version".into(),
        meta.generator_version.clone().into(),
    );
    h.insert(
        "contract_version".into(),
        meta.contract_version.clone().into(),
    );
    h.insert("body_sha256".into(), sha.into());
    if let Some(c) = meta.coverage_complete {
        h.insert("coverage_complete".into(), c.into());
    }
    if let Some(t) = &meta.created_at {
        h.insert("created_at".into(), t.clone().into());
    }
    let bytes = encode_file(&Value::Object(h), &body);
    ProbeFile::from_bytes(bytes.clone())?;
    Ok(bytes)
}
