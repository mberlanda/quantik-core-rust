# Opening Book Engine Integration And Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SQLite opening book a contracted cross-language artifact and let Quantik engines use it as a safe first-choice move source before falling back to search.

**Architecture:** Treat `quantik-core-contracts` as the source of truth for the on-disk opening-book SQLite schema and semantics. In `quantik-core-rust`, add schema metadata/conformance tests, a small book-probing API that only returns legal moves in safe orientations, then wire that API into minimax/MCTS/beam entry points and benchmark adapters. The first implementation preserves the current representative-only safety rule; a final planned task prepares transform-aware lookup as the next compatibility expansion.

**Tech Stack:** Rust 2021, `rusqlite`, existing Quantik `State`/`Bitboard`/`Move` APIs, SQLite as local ACID schema-on-write storage, JSON Schema/contract docs in `quantik-core-contracts`, existing `cross_engine_benchmark` scripts.

---

## Current Facts To Preserve

- Rust opening book lives in `quantik-core-rust/crates/quantik-core/src/opening_book.rs`.
- Current SQLite tables are `positions`, `best_moves`, and `position_edges`.
- Current solved-reference reuse is implemented in `quantik-core-rust/crates/quantik-core/src/bench/book_export.rs`.
- Solved best moves are currently safe only when the queried board is its own canonical representative, because moves are stored as `(shape, position)` in one board orientation while rows are keyed by the shared canonical key.
- Rust canonicalization is D4 geometry plus shape relabeling, with no color swap.
- `quantik-core-contracts/docs/storage-representations.md` currently says SQLite is preferred for opening books and state DAGs, but it does not define an `opening-book.v1` contract or the actual table structure.
- The database cheatsheet reference emphasizes why SQLite is a good fit here: single-file, serverless, schema-on-write, ACID local storage with excellent read-heavy behavior, while not being the bulk ML/tensor format. Keep that framing in the contracts storage docs.

## File Structure

### `quantik-core-contracts`

- Modify: `contracts.json`
  - Add `opening_book` contract entry with id `opening-book.v1`.
- Create: `docs/opening-book-v1.md`
  - Normative SQLite schema, semantics, invariants, indexes, migration rules, and representative-only caveat.
- Modify: `docs/storage-representations.md`
  - Expand SQLite section to reference `opening-book.v1` and clarify why SQLite is for graph/index/opening-book workflows, not ML corpora.
- Create: `schemas/opening-book-sqlite-v1.json`
  - Machine-readable description of tables/columns/indexes. This is not a JSON data payload schema; it is a schema manifest for validating implementation conformance tests.
- Modify: `scripts/validate_contracts.py`
  - Validate the new schema manifest enough to catch missing required table/column fields.

### `quantik-core-rust`

- Modify: `crates/quantik-core/src/opening_book.rs`
  - Add contract constants.
  - Create/update a `book_metadata` table.
  - Add `OpeningBookDatabase::schema_version`, `contract_id`, `best_moves_for_state`, and `select_book_move`.
  - Validate legal moves before returning a book hit.
- Modify: `crates/quantik-core/src/minimax.rs`
  - Add optional opening-book usage at root search.
- Modify: `crates/quantik-core/src/mcts.rs`
  - Add optional opening-book usage at root search.
- Modify: `crates/quantik-core/src/beam_search.rs`
  - Add optional opening-book usage at root search.
- Modify: `crates/quantik-core/src/bench/adapters.rs`
  - Let adapters use a book path when configured.
- Modify: `crates/quantik-core/src/bin/cross_engine_benchmark.rs`
  - Add `run --book <path>` and include it in run config.
- Modify: `scripts/generate_observations.sh`, `scripts/generate_h2h_stats.sh`, `scripts/plan_runs.sh`, `scripts/README.md`
  - Thread `--book` through observation/h2h workflows and examples.
- Test: existing module tests plus new tests in `opening_book.rs`, adapter tests, and script tests.

---

## Task 1: Add The Opening Book Contract

**Files:**
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-contracts/contracts.json`
- Create: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-contracts/docs/opening-book-v1.md`
- Create: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-contracts/schemas/opening-book-sqlite-v1.json`
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-contracts/docs/storage-representations.md`

- [ ] **Step 1: Add contract manifest entry**

Edit `contracts.json` and add this object inside `contracts`:

```json
"opening_book": {
  "id": "opening-book.v1",
  "major": 1,
  "schema": "schemas/opening-book-sqlite-v1.json",
  "docs": "docs/opening-book-v1.md"
}
```

- [ ] **Step 2: Create the schema manifest**

Create `schemas/opening-book-sqlite-v1.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://github.com/mberlanda/quantik-core-contracts/schemas/opening-book-sqlite-v1.json",
  "title": "Quantik opening-book SQLite schema v1",
  "type": "object",
  "additionalProperties": false,
  "required": ["contract_id", "major", "storage_engine", "tables", "indexes"],
  "properties": {
    "contract_id": { "const": "opening-book.v1" },
    "major": { "const": 1 },
    "storage_engine": { "const": "sqlite" },
    "tables": {
      "type": "object",
      "additionalProperties": false,
      "required": ["book_metadata", "positions", "best_moves", "position_edges"],
      "properties": {
        "book_metadata": { "$ref": "#/$defs/table" },
        "positions": { "$ref": "#/$defs/table" },
        "best_moves": { "$ref": "#/$defs/table" },
        "position_edges": { "$ref": "#/$defs/table" }
      }
    },
    "indexes": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["name", "table", "columns", "unique"],
        "properties": {
          "name": { "type": "string" },
          "table": { "type": "string" },
          "columns": {
            "type": "array",
            "minItems": 1,
            "items": { "type": "string" }
          },
          "unique": { "type": "boolean" }
        }
      }
    }
  },
  "$defs": {
    "table": {
      "type": "object",
      "additionalProperties": false,
      "required": ["columns", "primary_key"],
      "properties": {
        "columns": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["name", "type", "nullable"],
            "properties": {
              "name": { "type": "string" },
              "type": { "type": "string" },
              "nullable": { "type": "boolean" },
              "default": {}
            }
          }
        },
        "primary_key": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    }
  },
  "contract_id": "opening-book.v1",
  "major": 1,
  "storage_engine": "sqlite",
  "tables": {
    "book_metadata": {
      "columns": [
        { "name": "key", "type": "TEXT", "nullable": false },
        { "name": "value", "type": "TEXT", "nullable": false }
      ],
      "primary_key": ["key"]
    },
    "positions": {
      "columns": [
        { "name": "canonical_key", "type": "BLOB", "nullable": false },
        { "name": "qfen", "type": "TEXT", "nullable": false },
        { "name": "depth", "type": "INTEGER", "nullable": false },
        { "name": "evaluation", "type": "REAL", "nullable": false },
        { "name": "visit_count", "type": "INTEGER", "nullable": false },
        { "name": "win_count_p0", "type": "INTEGER", "nullable": false },
        { "name": "win_count_p1", "type": "INTEGER", "nullable": false },
        { "name": "draw_count", "type": "INTEGER", "nullable": false },
        { "name": "is_terminal", "type": "INTEGER", "nullable": false, "default": 0 },
        { "name": "symmetry_count", "type": "INTEGER", "nullable": false, "default": 0 },
        { "name": "created_at", "type": "TIMESTAMP", "nullable": true, "default": "CURRENT_TIMESTAMP" },
        { "name": "solved", "type": "INTEGER", "nullable": false, "default": 0 },
        { "name": "game_value", "type": "INTEGER", "nullable": true }
      ],
      "primary_key": ["canonical_key"]
    },
    "best_moves": {
      "columns": [
        { "name": "canonical_key", "type": "BLOB", "nullable": false },
        { "name": "move_rank", "type": "INTEGER", "nullable": false },
        { "name": "shape", "type": "INTEGER", "nullable": false },
        { "name": "position", "type": "INTEGER", "nullable": false }
      ],
      "primary_key": ["canonical_key", "move_rank"]
    },
    "position_edges": {
      "columns": [
        { "name": "parent_key", "type": "BLOB", "nullable": false },
        { "name": "child_key", "type": "BLOB", "nullable": false }
      ],
      "primary_key": ["parent_key", "child_key"]
    }
  },
  "indexes": [
    { "name": "idx_depth", "table": "positions", "columns": ["depth"], "unique": false },
    { "name": "idx_visit_count", "table": "positions", "columns": ["visit_count"], "unique": false },
    { "name": "idx_edges_child", "table": "position_edges", "columns": ["child_key"], "unique": false }
  ]
}
```

- [ ] **Step 3: Document semantics**

Create `docs/opening-book-v1.md` with these sections:

```markdown
# Opening Book SQLite v1

This document defines `opening-book.v1`, the SQLite opening-book contract.

## Storage Choice

SQLite is the contracted local store for opening books because Quantik book
workloads need a portable, single-file, strongly consistent graph/index store.
This is a schema-on-write artifact with ACID transactions. It is not the bulk ML
training corpus format; JSONL/Arrow/Parquet/tensor stores remain the contracted
formats for training data.

## Canonicalization Scope

Rows are keyed by the 18-byte `State::canonical_key()` envelope:

- byte 0: version `1`
- byte 1: canonical flag `2`
- bytes 2..17: eight little-endian `u16` planes after Rust-compatible
  D4 geometry plus shape relabeling canonicalization

`opening-book.v1` does not include color-swap canonicalization.

## Tables

### `book_metadata`

`key TEXT PRIMARY KEY`, `value TEXT NOT NULL`.

Required keys:

- `contract_id = opening-book.v1`
- `schema_major = 1`
- `contracts_release = 1.0.0`
- `canonicalization = d4_shape_permutation_no_color_swap`

### `positions`

One row per canonical position:

| Column | Type | Meaning |
| --- | --- | --- |
| `canonical_key` | BLOB primary key | 18-byte canonical state key |
| `qfen` | TEXT | QFEN for the stored board orientation |
| `depth` | INTEGER | Pieces placed / ply depth |
| `evaluation` | REAL | Heuristic or solved value |
| `visit_count` | INTEGER | Search/book visits |
| `win_count_p0` | INTEGER | Player-0 wins from rollouts/search |
| `win_count_p1` | INTEGER | Player-1 wins from rollouts/search |
| `draw_count` | INTEGER | Reserved; Quantik has no draw outcome |
| `is_terminal` | INTEGER | `0=interior`, `1=win_p0`, `2=win_p1`, `3=stalemate` |
| `symmetry_count` | INTEGER | Orbit size / representative multiplicity |
| `created_at` | TIMESTAMP | Creation timestamp |
| `solved` | INTEGER | `1` when `game_value` and full optimal move set are exact |
| `game_value` | INTEGER nullable | Exact value for side to move: `+1` or `-1` |

### `best_moves`

Ordered candidate moves for a position:

| Column | Type | Meaning |
| --- | --- | --- |
| `canonical_key` | BLOB | Parent `positions.canonical_key` |
| `move_rank` | INTEGER | Rank starting at 1 |
| `shape` | INTEGER | Shape `0..3` |
| `position` | INTEGER | Board position `0..15` |

The moving player is not stored; it is derived from the queried legal game
state's `side_to_move`.

### `position_edges`

Parent-child canonical-key graph edges:

| Column | Type | Meaning |
| --- | --- | --- |
| `parent_key` | BLOB | Parent `positions.canonical_key` |
| `child_key` | BLOB | Child `positions.canonical_key` |

## Representative-Only Move Safety

In v1, stored moves are in the stored `qfen` orientation. Because
`canonical_key` can represent multiple D4-equivalent orientations and the schema
does not store a transform id, implementations must not serve `best_moves`
unless the queried board is its own canonical representative. Non-representative
queries must fall back to search.

## Migration Rules

Readers must tolerate older books missing `book_metadata`, `solved`, or
`game_value` by adding missing columns/tables idempotently. Missing `solved`
means `false`; missing `game_value` means `null`.
```

- [ ] **Step 4: Expand storage considerations**

In `docs/storage-representations.md`, replace the SQLite Decision paragraph with:

```markdown
Decision:

SQLite is the preferred contracted store for opening books and local state DAGs.
It gives Quantik a portable, single-file, schema-on-write, ACID graph/index
artifact with ordinary SQL inspection. See `docs/opening-book-v1.md` for the
normative `opening-book.v1` schema. SQLite remains the wrong fit for the main
ML training corpus: write Parquet/Arrow in batches and derive tensor stores for
training epochs.
```

- [ ] **Step 5: Validate contracts**

Run:

```bash
cd /Users/mauroberlanda/Code/quantik-ns/quantik-core-contracts
rtk python3 scripts/validate_contracts.py
```

Expected: exits 0 and validates `contracts.json`.

- [ ] **Step 6: Commit**

```bash
cd /Users/mauroberlanda/Code/quantik-ns/quantik-core-contracts
rtk git add contracts.json docs/opening-book-v1.md docs/storage-representations.md schemas/opening-book-sqlite-v1.json
rtk git commit -m "Document opening book contract"
```

---

## Task 2: Make Rust Opening Books Self-Describing

**Files:**
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust/crates/quantik-core/src/opening_book.rs`

- [ ] **Step 1: Write failing metadata test**

Add this test to the `tests` module:

```rust
#[test]
fn opening_book_records_contract_metadata() {
    let path = temp_db_path();
    let config = OpeningBookConfig {
        database_path: path.clone(),
        ..Default::default()
    };
    let db = OpeningBookDatabase::open(&config).unwrap();

    assert_eq!(db.contract_id().unwrap(), "opening-book.v1");
    assert_eq!(db.schema_major().unwrap(), 1);
    assert_eq!(
        db.metadata_value("canonicalization").unwrap().as_deref(),
        Some("d4_shape_permutation_no_color_swap")
    );

    fs::remove_file(&path).ok();
}
```

- [ ] **Step 2: Verify red**

Run:

```bash
cd /Users/mauroberlanda/Code/quantik-ns/quantik-core-rust
rtk cargo test opening_book_records_contract_metadata -- --nocapture
```

Expected: fails because `contract_id`, `schema_major`, and `metadata_value` do not exist.

- [ ] **Step 3: Add constants and metadata table**

Add near the top of `opening_book.rs`:

```rust
pub const OPENING_BOOK_CONTRACT_ID: &str = "opening-book.v1";
pub const OPENING_BOOK_SCHEMA_MAJOR: i32 = 1;
pub const OPENING_BOOK_CONTRACTS_RELEASE: &str = "1.0.0";
pub const OPENING_BOOK_CANONICALIZATION: &str = "d4_shape_permutation_no_color_swap";
```

Add this table to the `execute_batch` schema block:

```sql
CREATE TABLE IF NOT EXISTS book_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

After the idempotent migrations, insert metadata:

```rust
for (key, value) in [
    ("contract_id", OPENING_BOOK_CONTRACT_ID),
    ("schema_major", "1"),
    ("contracts_release", OPENING_BOOK_CONTRACTS_RELEASE),
    ("canonicalization", OPENING_BOOK_CANONICALIZATION),
] {
    conn.execute(
        "INSERT OR REPLACE INTO book_metadata (key, value) VALUES (?1, ?2)",
        params![key, value],
    )?;
}
```

Add methods:

```rust
pub fn metadata_value(&self, key: &str) -> SqlResult<Option<String>> {
    let result = self.conn.query_row(
        "SELECT value FROM book_metadata WHERE key = ?1",
        params![key],
        |row| row.get::<_, String>(0),
    );
    match result {
        Ok(value) => Ok(Some(value)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(err) => Err(err),
    }
}

pub fn contract_id(&self) -> SqlResult<String> {
    Ok(self
        .metadata_value("contract_id")?
        .unwrap_or_else(|| OPENING_BOOK_CONTRACT_ID.to_string()))
}

pub fn schema_major(&self) -> SqlResult<i32> {
    Ok(self
        .metadata_value("schema_major")?
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(OPENING_BOOK_SCHEMA_MAJOR))
}
```

- [ ] **Step 4: Verify green**

Run:

```bash
rtk cargo test opening_book_records_contract_metadata -- --nocapture
```

Expected: passes.

- [ ] **Step 5: Run opening-book tests**

```bash
rtk cargo test opening_book::tests -- --nocapture
```

Expected: passes.

- [ ] **Step 6: Commit**

```bash
rtk git add crates/quantik-core/src/opening_book.rs
rtk git commit -m "Add opening book schema metadata"
```

---

## Task 3: Add Safe Book Move Lookup API

**Files:**
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust/crates/quantik-core/src/opening_book.rs`

- [ ] **Step 1: Write failing lookup tests**

Add:

```rust
#[test]
fn select_book_move_returns_legal_current_player_move_for_representative() {
    let path = temp_db_path();
    let config = OpeningBookConfig {
        database_path: path.clone(),
        ..Default::default()
    };
    let db = OpeningBookDatabase::open(&config).unwrap();
    let state = State::empty();
    db.add_solved_position(&state, 1, &[(0, 0), (1, 5)]).unwrap();

    let hit = db.select_book_move(&state).unwrap().unwrap();
    assert_eq!(hit.mv.player, 0);
    assert_eq!(hit.mv.shape, 0);
    assert_eq!(hit.mv.position, 0);
    assert!(hit.entry.solved);

    fs::remove_file(&path).ok();
}

#[test]
fn select_book_move_ignores_illegal_or_unsafe_rows() {
    let path = temp_db_path();
    let config = OpeningBookConfig {
        database_path: path.clone(),
        ..Default::default()
    };
    let db = OpeningBookDatabase::open(&config).unwrap();
    let state = State::empty();
    db.add_solved_position(&state, 1, &[(0, 99)]).unwrap();

    assert!(db.select_book_move(&state).unwrap().is_none());

    fs::remove_file(&path).ok();
}
```

- [ ] **Step 2: Verify red**

Run:

```bash
rtk cargo test select_book_move -- --nocapture
```

Expected: fails because `select_book_move` and `BookHit` do not exist.

- [ ] **Step 3: Implement the API**

Import move helpers:

```rust
use crate::game::current_player;
use crate::moves::{generate_legal_moves, Move};
```

Add structs:

```rust
#[derive(Clone, Debug)]
pub struct BookHit {
    pub entry: OpeningBookEntry,
    pub mv: Move,
    pub source: BookHitSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BookHitSource {
    ExactSolved,
    RankedBookMove,
}
```

Add method:

```rust
pub fn select_book_move(&self, state: &State) -> SqlResult<Option<BookHit>> {
    if state.canonical_payload() != state.bb.to_le_bytes() {
        return Ok(None);
    }

    let Some(entry) = self.get_position(state)? else {
        return Ok(None);
    };

    let Some(player) = current_player(&state.bb) else {
        return Ok(None);
    };
    let legal = generate_legal_moves(&state.bb);

    for &(shape, position) in &entry.best_moves {
        if !(0..=3).contains(&shape) || !(0..=15).contains(&position) {
            continue;
        }
        let mv = Move::new(player, shape as u8, position as u8);
        if legal.contains(&mv) {
            let source = if entry.solved {
                BookHitSource::ExactSolved
            } else {
                BookHitSource::RankedBookMove
            };
            return Ok(Some(BookHit { entry, mv, source }));
        }
    }

    Ok(None)
}
```

- [ ] **Step 4: Verify green**

Run:

```bash
rtk cargo test select_book_move -- --nocapture
```

Expected: passes.

- [ ] **Step 5: Commit**

```bash
rtk git add crates/quantik-core/src/opening_book.rs
rtk git commit -m "Add safe opening book move lookup"
```

---

## Task 4: Add Engine-Level Book-First Selection

**Files:**
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust/crates/quantik-core/src/minimax.rs`
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust/crates/quantik-core/src/mcts.rs`
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust/crates/quantik-core/src/beam_search.rs`

- [ ] **Step 1: Write failing minimax test**

In `minimax.rs`, add a test that builds a temp book with empty-board best move `(0, 0)`, constructs `MinimaxEngine` with that book, searches empty state, and expects `best_move == Move::new(0, 0, 0)` and `nodes == 0`.

```rust
#[test]
fn minimax_uses_opening_book_before_search() {
    let path = std::env::temp_dir()
        .join(format!("quantik_minimax_book_{}.db", std::process::id()));
    let db = crate::opening_book::OpeningBookDatabase::open(&crate::opening_book::OpeningBookConfig {
        database_path: path.to_string_lossy().to_string(),
        ..Default::default()
    })
    .unwrap();
    db.add_solved_position(&State::empty(), 1, &[(0, 0)]).unwrap();

    let mut engine = MinimaxEngine::new(MinimaxConfig {
        opening_book_path: Some(path.to_string_lossy().to_string()),
        ..Default::default()
    });
    let result = engine.search(&State::empty()).unwrap();

    assert_eq!(result.best_move, Move::new(0, 0, 0));
    assert_eq!(result.nodes, 0);

    std::fs::remove_file(path).ok();
}
```

- [ ] **Step 2: Verify red**

Run:

```bash
rtk cargo test minimax_uses_opening_book_before_search -- --nocapture
```

Expected: fails because `opening_book_path` does not exist.

- [ ] **Step 3: Add config fields and root probe**

For each engine config, add:

```rust
pub opening_book_path: Option<String>,
```

At the start of each root `search` function, before expensive search:

```rust
if let Some(path) = &self.config.opening_book_path {
    let db = crate::opening_book::OpeningBookDatabase::open(&crate::opening_book::OpeningBookConfig {
        database_path: path.clone(),
        ..Default::default()
    })
    .map_err(|e| format!("open opening book {path}: {e}"))?;
    if let Some(hit) = db
        .select_book_move(state)
        .map_err(|e| format!("opening book lookup {path}: {e}"))?
    {
        return Ok(/* engine-specific result with hit.mv and zero search work */);
    }
}
```

Use each engine's existing result struct. For fields that need scores, use:

- exact solved hit: `hit.entry.game_value.unwrap_or(hit.entry.evaluation as i32) as f64`
- ranked unsolved hit: `hit.entry.evaluation`

- [ ] **Step 4: Add MCTS and beam tests**

Mirror the minimax test in `mcts.rs` and `beam_search.rs`, asserting the selected root move comes from the book and expensive work counters are zero or minimal according to each result type.

- [ ] **Step 5: Verify engine tests**

Run:

```bash
rtk cargo test uses_opening_book_before_search -- --nocapture
```

Expected: all engine book-first tests pass.

- [ ] **Step 6: Commit**

```bash
rtk git add crates/quantik-core/src/minimax.rs crates/quantik-core/src/mcts.rs crates/quantik-core/src/beam_search.rs
rtk git commit -m "Use opening book before engine search"
```

---

## Task 5: Thread Book Usage Through Benchmark Adapters And Scripts

**Files:**
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust/crates/quantik-core/src/bench/adapters.rs`
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust/crates/quantik-core/src/bin/cross_engine_benchmark.rs`
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust/scripts/generate_observations.sh`
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust/scripts/generate_h2h_stats.sh`
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust/scripts/plan_runs.sh`
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust/scripts/README.md`
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust/crates/quantik-core/tests/script_tools.rs`

- [ ] **Step 1: Write failing CLI dry-run script test**

Add to `script_tools.rs`:

```rust
#[test]
fn dry_run_observations_threads_book_path() {
    let (success, text) = run_script(
        "generate_observations.sh",
        &[
            "--dataset",
            "benchmarks/positions-v1.json",
            "--output",
            "benchmarks/results/dev.json",
            "--checkpoint-dir",
            "benchmarks/results/dev-ckpt",
            "--engines",
            "mcts,minmax",
            "--book",
            "benchmarks/results/book.db",
            "--dry-run",
        ],
    );

    assert!(success, "dry run failed:\n{text}");
    assert!(text.contains("--book benchmarks/results/book.db"), "{text}");
}
```

- [ ] **Step 2: Verify red**

Run:

```bash
rtk cargo test --test script_tools dry_run_observations_threads_book_path -- --nocapture
```

Expected: fails because scripts do not accept `--book`.

- [ ] **Step 3: Add adapter and CLI field**

Add `book_path: Option<PathBuf>` or `Option<String>` to `RunArgs`, adapter constructors, and engine configs. In `cross_engine_benchmark.rs`, add:

```rust
#[arg(long)]
book: Option<PathBuf>,
```

Include `"book": book_path` in `run_config`.

- [ ] **Step 4: Thread scripts**

In `generate_observations.sh` and `generate_h2h_stats.sh`, parse:

```bash
--book) book="$2"; shift 2 ;;
```

When non-empty, append:

```bash
cmd+=(--book "$book")
```

In `plan_runs.sh matrix`, add `--book PATH` and include it in generated commands when set.

- [ ] **Step 5: Verify scripts**

Run:

```bash
rtk cargo test --test script_tools -- --nocapture
rtk scripts/generate_observations.sh --dataset benchmarks/positions-v1.json --output benchmarks/results/dev.json --checkpoint-dir benchmarks/results/dev-ckpt --engines mcts,minmax --book benchmarks/results/book.db --dry-run
```

Expected: test passes; dry-run command includes `--book benchmarks/results/book.db`.

- [ ] **Step 6: Commit**

```bash
rtk git add crates/quantik-core/src/bench/adapters.rs crates/quantik-core/src/bin/cross_engine_benchmark.rs scripts crates/quantik-core/tests/script_tools.rs
rtk git commit -m "Thread opening book through benchmark runs"
```

---

## Task 6: Add Contract Conformance Tests In Rust

**Files:**
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust/crates/quantik-core/src/opening_book.rs`

- [ ] **Step 1: Write schema introspection test**

Add:

```rust
#[test]
fn sqlite_schema_matches_opening_book_v1_contract() {
    let path = temp_db_path();
    let config = OpeningBookConfig {
        database_path: path.clone(),
        ..Default::default()
    };
    let _db = OpeningBookDatabase::open(&config).unwrap();
    let conn = Connection::open(&path).unwrap();

    let table_columns = |table: &str| -> Vec<String> {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})")).unwrap();
        stmt.query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    };

    assert_eq!(
        table_columns("positions"),
        vec![
            "canonical_key",
            "qfen",
            "depth",
            "evaluation",
            "visit_count",
            "win_count_p0",
            "win_count_p1",
            "draw_count",
            "is_terminal",
            "symmetry_count",
            "created_at",
            "solved",
            "game_value",
        ]
    );
    assert_eq!(
        table_columns("best_moves"),
        vec!["canonical_key", "move_rank", "shape", "position"]
    );
    assert_eq!(
        table_columns("position_edges"),
        vec!["parent_key", "child_key"]
    );
    assert_eq!(table_columns("book_metadata"), vec!["key", "value"]);

    fs::remove_file(&path).ok();
}
```

- [ ] **Step 2: Verify test**

Run:

```bash
rtk cargo test sqlite_schema_matches_opening_book_v1_contract -- --nocapture
```

Expected: passes after Task 2.

- [ ] **Step 3: Run full Rust tests**

```bash
rtk cargo test
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
rtk git add crates/quantik-core/src/opening_book.rs
rtk git commit -m "Assert opening book schema contract"
```

---

## Task 7: Plan The Transform-Aware Upgrade

**Files:**
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust/docs/BENCHMARKS.md`
- Modify: `/Users/mauroberlanda/Code/quantik-ns/quantik-core-contracts/docs/opening-book-v1.md`

- [ ] **Step 1: Add explicit follow-up note**

In both docs, add:

```markdown
## Transform-Aware Lookup Follow-Up

`opening-book.v1` is representative-only for move serving. To serve every
orientation, the next compatible extension must expose the canonical transform
used for each query and store enough transform metadata to translate
`best_moves` from stored orientation back to query orientation. The extension
must define stable D4 indices, stable shape permutation encoding, inverse move
mapping, and cross-language parity tests before engines may serve book moves for
non-representative boards.
```

- [ ] **Step 2: Create a future implementation issue**

Use GitHub issue or local plan title:

```text
Add transform-aware opening-book lookup
```

Acceptance criteria:

- Rust `SymmetryHandler` exposes `(canonical_bitboard, d4_index, shape_perm)`.
- Move mapping works query -> canonical and canonical -> query.
- Python/Rust agree on D4 index and shape permutation encoding.
- `opening-book.v1` docs gain an additive `transform` metadata section or a future `opening-book.v2` if old readers cannot ignore it safely.
- Book hits for non-representative boards return legal moves in the queried orientation.

- [ ] **Step 3: Commit docs**

```bash
rtk git add docs/BENCHMARKS.md ../quantik-core-contracts/docs/opening-book-v1.md
rtk git commit -m "Document transform-aware opening book follow-up"
```

---

## Final Verification

- [ ] **Contracts validation**

```bash
cd /Users/mauroberlanda/Code/quantik-ns/quantik-core-contracts
rtk python3 scripts/validate_contracts.py
```

Expected: exits 0.

- [ ] **Rust full suite**

```bash
cd /Users/mauroberlanda/Code/quantik-ns/quantik-core-rust
rtk cargo test
```

Expected: all tests pass.

- [ ] **Smoke book-backed observation dry run**

```bash
rtk scripts/generate_observations.sh \
  --dataset benchmarks/positions-v1.json \
  --output benchmarks/results/book-backed.json \
  --checkpoint-dir benchmarks/results/book-backed-ckpt \
  --engines mcts,minimax \
  --book benchmarks/results/book.db \
  --dry-run
```

Expected: command includes `--book benchmarks/results/book.db`.

## Spec Coverage Self-Review

- Opening book imported into engines: Tasks 3-5.
- Incrementally smarter players: engines use book first and fallback to search; book can grow via existing export/search flows.
- Contracts describe actual opening-book structure: Tasks 1, 2, and 6.
- Storage considerations expanded with the database cheatsheet framing: Task 1 Step 4.
- Representative-only safety is explicit and preserved: Tasks 1, 3, and 7.
- Parameterized generation workflows keep book support: Task 5.

No placeholders remain; every implementation task has target files, expected tests, and commit steps.

