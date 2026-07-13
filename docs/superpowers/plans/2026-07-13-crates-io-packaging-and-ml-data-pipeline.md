# crates.io Packaging + ML Self-Play Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Task 3 is human-only and cannot be delegated to an agent** — it requires the repo owner's personal crates.io account and login token.

**Goal:** Publish `quantik-core` to crates.io mirroring the conventions already established by the sibling Python package (`quantik-core` on PyPI), then build the first concrete slice of the ML roadmap: exposing MCTS's visit-count policy distribution and a self-play dataset exporter, so a future Python training pipeline has real data to consume instead of a speculative format.

**Architecture:** Part A (Tasks 1-4) is pure packaging/CI work on the existing `crates/quantik-core` crate — no behavior changes, only metadata, a LICENSE file, and a GitHub Actions publish workflow using crates.io's OIDC Trusted Publishing (the same trust model, mechanically, as the Python package's PyPI trusted publishing). Part B (Tasks 5-6) is the first Rust-only increment of the ML data pipeline: a new `MCTSEngine` accessor exposing per-root-move visit counts (the raw material for an AlphaZero-style soft policy target, which `search()` today throws away in favor of a single argmax move), and a self-play exporter that plays full games and writes one JSONL training row per ply using the existing canonical-JSON encoder for cross-language byte compatibility.

**Tech Stack:** Rust (existing crate, no new dependencies for Tasks 1-2, 5-6), GitHub Actions (`rust-lang/crates-io-auth-action@v1` for Task 2), crates.io (manual account actions in Task 3).

## Global Constraints

- CI gates (existing, `.github/workflows/rust.yml`): `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo test --workspace --all-targets --all-features --locked`, `cargo build --workspace --all-targets --release --locked`. Every task must pass all four before merge.
- The crate name `quantik-core` is confirmed available on crates.io as of this plan (checked via `GET https://crates.io/api/v1/crates/quantik-core` → `crate does not exist`) — mirrors the Python package's PyPI name exactly. Publish under this name, not a variant.
- crates.io Trusted Publishing requires an initial **manual** `cargo publish` before a Trusted Publisher Configuration can be created in the crates.io UI (verified against RFC 3691, the crates.io trusted-publishing RFC). This means Task 3 cannot be automated by an agent — it needs the repo owner's own `cargo login` session. Tasks 1-2 prepare everything needed; Task 3 is a short manual runbook for the human; Task 4 verifies the automated path afterward.
- Quantik has no draw outcome (established fact, verified this repository — see `docs/benchmarks/quantik-game-tree-census-2026-07-13.md`): every terminal position is a decisive win. Any dataset schema for self-play data must encode `value` as strictly `+1.0` or `-1.0`, never `0.0` for "draw."
- New JSONL dataset rows must be written via `bench::canonical::canonical_json`, matching every other JSONL artifact in this codebase (`bench/checkpoint.py`'s Rust port), so the format stays byte-compatible with a future Python reader without a second encoder needing to be written.
- `MCTSEngine::search`'s existing public signature and behavior must not change — Task 5 is strictly additive (a new method), since `search` already has callers and tests elsewhere in the crate (`bench/adapters.rs`'s `MCTSAdapter`).

---

## Progress Ledger (updated as tasks merge)

| Task | PR | Status |
|---|---|---|
| 1 crates.io metadata (Cargo.toml, LICENSE) | #16 | MERGED (combined with Task 2 and README updates in one PR) |
| 2 Trusted-publishing CI workflow | #16 | MERGED; `publish-crate` job present but will fail until Task 3 is done |
| 3 Manual initial publish + Trusted Publisher config (human-only) | | **Blocked on repo owner** — see runbook below |
| 4 Verify automated publish via a patch release | | Not started |
| 5 Expose MCTS root-move visit-count distribution | | Not started |
| 6 Self-play dataset exporter | | Not started |

## Delegation Protocol

Same pipeline used for the rest of this repo's incremental work: implementation delegated to a **Sonnet** subagent, each finished PR reviewed by an **Opus** subagent before merge, orchestrator merges after CI is green. **Exception: Task 3 is not delegable** — it requires the crates.io account owner's own login session and cannot be performed by any agent, subagent or otherwise. Do not attempt to script around this; it is a deliberate security boundary (the same one PyPI's trusted publishing enforces for the Python package).

**Implementation subagent contract** (one task per agent):
- Branch from up-to-date `main`.
- Implement the task's steps exactly as written.
- Run all four CI gates locally before opening a PR.
- Commit message per the task, PR description summarizing the change and linking back to this plan file.
- Do not merge — hand off for review.

**Review subagent contract:**
- Review the PR diff against this task's spec and the Global Constraints above.
- Flag anything that changes `MCTSEngine::search`'s existing behavior (Task 5 must be purely additive).
- Flag any dataset row that could ever encode `value: 0.0` (Task 6 — no draws exist, a zero value is a bug, not a legitimate outcome).

---

## Part A: crates.io Packaging

### Task 1: Crate metadata for crates.io publication

**Files:**
- Modify: `crates/quantik-core/Cargo.toml`
- Create: `LICENSE` (repo root)
- Create: `crates/quantik-core/README.md` (crate-local, since `cargo publish` only packages files inside the crate directory — the repo-root `README.md` is workspace-level documentation and is not reachable by a package-relative `readme` path across the workspace boundary without extra verification, so this task creates a crate-scoped README instead of guessing that cross-directory paths work)

**Interfaces:**
- Consumes: nothing (metadata-only change).
- Produces: a `crates/quantik-core/Cargo.toml` with every field `cargo publish --dry-run` requires satisfied; a `LICENSE` file at the repo root; a `crates/quantik-core/README.md` crates.io will render on the crate's page.

- [ ] **Step 1: Add package metadata to `crates/quantik-core/Cargo.toml`**

Replace the `[package]` section (currently just `name`, `version`, `edition`, `description`) with:

```toml
[package]
name = "quantik-core"
version = "0.1.0"
edition = "2021"
description = "High-performance Quantik board game engine"
license = "MIT"
readme = "README.md"
repository = "https://github.com/mberlanda/quantik-core-rust"
homepage = "https://github.com/mberlanda/quantik-core-rust"
documentation = "https://docs.rs/quantik-core"
keywords = ["quantik", "board-games", "mcts", "bitboards", "game-ai"]
categories = ["game-development", "algorithms"]
authors = ["Mauro Berlanda <mauro.berlanda@gmail.com>"]
```

Notes on choices, so a reviewer isn't left guessing:
- `version` stays `0.1.0` rather than mirroring the Python package's `1.0.0` — this is a brand-new public crate with no external consumers yet, and starting pre-1.0 while the API can still shift without a breaking-change ceremony is standard Rust practice, not a departure from "the same convention." The *naming, license, and CI shape* mirror Python; the version number is an independent decision.
- `keywords` is capped at 5 by crates.io (verified: Python's `pyproject.toml` lists 9; this is the closest 5 that survive the cap without becoming redundant with each other).
- `categories` uses crates.io's actual category slugs (verified via `GET /api/v1/categories`): `game-development` (closest match to Python's "Games/Entertainment :: Board Games") and `algorithms` (closest match to "Scientific/Engineering :: Artificial Intelligence" — this crate's minimax/MCTS/beam search are literally algorithm implementations, `game-engines` is for one-stop-shop engines like Bevy and doesn't fit a focused state/search library).
- `rust-version` (MSRV) is deliberately omitted: no MSRV has been verified against an older toolchain in this repository, and asserting an unverified floor would be worse than omitting the field. If a future task verifies one (e.g. via `cargo-msrv` or CI matrix testing), add it then.

- [ ] **Step 2: Create the LICENSE file**

Create `LICENSE` at the repo root (mirrors the Python package's MIT license text exactly, same copyright holder, this repo's own creation year):

```
MIT License

Copyright (c) 2026 Mauro Berlanda

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Create a crate-local README for the crates.io package page**

Create `crates/quantik-core/README.md`. This is intentionally short — it's what renders on crates.io and docs.rs, not the full repo documentation (which stays at the repo-root `README.md` for GitHub visitors):

```markdown
# quantik-core

High-performance Rust engine for the [Quantik](https://github.com/mberlanda/quantik-core-rust) board game: bitboard state, QFEN notation, canonical symmetry-reduced keys, and minimax/MCTS/beam-search engines.

Rust counterpart to the [Python `quantik-core` package](https://pypi.org/project/quantik-core/) — same core model (a tiny bitboard state, QFEN for human-readable positions, canonical binary keys for search caches and databases), byte-compatible canonical keys across both languages.

See the [repository README](https://github.com/mberlanda/quantik-core-rust#readme) for full documentation, and [`docs/BENCHMARKS.md`](https://github.com/mberlanda/quantik-core-rust/blob/main/docs/BENCHMARKS.md) for cross-engine performance data.

## License

MIT
```

- [ ] **Step 4: Verify the package builds and dry-run publishes cleanly**

```bash
cd crates/quantik-core
cargo package --list
```

Expected: a file listing that includes `Cargo.toml`, `README.md`, `src/**/*.rs`, `examples/**/*.rs` — and does **not** error about a missing or unresolvable `readme` path.

```bash
cargo publish --dry-run
```

Expected: `Packaging quantik-core v0.1.0` followed by `Uploading quantik-core v0.1.0 (dry run)` with no errors. If this fails on the `readme` field specifically, that confirms the cross-directory-path concern above — the fix is already applied by Step 3 (crate-local README), so a failure here means Step 3 wasn't done correctly, not that a new approach is needed.

- [ ] **Step 5: Run full CI gates**

```bash
cd /path/to/quantik-core-rust
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features --locked
cargo build --workspace --all-targets --release --locked
```

Expected: all four pass with no changes needed (this task touches only metadata files, not source).

- [ ] **Step 6: Commit**

```bash
git add crates/quantik-core/Cargo.toml crates/quantik-core/README.md LICENSE
git commit -m "chore: add crates.io publication metadata, LICENSE, and crate README"
```

---

### Task 2: Trusted-publishing GitHub Actions workflow

**Files:**
- Modify: `.github/workflows/rust.yml`

**Interfaces:**
- Consumes: Task 1's `Cargo.toml` metadata (the workflow will fail to publish without it, but that failure won't surface until Task 4, since this workflow only runs on a GitHub Release).
- Produces: a `publish-crate` job that runs on `release: published`, gated behind Task 3's manual crates.io configuration.

- [ ] **Step 1: Add the publish job to `.github/workflows/rust.yml`**

Add this job at the end of the existing workflow file (after `release-artifacts`), matching the verified crates.io Trusted Publishing pattern (RFC 3691) — structurally the same shape as the Python package's `publish.yml` (test gate, then publish gate, triggered by a published GitHub Release):

```yaml
  publish-crate:
    name: Publish to crates.io
    runs-on: ubuntu-latest
    needs: ci
    if: github.event_name == 'release' && github.event.action == 'published'
    environment: release
    permissions:
      id-token: write
      contents: read

    steps:
      - name: Checkout
        uses: actions/checkout@v6

      - name: Install Rust toolchain
        run: rustup toolchain install stable --profile minimal

      - name: Use stable Rust
        run: rustup default stable

      - name: Authenticate with crates.io
        id: auth
        uses: rust-lang/crates-io-auth-action@v1

      - name: Publish to crates.io
        run: cargo publish -p quantik-core
        env:
          CARGO_REGISTRY_TOKEN: ${{ steps.auth.outputs.token }}
```

Notes:
- `on: release: types: [published]` is not yet in this workflow's top-level `on:` block — add it alongside the existing `pull_request`/`push`/`workflow_dispatch` triggers, so the new job actually has an event to react to:

```yaml
on:
  pull_request:
  push:
    branches:
      - main
    tags:
      - "v*"
  release:
    types: [published]
  workflow_dispatch:
```

- `environment: release` matches the RFC's recommended pattern (lets the repo owner add required-reviewer protection on this specific GitHub Actions environment later, without any workflow changes) — it does not need to exist yet for this workflow to be syntactically valid; GitHub creates environments implicitly on first use, but **this job will fail at the `crates-io-auth-action` step until Task 3's manual Trusted Publisher configuration exists on crates.io** — that dependency is expected and resolved by Task 3, verified by Task 4.
- `needs: ci` reuses the existing `ci` job (fmt/clippy/test/build) already defined earlier in this file — do not duplicate those steps.

- [ ] **Step 2: Validate workflow YAML syntax**

```bash
cd /path/to/quantik-core-rust
ruby -ryaml -e "
doc = YAML.load_file('.github/workflows/rust.yml')
raise 'publish-crate job missing' unless doc['jobs'].key?('publish-crate')
job = doc['jobs']['publish-crate']
raise 'needs wrong' unless job['needs'] == 'ci'
raise 'id-token missing' unless job['permissions']['id-token'] == 'write'
on_key = doc['on'] || doc[true]
raise 'release trigger missing' unless on_key.key?('release')
puts 'YAML valid, publish-crate job present with correct needs/permissions'
"
```

(Ruby's YAML library ships with macOS by default — no `pip install` needed, which matters here since this machine's Python is externally-managed and blocks ad hoc `pip install pyyaml` without `--break-system-packages`.)

Expected: `YAML valid, publish-crate job present with correct needs/permissions`. This checks syntax and structure only, not full GitHub Actions semantics — the job cannot be fully exercised until Task 4, since it requires a real GitHub Release event. This exact command was run against a scratch copy of the edited file while writing this plan, confirming Step 1's YAML is correct as written.

- [ ] **Step 3: Run full CI gates**

Same four commands as Task 1 Step 5 — this task doesn't touch Rust source, so nothing should change.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/rust.yml
git commit -m "ci: add crates.io trusted-publishing workflow, gated on GitHub Release"
```

---

### Task 3 (HUMAN-ONLY — not agent-delegable): Manual initial publish + Trusted Publisher configuration

This task requires the repo owner's personal crates.io account. No agent can perform it — crates.io intentionally requires a human-authenticated first publish before any automation can be trusted (the same reason PyPI required a manual first publish for the Python package before trusted publishing could be enabled there).

**Runbook for the repo owner:**

1. Merge Tasks 1 and 2 to `main` first (so the published crate's source matches what's actually in the repo).
2. Log in to crates.io locally, if not already:
   ```bash
   cargo login
   ```
   (Follow the prompt to paste an API token generated at https://crates.io/settings/tokens.)
3. From the repo root:
   ```bash
   cd crates/quantik-core
   cargo publish
   ```
   This performs the first, manual publish of `quantik-core` v0.1.0 to crates.io.
4. In the crates.io web UI, go to the `quantik-core` crate page → Settings → Trusted Publishing, and add a GitHub Actions publisher with:
   - **Repository owner:** `mberlanda`
   - **Repository name:** `quantik-core-rust`
   - **Workflow filename:** `rust.yml`
   - **Environment name:** `release` (matches the `environment: release` set in Task 2's job)
5. No further local `cargo publish` should be needed after this — future releases go through Task 2's GitHub Actions job.

**Update the Progress Ledger** once done (mark Task 3 complete with the publish date), so Task 4 knows it's safe to proceed.

---

### Task 4: Verify the automated publish path with a patch release

**Files:**
- Modify: `crates/quantik-core/Cargo.toml` (version bump only)
- Modify: `CHANGELOG.md` (release entry)

**Interfaces:**
- Consumes: Task 3's Trusted Publisher configuration.
- Produces: confirmation that `git tag vX.Y.Z` + a published GitHub Release triggers a successful, tokenless crates.io publish.

- [ ] **Step 1: Bump the version and move the CHANGELOG's Unreleased section**

In `crates/quantik-core/Cargo.toml`, bump `version = "0.1.0"` to `version = "0.1.1"`.

In `CHANGELOG.md`, rename the current `## Unreleased` heading to `## 0.1.1 - <today's date>` and add a fresh empty `## Unreleased` section above it, matching this repo's existing CHANGELOG convention (already visible in the file's current structure).

- [ ] **Step 2: Run full CI gates, commit, push, tag**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features --locked
cargo build --workspace --all-targets --release --locked
git add crates/quantik-core/Cargo.toml CHANGELOG.md
git commit -m "chore: release quantik-core v0.1.1"
git push
git tag v0.1.1
git push origin v0.1.1
```

- [ ] **Step 3: Publish a GitHub Release from the tag**

```bash
gh release create v0.1.1 --title v0.1.1 --generate-notes
```

- [ ] **Step 4: Confirm the workflow ran and published successfully**

```bash
gh run list --workflow=rust.yml --limit 3
gh run view --log <run-id> | grep -A5 "Publish to crates.io"
```

Expected: the `publish-crate` job succeeded, and:

```bash
curl -s -H "User-Agent: quantik-core-rust-verify" "https://crates.io/api/v1/crates/quantik-core/0.1.1"
```

returns the new version, confirming the tokenless, trusted-publishing path works end-to-end.

- [ ] **Step 5: Update the Progress Ledger**

Mark Tasks 1-4 complete in this plan's Progress Ledger table.

---

## Part B: ML Data Pipeline, First Increment (Rust-only)

### Task 5: Expose MCTS root-move visit-count distribution

**Files:**
- Modify: `crates/quantik-core/src/mcts.rs`

**Interfaces:**
- Consumes: `MCTSEngine`'s existing private `nodes: Vec<MCTSNode>` field (already populated by `search()`; this task only adds a read accessor, no changes to search logic).
- Produces: `MCTSEngine::root_move_visits(&self) -> Vec<(Move, u32)>` — for later tasks (Task 6, and the future Python training pipeline) to build a normalized policy target (`visits / total_visits` per legal move) instead of just the single best move `search()` already returns.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `crates/quantik-core/src/mcts.rs` (near the other `MCTSEngine` tests):

```rust
#[test]
fn root_move_visits_covers_every_legal_move_and_sums_to_iterations() {
    let bb = Bitboard::EMPTY;
    let legal = generate_legal_moves(&bb);

    let mut engine = MCTSEngine::new(MCTSConfig {
        max_iterations: 2000,
        seed: Some(7),
        ..Default::default()
    });
    let (best_move, _win_prob) = engine.search(&bb).expect("legal moves exist");

    let visits = engine.root_move_visits();

    // Every legal root move was expanded at least once (2000 iterations
    // against a 64-move branching factor is far more than one pass).
    assert_eq!(visits.len(), legal.len());
    let visited_moves: std::collections::HashSet<Move> =
        visits.iter().map(|(mv, _)| *mv).collect();
    for mv in &legal {
        assert!(visited_moves.contains(mv), "missing {mv:?} from root_move_visits");
    }

    // Visit counts sum to the iterations actually performed (root gets
    // one visit per iteration via the selection pass starting there).
    let total_visits: u32 = visits.iter().map(|(_, v)| v).sum();
    assert_eq!(total_visits, 2000);

    // The move search() actually returned must be among the visited
    // moves, and must have the maximum visit count (search() picks by
    // visit count, not raw value).
    let best_visits = visits
        .iter()
        .find(|(mv, _)| *mv == best_move)
        .map(|(_, v)| *v)
        .unwrap();
    assert!(visits.iter().all(|(_, v)| *v <= best_visits));
}

#[test]
fn root_move_visits_empty_before_search() {
    let engine = MCTSEngine::new(MCTSConfig::default());
    assert!(engine.root_move_visits().is_empty());
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test --lib root_move_visits -p quantik-core
```

Expected: FAIL with `no method named 'root_move_visits' found for struct 'MCTSEngine'`.

- [ ] **Step 3: Implement `root_move_visits`**

In `crates/quantik-core/src/mcts.rs`, inside `impl MCTSEngine`, add this method near `search` (after it):

```rust
    /// Visit-count distribution over the root's legal moves from the most
    /// recent `search()` call — the raw material for an AlphaZero-style
    /// soft policy target (`visits / total_visits` per move), as opposed
    /// to the single argmax move `search()` returns. Empty if `search()`
    /// returned `None` (no legal moves) or hasn't been called yet.
    pub fn root_move_visits(&self) -> Vec<(Move, u32)> {
        let Some(root) = self.nodes.first() else {
            return Vec::new();
        };
        root.children
            .iter()
            .map(|&child_idx| {
                let child = &self.nodes[child_idx];
                (
                    child.mv.expect("child node always has a move"),
                    child.visit_count,
                )
            })
            .collect()
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cargo test --lib root_move_visits -p quantik-core
```

Expected: `test result: ok. 2 passed`.

- [ ] **Step 5: Run full CI gates**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features --locked
cargo build --workspace --all-targets --release --locked
```

- [ ] **Step 6: Commit**

```bash
git add crates/quantik-core/src/mcts.rs
git commit -m "feat: expose MCTSEngine::root_move_visits for policy-target extraction"
```

---

### Task 6: Self-play dataset exporter

**Files:**
- Create: `crates/quantik-core/examples/selfplay_export.rs`

**Interfaces:**
- Consumes: `MCTSEngine::search` and `MCTSEngine::root_move_visits` (Task 5); `bench::canonical::canonical_json` (existing, for byte-compatible JSONL rows); `State::to_qfen`, `generate_legal_moves`, `apply_move`, `check_winner`/`has_winning_line`, `current_player` (all existing).
- Produces: a JSONL file, one training example per ply per self-played game, in a schema documented below — the format a future Python loader will read.

- [ ] **Step 1: Write the exporter**

Create `crates/quantik-core/examples/selfplay_export.rs`:

```rust
//! Self-play data exporter: plays full games with MCTSEngine on both
//! sides, and writes one JSONL training row per ply — position, the
//! visit-count policy target (Task 5's `root_move_visits`), and the
//! eventual game outcome from that ply's mover perspective. Quantik has
//! no draws (see docs/benchmarks/quantik-game-tree-census-2026-07-13.md),
//! so `value` is always exactly +1.0 or -1.0, never 0.0.
//!
//! Row schema (one per line, compact canonical JSON):
//! {
//!   "game_id": u64,
//!   "ply": u32,
//!   "qfen": string,
//!   "side_to_move": 0 | 1,
//!   "policy": [{"shape": u8, "position": u8, "visits": u32}, ...],
//!   "value": 1.0 | -1.0   // outcome for `side_to_move`, decided in hindsight
//! }
//!
//! Usage:
//!   cargo run --release --example selfplay_export -- \
//!     --games 100 --iterations 2000 --seed 20260713 \
//!     --out benchmarks/results/selfplay.jsonl

use quantik_core::bench::canonical::canonical_json;
use quantik_core::bitboard::Bitboard;
use quantik_core::game::{check_winner, current_player, has_winning_line, WinStatus};
use quantik_core::mcts::{MCTSConfig, MCTSEngine};
use quantik_core::moves::{apply_move, generate_legal_moves};
use quantik_core::state::State;
use serde_json::json;
use std::io::Write;

struct Args {
    games: u32,
    iterations: u32,
    seed: u64,
    out: String,
}

fn parse_args() -> Args {
    let mut games = 100u32;
    let mut iterations = 2000u32;
    let mut seed = 20260713u64;
    let mut out = "benchmarks/results/selfplay.jsonl".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--games" => games = it.next().unwrap().parse().unwrap(),
            "--iterations" => iterations = it.next().unwrap().parse().unwrap(),
            "--seed" => seed = it.next().unwrap().parse().unwrap(),
            "--out" => out = it.next().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    Args {
        games,
        iterations,
        seed,
        out,
    }
}

struct PendingRow {
    ply: u32,
    qfen: String,
    side_to_move: u8,
    policy: Vec<(u8, u8, u32)>, // (shape, position, visits)
}

/// Play one self-play game to completion, returning one pending row per
/// ply (value filled in afterward, once the winner is known).
fn play_game(seed: u64, iterations: u32) -> (Vec<PendingRow>, WinStatus) {
    let mut bb = Bitboard::EMPTY;
    let mut rows = Vec::new();
    let mut ply = 0u32;

    loop {
        if has_winning_line(&bb) {
            return (rows, check_winner(&bb));
        }
        let legal = generate_legal_moves(&bb);
        if legal.is_empty() {
            // No legal moves: the side to move loses (see Global
            // Constraints — this is a decisive result, never a draw).
            let loser = current_player(&bb).unwrap();
            let winner = if loser == 0 {
                WinStatus::Player1Wins
            } else {
                WinStatus::Player0Wins
            };
            return (rows, winner);
        }

        let side_to_move = current_player(&bb).unwrap();
        let mut engine = MCTSEngine::new(MCTSConfig {
            max_iterations: iterations,
            seed: Some(seed.wrapping_add(ply as u64)),
            ..Default::default()
        });
        let (best_move, _) = engine.search(&bb).expect("legal moves exist");
        let policy: Vec<(u8, u8, u32)> = engine
            .root_move_visits()
            .into_iter()
            .map(|(mv, visits)| (mv.shape, mv.position, visits))
            .collect();

        rows.push(PendingRow {
            ply,
            qfen: State::new(bb).to_qfen(),
            side_to_move,
            policy,
        });

        bb = apply_move(&bb, &best_move);
        ply += 1;
    }
}

fn main() {
    let args = parse_args();
    if let Some(parent) = std::path::Path::new(&args.out).parent() {
        std::fs::create_dir_all(parent).expect("mkdir output dir");
    }
    let mut file = std::fs::File::create(&args.out).expect("create output file");

    for game_id in 0..args.games {
        let (rows, winner) = play_game(args.seed.wrapping_add(game_id as u64 * 1000), args.iterations);
        assert_ne!(winner, WinStatus::NoWin, "game must resolve to a decisive winner");

        for row in rows {
            let value = match (winner, row.side_to_move) {
                (WinStatus::Player0Wins, 0) => 1.0,
                (WinStatus::Player0Wins, 1) => -1.0,
                (WinStatus::Player1Wins, 0) => -1.0,
                (WinStatus::Player1Wins, 1) => 1.0,
                (WinStatus::NoWin, _) => unreachable!(),
            };
            let policy_json: Vec<_> = row
                .policy
                .iter()
                .map(|(shape, position, visits)| {
                    json!({"shape": shape, "position": position, "visits": visits})
                })
                .collect();
            let record = json!({
                "game_id": game_id,
                "ply": row.ply,
                "qfen": row.qfen,
                "side_to_move": row.side_to_move,
                "policy": policy_json,
                "value": value,
            });
            writeln!(file, "{}", canonical_json(&record)).expect("write row");
        }

        if (game_id + 1) % 10 == 0 || game_id + 1 == args.games {
            println!("{}/{} games exported -> {}", game_id + 1, args.games, args.out);
        }
    }
}
```

- [ ] **Step 2: Build it**

```bash
cargo build --release --example selfplay_export -p quantik-core
```

Expected: clean build.

- [ ] **Step 3: Run a small export and verify the output**

```bash
./target/release/examples/selfplay_export --games 5 --iterations 500 --seed 1 --out /tmp/selfplay-smoke.jsonl
wc -l /tmp/selfplay-smoke.jsonl
python3 -c "
import json
rows = [json.loads(l) for l in open('/tmp/selfplay-smoke.jsonl')]
assert all(r['value'] in (1.0, -1.0) for r in rows), 'found a non-decisive value'
assert all(len(r['policy']) > 0 for r in rows), 'found an empty policy'
games = set(r['game_id'] for r in rows)
assert games == set(range(5)), games
print(f'{len(rows)} rows across {len(games)} games, all decisive, all schema-valid')
"
```

Expected: prints a row count with no assertion errors — confirms every row has a decisive (`+1.0`/`-1.0`) value and a non-empty policy, matching the Global Constraints.

- [ ] **Step 4: Run full CI gates**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features --locked
cargo build --workspace --all-targets --release --locked
```

- [ ] **Step 5: Commit**

```bash
git add crates/quantik-core/examples/selfplay_export.rs
git commit -m "feat: add self-play dataset exporter (JSONL, MCTS policy + decisive value targets)"
```

---

## Roadmap beyond this plan (not detailed here — separate future plans)

The remaining ML phases are deliberately **not** broken into bite-sized steps in this document, per the writing-plans scope-check: they're a different subsystem (Python, not Rust), and their concrete design depends on what Task 6's exported data actually looks like once real self-play runs exist — writing fake bite-sized training-loop code now would violate the "no placeholders" rule in spirit even if each line compiled.

- **Phase C — Python tensor encoding + dataset loader.** Consume Task 6's JSONL schema; encode `qfen` into the 9-channel board tensor (8 colour/shape planes + 1 side-to-move plane), normalize `policy` visit counts into a probability distribution over the 64 `(shape, position)` actions (masking illegal ones), keep `value` as-is.
- **Phase D — Small policy/value network + imitation-learning training loop.** PyTorch, trained on Colab per the original report's guidance (few conv layers, policy + value heads, cross-entropy + MSE loss) — but now against real Task 6 data instead of a speculative format.
- **Phase E — Evaluation.** Wire the trained network in as a new adapter in the *existing* `cross_engine_benchmark` harness (Python side has a pluggable `EngineAdapter`-equivalent already) so it gets measured against random/MCTS/minimax with the same Wilson-CI-backed agreement/cost/h2h methodology already built and trusted in this project, rather than inventing a separate Elo-ladder tool from scratch.

Each of these should get its own `docs/superpowers/plans/YYYY-MM-DD-<name>.md` once the prior phase's real output is available to design against.
