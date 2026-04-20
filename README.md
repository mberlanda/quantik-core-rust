# quantik-core-rust

A high-performance Quantik board game engine written in Rust.

## Overview

Quantik is a two-player abstract strategy game on a 4x4 grid with four shapes (A, B, C, D).
Players alternate placing pieces. A line (row, column, or 2x2 zone) wins when it contains
all four different shapes, regardless of which player placed them. A player may not place a
shape on a line where the opponent already has that same shape.

## Modules


| Module         | Description                                             |
| -------------- | ------------------------------------------------------- |
| `bitboard`     | 128-bit bitboard representation using `[u16; 8]`        |
| `constants`    | Win masks, game limits, version flags                   |
| `moves`        | Move struct, validation, legal move generation          |
| `game`         | Win detection, turn logic, game-over checks             |
| `qfen`         | QFEN (Quantik FEN) encode/decode                        |
| `state`        | State struct with binary pack/unpack and canonical keys |
| `symmetry`     | D4 + shape permutation canonicalization with LUT        |
| `board`        | High-level `QuantikBoard` with inventory and undo       |
| `mcts`         | Monte Carlo Tree Search engine (UCB1)                   |
| `opening_book` | SQLite-backed opening book database                     |


## Building

```sh
cargo build --release
```

## Testing

```sh
cargo test
```

## IDDFS Opening Book Builder

The `bench_bfs` binary builds an opening book using hybrid iterative-deepening DFS
(IDDFS) with a persistent SQLite transposition table. The algorithm provides BFS-like
completeness (discovering positions at their shallowest depth) without holding the
full frontier in RAM.

### Algorithm

**Phase 1 — Exhaustive (depths 0..N):** For each depth limit from 1 to N, a full
DFS pass is run from the root. An in-memory `HashMap` serves as the transposition
table: a position is skipped if its `searched_depth` already covers the remaining
depth budget. Results are flushed to SQLite in batches.

**Phase 2 — Selective (depths N+1..M):** After the exhaustive phase, expansion
continues with the same IDDFS loop. A future extension can add priority-queue
ordering (high symmetry count first, near forced wins, high uncertainty).

### Usage

```sh
cargo run --release --bin bench_bfs -- [OPTIONS] <DEPTH>
```

| Option                  | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| `<DEPTH>`               | Maximum depth to explore (required)                     |
| `--db <path>`           | SQLite database path (default: `quantik_book.db`)       |
| `--resume`              | Resume from existing database (uses `searched_depth`)   |
| `--max-positions N`     | Stop after N total positions (dropout)                  |
| `--exhaustive-depth N`  | Depth for exhaustive expansion (default: same as depth) |
| `--batch-size N`        | SQLite transaction batch size (default: 50000)          |
| `--quiet`               | Only print the final summary                            |

### Examples

```sh
# IDDFS to depth 4 (11,739 canonical positions)
cargo run --release --bin bench_bfs -- 4

# IDDFS to depth 6, custom database (1,037,643 positions)
cargo run --release --bin bench_bfs -- 6 --db quantik_depth6.db

# Explore up to 500k positions, then resume later
cargo run --release --bin bench_bfs -- 8 --max-positions 500000
cargo run --release --bin bench_bfs -- 8 --resume

# Exhaustive to depth 4, then selective to depth 8
cargo run --release --bin bench_bfs -- 8 --exhaustive-depth 4
```

### Database Schema

The SQLite database uses two tables:

**positions** — one row per canonical position:

| Column           | Type    | Description                                      |
| ---------------- | ------- | ------------------------------------------------ |
| `canonical_key`  | BLOB    | 18-byte canonical key (primary key)              |
| `depth`          | INTEGER | Shallowest depth at which position was found     |
| `is_terminal`    | INTEGER | 1 if game over                                   |
| `winner`         | INTEGER | 0 or 1 if terminal, NULL otherwise               |
| `symmetry_count` | INTEGER | Orbit size under the 192-element symmetry group  |
| `searched_depth` | INTEGER | How deeply this position has been analyzed        |
| `score`          | REAL    | Evaluation score (reserved for future use)       |
| `status`         | INTEGER | 0 = unexplored, 1 = expanded, 2 = dropped       |

**edges** — parent-child move graph (many parents can reach the same child):

| Column       | Type | Description                  |
| ------------ | ---- | ---------------------------- |
| `parent_key` | BLOB | Parent position canonical key |
| `child_key`  | BLOB | Child position canonical key  |
| `move`       | TEXT | Move string, e.g. `P0S2P5`   |

### Known Position Counts

| Depth | Canonical Positions |
| ----- | ------------------- |
| 0     | 1                   |
| 1     | 3                   |
| 2     | 51                  |
| 3     | 726                 |
| 4     | 10,958              |
| 5     | 106,216             |
| 6     | 919,688             |
| **Total (depth 6)** | **1,037,643** |
