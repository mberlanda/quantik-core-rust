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

## BFS / DFS Position Enumerator

The `bench_bfs` binary enumerates all reachable Quantik positions using BFS or DFS,
storing results in a SQLite database with parent tracking, terminal detection, and
symmetry orbit sizes.

### Usage

```sh
cargo run --release --bin bench_bfs -- [OPTIONS] <DEPTH>
```

| Option               | Description                                    |
| -------------------- | ---------------------------------------------- |
| `<DEPTH>`            | Maximum depth to explore (required)            |
| `--db <path>`        | SQLite database path (default: quantik_bfs.db) |
| `--dfs`              | Use DFS instead of BFS                         |
| `--resume`           | Resume from an existing database               |
| `--max-positions N`  | Stop after N total positions (dropout)         |
| `--quiet`            | Only print the final summary                   |

### Examples

```sh
# BFS to depth 4
cargo run --release --bin bench_bfs -- 4

# DFS to depth 6, custom database
cargo run --release --bin bench_bfs -- 6 --dfs --db quantik_dfs.db

# Explore up to 500k positions, then resume later
cargo run --release --bin bench_bfs -- 8 --max-positions 500000
cargo run --release --bin bench_bfs -- 8 --resume
```

### Database Schema

The SQLite database stores one row per canonical position:

| Column           | Type    | Description                                      |
| ---------------- | ------- | ------------------------------------------------ |
| `canonical_key`  | BLOB    | 18-byte canonical key (primary key)              |
| `parent_key`     | BLOB    | Parent position's canonical key                  |
| `parent_move`    | TEXT    | Move string, e.g. `P0S2P5`                       |
| `depth`          | INTEGER | BFS depth from root                              |
| `is_terminal`    | INTEGER | 1 if game over                                   |
| `winner`         | INTEGER | 0 or 1 if terminal, NULL otherwise               |
| `symmetry_count` | INTEGER | Orbit size under the 192-element symmetry group  |
| `status`         | INTEGER | 0 = frontier, 1 = expanded, 2 = dropped         |

## Hybrid Opening Book Builder

The `book_builder` binary generates a production-quality opening book through three phases:

1. **Exhaustive canonical BFS** (depth 0–6): enumerates ALL reachable positions
2. **Selective iterative-deepening DFS** (depth 7–10): expands top-K moves per position using a heuristic
3. **MCTS-style deepening** (depth 11+): focuses on high-value uncertain lines

After all phases, a compact book table is exported with ranked moves and scores.

### Usage

```sh
cargo run --release --bin book_builder -- [OPTIONS]
```

| Option                  | Description                                      |
| ----------------------- | ------------------------------------------------ |
| `--db <path>`           | SQLite database path (default: quantik_book.db)  |
| `--exhaustive-depth N`  | Depth for exhaustive BFS phase (default: 6)      |
| `--selective-depth N`   | Max depth for selective DFS phase (default: 10)   |
| `--top-k N`             | Top K moves to expand in selective phase (default: 6) |
| `--mcts-iterations N`   | MCTS iterations for evaluation (default: 500)    |
| `--max-positions N`     | Stop after N total positions                     |
| `--resume`              | Resume from existing database                    |
| `--quiet`               | Minimal output                                   |

### Examples

```sh
# Full build with defaults (BFS to 6, selective DFS to 10, MCTS 500 iters)
cargo run --release --bin book_builder

# Quick test: BFS only to depth 4, no MCTS
cargo run --release --bin book_builder -- --exhaustive-depth 4 --selective-depth 4 --mcts-iterations 0

# Deep selective search with wider branching
cargo run --release --bin book_builder -- --selective-depth 12 --top-k 8

# Resume a previous build with more MCTS iterations
cargo run --release --bin book_builder -- --resume --mcts-iterations 2000

# Limit total positions explored
cargo run --release --bin book_builder -- --max-positions 500000
```

### Book Builder Database Schema

The book builder uses three tables:

**positions** — one row per canonical position:

| Column           | Type    | Description                                      |
| ---------------- | ------- | ------------------------------------------------ |
| `canonical_key`  | BLOB    | 18-byte canonical key (primary key)              |
| `depth`          | INTEGER | BFS depth from root                              |
| `is_terminal`    | INTEGER | 1 if game over                                   |
| `winner`         | INTEGER | 0 or 1 if terminal, NULL otherwise               |
| `symmetry_count` | INTEGER | Orbit size under the 192-element symmetry group  |
| `searched_depth` | INTEGER | Deepest remaining depth analyzed                 |
| `score`          | REAL    | Evaluation (-1.0 to 1.0, NULL if unknown)        |
| `visits`         | INTEGER | MCTS visit count                                 |
| `best_move`      | TEXT    | Best known move string, e.g. `P0S2P5`            |
| `status`         | INTEGER | 0=frontier, 1=expanded, 2=dropped, 3=solved      |

**edges** — parent-child move graph:

| Column       | Type | Description                  |
| ------------ | ---- | ---------------------------- |
| `parent_key` | BLOB | Parent position canonical key |
| `child_key`  | BLOB | Child position canonical key  |
| `move`       | TEXT | Move string, e.g. `P0S2P5`   |

**book** — compact opening book with ranked moves:

| Column          | Type | Description                                           |
| --------------- | ---- | ----------------------------------------------------- |
| `canonical_key` | BLOB | Position canonical key (primary key)                  |
| `moves_json`    | TEXT | JSON array of ranked moves with scores and depths     |

