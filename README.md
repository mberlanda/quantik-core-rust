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

