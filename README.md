# quantik-core-rust

A high-performance Quantik board game engine written in Rust.

## Overview

Quantik is a two-player abstract strategy game on a 4x4 grid with four shapes (A, B, C, D).
Players alternate placing pieces. A line (row, column, or 2x2 zone) wins when it contains
all four different shapes, regardless of which player placed them. A player may not place a
shape on a line where the opponent already has that same shape.

## Modules

| Module | Description |
|--------|-------------|
| `bitboard` | 128-bit bitboard representation using `[u16; 8]` |
| `constants` | Win masks, game limits, version flags |
| `moves` | Move struct, validation, legal move generation |
| `game` | Win detection, turn logic, game-over checks |
| `qfen` | QFEN (Quantik FEN) encode/decode |
| `state` | State struct with binary pack/unpack and canonical keys |
| `symmetry` | D4 + shape permutation canonicalization with LUT |
| `board` | High-level `QuantikBoard` with inventory and undo |
| `mcts` | Monte Carlo Tree Search engine (UCB1) |
| `opening_book` | SQLite-backed opening book database |

## Building

```sh
cargo build --release
```

## Testing

```sh
cargo test
```
