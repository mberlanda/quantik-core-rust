# quantik-core-rust

A high-performance Rust engine for the Quantik board game, designed for fast
move generation, win detection, Monte Carlo search, symmetry-aware storage, and
opening-book construction.

This crate is the Rust counterpart to the Python Quantik core documentation. It
keeps the same core model: a tiny bitboard state, QFEN for human-readable
positions, and canonical binary keys for search caches and databases.

## Agent Context

Use this section as the shared context for coding agents, analysis agents, and
engine work.

- Quantik is a deterministic, perfect-information, two-player abstract strategy
  game on a 4x4 board.
- Player 0 moves first. In low-level state operations, the side to move is
  derived from piece counts: equal counts means player 0, and player 0 having
  one extra piece means player 1.
- Each player owns two copies of each of four shapes: A, B, C, and D.
- The win condition is color-agnostic: a row, column, or 2x2 zone wins when all
  four shapes are present in that line, regardless of which player placed them.
- The winner is the last player to move into the winning position.
- Legal move generation must enforce the core placement constraints: target
  square is empty, the player still has that shape available, and the opponent
  does not already have the same shape in any row, column, or 2x2 zone touched
  by the target. Board-level play also enforces the current player.
- `QFEN` is the debugging and fixture language. The 18-byte binary state and
  canonical key are the storage and transposition-table language.
- Rust canonicalization currently preserves player/color identity and normalizes
  over board D4 symmetries plus shape relabeling: `8 * 24 = 192` transforms.
  Do not assume color-swap canonicalization unless the implementation changes.
- Prefer `QuantikBoard` for game-play workflows because it tracks inventories,
  current player, move history, undo, game result, and stalemate handling.
  Prefer `State` and `Bitboard` for serialization, canonical keys, and search
  storage.

## Game Rules

### Board

Quantik is played on a 4x4 grid. Squares are indexed row-major from 0 to 15:

```text
r\c  0  1  2  3
 0   0  1  2  3
 1   4  5  6  7
 2   8  9 10 11
 3  12 13 14 15
```

The twelve scoring lines are:

- 4 rows
- 4 columns
- 4 zones, where each zone is a 2x2 quadrant

### Pieces

There are four shape types:

| Shape index | Letter |
| ----------- | ------ |
| 0           | A      |
| 1           | B      |
| 2           | C      |
| 3           | D      |

Each player has two pieces of each shape. Player 0 pieces are written with
uppercase letters in QFEN; player 1 pieces are written with lowercase letters.

### Legal Move

A move is `(player, shape, position)`. It is legal when:

1. `position` is empty.
2. `player` has at least one remaining piece of `shape`.
3. The opponent has no piece of the same `shape` in the target position's row,
   column, or 2x2 zone.
4. `player` is the current player.

Example: if player 0 has an `A` at position 0, player 1 cannot place `a` in
positions that share the top row, left column, or top-left 2x2 zone with
position 0. Player 1 may still place `a` elsewhere.

### Victory And Termination

A board is winning when any row, column, or 2x2 zone contains all four shapes
A, B, C, and D. Colors do not matter for line completion:

```text
A b C d  <- all four shapes in one row, so the last mover wins
. . . .
. . . .
. . . .
```

If no legal moves remain and no player has already won, `QuantikBoard` treats
the position as stalemate: the player who cannot move loses.

## QFEN Notation

QFEN is a compact, FEN-like notation for fixtures, logs, and examples. It has
four slash-separated ranks from top to bottom, with four characters per rank:

```text
<rank0>/<rank1>/<rank2>/<rank3>
```

Characters:

| Character | Meaning                         |
| --------- | ------------------------------- |
| `A`-`D`   | Player 0 shape A-D              |
| `a`-`d`   | Player 1 shape A-D              |
| `.`       | Empty square                    |
| `/`       | Rank separator                  |

Examples:

```text
..../..../..../....   empty board
A.bC/..../d..B/...a   mixed position
AbCd/..../..../....   complete top row
```

## Rust Design

The engine is centered on a compact, portable state representation that supports
constant-time bit operations and stable storage keys.

### Bitboard

`Bitboard` stores eight disjoint 16-bit planes:

```text
[C0S0, C0S1, C0S2, C0S3, C1S0, C1S1, C1S2, C1S3]
```

Each bit index is a board square. The occupied mask is the bitwise OR of all
planes. Shape unions are computed as:

```text
U[shape] = B[player0][shape] | B[player1][shape]
```

This layout keeps the runtime state at 16 bytes and makes legality and win
checks small bitwise operations.

### State And Binary Format

`State` wraps a `Bitboard` for serialization, QFEN conversion, and canonical
keys. The binary format is 18 bytes:

```text
byte 0      version (= 1)
byte 1      flags, with FLAG_CANON = 1 << 1
bytes 2-17  8 little-endian u16 planes
```

`State::pack(flags)` writes this format. `State::unpack(data)` accepts an
18-byte-or-longer buffer and validates the version byte.

`State::canonical_key()` returns the same 18-byte envelope with the canonical
flag set and the canonical 16-byte payload.

### Canonicalization

The Rust canonical key is the lexicographically smallest little-endian payload
across:

- the 8 D4 board symmetries: rotations and reflections
- the 24 shape permutations

That gives 192 candidates per position. A 16-bit permutation lookup table is
built once on first use, using about 1 MB, so geometry transforms become table
lookups.

Player colors are not swapped during Rust canonicalization. This preserves
player identity and side-to-move semantics for engine and book workflows.

### High-Level Board

`QuantikBoard` is the safest API for playing or simulating games. It provides:

- inventory tracking for each player's two copies of each shape
- current-player tracking
- legal move filtering
- `play_move`, `undo_move`, and `undo_moves`
- QFEN conversion
- `GameResult` with win and stalemate handling

### Search And Storage

The crate includes:

- `MCTSEngine` for Monte Carlo Tree Search using UCB1 selection
- `OpeningBookDatabase` for SQLite-backed canonical position storage
- `bench_bfs` for IDDFS-style opening-book generation
- `book_builder` and `validate` binaries for book and data workflows

## Modules

| Module         | Description                                             |
| -------------- | ------------------------------------------------------- |
| `bitboard`     | 128-bit bitboard representation using `[u16; 8]`        |
| `constants`    | Win masks, game limits, version flags                   |
| `moves`        | Move struct, validation, legal move generation          |
| `game`         | Win detection, turn logic, game-over checks             |
| `qfen`         | QFEN encode/decode                                      |
| `state`        | State struct with binary pack/unpack and canonical keys |
| `symmetry`     | D4 + shape permutation canonicalization with LUT        |
| `board`        | High-level `QuantikBoard` with inventory and undo       |
| `mcts`         | Monte Carlo Tree Search engine using UCB1               |
| `opening_book` | SQLite-backed opening book database                     |

## Quick Start

```rust
use quantik_core::board::{GameResult, QuantikBoard};
use quantik_core::moves::Move;
use quantik_core::state::State;

fn main() -> Result<(), String> {
    let mut board = QuantikBoard::new();
    board.play_move(Move::new(0, 0, 0))?;
    board.play_move(Move::new(1, 1, 5))?;

    println!("QFEN: {}", board.to_qfen());
    println!("Current player: {}", board.current_player());
    println!("Result: {:?}", board.game_result());

    let state = State::from_qfen("A.bC/..../d..B/...a")?;
    let key = state.canonical_key();
    println!("Canonical key is {} bytes", key.len());

    assert_eq!(GameResult::Ongoing, board.game_result());
    Ok(())
}
```

## Building

```sh
cargo build --release
```

## Testing

```sh
cargo test
```

## IDDFS Opening Book Builder

The `bench_bfs` binary builds an opening book using hybrid iterative-deepening
DFS (IDDFS) with a persistent SQLite transposition table. The algorithm provides
BFS-like completeness by discovering positions at their shallowest depth without
holding the full frontier in RAM.

### Algorithm

**Phase 1 - Exhaustive (depths 0..N):** For each depth limit from 1 to N, a full
DFS pass is run from the root. An in-memory `HashMap` serves as the transposition
table: a position is skipped if its `searched_depth` already covers the remaining
depth budget. Results are flushed to SQLite in batches.

**Phase 2 - Selective (depths N+1..M):** After the exhaustive phase, expansion
continues with the same IDDFS loop. A future extension can add priority-queue
ordering for high symmetry count, near forced wins, or high uncertainty.

### Usage

```sh
cargo run --release --bin bench_bfs -- [OPTIONS] <DEPTH>
```

| Option                  | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| `<DEPTH>`               | Maximum depth to explore (required)                     |
| `--db <path>`           | SQLite database path (default: `quantik_book.db`)       |
| `--resume`              | Resume from existing database using `searched_depth`    |
| `--max-positions N`     | Stop after N total positions                            |
| `--exhaustive-depth N`  | Depth for exhaustive expansion (default: same as depth) |
| `--batch-size N`        | SQLite transaction batch size (default: 50000)          |
| `--quiet`               | Only print the final summary                            |

### Examples

```sh
# IDDFS to depth 4
cargo run --release --bin bench_bfs -- 4

# IDDFS to depth 6, custom database
cargo run --release --bin bench_bfs -- 6 --db quantik_depth6.db

# Explore up to 500k positions, then resume later
cargo run --release --bin bench_bfs -- 8 --max-positions 500000
cargo run --release --bin bench_bfs -- 8 --resume

# Exhaustive to depth 4, then selective to depth 8
cargo run --release --bin bench_bfs -- 8 --exhaustive-depth 4
```

### Database Schema

The SQLite database uses two tables:

**positions** - one row per canonical position:

| Column           | Type    | Description                                      |
| ---------------- | ------- | ------------------------------------------------ |
| `canonical_key`  | BLOB    | 18-byte canonical key (primary key)              |
| `depth`          | INTEGER | Shallowest depth at which position was found     |
| `is_terminal`    | INTEGER | 1 if game over                                   |
| `winner`         | INTEGER | 0 or 1 if terminal, NULL otherwise               |
| `symmetry_count` | INTEGER | Orbit size under the 192-element symmetry group  |
| `searched_depth` | INTEGER | How deeply this position has been analyzed       |
| `score`          | REAL    | Evaluation score (reserved for future use)       |
| `status`         | INTEGER | 0 = unexplored, 1 = expanded, 2 = dropped        |

**edges** - parent-child move graph. Many parents can reach the same child:

| Column       | Type | Description                   |
| ------------ | ---- | ----------------------------- |
| `parent_key` | BLOB | Parent position canonical key |
| `child_key`  | BLOB | Child position canonical key  |
| `move`       | TEXT | Move string, e.g. `P0S2P5`    |

### Known Position Counts

These counts are for the current Rust canonicalization scope, which preserves
player color and uses the 192-element D4 x shape-permutation symmetry group.

| Depth               | Canonical Positions |
| ------------------- | ------------------- |
| 0                   | 1                   |
| 1                   | 3                   |
| 2                   | 51                  |
| 3                   | 726                 |
| 4                   | 10,958              |
| 5                   | 106,216             |
| 6                   | 919,688             |
| **Total (depth 6)** | **1,037,643**       |
