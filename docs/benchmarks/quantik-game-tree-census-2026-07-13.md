# Quantik's canonical game tree: a depth-by-depth census

Extends the depth-4 symmetry survey (see
[`depth4-canonical-solve-cost-2026-07-13.md`](depth4-canonical-solve-cost-2026-07-13.md))
to every depth of the game: how many canonical positions exist at each
ply, how the game actually ends (line completion vs. running out of legal
moves), and whether a fully-packed 16-piece board is reachable at all.

## Methodology

Canonical-key BFS, one ply at a time, starting from the empty board:

1. At each ply, take every canonical representative currently in the
   frontier and generate its legal children.
2. Any child with a completed line (`has_winning_line`) is a decisive
   win and is removed from play — it does not get expanded further.
3. Every surviving child is folded onto its canonical representative
   (`SymmetryHandler::find_canonical`, the existing 192-element group:
   8 board symmetries x 24 shape relabelings). Multiplicity — how many
   raw, non-canonical move sequences collapse onto that representative —
   is accumulated by summing the multiplicities of every parent that
   produced it, the same path-count technique `docs/BEAM_SEARCH.md`
   documents for beam search.
4. The deduplicated result becomes next ply's frontier.

The critical property that makes this tractable at all: folding onto
canonical representatives *at every level*, not just at the end, means
the work done at each step is bounded by the *canonical* state count of
the previous ply, not the *raw* one. A brute-force approach that
enumerates raw sequences before reducing hits combinatorial explosion
almost immediately (see validation below); this doesn't, because it
never materializes the raw sequences at all.

## Validation against independently-computed numbers

Two independent checks, both exact matches:

**Against this repo's own standalone depth-4 survey**
(`examples/depth4_orbit_histogram.rs`, now folded into this sweep):
10,946 canonical nonterminal states, same orbit-size histogram
(8:2, 16:15, 24:8, 32:80, 48:208, 96:3771, 192:6862).

**Against the Python worktree's `SYMMETRY_REDUCTION_DEMONSTRATION.md`**,
which independently computed depths 1-5 via full raw enumeration (no
canonical folding until the end) — a completely different algorithm,
implementation, and language:

| Depth | Python: unique canonical | This sweep: canonical states | Match |
|---|---|---|---|
| 1 | 3 | 3 | Yes |
| 2 | 51 | 51 | Yes |
| 3 | 726 | 726 | Yes |
| 4 | 10,946 | 10,946 | Yes |
| 5 | 105,632 | 105,632 | Yes |

Win-attribution counts match too — e.g. depth 5: both compute exactly
1,050,624 raw P0-win transitions. This is about as strong a
cross-validation as two independently-written implementations can give
each other.

Notably, Python's table **stops at depth 5** — its methodology (reduce
after fully enumerating) needs to materialize 231,883,776 raw
transitions to get there. This sweep's canonical-fold-at-every-level
approach reached depth 5 in 2.9 seconds.

## The census: depths 1-8

| Depth | Canonical states | Raw boards (ongoing) | P0 wins (raw) | P1 wins (raw) | States w/ legal moves | Mean orbit size | Elapsed |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 64 | 0 | 0 | 3 | 21.3 | 0.02s |
| 2 | 51 | 3,392 | 0 | 0 | 51 | 66.5 | 0.001s |
| 3 | 726 | 167,552 | 0 | 0 | 726 | 115.4 | 0.02s |
| 4 | 10,946 | 6,770,048 | 0 | 6,912 | 10,946 | 154.6 | 0.27s |
| 5 | 105,632 | 230,833,152 | 1,050,624 | 0 | 105,632 | 182.1 | 2.9s |
| 6 | 901,916 | 6,159,946,752 | 0 | 81,653,760 | 901,916 | 189.7 | 26.0s |
| 7 | 4,658,465 | 128,513,710,080 | 3,886,838,784 | 0 | 4,658,465 | 191.6 | 140.1s |
| 8 | 17,900,160 | 1,978,186,364,928 | 0 | 118,862,401,536 | **17,894,928** | 191.9 | 570.1s |

A few things worth reading closely:

- **Wins alternate strictly between P0 and P1 by depth parity.** At even
  depths (4, 6, 8) only P1 wins are possible; at odd depths (5, 7) only
  P0 wins. This falls straight out of `check_winner`'s tiebreak (whoever
  has *more* pieces on the board is credited the win, and that's always
  whoever just moved) combined with strict alternation starting with P0
  — the mover at ply *k* is P0 if *k* is odd, P1 if *k* is even.
- **Mean orbit size keeps climbing toward the 192 ceiling** (21.3 to
  191.9 across depths 1-8) — as more pieces go down, positions have
  progressively less accidental symmetry, so canonicalization's payoff
  (raw-to-canonical ratio) actually *grows* with depth, not shrinks.
- **Depth 8 is the first depth where `canonical_states` and `states with
  legal moves` diverge**: 17,900,160 vs. 17,894,928 — a gap of 5,232.
  Those are canonical positions where the side to move has zero legal
  placements without ever having completed a line: a genuine mid-game
  loss-by-exhaustion, confirmed to first appear at exactly depth 8, with
  only half the board (8 of 16 cells) filled.

## Why the sweep stops at depth 8

Growth is steep and, through depth 8, still accelerating in absolute
terms even as its *rate* slows: 5.16x (6→7) then 3.84x (7→8) in
canonical-state count, with elapsed time growing by similar factors
(5.7x, then 4.1x). Depth 8 alone took 570 seconds and peaked at 1.85GB
resident memory. A follow-up attempt at depth 9, given a 40-minute
budget, did not complete — it was still running when the cap was hit,
having already exceeded the 1,660 seconds left after depth 8. **Full
exhaustive enumeration through depth 16 is not practical with this
approach**, for the same underlying reason the depth-4 exact-solve
survey needed to be sampled rather than exhaustive: canonical folding
tames the *raw* combinatorial explosion, but the *canonical* state count
itself is still large enough, deep enough into the game, to make full
enumeration expensive.

## A sharper question, answered directly instead: can the board ever fill completely?

Rather than pushing the exhaustive sweep further to settle this, a much
cheaper targeted approach answers it directly: search for a *single*
legal 16-move sequence that never completes a line, using ordinary
backtracking DFS with the real legal-move generator (so every
intermediate state already respects turn alternation, the per-shape
piece cap, and the opponent-shape blocking rule) instead of enumerating
every possibility.

**First, a pure combinatorial check** (ignoring legality/reachability
entirely): of all 63,063,000 ways to place exactly 4 of each of the 4
shapes onto the 16 cells (forced, once the board is full, since
`MAX_PIECES_PER_SHAPE=2` per player x 2 players = 4 per shape, and
4 shapes x 4 = 16 = board size), **11,449,080 (18.2%) have zero
complete lines.** So it's not combinatorially impossible in the
abstract — the question genuinely depends on reachability under real
play, not just counting.

**Then, the constructive search**: randomized backtracking DFS from the
empty board, retried with different random orderings on failure. It
found a witness on the **3rd attempt** (~28s, 169M nodes): a fully
verified, legal 16-move sequence — turns alternate correctly, each
player ends with exactly 2 of each shape, all 16 cells are used exactly
once, and no line is ever completed along the way.

The witness sequence (`player:shape:position`, ply order):

```
0:2:12  1:3:7   0:0:6   1:0:0   0:1:3   1:3:15  0:3:9   1:0:8
0:2:13  1:2:10  0:3:1   1:2:11  0:1:2   1:1:4   0:0:14  1:1:5
```

Verified: turns strictly alternate 0/1; each player places exactly 2 of
each of the 4 shapes; all 16 board positions (0-15) are used exactly
once; `has_winning_line` is false after every one of the 16 moves.

**So yes — a completely full board with no winning line is reachable.**
But this does not produce a draw. After the 16th piece, it's P0's turn
again (ply 17): the board has zero empty cells, so `generate_legal_moves`
returns empty — not from being blocked, simply because there's nowhere
left to place anything. The same "no legal moves ⇒ mover loses" rule
that handles mid-game exhaustion (confirmed first occurring at depth 8,
above) applies here without any special-casing: `board.rs::game_result`
falls through `check_winner`'s `NoWin` into the `legal_moves().is_empty()`
branch and correctly resolves to **Player1Wins** — the player who placed
the 16th and final piece is credited the win, purely as a side effect of
the same rule, with zero extra code needed for "board full" as a
distinct case.

## How the "no draws" rule is actually implemented

Confirmed by direct inspection, not assumption: `game.rs::check_winner`
deliberately handles *only* the completed-line condition. Every consumer
that needs full termination semantics — `minimax.rs::negamax`,
`mcts.rs`'s node expansion and rollout, `board.rs::game_result`,
`beam_search.rs::rollout` — separately checks `generate_legal_moves(bb)
.is_empty()` and attributes the loss to whichever player is stuck. This
is a consistent, deliberate layering across every engine in the crate,
not an accident of one code path: draws are structurally impossible
because "ran out of moves" is defined as a loss everywhere the rule is
checked, whether that happens from being blocked mid-game or from the
board simply filling up.

## Connecting back to the depth-4 solve-cost investigation

This census puts the earlier ~12.7-day depth-4 exact-solve projection in
context: depth 4 (10,946 canonical states) is the *smallest* depth
beyond the trivial opening moves. Depth 5 already has 105,632 — roughly
10x more — and depth 6 nearly 100x more. Exhaustively exact-solving any
depth beyond 4 with the current unshared-transposition architecture
would be correspondingly more expensive, not less, reinforcing that a
sampled-survey approach (or the shared-transposition-cache investment
discussed in the depth-4 doc) is the only practical path to book-building
beyond the shallowest depths.

## Reproduction

```bash
cd quantik-core-rust
cargo build --release --example depth_sweep
./target/release/examples/depth_sweep   # prints CSV to stdout, per-depth orbit histograms to stderr
```

The full-board reachability search and the pure combinatorial check were
one-off diagnostic scripts, removed after their findings were folded
into this document; the DFS witness sequence is recorded above in full
and can be replayed move-by-move against `board.rs` to re-verify.
