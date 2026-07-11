# Parametrizable Beam Search for Quantik MCTS — Design

**Date:** 2026-07-07
**Status:** Approved (autonomous orchestration per user /goal directive)
**Branch:** `feat/mcts-beam-search`

## Problem

The existing `MCTSEngine` (UCT) samples the game tree stochastically and its
random playouts stop at `max_depth` without guaranteeing terminal resolution.
Exhaustive expansion is infeasible: Quantik has ~2.2 × 10^12 legal move
sequences (23.6M unique canonical states through depth 8 alone, see
`GAME_TREE_ANALYSIS.md`). We need a search mode that:

1. **Reaches end leaves** — descends to true terminal states (win / loss by
   blocked player), up to the full game length of 16 plies.
2. **Bounds memory** — keeps a memory-efficient representation of the game,
   reusing the 18-byte `State.pack()` encoding and the 64-byte
   `CompactGameTree` nodes already used by MCTS.
3. **Is parametrizable** — beam width, depth, evaluation policy, rollout
   budget, seed are all configuration.

## Approaches considered

- **A. Standalone beam engine sharing `CompactGameTree` (chosen).** A
  level-by-level frontier search: expand all children of the current frontier,
  score them, keep the top `beam_width`, insert only survivors into the
  compact tree. Memory is O(beam_width × depth) tree nodes. Composes with
  MCTS by accepting an existing `CompactGameTree` (e.g. `mcts_engine.tree`)
  so beam results (terminal flags, values) enrich the same transposition
  structure.
- **B. Beam-limited MCTS (progressive pruning inside `MCTSEngine`).**
  Rejected: changes the semantics of the existing, tested engine; UCT and
  beam pruning interact badly at low iteration counts; harder to reason about.
- **C. Post-hoc beam over MCTS statistics.** Rejected: needs MCTS to have
  visited deep nodes already, so it cannot *guarantee* reaching terminal
  leaves — the original goal.

## Design

New module `src/quantik_core/beam_search.py` (public via
`quantik_core.beam_search`, same convention as `quantik_core.mcts` which is
not re-exported at package top level).

### Configuration

```python
@dataclass
class BeamSearchConfig:
    beam_width: int = 64            # frontier nodes kept per depth (>= 1)
    max_depth: int = 16             # plies from root; 16 = full Quantik game
    rollouts_per_candidate: int = 8 # rollout budget for the default evaluator (>= 1)
    random_seed: Optional[int] = None
    evaluator: Optional[Callable[[State], float]] = None
    initial_tree_capacity: int = 4096
```

- `evaluator(state) -> float` returns a value in [-1, 1] **from player 0's
  perspective** (same convention as `MCTSEngine._simulate` /
  `terminal_value`). When `None`, the engine uses its built-in random-rollout
  evaluator: mean of `rollouts_per_candidate` uniform playouts, each rolled
  out to true terminal (a playout never needs more than 16 plies, so no
  depth cutoff heuristic is required).
- Config values are validated in `BeamSearchEngine.__init__`
  (`beam_width >= 1`, `1 <= max_depth <= 16`, `rollouts_per_candidate >= 1`);
  invalid values raise `ValueError`.
- `random_seed` seeds a **private** `random.Random` instance (unlike
  `MCTSConfig`, which seeds the global RNG — do not copy that behaviour;
  a private RNG keeps the engine reproducible and side-effect free).

### Engine and algorithm

```python
class BeamSearchEngine:
    def __init__(self, config: BeamSearchConfig,
                 tree: Optional[CompactGameTree] = None): ...
    def search(self, initial_state: State) -> BeamSearchResult: ...
    def get_statistics(self) -> dict: ...
```

Frontier entry: `(node_id, bitboard, moves)` where `moves` is the tuple of
`Move`s from the root (≤ 16 small dataclasses — this is what makes principal
variations reconstructible without storing moves in the 64-byte node, which
has no edge-move field).

Per depth level `d` (players strictly alternate in Quantik; every node at a
given depth has the same side to move):

1. **Expand:** for each frontier entry, generate legal moves
   (`generate_legal_moves`), apply each, and classify the child with
   `check_game_winner`. A frontier state with *no* legal moves is terminal —
   the player to move loses (same rule as `MCTSEngine._expand`).
2. **Terminal handling:** terminal children are recorded into the tree with
   `NODE_FLAG_TERMINAL` + `NODE_FLAG_WINNING_P0/P1` and `terminal_value`
   (±1.0), collected as result leaves, and **not** carried into the next
   frontier.
3. **Deduplicate:** non-terminal candidates are deduplicated per depth by
   `State.canonical_key()` (symmetry-aware — reduction factors of 21×–10⁵×
   per `GAME_TREE_ANALYSIS.md`; first path encountered wins).
4. **Score:** each surviving candidate is evaluated with the evaluator and
   ranked **from the perspective of the player who just moved** (score =
   value for P0-mover, −value for P1-mover). This gives beam search an
   adversarially sensible frontier at every level instead of optimizing one
   fixed player throughout.
5. **Prune:** keep the top `beam_width` candidates (stable ordering for ties,
   so seeded runs are deterministic). **Only survivors are inserted** into
   the `CompactGameTree` (via `add_child_node`, which also merges
   transpositions); pruned candidates never allocate a node. Survivors'
   `best_value`/`visit_count` are updated with their evaluation so the shared
   tree is useful to MCTS afterwards.
6. Stop when the frontier is empty (all lines resolved to terminal) or
   `max_depth` is reached.

### Result

```python
@dataclass
class BeamLeaf:
    moves: Tuple[Move, ...]   # principal variation from the root
    value: float              # P0 perspective; ±1.0 for terminal leaves
    depth: int
    is_terminal: bool

@dataclass
class BeamSearchResult:
    best_leaf: Optional[BeamLeaf]      # best for the ROOT player to move
    terminal_leaves: List[BeamLeaf]    # all terminals discovered, best first
    reached_terminal: bool
    max_depth_reached: int
    stats: Dict[str, int]              # candidates_generated, candidates_deduped,
                                       # nodes_inserted, nodes_pruned,
                                       # evaluations, memory_usage
```

`best_leaf` ranks all collected leaves (terminals plus, if the search hit
`max_depth` with a live frontier, the final frontier entries) by value from
the **root player's** perspective. `search` raises `ValueError` if the root
state is already terminal.

### Memory model

- ≤ `beam_width` non-terminal nodes inserted per depth + terminal leaves
  found among their children ⇒ tree nodes grow O(beam_width × depth), each
  64 bytes, versus the unbounded growth of exhaustive expansion.
- Frontier bookkeeping is O(beam_width) packed states + move tuples.
- `get_statistics()` mirrors `MCTSEngine.get_statistics()` (delegates to
  `tree.memory_usage()` / `tree.get_stats()`).

### Error handling

- Config validation as above (`ValueError`).
- Root already terminal → `ValueError` with clear message.
- Evaluator returning values outside [-1, 1] is clamped (defensive; documented).

### Testing (tests/test_beam_search.py)

Follow `tests/test_mcts.py` structure (class-per-concern, seeded configs).
Project gate: full suite ≥ 90% coverage (`./dev-check.sh`), mypy + flake8 clean.

1. Config defaults, custom values, validation errors.
2. Immediate win found: near-win fixture (reuse QFEN patterns from
   `test_mcts.py`) → `best_leaf` is the winning terminal at depth 1 with the
   correct move.
3. Full-game reachability: from the empty board with small beam
   (e.g. width 4, seeded), `reached_terminal` is True and every terminal
   leaf's PV replays legally (`apply_move` chain) to a state where
   `check_game_winner` ≠ NO_WIN or the mover is blocked.
4. Symmetry dedup: from the empty board, depth-1 candidates collapse from
   64 moves to 3 canonical states (per `GAME_TREE_ANALYSIS.md`).
5. Memory bound: nodes_inserted ≤ beam_width × depth + terminal leaf count;
   compare stats for beam_width 2 vs 16.
6. Determinism: same seed ⇒ identical result; different seed may differ.
7. Pluggable evaluator: custom callable is used (call count > 0) and biases
   the beam as expected; out-of-range values clamped.
8. Adversarial perspective: a position where the side to move at depth 1 is
   P1 and P1 has a winning reply — the beam must keep it (score is
   mover-relative, not P0-fixed).
9. Shared tree integration: pass `MCTSEngine(config).tree` (or a fresh
   `CompactGameTree`), run beam search, verify terminal flags/values were
   written into it and node count grew within the bound.
10. Root terminal / no-legal-moves root → `ValueError`.

### Out of scope (YAGNI)

- No changes to `MCTSEngine` or `CompactGameTree`.
- No top-level `__init__.py` re-export (matches `mcts` precedent).
- No parallel evaluation, no persistence of beam results.
