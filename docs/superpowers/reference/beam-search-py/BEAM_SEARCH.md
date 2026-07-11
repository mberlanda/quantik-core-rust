# Beam Search Implementation

This document describes the parametrizable beam search engine in Quantik Core: a memory-bounded search mode that guarantees reaching true terminal game states, complementing the statistical sampling of MCTS.

## Overview

`MCTSEngine` samples the game tree stochastically and its random playouts stop at `max_depth` without guaranteeing terminal resolution. Exhaustive expansion of the full game tree is infeasible (Quantik has ~2.2 × 10^12 legal move sequences). `BeamSearchEngine` fills the gap: a level-by-level frontier search that always descends to true terminal states (win, loss by blocked player) while bounding memory to a configurable width.

### Algorithm Phases

Per depth level (players strictly alternate, so every node at a given depth shares the same side to move):

1. **Expand**: generate legal moves for each frontier entry and apply them. A frontier state with no legal moves is itself terminal (the player to move loses).
2. **Classify**: any child that completes a winning line is recorded immediately as a terminal leaf and inserted into the tree — regardless of the beam width.
3. **Deduplicate**: remaining non-terminal candidates are deduplicated per depth by `State.canonical_key()` (symmetry-aware; first path encountered wins).
4. **Score**: each surviving candidate is evaluated and ranked **mover-relative** — from the perspective of the player who just moved (`score = value` for a P0 move, `-value` for a P1 move) — so the beam stays adversarially sensible at every level instead of optimizing one fixed player throughout.
5. **Prune**: only the top `beam_width` candidates survive (stable ordering, so seeded runs are deterministic); pruned candidates never allocate a tree node.
6. **Insert**: survivors are added to the shared `CompactGameTree` via `add_child_node` and become the next depth's frontier.

Search stops when the frontier empties (every line resolved to a terminal) or `max_depth` is reached.

## Quick Start

```python
from quantik_core import State
from quantik_core.beam_search import BeamSearchEngine, BeamSearchConfig

config = BeamSearchConfig(beam_width=8, max_depth=16, random_seed=42)
engine = BeamSearchEngine(config)

state = State.from_qfen("..../..../..../....")
result = engine.search(state)

print(f"Reached terminal: {result.reached_terminal}")
print(f"Best line: {result.best_leaf.moves}")
print(f"Value (P0 perspective): {result.best_leaf.value}")
```

## Configuration

### BeamSearchConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beam_width` | int | 64 | Frontier nodes kept per depth (>= 1) |
| `max_depth` | int | 16 | Plies from root; 16 = full Quantik game (1-16) |
| `rollouts_per_candidate` | int | 8 | Rollout budget for the built-in evaluator (>= 1) |
| `random_seed` | int\|None | None | Seeds a **private** `random.Random` — the global RNG is never touched |
| `evaluator` | `Callable[[State], float]`\|None | None | Custom evaluator; falls back to random-rollout scoring when omitted |
| `initial_tree_capacity` | int | 4096 | Initial `CompactGameTree` node capacity |

Invalid values (`beam_width < 1`, `max_depth` outside `1..16`, `rollouts_per_candidate < 1`) raise `ValueError` from `BeamSearchEngine.__init__`.

### Evaluator Contract

`evaluator(state) -> float` returns a value in `[-1, 1]` from **player 0's perspective**, matching `MCTSEngine`'s convention. Values outside that range are clamped. When omitted, the engine uses the mean of `rollouts_per_candidate` uniform random playouts, each rolled out to a true terminal — a Quantik playout never exceeds 16 plies, so no depth-cutoff heuristic is needed.

## Result

```python
@dataclass
class BeamLeaf:
    moves: Tuple[Move, ...]   # principal variation from the root
    value: float              # P0 perspective; +/-1.0 for terminal leaves
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

`best_leaf` ranks every collected leaf — terminals plus, if the search hit `max_depth` with a live frontier, the final frontier entries — from the **root player's** fixed perspective (unlike the mover-relative pruning score used level by level). `search()` raises `ValueError` if the root state is already terminal or has no legal moves.

## Memory Model

Only surviving candidates allocate a tree node, so growth is bounded by `beam_width x depth` non-terminal nodes plus however many terminal leaves were discovered along the way — versus the unbounded growth of exhaustive expansion. Each node is the same 64-byte `CompactGameTreeNode` used by MCTS. `get_statistics()` mirrors `MCTSEngine.get_statistics()`, delegating to `tree.memory_usage()` and `tree.get_stats()`.

## Sharing a Tree with MCTS

`BeamSearchEngine` accepts an existing `CompactGameTree` (e.g. `mcts_engine.tree`) so both engines can enrich the same transposition structure:

```python
from quantik_core.mcts import MCTSEngine, MCTSConfig
from quantik_core.beam_search import BeamSearchEngine, BeamSearchConfig

mcts_engine = MCTSEngine(MCTSConfig())
beam_engine = BeamSearchEngine(BeamSearchConfig(beam_width=8), tree=mcts_engine.tree)
```

**Caveat**: `CompactGameTree.create_root_node` hardcodes the root's `player_turn` to 0 and alternates from there. When sharing a tree, root the beam search at a position where **player 0 is to move** — otherwise every node's `player_turn` is inverted, which would corrupt MCTS's UCB calculation if it later resumes on the same tree.

Also note that `CompactGameTree`'s own transposition key is the literal `State.pack()` bytes (its `canonical_state_data` field is *not* symmetry-reduced), while beam search's own deduplication (step 3 above) uses the coarser `State.canonical_key()`. The shared tree therefore isn't itself symmetry-reduced — beam search just feeds it canonically-distinct representatives per depth.

## Comparison with MCTS

| Aspect | Beam Search | MCTS |
|--------|-------------|------|
| **Terminal guarantee** | Always reaches true terminals (within `max_depth`) | Not guaranteed; playouts stop at `max_depth` |
| **Exploration** | Deterministic frontier expansion, mover-relative pruning | Stochastic UCB1 sampling |
| **Memory** | O(beam_width x depth) | Grows with iteration count |
| **Best for** | Exhaustive-ish tactical verification under a memory budget | General-purpose anytime search |

## Examples

See `examples/beam_search_demo.py` for complete working examples:

- Full-depth search from the empty board reaching a true terminal
- Tactical (immediate win) position analysis
- Beam width sweep demonstrating the memory bound
- Pluggable custom evaluator
