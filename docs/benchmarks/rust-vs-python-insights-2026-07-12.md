# Cross-engine benchmark insights: Rust vs Python (2026-07-12)

Consolidates two comparison runs — a small `fixed`-family smoke test and a
larger `native`-family production-scale run — into one narrative. Both
compare the ported Rust benchmark harness against the Python original it
was ported from, on identical configurations.

**Not** the canonical full benchmark: see [`docs/BENCHMARKS.md`](../BENCHMARKS.md)
for the authoritative fixed-1s-seeds30 Rust run (240 games/pairing, 30
seeds). The two runs here exist to isolate the *language/runtime*
performance gap specifically, using matched configurations on both sides.

## Headline numbers

| | Smoke (`fixed`, 0.3s/move, 3 seeds) | Production (`native`, 30 seeds) |
|---|---|---|
| Total/agreement wall time | Rust 82.3s vs Python 278.5s (**3.4x**) | Rust 25.98s vs Python 5,688.51s measured engine time (**~219x**) |
| Why the ratio differs | Shared wall-clock budget caps *both* languages' search work, partially masking the gap | Iteration/depth caps hold search *work* constant across languages — this isolates raw interpreter overhead |
| beam engine | 5.6x more nodes explored by Rust in the same budget | **266x** wall-time gap, comparable node counts (same fixed work) |
| mcts engine | 7.5x more nodes explored by Rust in the same budget | **912x** wall-time gap, comparable node counts |
| minimax engine | 8.3x more nodes explored by Rust in the same budget | Only **2.6x** wall-time gap (same 1.0s cap both sides) — Rust instead does **4.9x more nodes** in that shared budget |

The two runs tell a consistent story from two different angles: whenever
both languages share an actual wall-clock budget (minimax in both runs;
every engine in the smoke run), the visible gap compresses because the
budget caps the *work*, and Rust's advantage shows up as *more search per
second* rather than *less wall time*. Whenever the budget is removed and
work is held constant by iteration/depth counts instead (mcts and beam
under `native`), the gap balloons to 266x–912x — that's the actual
CPython-interpreter-overhead-per-node number, undiluted by a shared
clock.

## Run 1: `fixed` family smoke test (small, same-machine, clean)

**Methodology**: identical dataset (sha256-verified byte match), identical
CLI params (`--family fixed --time-limit 0.3 --seeds 3 --h2h-positions 4
--h2h-seeds 2`), both run today on this machine (macOS-aarch64, 8 CPUs).
Workload: 360 agreement observations, 96 h2h games, both sides.

### Wall-clock time

| | Rust | Python | Ratio |
|---|---|---|---|
| Total wall time | 82.3s | 278.5s (after retrying past a known MCTS preflight-determinism flake) | 3.4x |

### Per-engine latency: p50/p95

| Engine | Moves | Rust p50 | Rust p95 | Python p50 | Python p95 | Rust median nodes | Python median nodes |
|---|---|---|---|---|---|---|---|
| beam | 108 | 0.0023s | 0.2135s | 0.3219s | 3.2043s | 528 | 94 |
| mcts | 108 | 0.3000s | 0.3014s | 0.3025s | 0.3078s | 451 | 60 |
| minimax | 36 | 0.0156s | 0.3037s | 0.3234s | 0.7670s | 4397 | 530 |
| random | 108 | 0.0000s | 0.0000s | 0.0001s | 0.0002s | - | - |

mcts is time-bounded by design in both languages (p50 converges to the
0.3s budget either way), but Rust still packs 7.5x more nodes into that
window. beam and minimax finish well inside budget in Rust (low
milliseconds) while Python consistently *consumes the entire budget* and
still explores several times fewer nodes. Python's beam p95 (3.20s) is an
order of magnitude above its own p50 — the time-limit is only checked
between beam levels, so a slow level overshoots noticeably more in Python.

### Win distribution (96 games, n=16/pairing)

```
minimax vs mcts:    rust 62.5%/37.5%   python 87.5%/12.5%
minimax vs beam:    rust 87.5%/12.5%   python 75.0%/25.0%
minimax vs random:  rust 87.5%/12.5%   python 87.5%/12.5%
mcts vs beam:       rust 87.5%/12.5%   python 31.2%/68.8%   <- flips outright
mcts vs random:     rust 87.5%/12.5%   python 87.5%/12.5%
beam vs random:     rust 43.8%/56.2%   python 81.2%/18.8%
```

`mcts vs beam` flips entirely between languages (Rust: mcts 14-2; Python:
beam 11-5), tracking the node-count gap above — "beam" isn't the same
opponent strength across languages even at matched wall-clock settings.
At n=16 this is within reach of sampling noise; flagged as a hypothesis
to check at scale, not a conclusion (see Run 2 below, where the larger
n=160 sample confirms a similar-direction but not identical pattern).

## Run 2: `native` family, seeds=30 (larger, mixed-provenance)

**Methodology and a hardware caveat**: Rust was regenerated fresh, today,
locally (macOS-aarch64, 8 CPUs). Python was **not recomputed** — the
existing checkpoint artifacts already in the Python worktree
(`benchmarks/results/native-seeds30-agreement.ckpt` and
`native-seeds30-h2h16x5.ckpt`) were reused as-is, produced earlier on a
*different machine* (`Linux-6.6.122+-x86_64`, 2 CPUs, Python 3.12.13).
This is **not a same-hardware comparison**. Where possible this is
sidestepped by comparing each engine call's own measured `wall_time_s`
(self-reported, immune to whatever else was happening on either machine)
rather than trusting outer process wall-clock, which for the Python
artifacts spans real-world hours that likely include idle/resume gaps.

Config (verbatim from the Python artifacts' manifests): `family=native
seeds=30`, `minimax(depth=16, time=1.0s)`, `mcts(iterations=1000,
depth=16, c=1.414)`, `beam(width=16, depth=12)`, `random`. Unlike `fixed`,
only minimax has a time cap here — mcts and beam run to a fixed
iteration/depth regardless of wall time, making this run a purer measure
of interpreter overhead per unit of algorithmic work.

### Agreement phase: 3,276 observations

Sum of each engine call's own measured `wall_time_s`:

| | Rust | Python | Ratio |
|---|---|---|---|
| Total measured engine time | 25.98s | 5,688.51s (~94.8 min) | ~219x |

| Engine | n | Rust sum | Python sum | Ratio | Rust p50 | Python p50 | Rust p95 | Python p95 | Rust median nodes | Python median nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| beam | 1080 | 11.48s | 3,050.70s | 266x | 0.0013s | 0.4110s | 0.069s | 16.72s | 159 | 144.5 |
| mcts | 1080 | 2.86s | 2,607.09s | 912x | 0.0010s | 1.3576s | 0.010s | 7.04s | 257.5 | 168 |
| minimax | 36 | 11.64s | 30.51s | 2.6x | 0.0190s | 1.0335s | 1.004s | 1.724s | 4396.5 | 892.5 |
| random | 1080 | 0.002s | 0.20s | ~90x | 0.000002s | 0.000145s | 0.000002s | 0.000380s | - | - |

Since mcts/beam do a fixed, language-independent amount of work here
(iteration/depth capped, not time-capped), node counts come out
comparable between languages (Rust explores 1.1-1.5x more, from its own
faster inner loop resolving more early-exit/pruning opportunities — not
from a bigger budget). The 266x-912x wall-time gap is therefore almost
entirely CPython interpreter overhead per unit of search, not a
difference in how much searching happened — a cleaner signal than the
`fixed`-family run, where the shared time budget partially masked the gap
by capping both languages' work, not just their time.

Minimax, bounded by the same 1.0s wall-clock cap in both languages, shows
the smallest gap (2.6x) and does it by using that shared budget for 4.9x
more nodes — the same "same time, more work done" story as Run 1.

### Head-to-head phase: 960 games

**Rust**: clean single run, 234.37s total process time, ~26s of which is
the re-run agreement phase, leaving **~208s (~3.5 min) for 960 games**
(≈0.22s/game).

**Python**: `h2h.jsonl` carries no per-game timing field in either
language, so no clean way exists to isolate h2h-only time from the
existing artifact without recomputing (which we were instructed not to
do). The only available number is the checkpoint manifest's outer span
(started_at to updated_at): **6h 19m 32s** — flagged as an **unreliable
upper bound**, not a clean measurement, since that checkpoint's
`observations.jsonl` is byte-identical to the separate agreement-only
checkpoint's, suggesting it was merged in (the worktree's
`benchmarks/README.md` documents exactly this workflow) rather than
independently recomputed inside that span. **No reliable h2h-only ratio
can be quoted.**

### Win distribution (960 games, n=160/pairing)

```
beam vs mcts:        rust 16.2%/83.8%   python 51.2%/48.8%
beam vs minimax:      rust 16.2%/83.8%   python 35.0%/65.0%
beam vs random:       rust 49.4%/50.6%   python 82.5%/17.5%
mcts vs minimax:      rust 40.6%/59.4%   python 34.4%/65.6%
mcts vs random:       rust 84.4%/15.6%   python 80.0%/20.0%
minimax vs random:    rust 84.4%/15.6%   python 83.8%/16.2%
```

Deltas are large enough at n=160/pairing to be more than noise for
several pairings (e.g. beam vs random: 49.4% vs 82.5%), reinforcing the
Run 1 finding: "beam" is not a fixed opponent strength across languages
even under matched configuration — it's determined by how much a
language's runtime lets it actually search within the given time/iteration
budget.

**A flagged anomaly, not a conclusion**: in the Rust results only, `beam
vs mcts` (26-134) and `beam vs minimax` (26-134) landed on the exact same
win/loss split, and so did `mcts vs random` (135-25) and `minimax vs
random` (135-25) — verified two independent ways from the raw rows, not
a reporting bug. Not present in the Python data. A plausible, *unverified*
explanation: on these mostly-deep positions, both near-optimal native
engines may converge to the same effectively-optimal move against a weak
opponent, making the winner determined by who objectively has the winning
position rather than by which strong engine is playing it. Not confirmed
by move-by-move inspection — a lead, not a finding.

## Synthesis

1. **The core engineering result is robust and repeats across both runs
   at two very different scales**: whenever both languages share a
   wall-clock budget, Rust converts that same time into several times
   more search (5-8x nodes in Run 1, 4.9x in Run 2's minimax). Whenever
   the budget is removed and iteration/depth counts hold work constant,
   the *time* gap is enormous instead (266x-912x in Run 2).
2. **h2h game timing is not currently measurable in Python** (no
   per-game field in `h2h.jsonl`, and no clean historical baseline
   exists) — a real instrumentation gap. Closing it would need either a
   schema addition (per-game wall time) or a fresh, deliberately-timed
   Python run.
3. **Engine-strength win-rate comparisons across languages are
   confounded by search-depth differences**, not just RNG divergence —
   "beam" and even, at n=160, "mcts vs minimax" shift meaningfully
   between languages under nominally identical configuration. Treat
   cross-language win-rate deltas as evidence about *runtime
   capability*, not about the *algorithms* being compared.
4. **Same-hardware Python data would strengthen Run 2** — its Python
   side came from a different (2-CPU Linux cloud) machine. The
   `wall_time_s`-sum methodology mostly compensates by measuring each
   engine call's own work rather than trusting outer process time, but a
   true same-machine, same-day run would remove the last caveat.

## Reproduction

```bash
# Smoke run (fixed family)
cd quantik-core-rust && cargo build --release --bin cross_engine_benchmark
./target/release/cross_engine_benchmark run \
  --dataset benchmarks/positions-v1.json \
  --family fixed --time-limit 0.3 --seeds 3 --h2h-positions 4 --h2h-seeds 2 \
  --output /tmp/rust-smoke.json

cd quantik-core-py/.claude/worktrees/cross-engine-benchmark-24
.venv/bin/python examples/cross_engine_benchmark.py run \
  --dataset benchmarks/positions-v1.json \
  --family fixed --time-limit 0.3 --seeds 3 --h2h-positions 4 --h2h-seeds 2 \
  --output /tmp/python-smoke.json
# Retry if preflight reports an MCTS non-determinism message — documented flake, not a data-integrity issue.

# Production-scale run (native family, seeds=30) — Rust side
cd quantik-core-rust
./target/release/cross_engine_benchmark run \
  --dataset benchmarks/positions-v1.json \
  --family native --seeds 30 --h2h-positions 16 --h2h-seeds 5 \
  --minimax-depth 16 --minimax-time 1.0 \
  --mcts-iterations 1000 --mcts-depth 16 --mcts-exploration 1.414 \
  --beam-width 16 --beam-depth 12 \
  --checkpoint-dir /tmp/rust-native-h2h16x5.ckpt \
  --output /tmp/rust-native-h2h16x5.json
# Python side for this config already exists at
# quantik-core-py/.claude/worktrees/cross-engine-benchmark-24/benchmarks/results/native-seeds30-h2h16x5.ckpt
```

## Source data

- Smoke run: regenerated ad hoc for this comparison, not committed.
- Production run, Rust: regenerated ad hoc for this comparison, not
  committed.
- Production run, Python: reused, not regenerated —
  `quantik-core-py/.claude/worktrees/cross-engine-benchmark-24/benchmarks/results/native-seeds30-agreement.ckpt`
  and `.../native-seeds30-h2h16x5.ckpt`.
