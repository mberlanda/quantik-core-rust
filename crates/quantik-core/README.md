# quantik-core

High-performance Rust engine for the [Quantik](https://github.com/mberlanda/quantik-core-rust) board game: bitboard state, QFEN notation, canonical symmetry-reduced keys, and minimax/MCTS/beam-search engines.

Rust companion to the [Python `quantik-core` package](https://pypi.org/project/quantik-core/) — same core model (a tiny bitboard state, QFEN for human-readable positions, canonical binary keys for search caches and databases), byte-compatible canonical keys across both languages. See the [repository README](https://github.com/mberlanda/quantik-core-rust#readme) for guidance on choosing between the two.

See the [full repository documentation](https://github.com/mberlanda/quantik-core-rust#readme) for everything else, and [`docs/BENCHMARKS.md`](https://github.com/mberlanda/quantik-core-rust/blob/main/docs/BENCHMARKS.md) for cross-engine performance data.

## License

MIT
