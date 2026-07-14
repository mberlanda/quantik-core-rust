//! Cross-engine benchmark harness (port of the Python `benchmarks/` package).
//!
//! Compares the minimax, MCTS, beam-search, and uniform-random engines on a
//! shared, versioned, checksummed position dataset under methodologically
//! consistent conditions. See `docs/BENCHMARKS.md`.

pub mod adapters;
pub mod agreement;
pub mod book_export;
pub mod bundle;
pub mod canonical;
pub mod checkpoint;
pub mod contracts;
pub mod correctness;
pub mod dataset;
pub mod head_to_head;
pub mod metrics;
pub mod reference;
pub mod report;
pub mod stability;
