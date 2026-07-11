//! Fitted-linear handcrafted evaluation for non-terminal Quantik positions.
//!
//! Scores a position as the dot product of a small, hand-designed feature
//! vector and a fitted weight vector (`EvalConfig::weights`). `features` is
//! a pure function of `(bb, player)`. Port of the Python
//! `quantik_core.evaluation` module; feature semantics must stay identical
//! so fitted weights remain interchangeable between implementations.

use crate::bitboard::Bitboard;
use crate::constants::WIN_MASKS;
use crate::game::current_player;
use crate::moves::generate_legal_moves;

/// Feature vector layout produced by [`features`], in order.
pub const FEATURE_NAMES: [&str; 6] = [
    "threat_own",
    "threat_opp",
    "threat_shared",
    "mobility_diff",
    "build_two",
    "build_one",
];

const SEEDED_WEIGHTS: [f64; 6] = [100.0, -100.0, 20.0, 3.0, 2.0, 0.0];

/// Weights for the fitted-linear evaluation and the terminal win bonus.
///
/// `weights` follows [`FEATURE_NAMES`] order. `win` is not consumed by
/// [`evaluate`] (which only scores non-terminal positions) — it is used by
/// search engines built on top of this module to score forced wins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalConfig {
    pub weights: [f64; 6],
    pub win: f64,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            weights: SEEDED_WEIGHTS,
            win: 10_000.0,
        }
    }
}

/// Whether `player` may place `shape` at the empty `position`.
///
/// Mirrors the win-line rule (a shape cannot be placed on a line where the
/// opponent already holds the same shape) without the occupancy check,
/// since callers only ask this for cells already known to be empty.
fn placement_is_legal(bb: &Bitboard, player: u8, shape: u8, position: u8) -> bool {
    let opponent_shape_bits = bb.planes[Bitboard::plane_index(1 - player, shape)];
    let position_mask = 1u16 << position;
    for &mask in &WIN_MASKS {
        if (position_mask & mask != 0) && (opponent_shape_bits & mask != 0) {
            return false;
        }
    }
    true
}

/// Count `player`'s legal moves, 0 if it is not `player`'s turn.
///
/// Quantik is strictly turn-alternating, so this is 0 whenever `player` is
/// not the side to move.
pub fn count_legal_moves(bb: &Bitboard, player: u8) -> usize {
    if current_player(bb) != Some(player) {
        return 0;
    }
    generate_legal_moves(bb).len()
}

/// Compute the 6-dimensional handcrafted feature vector for `bb`.
///
/// Features are from `player`'s perspective (see [`FEATURE_NAMES`]), but
/// `player` need not be the side to move: it is simply the perspective the
/// caller wants scored. `bb` should be non-terminal.
pub fn features(bb: &Bitboard, player: u8) -> [f64; 6] {
    let counts: [[u32; 4]; 2] =
        std::array::from_fn(|p| std::array::from_fn(|s| bb.shape_piece_count(p as u8, s as u8)));
    let side_to_move = current_player(bb).unwrap_or(0);
    let sign = if side_to_move == player { 1.0 } else { -1.0 };

    let union_all = bb.occupied();
    let shape_unions: [u16; 4] = std::array::from_fn(|s| bb.planes[s] | bb.planes[s + 4]);

    let mut threat_own = 0.0;
    let mut threat_opp = 0.0;
    let mut threat_shared = 0.0;
    let mut build_two = 0.0;
    let mut build_one = 0.0;

    for &mask in &WIN_MASKS {
        let present: Vec<u8> = (0..4u8)
            .filter(|&s| shape_unions[s as usize] & mask != 0)
            .collect();
        let occupied = (union_all & mask).count_ones();

        if (present.len() as u32) < occupied {
            continue; // dead line: some shape repeats, can never be 4-distinct
        }

        if occupied == 3 {
            let missing_shape = (0..4u8).find(|s| !present.contains(s)).unwrap();
            let empty_position = (mask & !union_all).trailing_zeros() as u8;
            let completable: [bool; 2] = std::array::from_fn(|side| {
                counts[side][missing_shape as usize] < 2
                    && placement_is_legal(bb, side as u8, missing_shape, empty_position)
            });
            if completable[player as usize] {
                threat_own += 1.0;
            }
            if completable[1 - player as usize] {
                threat_opp += 1.0;
            }
            if completable[0] && completable[1] {
                threat_shared += sign;
            }
        } else if occupied == 2 {
            build_two += sign;
        } else if occupied == 1 {
            build_one += sign;
        }
    }

    let mobility_diff =
        count_legal_moves(bb, player) as f64 - count_legal_moves(bb, 1 - player) as f64;

    [
        threat_own,
        threat_opp,
        threat_shared,
        mobility_diff,
        build_two,
        build_one,
    ]
}

/// Score a non-terminal position as `cfg.weights · features(bb, player)`.
pub fn evaluate(bb: &Bitboard, player: u8, cfg: &EvalConfig) -> f64 {
    features(bb, player)
        .iter()
        .zip(cfg.weights.iter())
        .map(|(f, w)| f * w)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_features_are_zero_except_nothing() {
        // No lines occupied, both mobility counts resolve to 64 - 0.
        let feats = features(&Bitboard::EMPTY, 0);
        assert_eq!(feats[0], 0.0); // threat_own
        assert_eq!(feats[1], 0.0); // threat_opp
        assert_eq!(feats[2], 0.0); // threat_shared
        assert_eq!(feats[3], 64.0); // p0 to move: 64 legal moves vs 0
        assert_eq!(feats[4], 0.0); // build_two
        assert_eq!(feats[5], 0.0); // build_one
    }

    /// A@0, B@1, C@2 (all p0), d@8, d@13 (p1). Row 0 misses D at position 3;
    /// p1 already spent both d pieces, so only p0 can complete the line.
    fn one_sided_threat_board() -> Bitboard {
        Bitboard::EMPTY
            .with_move(0, 0, 0)
            .with_move(1, 3, 8)
            .with_move(0, 2, 2)
            .with_move(1, 3, 13)
            .with_move(0, 1, 1)
    }

    #[test]
    fn one_sided_threat_counts() {
        let bb = one_sided_threat_board();
        // p0 = 3 pieces, p1 = 2 pieces: p1 to move, so sign for player 0 is -1.
        assert_eq!(current_player(&bb), Some(1));
        let n = generate_legal_moves(&bb).len() as f64;

        let feats_p0 = features(&bb, 0);
        // Lines: row0 threat (p0 only); col0, col1, zone TL build_two (3);
        // row2, row3, col2, zone TR build_one (4); zone BL dead (two d's).
        assert_eq!(feats_p0[0], 1.0, "threat_own for p0");
        assert_eq!(feats_p0[1], 0.0, "threat_opp for p0");
        assert_eq!(feats_p0[2], 0.0, "threat_shared");
        assert_eq!(feats_p0[3], -n, "mobility: not p0's turn");
        assert_eq!(feats_p0[4], -3.0, "build_two with sign -1");
        assert_eq!(feats_p0[5], -4.0, "build_one with sign -1");

        let feats_p1 = features(&bb, 1);
        assert_eq!(feats_p1[0], 0.0, "threat_own for p1");
        assert_eq!(feats_p1[1], 1.0, "threat_opp for p1");
        assert_eq!(feats_p1[3], n, "mobility: p1's turn");
        assert_eq!(feats_p1[4], 3.0);
        assert_eq!(feats_p1[5], 4.0);
    }

    #[test]
    fn features_do_not_mutate_input() {
        let bb = one_sided_threat_board();
        let before = bb;
        let _ = features(&bb, 0);
        assert_eq!(bb, before);
    }

    #[test]
    fn evaluate_matches_hand_computed_dot_product() {
        let bb = one_sided_threat_board();
        let cfg = EvalConfig::default();
        let feats = features(&bb, 0);
        let expected: f64 = feats
            .iter()
            .zip(cfg.weights.iter())
            .map(|(f, w)| f * w)
            .sum();
        assert_eq!(evaluate(&bb, 0, &cfg), expected);
        // Concrete: 1*100 + 0*-100 + 0*20 + (-n)*3 + -3*2 + -4*0
        let n = generate_legal_moves(&bb).len() as f64;
        assert_eq!(expected, 100.0 - 3.0 * n - 6.0);
    }

    #[test]
    fn count_legal_moves_zero_off_turn() {
        assert_eq!(count_legal_moves(&Bitboard::EMPTY, 1), 0);
        assert_eq!(count_legal_moves(&Bitboard::EMPTY, 0), 64);
    }

    #[test]
    fn shared_threat_is_counted_for_both() {
        // A@0, b@1, C@2: row 0 misses D at 3, both sides still hold D pieces.
        let bb = Bitboard::EMPTY
            .with_move(0, 0, 0)
            .with_move(1, 1, 1)
            .with_move(0, 2, 2);
        // p1 to move: sign for player 1 is +1.
        let feats = features(&bb, 1);
        assert_eq!(feats[0], 1.0);
        assert_eq!(feats[1], 1.0);
        assert_eq!(feats[2], 1.0);
    }
}
