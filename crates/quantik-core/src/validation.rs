//! Shared game-state validation, used at both the state/board constructor
//! boundary and the adapter/portability-report boundary.
//!
//! See quantik-core-contracts' `docs/game-state.md#invalid-state-validation-boundaries`
//! for the normative definition of what each boundary must check, and
//! `fixtures/invalid-states/invalid-state-v1.json` for golden cases. The
//! parser boundary (QFEN string well-formedness) is handled separately in
//! `qfen.rs` and is not part of this module: it rejects malformed QFEN
//! strings before any `Bitboard` exists to validate here.

use crate::bitboard::Bitboard;
use crate::constants::{MAX_PIECES_PER_SHAPE, WIN_MASKS};
use crate::game::current_player;

/// Why a bitboard failed full game-state validation. Names match the
/// `ValidationResult` vocabulary shared with `quantik-core-py` and
/// `invalid-state-fixtures.v1` (`code()`), so cross-language reports and
/// fixture-driven tests can compare on the reason, not just success/failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvalidStateReason {
    PieceOverlap,
    ShapeCountExceeded,
    TurnBalanceInvalid,
    IllegalPlacement,
}

impl InvalidStateReason {
    /// The `invalid-state-fixtures.v1` / `ValidationResult` name.
    pub fn code(&self) -> &'static str {
        match self {
            InvalidStateReason::PieceOverlap => "PIECE_OVERLAP",
            InvalidStateReason::ShapeCountExceeded => "SHAPE_COUNT_EXCEEDED",
            InvalidStateReason::TurnBalanceInvalid => "TURN_BALANCE_INVALID",
            InvalidStateReason::IllegalPlacement => "ILLEGAL_PLACEMENT",
        }
    }
}

impl std::fmt::Display for InvalidStateReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            InvalidStateReason::PieceOverlap => "bitboards contain overlapping pieces",
            InvalidStateReason::ShapeCountExceeded => "bitboards exceed max pieces per shape",
            InvalidStateReason::TurnBalanceInvalid => "invalid turn balance",
            InvalidStateReason::IllegalPlacement => {
                "bitboards contain illegal same-shape line conflict"
            }
        };
        f.write_str(message)
    }
}

/// Full game-state validation: piece-count/inventory limits, overlap,
/// turn balance, and same-shape line conflicts. Returns the side to move on
/// success.
///
/// Check order matches `quantik_core.state_validator.validate_game_state`
/// in `quantik-core-py` exactly (overlap/inventory during a single pass over
/// the 8 planes, then turn balance, then placement legality), so that a
/// bitboard invalid in more than one way is reported with the same reason in
/// both languages.
pub fn validate_bitboard_state(bb: &Bitboard) -> Result<u8, InvalidStateReason> {
    let mut occupied = 0u16;
    for plane in bb.planes.iter() {
        if plane.count_ones() > MAX_PIECES_PER_SHAPE as u32 {
            return Err(InvalidStateReason::ShapeCountExceeded);
        }
        if occupied & plane != 0 {
            return Err(InvalidStateReason::PieceOverlap);
        }
        occupied |= plane;
    }

    let side_to_move = current_player(bb).ok_or(InvalidStateReason::TurnBalanceInvalid)?;

    for shape in 0..4 {
        let p0 = bb.planes[shape];
        let p1 = bb.planes[shape + 4];
        for &line in &WIN_MASKS {
            if (p0 & line != 0) && (p1 & line != 0) {
                return Err(InvalidStateReason::IllegalPlacement);
            }
        }
    }

    Ok(side_to_move)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bb_from_planes(planes: [u16; 8]) -> Bitboard {
        Bitboard::new(planes)
    }

    #[test]
    fn valid_state_returns_side_to_move() {
        // One player-0 piece placed: player 1 to move.
        let bb = bb_from_planes([1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(validate_bitboard_state(&bb), Ok(1));
    }

    #[test]
    fn rejects_piece_overlap() {
        // Player 0 shape A and shape B both occupy position 0.
        let bb = bb_from_planes([1, 1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            validate_bitboard_state(&bb),
            Err(InvalidStateReason::PieceOverlap)
        );
    }

    #[test]
    fn rejects_shape_count_exceeded() {
        // Player 0 shape A at three positions; inventory allows at most 2.
        let bb = bb_from_planes([0b111, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            validate_bitboard_state(&bb),
            Err(InvalidStateReason::ShapeCountExceeded)
        );
    }

    #[test]
    fn rejects_turn_balance_invalid() {
        // Two player-0 pieces (shape A at position 0, shape B at position 1;
        // distinct planes and positions, so neither overlap nor shape-count
        // checks preempt this), zero player-1 pieces: difference of 2.
        let bb = bb_from_planes([0b01, 0b10, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            validate_bitboard_state(&bb),
            Err(InvalidStateReason::TurnBalanceInvalid)
        );
    }

    #[test]
    fn rejects_illegal_placement_cross_player_same_shape_line() {
        // Player 0 shape A at position 0, player 1 shape A at position 1: same row.
        let bb = bb_from_planes([0b01, 0, 0, 0, 0b10, 0, 0, 0]);
        assert_eq!(
            validate_bitboard_state(&bb),
            Err(InvalidStateReason::IllegalPlacement)
        );
    }
}
