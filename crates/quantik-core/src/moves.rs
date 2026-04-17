use crate::bitboard::Bitboard;
use crate::constants::{MAX_PIECES_PER_SHAPE, NUM_SHAPES, WIN_MASKS};
use crate::game::current_player;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Move {
    pub player: u8,
    pub shape: u8,
    pub position: u8,
}

impl Move {
    pub fn new(player: u8, shape: u8, position: u8) -> Self {
        debug_assert!(player <= 1);
        debug_assert!(shape < 4);
        debug_assert!(position < 16);
        Self { player, shape, position }
    }
}

/// Check whether `(player, shape, position)` is legal on `bb`.
pub fn is_move_legal(bb: &Bitboard, player: u8, shape: u8, position: u8) -> bool {
    if bb.is_position_occupied(position) {
        return false;
    }
    if bb.shape_piece_count(player, shape) >= MAX_PIECES_PER_SHAPE as u32 {
        return false;
    }
    let opponent = 1 - player;
    let opp_bits = bb.planes[Bitboard::plane_index(opponent, shape)];
    let pos_mask = 1u16 << position;
    for &wm in &WIN_MASKS {
        if (pos_mask & wm != 0) && (opp_bits & wm != 0) {
            return false;
        }
    }
    true
}

/// Generate all legal moves for the current player.
pub fn generate_legal_moves(bb: &Bitboard) -> Vec<Move> {
    let player = match current_player(bb) {
        Some(p) => p,
        None => return Vec::new(),
    };
    let occupied = bb.occupied();
    let mut moves = Vec::new();

    for shape in 0..NUM_SHAPES as u8 {
        if bb.shape_piece_count(player, shape) >= MAX_PIECES_PER_SHAPE as u32 {
            continue;
        }
        let opp_bits = bb.planes[Bitboard::plane_index(1 - player, shape)];
        for pos in 0..16u8 {
            if occupied & (1u16 << pos) != 0 {
                continue;
            }
            let pos_mask = 1u16 << pos;
            let blocked = WIN_MASKS.iter().any(|&wm| {
                (pos_mask & wm != 0) && (opp_bits & wm != 0)
            });
            if !blocked {
                moves.push(Move::new(player, shape, pos));
            }
        }
    }
    moves
}

/// Apply a move (assumes legality).
#[inline]
pub fn apply_move(bb: &Bitboard, mv: &Move) -> Bitboard {
    bb.with_move(mv.player, mv.shape, mv.position)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_legal_moves() {
        let moves = generate_legal_moves(&Bitboard::EMPTY);
        // 4 shapes * 16 positions = 64
        assert_eq!(moves.len(), 64);
        assert!(moves.iter().all(|m| m.player == 0));
    }

    #[test]
    fn occupied_position_blocked() {
        let bb = Bitboard::EMPTY.with_move(0, 0, 0);
        assert!(!is_move_legal(&bb, 1, 1, 0));
    }

    #[test]
    fn opponent_shape_on_line_blocked() {
        // Player 0 places shape A at position 0 (row 0, zone top-left)
        let bb = Bitboard::EMPTY.with_move(0, 0, 0);
        // Player 1 cannot place shape A anywhere on row 0 (positions 1,2,3),
        // column 0 (positions 4,8,12), or zone top-left (positions 1,4,5)
        assert!(!is_move_legal(&bb, 1, 0, 1)); // same row
        assert!(!is_move_legal(&bb, 1, 0, 4)); // same col & zone
        // But player 1 can place shape A at position 10 (no shared line)
        assert!(is_move_legal(&bb, 1, 0, 10));
    }

    #[test]
    fn max_pieces_per_shape() {
        let bb = Bitboard::EMPTY
            .with_move(0, 0, 0)
            .with_move(1, 1, 5)
            .with_move(0, 0, 10);
        // Player 0 already has 2 A-pieces
        assert!(!is_move_legal(&bb, 0, 0, 15));
        // But can still place B
        assert!(is_move_legal(&bb, 0, 1, 15));
    }
}
