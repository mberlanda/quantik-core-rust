use crate::bitboard::Bitboard;
use crate::constants::{NUM_SHAPES, WIN_MASKS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WinStatus {
    NoWin,
    Player0Wins,
    Player1Wins,
}

/// Check whether any win line contains all 4 distinct shapes (regardless of color).
pub fn has_winning_line(bb: &Bitboard) -> bool {
    let shape_unions: [u16; NUM_SHAPES] = std::array::from_fn(|s| {
        bb.planes[s] | bb.planes[s + 4]
    });

    WIN_MASKS.iter().any(|&mask| {
        shape_unions.iter().all(|&su| su & mask != 0)
    })
}

/// Determine winner.  The last player to move (the one with more pieces on the
/// board) is credited with the win.
pub fn check_winner(bb: &Bitboard) -> WinStatus {
    if !has_winning_line(bb) {
        return WinStatus::NoWin;
    }
    let p0 = bb.player_piece_count(0);
    let p1 = bb.player_piece_count(1);
    if p0 > p1 {
        WinStatus::Player0Wins
    } else {
        WinStatus::Player1Wins
    }
}

/// Whose turn is it?  Player 0 moves first; after that they alternate.
/// Returns `None` if the piece counts are inconsistent.
pub fn current_player(bb: &Bitboard) -> Option<u8> {
    let p0 = bb.player_piece_count(0);
    let p1 = bb.player_piece_count(1);
    if p0 == p1 {
        Some(0)
    } else if p0 == p1 + 1 {
        Some(1)
    } else {
        None
    }
}

pub fn is_game_over(bb: &Bitboard) -> bool {
    check_winner(bb) != WinStatus::NoWin
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_no_win() {
        assert_eq!(check_winner(&Bitboard::EMPTY), WinStatus::NoWin);
        assert_eq!(current_player(&Bitboard::EMPTY), Some(0));
    }

    #[test]
    fn row_0_win() {
        // Place A(p0) at 0, B(p1) at 1, C(p0) at 2, D(p1) at 3
        let bb = Bitboard::EMPTY
            .with_move(0, 0, 0)  // A at pos 0
            .with_move(1, 1, 1)  // b at pos 1
            .with_move(0, 2, 2)  // C at pos 2
            .with_move(1, 3, 3); // d at pos 3
        assert!(has_winning_line(&bb));
        // p0 has 2, p1 has 2 → p1 wins (equal means last mover is p1 when 4 total)
        // Actually: p0=2, p1=2 → p0==p1 so winner is p1 per check_winner logic
        assert_eq!(check_winner(&bb), WinStatus::Player1Wins);
    }

    #[test]
    fn current_player_alternates() {
        let bb = Bitboard::EMPTY;
        assert_eq!(current_player(&bb), Some(0));
        let bb = bb.with_move(0, 0, 0);
        assert_eq!(current_player(&bb), Some(1));
        let bb = bb.with_move(1, 1, 5);
        assert_eq!(current_player(&bb), Some(0));
    }
}
