use crate::bitboard::Bitboard;
use crate::constants::MAX_PIECES_PER_SHAPE;
use crate::game::{check_winner, current_player, WinStatus};
use crate::moves::{apply_move, generate_legal_moves, is_move_legal, Move};
use crate::qfen::{bb_from_qfen, bb_to_qfen};
use crate::state::State;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameResult {
    Ongoing,
    Player0Wins,
    Player1Wins,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlayerInventory {
    pub remaining: [u8; 4], // shapes A, B, C, D
}

impl PlayerInventory {
    pub fn full() -> Self {
        Self {
            remaining: [MAX_PIECES_PER_SHAPE; 4],
        }
    }

    pub fn total(&self) -> u8 {
        self.remaining.iter().sum()
    }

    pub fn has_shape(&self, shape: u8) -> bool {
        self.remaining[shape as usize] > 0
    }

    pub fn use_shape(&mut self, shape: u8) {
        debug_assert!(self.remaining[shape as usize] > 0);
        self.remaining[shape as usize] -= 1;
    }

    pub fn return_shape(&mut self, shape: u8) {
        debug_assert!(self.remaining[shape as usize] < MAX_PIECES_PER_SHAPE);
        self.remaining[shape as usize] += 1;
    }
}

struct MoveRecord {
    mv: Move,
    prev_bb: Bitboard,
    prev_inventories: [PlayerInventory; 2],
}

/// High-level Quantik board with inventory tracking, move history, and undo.
pub struct QuantikBoard {
    bb: Bitboard,
    inventories: [PlayerInventory; 2],
    current_player: u8,
    history: Vec<MoveRecord>,
}

impl QuantikBoard {
    pub fn new() -> Self {
        Self {
            bb: Bitboard::EMPTY,
            inventories: [PlayerInventory::full(), PlayerInventory::full()],
            current_player: 0,
            history: Vec::new(),
        }
    }

    pub fn from_bitboard(bb: Bitboard) -> Result<Self, String> {
        let cp = current_player(&bb).ok_or("Invalid turn balance")?;
        let mut invs = [PlayerInventory::full(), PlayerInventory::full()];
        for player in 0..2u8 {
            for shape in 0..4u8 {
                let used = bb.shape_piece_count(player, shape) as u8;
                if used > MAX_PIECES_PER_SHAPE {
                    return Err(format!(
                        "Player {} has {} pieces of shape {} (max {})",
                        player, used, shape, MAX_PIECES_PER_SHAPE
                    ));
                }
                invs[player as usize].remaining[shape as usize] = MAX_PIECES_PER_SHAPE - used;
            }
        }
        Ok(Self {
            bb,
            inventories: invs,
            current_player: cp,
            history: Vec::new(),
        })
    }

    pub fn from_qfen(qfen: &str) -> Result<Self, String> {
        let bb = bb_from_qfen(qfen)?;
        Self::from_bitboard(bb)
    }

    // ── accessors ────────────────────────────────────────────────────

    pub fn bitboard(&self) -> &Bitboard {
        &self.bb
    }

    pub fn state(&self) -> State {
        State::new(self.bb)
    }

    pub fn current_player(&self) -> u8 {
        self.current_player
    }

    pub fn inventories(&self) -> &[PlayerInventory; 2] {
        &self.inventories
    }

    pub fn move_count(&self) -> usize {
        self.history.len()
    }

    pub fn last_move(&self) -> Option<&Move> {
        self.history.last().map(|r| &r.mv)
    }

    pub fn to_qfen(&self) -> String {
        bb_to_qfen(&self.bb)
    }

    // ── game status ──────────────────────────────────────────────────

    pub fn game_result(&self) -> GameResult {
        match check_winner(&self.bb) {
            WinStatus::Player0Wins => return GameResult::Player0Wins,
            WinStatus::Player1Wins => return GameResult::Player1Wins,
            WinStatus::NoWin => {}
        }
        if self.legal_moves().is_empty() {
            // stalemate → the player who cannot move loses
            if self.current_player == 0 {
                GameResult::Player1Wins
            } else {
                GameResult::Player0Wins
            }
        } else {
            GameResult::Ongoing
        }
    }

    pub fn is_game_over(&self) -> bool {
        self.game_result() != GameResult::Ongoing
    }

    // ── move generation ──────────────────────────────────────────────

    pub fn legal_moves(&self) -> Vec<Move> {
        let raw = generate_legal_moves(&self.bb);
        raw.into_iter()
            .filter(|m| self.inventories[m.player as usize].has_shape(m.shape))
            .collect()
    }

    pub fn is_legal(&self, mv: &Move) -> bool {
        mv.player == self.current_player
            && self.inventories[mv.player as usize].has_shape(mv.shape)
            && is_move_legal(&self.bb, mv.player, mv.shape, mv.position)
    }

    // ── play / undo ──────────────────────────────────────────────────

    pub fn play_move(&mut self, mv: Move) -> Result<(), String> {
        if self.is_game_over() {
            return Err("Game is already over".into());
        }
        if !self.is_legal(&mv) {
            return Err("Illegal move".into());
        }

        let record = MoveRecord {
            mv,
            prev_bb: self.bb,
            prev_inventories: self.inventories,
        };

        self.bb = apply_move(&self.bb, &mv);
        self.inventories[mv.player as usize].use_shape(mv.shape);
        self.current_player = 1 - self.current_player;
        self.history.push(record);
        Ok(())
    }

    pub fn undo_move(&mut self) -> bool {
        if let Some(record) = self.history.pop() {
            self.bb = record.prev_bb;
            self.inventories = record.prev_inventories;
            self.current_player = record.mv.player;
            true
        } else {
            false
        }
    }

    pub fn undo_moves(&mut self, count: usize) -> usize {
        let mut undone = 0;
        for _ in 0..count {
            if self.undo_move() {
                undone += 1;
            } else {
                break;
            }
        }
        undone
    }
}

impl Default for QuantikBoard {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for QuantikBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "QFEN: {}", self.to_qfen())?;
        writeln!(f, "Current player: {}", self.current_player)?;
        writeln!(f, "Move count: {}", self.move_count())?;
        for p in 0..2 {
            let inv = &self.inventories[p];
            writeln!(
                f,
                "Player {} inventory: A={} B={} C={} D={}",
                p, inv.remaining[0], inv.remaining[1], inv.remaining[2], inv.remaining[3]
            )?;
        }
        let result = self.game_result();
        if result != GameResult::Ongoing {
            writeln!(f, "Game result: {:?}", result)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_board_is_empty() {
        let board = QuantikBoard::new();
        assert_eq!(board.to_qfen(), "..../..../..../....");
        assert_eq!(board.current_player(), 0);
        assert_eq!(board.move_count(), 0);
        assert!(!board.is_game_over());
    }

    #[test]
    fn play_and_undo() {
        let mut board = QuantikBoard::new();
        let mv = Move::new(0, 0, 0);
        board.play_move(mv).unwrap();
        assert_eq!(board.current_player(), 1);
        assert_eq!(board.move_count(), 1);
        assert!(board.undo_move());
        assert_eq!(board.current_player(), 0);
        assert_eq!(board.move_count(), 0);
    }

    #[test]
    fn inventory_decreases_on_play() {
        let mut board = QuantikBoard::new();
        assert_eq!(board.inventories()[0].remaining[0], 2);
        board.play_move(Move::new(0, 0, 0)).unwrap();
        assert_eq!(board.inventories()[0].remaining[0], 1);
    }

    #[test]
    fn illegal_move_rejected() {
        let mut board = QuantikBoard::new();
        // wrong player
        let mv = Move::new(1, 0, 0);
        assert!(board.play_move(mv).is_err());
    }

    #[test]
    fn from_qfen_round_trip() {
        let board = QuantikBoard::from_qfen("A.../..../..../....").unwrap();
        assert_eq!(board.current_player(), 1);
        assert_eq!(board.inventories()[0].remaining[0], 1);
    }

    #[test]
    fn win_detection() {
        let mut board = QuantikBoard::new();
        // Build a row-0 win: A(0,0) b(1,1) C(0,2) d(1,3)
        // But we need to place them on row 0 without conflicting lines.
        // p0 A@0, p1 b@1, p0 C@2, p1 d@3
        board.play_move(Move::new(0, 0, 0)).unwrap(); // A at 0
        board.play_move(Move::new(1, 1, 1)).unwrap(); // b at 1
        board.play_move(Move::new(0, 2, 2)).unwrap(); // C at 2
        board.play_move(Move::new(1, 3, 3)).unwrap(); // d at 3
        assert!(board.is_game_over());
    }
}
