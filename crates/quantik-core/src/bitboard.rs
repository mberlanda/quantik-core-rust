use crate::constants::NUM_PLANES;

/// 128-bit bitboard: 8 planes of u16 for a 4×4 Quantik board.
///
/// Layout: `[C0S0, C0S1, C0S2, C0S3, C1S0, C1S1, C1S2, C1S3]`
/// where C = color (0 = player 0, 1 = player 1) and S = shape (0..3 → A..D).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(C, align(16))]
pub struct Bitboard {
    pub planes: [u16; NUM_PLANES],
}

impl Bitboard {
    pub const EMPTY: Self = Self {
        planes: [0; NUM_PLANES],
    };

    #[inline]
    pub fn new(planes: [u16; NUM_PLANES]) -> Self {
        Self { planes }
    }

    #[inline]
    pub fn plane_index(player: u8, shape: u8) -> usize {
        (player as usize) * 4 + (shape as usize)
    }

    #[inline]
    pub fn occupied(&self) -> u16 {
        self.planes.iter().fold(0u16, |acc, &p| acc | p)
    }

    #[inline]
    pub fn is_position_occupied(&self, pos: u8) -> bool {
        self.occupied() & (1u16 << pos) != 0
    }

    /// Set one bit and return a new bitboard (functional style).
    #[inline]
    pub fn with_move(&self, player: u8, shape: u8, position: u8) -> Self {
        let mut bb = *self;
        bb.planes[Self::plane_index(player, shape)] |= 1u16 << position;
        bb
    }

    /// Total number of pieces for a given player.
    #[inline]
    pub fn player_piece_count(&self, player: u8) -> u32 {
        let base = (player as usize) * 4;
        (0..4).map(|s| self.planes[base + s].count_ones()).sum()
    }

    /// Piece count for a specific (player, shape) pair.
    #[inline]
    pub fn shape_piece_count(&self, player: u8, shape: u8) -> u32 {
        self.planes[Self::plane_index(player, shape)].count_ones()
    }

    /// Pack to 16 little-endian bytes.
    pub fn to_le_bytes(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        for (i, &plane) in self.planes.iter().enumerate() {
            let bytes = plane.to_le_bytes();
            buf[i * 2] = bytes[0];
            buf[i * 2 + 1] = bytes[1];
        }
        buf
    }

    /// Unpack from 16 little-endian bytes.
    pub fn from_le_bytes(buf: &[u8; 16]) -> Self {
        let mut planes = [0u16; NUM_PLANES];
        for i in 0..NUM_PLANES {
            planes[i] = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
        }
        Self { planes }
    }
}

impl Default for Bitboard {
    fn default() -> Self {
        Self::EMPTY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bitboard() {
        let bb = Bitboard::EMPTY;
        assert_eq!(bb.occupied(), 0);
        assert_eq!(bb.player_piece_count(0), 0);
        assert_eq!(bb.player_piece_count(1), 0);
    }

    #[test]
    fn with_move_sets_bit() {
        let bb = Bitboard::EMPTY.with_move(0, 0, 5);
        assert!(bb.is_position_occupied(5));
        assert!(!bb.is_position_occupied(0));
        assert_eq!(bb.shape_piece_count(0, 0), 1);
    }

    #[test]
    fn roundtrip_bytes() {
        let bb = Bitboard::new([1, 2, 3, 4, 5, 6, 7, 8]);
        let bytes = bb.to_le_bytes();
        let bb2 = Bitboard::from_le_bytes(&bytes);
        assert_eq!(bb, bb2);
    }

    #[test]
    fn plane_index_mapping() {
        assert_eq!(Bitboard::plane_index(0, 0), 0);
        assert_eq!(Bitboard::plane_index(0, 3), 3);
        assert_eq!(Bitboard::plane_index(1, 0), 4);
        assert_eq!(Bitboard::plane_index(1, 3), 7);
    }
}
