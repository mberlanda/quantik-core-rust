use crate::bitboard::Bitboard;
use crate::constants::{FLAG_CANON, VERSION};
use crate::qfen::{bb_from_qfen, bb_to_qfen};
use crate::symmetry::SymmetryHandler;

/// Serialisable game state wrapping a [`Bitboard`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct State {
    pub bb: Bitboard,
}

impl State {
    pub fn new(bb: Bitboard) -> Self {
        Self { bb }
    }

    pub fn empty() -> Self {
        Self { bb: Bitboard::EMPTY }
    }

    // ── binary (18 bytes: version + flags + 8×u16 LE) ───────────────

    pub fn pack(&self, flags: u8) -> [u8; 18] {
        let mut buf = [0u8; 18];
        buf[0] = VERSION;
        buf[1] = flags;
        let bb_bytes = self.bb.to_le_bytes();
        buf[2..18].copy_from_slice(&bb_bytes);
        buf
    }

    pub fn unpack(data: &[u8]) -> Result<Self, String> {
        if data.len() < 18 {
            return Err(format!("Buffer too small: need 18 bytes, got {}", data.len()));
        }
        if data[0] != VERSION {
            return Err(format!("Unsupported version {}", data[0]));
        }
        let mut bb_buf = [0u8; 16];
        bb_buf.copy_from_slice(&data[2..18]);
        Ok(Self {
            bb: Bitboard::from_le_bytes(&bb_buf),
        })
    }

    // ── QFEN ─────────────────────────────────────────────────────────

    pub fn to_qfen(&self) -> String {
        bb_to_qfen(&self.bb)
    }

    pub fn from_qfen(qfen: &str) -> Result<Self, String> {
        bb_from_qfen(qfen).map(|bb| Self { bb })
    }

    // ── canonical form ───────────────────────────────────────────────

    pub fn canonical_payload(&self) -> [u8; 16] {
        SymmetryHandler::canonical_payload(&self.bb)
    }

    pub fn canonical_key(&self) -> [u8; 18] {
        let mut key = [0u8; 18];
        key[0] = VERSION;
        key[1] = FLAG_CANON;
        key[2..18].copy_from_slice(&self.canonical_payload());
        key
    }

    pub fn symmetry_count(&self) -> usize {
        SymmetryHandler::orbit_size(&self.bb)
    }
}

impl Default for State {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let st = State::from_qfen("A.bC/..../d..B/...a").unwrap();
        let buf = st.pack(0);
        let st2 = State::unpack(&buf).unwrap();
        assert_eq!(st, st2);
    }

    #[test]
    fn canonical_key_starts_with_version_flag() {
        let key = State::empty().canonical_key();
        assert_eq!(key[0], VERSION);
        assert_eq!(key[1], FLAG_CANON);
    }

    #[test]
    fn qfen_roundtrip() {
        let qfen = "AbCd/..../..../....";
        let st = State::from_qfen(qfen).unwrap();
        assert_eq!(st.to_qfen(), qfen);
    }
}
