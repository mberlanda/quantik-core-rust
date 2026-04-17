use crate::bitboard::Bitboard;
use crate::constants::SHAPE_LETTERS;

/// Encode a bitboard as a QFEN string.
///
/// Format: 4 slash-separated ranks of 4 characters each.
/// Uppercase A-D for player 0, lowercase a-d for player 1, `.` for empty.
pub fn bb_to_qfen(bb: &Bitboard) -> String {
    let mut ranks = [String::new(), String::new(), String::new(), String::new()];

    for r in 0..4 {
        for c in 0..4 {
            let pos = r * 4 + c;
            let mut ch = '.';
            for color in 0..2u8 {
                for s in 0..4u8 {
                    if (bb.planes[Bitboard::plane_index(color, s)] >> pos) & 1 == 1 {
                        let letter = SHAPE_LETTERS[s as usize];
                        ch = if color == 0 {
                            letter
                        } else {
                            letter.to_ascii_lowercase()
                        };
                    }
                }
            }
            ranks[r].push(ch);
        }
    }
    ranks.join("/")
}

/// Decode a QFEN string into a bitboard.
pub fn bb_from_qfen(qfen: &str) -> Result<Bitboard, String> {
    let parts: Vec<&str> = qfen.split('/').collect();
    if parts.len() != 4 {
        return Err(format!("QFEN must have 4 ranks, got {}", parts.len()));
    }
    let mut planes = [0u16; 8];

    for (r, rank) in parts.iter().enumerate() {
        let chars: Vec<char> = rank.chars().collect();
        if chars.len() != 4 {
            return Err(format!(
                "Rank {} must have 4 chars, got {}",
                r,
                chars.len()
            ));
        }
        for (c, &ch) in chars.iter().enumerate() {
            if ch == '.' {
                continue;
            }
            let upper = ch.to_ascii_uppercase();
            let shape = match upper {
                'A' => 0u8,
                'B' => 1,
                'C' => 2,
                'D' => 3,
                _ => return Err(format!("Invalid char '{}' in QFEN", ch)),
            };
            let color: u8 = if ch.is_uppercase() { 0 } else { 1 };
            let pos = r * 4 + c;
            planes[Bitboard::plane_index(color, shape)] |= 1u16 << pos;
        }
    }
    Ok(Bitboard::new(planes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_qfen() {
        let bb = Bitboard::EMPTY;
        assert_eq!(bb_to_qfen(&bb), "..../..../..../....");
    }

    #[test]
    fn roundtrip_qfen() {
        let qfen = "AbCd/..../..../....";
        let bb = bb_from_qfen(qfen).unwrap();
        assert_eq!(bb_to_qfen(&bb), qfen);
    }

    #[test]
    fn mixed_position() {
        let qfen = "A.bC/..../d..B/...a";
        let bb = bb_from_qfen(qfen).unwrap();
        assert_eq!(bb_to_qfen(&bb), qfen);
    }

    #[test]
    fn invalid_qfen_rank_count() {
        assert!(bb_from_qfen("..../....").is_err());
    }

    #[test]
    fn invalid_qfen_char() {
        assert!(bb_from_qfen("X.../..../..../....").is_err());
    }
}
