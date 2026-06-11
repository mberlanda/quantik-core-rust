use crate::bitboard::Bitboard;
use std::sync::OnceLock;

/// Position mapping: for each of the 8 D4 symmetries, maps input position → output position.
type PosMap = [u8; 16];

const D4_MAPS: [PosMap; 8] = {
    let mut maps = [[0u8; 16]; 8];
    let mut i: u8 = 0;
    while i < 16 {
        let r = i / 4;
        let c = i % 4;

        maps[0][i as usize] = r * 4 + c; // id
        maps[1][i as usize] = c * 4 + (3 - r); // rot90
        maps[2][i as usize] = (3 - r) * 4 + (3 - c); // rot180
        maps[3][i as usize] = (3 - c) * 4 + r; // rot270
        maps[4][i as usize] = r * 4 + (3 - c); // reflV
        maps[5][i as usize] = (3 - r) * 4 + c; // reflH
        maps[6][i as usize] = c * 4 + r; // reflD
        maps[7][i as usize] = (3 - c) * 4 + (3 - r); // reflAD

        i += 1;
    }
    maps
};

/// All 24 permutations of shapes 0..3.
const SHAPE_PERMS: [[u8; 4]; 24] = generate_shape_perms();

const fn generate_shape_perms() -> [[u8; 4]; 24] {
    let mut perms = [[0u8; 4]; 24];
    let mut idx = 0;
    let mut a: u8 = 0;
    while a < 4 {
        let mut b: u8 = 0;
        while b < 4 {
            if b == a {
                b += 1;
                continue;
            }
            let mut c: u8 = 0;
            while c < 4 {
                if c == a || c == b {
                    c += 1;
                    continue;
                }
                let d = 6 - a - b - c; // the remaining element (0+1+2+3=6)
                perms[idx] = [a, b, c, d];
                idx += 1;
                c += 1;
            }
            b += 1;
        }
        a += 1;
    }
    perms
}

/// Pre-computed LUT: `PERM16_LUT[d4_idx][mask]` → permuted mask.
///
/// Built once on first access (~1 MB).  We use `Vec` instead of a fixed
/// array to avoid blowing the stack during initialisation.
struct Perm16Lut {
    tables: Vec<Vec<u16>>, // [8][65536]
}

fn build_perm16_lut() -> Perm16Lut {
    let mut tables: Vec<Vec<u16>> = Vec::with_capacity(8);
    for map in &D4_MAPS {
        let mut t = vec![0u16; 65536];
        for x in 0u32..65536 {
            let mut y = 0u16;
            let mut m = x as u16;
            let mut i = 0u8;
            while m != 0 {
                if m & 1 != 0 {
                    y |= 1u16 << map[i as usize];
                }
                i += 1;
                m >>= 1;
            }
            t[x as usize] = y;
        }
        tables.push(t);
    }
    Perm16Lut { tables }
}

static PERM16_LUT: OnceLock<Perm16Lut> = OnceLock::new();

fn lut() -> &'static Perm16Lut {
    PERM16_LUT.get_or_init(build_perm16_lut)
}

#[inline]
fn permute16(mask: u16, d4_idx: usize) -> u16 {
    lut().tables[d4_idx][mask as usize]
}

pub struct SymmetryHandler;

impl SymmetryHandler {
    /// Find the canonical (lexicographically smallest) bitboard under the
    /// 192-element symmetry group (8 D4 × 24 shape permutations, no color swap).
    pub fn find_canonical(bb: &Bitboard) -> Bitboard {
        let mut best: Option<[u16; 8]> = None;

        for d4_idx in 0..8 {
            let g0: [u16; 4] = std::array::from_fn(|s| permute16(bb.planes[s], d4_idx));
            let g1: [u16; 4] = std::array::from_fn(|s| permute16(bb.planes[s + 4], d4_idx));

            for perm in &SHAPE_PERMS {
                let candidate: [u16; 8] = [
                    g0[perm[0] as usize],
                    g0[perm[1] as usize],
                    g0[perm[2] as usize],
                    g0[perm[3] as usize],
                    g1[perm[0] as usize],
                    g1[perm[1] as usize],
                    g1[perm[2] as usize],
                    g1[perm[3] as usize],
                ];

                let is_better = match &best {
                    None => true,
                    Some(b) => le_bytes_less(&candidate, b),
                };
                if is_better {
                    best = Some(candidate);
                }
            }
        }
        Bitboard::new(best.unwrap_or([0; 8]))
    }

    /// 16-byte canonical payload (LE-packed planes of the canonical form).
    pub fn canonical_payload(bb: &Bitboard) -> [u8; 16] {
        Self::find_canonical(bb).to_le_bytes()
    }

    /// How many distinct boards in this orbit (1..192).
    pub fn orbit_size(bb: &Bitboard) -> usize {
        let mut seen = std::collections::HashSet::new();

        for d4_idx in 0..8 {
            let g0: [u16; 4] = std::array::from_fn(|s| permute16(bb.planes[s], d4_idx));
            let g1: [u16; 4] = std::array::from_fn(|s| permute16(bb.planes[s + 4], d4_idx));

            for perm in &SHAPE_PERMS {
                let candidate: [u16; 8] = [
                    g0[perm[0] as usize],
                    g0[perm[1] as usize],
                    g0[perm[2] as usize],
                    g0[perm[3] as usize],
                    g1[perm[0] as usize],
                    g1[perm[1] as usize],
                    g1[perm[2] as usize],
                    g1[perm[3] as usize],
                ];
                seen.insert(candidate);
            }
        }
        seen.len()
    }
}

/// Compare two `[u16; 8]` in little-endian byte order.
fn le_bytes_less(a: &[u16; 8], b: &[u16; 8]) -> bool {
    for i in 0..8 {
        let ab = a[i].to_le_bytes();
        let bb = b[i].to_le_bytes();
        for j in 0..2 {
            match ab[j].cmp(&bb[j]) {
                std::cmp::Ordering::Less => return true,
                std::cmp::Ordering::Greater => return false,
                std::cmp::Ordering::Equal => {}
            }
        }
    }
    false // equal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_permutation() {
        assert_eq!(permute16(1, 0), 1); // identity
    }

    #[test]
    fn canonical_empty_is_empty() {
        let canon = SymmetryHandler::find_canonical(&Bitboard::EMPTY);
        assert_eq!(canon, Bitboard::EMPTY);
    }

    #[test]
    fn canonical_is_idempotent() {
        let bb = Bitboard::EMPTY.with_move(0, 0, 0).with_move(1, 1, 5);
        let c1 = SymmetryHandler::find_canonical(&bb);
        let c2 = SymmetryHandler::find_canonical(&c1);
        assert_eq!(c1, c2);
    }

    #[test]
    fn rotated_boards_share_canonical_form() {
        // Place shape A (player 0) at the four corners – each is a rotation of the other.
        let corners = [0u8, 3, 12, 15];
        let canonicals: Vec<Bitboard> = corners
            .iter()
            .map(|&pos| {
                let bb = Bitboard::EMPTY.with_move(0, 0, pos);
                SymmetryHandler::find_canonical(&bb)
            })
            .collect();
        assert!(canonicals.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn orbit_size_single_piece() {
        let bb = Bitboard::EMPTY.with_move(0, 0, 0);
        let size = SymmetryHandler::orbit_size(&bb);
        // A single piece at a corner: 4 corners × 4 shape relabellings = 16
        assert_eq!(size, 16);
    }

    #[test]
    fn shape_perms_count() {
        assert_eq!(SHAPE_PERMS.len(), 24);
    }

    #[test]
    fn d4_maps_are_permutations() {
        for map in &D4_MAPS {
            let mut sorted = map.to_vec();
            sorted.sort();
            let expected: Vec<u8> = (0..16).collect();
            assert_eq!(sorted, expected);
        }
    }
}
