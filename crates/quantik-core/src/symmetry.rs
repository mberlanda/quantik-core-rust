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

/// Inverse of each D4 index: rotate90 <-> rotate270, every other element
/// (identity, rotate180, and all four reflections) is its own inverse.
const D4_INVERSE: [u8; 8] = [0, 3, 2, 1, 4, 5, 6, 7];

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

    /// Remap an `action-index.v1` value (`shape * 16 + position`) under one
    /// of the 192 canonicalization transforms.
    ///
    /// `transform_index` encodes `(d4_index, shape_perm)` as
    /// `d4_index * 24 + shape_perm_index`, where `shape_perm_index` indexes
    /// `SHAPE_PERMS` (lexicographic order of the 24 permutations of
    /// `(0, 1, 2, 3)`, matching `quantik-core-py`'s independently-generated
    /// `itertools.permutations` table). This is the same 192-element group
    /// `find_canonical`/`orbit_size` search over -- color swap is not part
    /// of it. See `docs/symmetry-transposition.md` in
    /// quantik-core-contracts for the normative definition and
    /// `fixtures/symmetry/symmetry-v1.json` for golden cases.
    ///
    /// Panics if `action_index >= 64` or `transform_index >= 192`.
    pub fn remap_action_index(action_index: u8, transform_index: u8) -> u8 {
        assert!(
            action_index < 64,
            "action_index must be 0..63, got {action_index}"
        );
        assert!(
            transform_index < 192,
            "transform_index must be 0..191, got {transform_index}"
        );

        let shape = action_index / 16;
        let position = action_index % 16;
        let d4_idx = (transform_index / 24) as usize;
        let perm = &SHAPE_PERMS[(transform_index % 24) as usize];

        let new_position = D4_MAPS[d4_idx][position as usize];
        // Inverse lookup, not perm[shape]: perm[k] names which *original*
        // shape moves into output slot k (see find_canonical above), so the
        // slot receiving old shape `shape` is perm.position(shape).
        let new_shape = perm
            .iter()
            .position(|&s| s == shape)
            .expect("SHAPE_PERMS entries are permutations of 0..3") as u8;
        new_shape * 16 + new_position
    }

    /// Returns the `transform_index` of the inverse transform.
    ///
    /// `remap_action_index(remap_action_index(a, t), inverse_transform_index(t)) == a`
    /// for every action index `a` and transform index `t`.
    ///
    /// Panics if `transform_index >= 192`.
    pub fn inverse_transform_index(transform_index: u8) -> u8 {
        assert!(
            transform_index < 192,
            "transform_index must be 0..191, got {transform_index}"
        );

        let d4_idx = (transform_index / 24) as usize;
        let perm = &SHAPE_PERMS[(transform_index % 24) as usize];

        let mut inverse_perm = [0u8; 4];
        for (i, &j) in perm.iter().enumerate() {
            inverse_perm[j as usize] = i as u8;
        }
        let inverse_perm_index = SHAPE_PERMS
            .iter()
            .position(|p| *p == inverse_perm)
            .expect("inverse of a permutation of 0..3 is itself one")
            as u8;

        D4_INVERSE[d4_idx] * 24 + inverse_perm_index
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

    fn apply_transform_to_bitboard(bb: &Bitboard, d4_idx: usize, perm: &[u8; 4]) -> Bitboard {
        let g0: [u16; 4] = std::array::from_fn(|s| permute16(bb.planes[s], d4_idx));
        let g1: [u16; 4] = std::array::from_fn(|s| permute16(bb.planes[s + 4], d4_idx));
        Bitboard::new([
            g0[perm[0] as usize],
            g0[perm[1] as usize],
            g0[perm[2] as usize],
            g0[perm[3] as usize],
            g1[perm[0] as usize],
            g1[perm[1] as usize],
            g1[perm[2] as usize],
            g1[perm[3] as usize],
        ])
    }

    #[test]
    fn identity_transform_is_a_no_op() {
        for action_index in 0..64u8 {
            assert_eq!(
                SymmetryHandler::remap_action_index(action_index, 0),
                action_index
            );
        }
    }

    #[test]
    #[should_panic(expected = "action_index must be 0..63")]
    fn remap_action_index_rejects_out_of_range_action_index() {
        SymmetryHandler::remap_action_index(64, 0);
    }

    #[test]
    #[should_panic(expected = "transform_index must be 0..191")]
    fn remap_action_index_rejects_out_of_range_transform_index() {
        SymmetryHandler::remap_action_index(0, 192);
    }

    #[test]
    fn round_trip_every_transform_and_action() {
        for transform_index in 0u8..192 {
            let inverse = SymmetryHandler::inverse_transform_index(transform_index);
            for action_index in 0u8..64 {
                let transformed =
                    SymmetryHandler::remap_action_index(action_index, transform_index);
                assert!(transformed < 64);
                let restored = SymmetryHandler::remap_action_index(transformed, inverse);
                assert_eq!(restored, action_index, "transform_index={transform_index}");
            }
        }
    }

    #[test]
    fn inverse_of_inverse_is_identity_transform() {
        for transform_index in 0u8..192 {
            let inverse = SymmetryHandler::inverse_transform_index(transform_index);
            assert_eq!(
                SymmetryHandler::inverse_transform_index(inverse),
                transform_index
            );
        }
    }

    #[test]
    fn remap_matches_direct_bitboard_transform() {
        // Cross-check remap_action_index (pure index arithmetic) against
        // actually transforming a single-piece bitboard the way
        // find_canonical does, for every one of the 192 transforms.
        let shape = 2u8;
        let position = 5u8;
        let action_index = shape * 16 + position;
        let bb = Bitboard::EMPTY.with_move(0, shape, position);

        for d4_idx in 0..8 {
            for (perm_idx, perm) in SHAPE_PERMS.iter().enumerate() {
                let transform_index = (d4_idx * 24 + perm_idx) as u8;
                let transformed_bb = apply_transform_to_bitboard(&bb, d4_idx, perm);

                let mut found = None;
                for (new_shape, plane) in transformed_bb.planes[0..4].iter().enumerate() {
                    if *plane != 0 {
                        found = Some((new_shape as u8, plane.trailing_zeros() as u8));
                    }
                }
                let (expected_shape, expected_position) =
                    found.expect("exactly one piece must remain after a symmetry transform");
                let expected_action_index = expected_shape * 16 + expected_position;

                assert_eq!(
                    SymmetryHandler::remap_action_index(action_index, transform_index),
                    expected_action_index,
                    "d4_idx={d4_idx} perm_idx={perm_idx}"
                );
            }
        }
    }

    #[test]
    fn golden_cases_from_contracts_fixture() {
        // Mirrors fixtures/symmetry/symmetry-v1.json's action_remap_cases in
        // quantik-core-contracts, which were generated from and cross-checked
        // against quantik-core-py. Kept here, rather than read from a sibling
        // checkout, so this crate's tests stay hermetic; a future change to
        // remap_action_index/inverse_transform_index must deliberately update
        // both this list and the shared fixture, not silently drift from it.
        let golden: [(u8, u8, u8, u8); 10] = [
            // (action_index, transform_index, expected_action_index, inverse_transform_index)
            (37, 0, 37, 0),     // identity
            (37, 24, 38, 72),   // rot90
            (37, 48, 42, 48),   // rot180
            (37, 72, 41, 24),   // rot270
            (37, 96, 38, 96),   // reflV
            (37, 120, 41, 120), // reflH
            (37, 144, 37, 144), // reflD
            (37, 168, 42, 168), // reflAD
            (37, 1, 53, 1),     // pure shape relabel (shapes 2<->3)
            (37, 25, 54, 73),   // rot90 + shape relabel
        ];
        for (action_index, transform_index, expected, expected_inverse) in golden {
            assert_eq!(
                SymmetryHandler::remap_action_index(action_index, transform_index),
                expected
            );
            assert_eq!(
                SymmetryHandler::inverse_transform_index(transform_index),
                expected_inverse
            );
        }
    }
}
