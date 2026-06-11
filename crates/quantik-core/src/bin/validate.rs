use quantik_core::bitboard::Bitboard;
use quantik_core::constants::{FLAG_CANON, VERSION};
use quantik_core::game::has_winning_line;
use quantik_core::moves::generate_legal_moves;
use quantik_core::qfen::{bb_from_qfen, bb_to_qfen};
use quantik_core::symmetry::SymmetryHandler;
use std::time::Instant;

type Fixture = (
    &'static str,
    &'static str,
    [u16; 8],
    bool,
    usize,
    &'static str,
);

fn canonical_key_bytes(bb: &Bitboard) -> [u8; 18] {
    let canon = SymmetryHandler::find_canonical(bb);
    let mut key = [0u8; 18];
    key[0] = VERSION;
    key[1] = FLAG_CANON;
    let bytes = canon.to_le_bytes();
    key[2..18].copy_from_slice(&bytes);
    key
}

fn main() {
    let fixtures: Vec<Fixture> = vec![
        (
            "empty",
            "..../..../..../....",
            [0, 0, 0, 0, 0, 0, 0, 0],
            false,
            64,
            "010200000000000000000000000000000000",
        ),
        (
            "single_corner",
            "A.../..../..../....",
            [1, 0, 0, 0, 0, 0, 0, 0],
            false,
            53,
            "010200000000000000100000000000000000",
        ),
        (
            "alternating",
            "Ab../..../..../....",
            [1, 0, 0, 0, 0, 2, 0, 0],
            false,
            50,
            "010200000000000000100000000000010000",
        ),
        (
            "winning_row",
            "AbCd/..../..../....",
            [1, 0, 4, 0, 0, 2, 0, 8],
            true,
            40,
            "010200000000000101000010100000000000",
        ),
        (
            "winning_col",
            "A.../B.../C.../D...",
            [1, 16, 256, 4096, 0, 0, 0, 0],
            true,
            0,
            "010200010010010010000000000000000000",
        ),
        (
            "midgame",
            "Ab../c.D./..../...A",
            [32769, 0, 0, 64, 0, 2, 16, 0],
            false,
            28,
            "010200000000000201800008004000000000",
        ),
        (
            "full",
            "AbcD/bCdA/cDaB/DaBc",
            [129, 18432, 32, 4616, 9216, 18, 33028, 64],
            true,
            0,
            "010200020018218084001840420000040021",
        ),
    ];

    println!("{{");
    println!("  \"language\": \"rust\",");
    println!("  \"results\": {{");
    let total = fixtures.len();

    for (idx, (name, qfen, planes, exp_win, exp_moves, exp_ck_hex)) in fixtures.iter().enumerate() {
        let bb = Bitboard::new(*planes);

        let encoded = bb_to_qfen(&bb);
        let decoded = bb_from_qfen(qfen).unwrap();
        let qfen_ok = encoded == *qfen && decoded == bb;

        let got_win = has_winning_line(&bb);
        let moves = generate_legal_moves(&bb);
        let got_moves = moves.len();

        let key = canonical_key_bytes(&bb);
        let got_ck_hex: String = key.iter().map(|b| format!("{:02x}", b)).collect();

        let iters = 10_000u64;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = SymmetryHandler::find_canonical(&bb);
        }
        let canon_ns = start.elapsed().as_nanos() as f64 / iters as f64;

        let start = Instant::now();
        for _ in 0..iters {
            let _ = generate_legal_moves(&bb);
        }
        let moves_ns = start.elapsed().as_nanos() as f64 / iters as f64;

        let start = Instant::now();
        for _ in 0..100_000u64 {
            let _ = has_winning_line(&bb);
        }
        let win_ns = start.elapsed().as_nanos() as f64 / 100_000.0;

        let comma = if idx < total - 1 { "," } else { "" };
        println!("    \"{}\": {{", name);
        println!("      \"qfen_roundtrip\": {},", qfen_ok);
        println!("      \"win_match\": {},", got_win == *exp_win);
        println!("      \"moves_match\": {},", got_moves == *exp_moves);
        println!("      \"canonical_match\": {},", got_ck_hex == *exp_ck_hex);
        println!("      \"got_moves\": {},", got_moves);
        println!("      \"got_canonical_hex\": \"{}\",", got_ck_hex);
        println!("      \"bench_canonical_ns_per_op\": {:.1},", canon_ns);
        println!("      \"bench_moves_ns_per_op\": {:.1},", moves_ns);
        println!("      \"bench_win_ns_per_op\": {:.1}", win_ns);
        println!("    }}{}", comma);
    }
    println!("  }}");
    println!("}}");
}
