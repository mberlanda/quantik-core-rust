pub const VERSION: u8 = 1;
pub const FLAG_CANON: u8 = 1 << 1;

pub const MAX_PIECES_PER_SHAPE: u8 = 2;
pub const NUM_SHAPES: usize = 4;
pub const NUM_PLAYERS: usize = 2;
pub const BOARD_SIZE: usize = 16;
pub const NUM_PLANES: usize = 8; // 2 players * 4 shapes

pub const ROW_MASKS: [u16; 4] = [0x000F, 0x00F0, 0x0F00, 0xF000];
pub const COLUMN_MASKS: [u16; 4] = [0x1111, 0x2222, 0x4444, 0x8888];
pub const ZONE_MASKS: [u16; 4] = [0x0033, 0x00CC, 0x3300, 0xCC00];

pub const WIN_MASKS: [u16; 12] = [
    // rows
    0x000F, 0x00F0, 0x0F00, 0xF000, // columns
    0x1111, 0x2222, 0x4444, 0x8888, // zones
    0x0033, 0x00CC, 0x3300, 0xCC00,
];

pub const SHAPE_LETTERS: [char; 4] = ['A', 'B', 'C', 'D'];
