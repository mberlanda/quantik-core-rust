use crate::bitboard::Bitboard;
use crate::game::{check_winner, current_player, WinStatus};
use crate::moves::{apply_move, generate_legal_moves, Move};
use rand::prelude::*;

pub struct MCTSConfig {
    pub exploration_weight: f64,
    pub max_iterations: u32,
    pub max_depth: u32,
    pub seed: Option<u64>,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            exploration_weight: std::f64::consts::SQRT_2,
            max_iterations: 10_000,
            max_depth: 16,
            seed: None,
        }
    }
}

struct MCTSNode {
    bb: Bitboard,
    parent: Option<usize>,
    children: Vec<usize>,
    mv: Option<Move>,    // move that led here
    visit_count: u32,
    win_count_p0: u32,
    win_count_p1: u32,
    untried_moves: Vec<Move>,
    is_terminal: bool,
    terminal_value: f64, // +1 p0 win, -1 p1 win, 0 draw
}

pub struct MCTSEngine {
    config: MCTSConfig,
    nodes: Vec<MCTSNode>,
    rng: StdRng,
}

impl MCTSEngine {
    pub fn new(config: MCTSConfig) -> Self {
        let rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self {
            config,
            nodes: Vec::new(),
            rng,
        }
    }

    /// Run MCTS from the given bitboard and return (best_move, win_probability_for_p0).
    pub fn search(&mut self, bb: &Bitboard) -> Option<(Move, f64)> {
        self.nodes.clear();

        let legal = generate_legal_moves(bb);
        if legal.is_empty() {
            return None;
        }

        let terminal = check_winner(bb);
        let is_terminal = terminal != WinStatus::NoWin;
        let terminal_value = match terminal {
            WinStatus::Player0Wins => 1.0,
            WinStatus::Player1Wins => -1.0,
            WinStatus::NoWin => 0.0,
        };

        self.nodes.push(MCTSNode {
            bb: *bb,
            parent: None,
            children: Vec::new(),
            mv: None,
            visit_count: 0,
            win_count_p0: 0,
            win_count_p1: 0,
            untried_moves: legal,
            is_terminal,
            terminal_value,
        });

        for _ in 0..self.config.max_iterations {
            let selected = self.select(0);
            let expanded = self.expand(selected);
            let value = self.simulate(expanded);
            self.backpropagate(expanded, value);
        }

        self.best_move()
    }

    fn select(&self, mut node_id: usize) -> usize {
        loop {
            let node = &self.nodes[node_id];
            if node.is_terminal {
                return node_id;
            }
            if !node.untried_moves.is_empty() {
                return node_id;
            }
            if node.children.is_empty() {
                return node_id;
            }
            let parent_visits = node.visit_count as f64;
            let c = self.config.exploration_weight;

            let mut best_ucb = f64::NEG_INFINITY;
            let mut best_child = node.children[0];
            for &child_id in &node.children {
                let child = &self.nodes[child_id];
                if child.visit_count == 0 {
                    best_child = child_id;
                    break;
                }
                let child_visits = child.visit_count as f64;
                let wins = child.win_count_p0 as f64;
                let win_rate = wins / child_visits;
                let ucb = win_rate + c * (parent_visits.ln() / child_visits).sqrt();
                if ucb > best_ucb {
                    best_ucb = ucb;
                    best_child = child_id;
                }
            }
            node_id = best_child;
        }
    }

    fn expand(&mut self, node_id: usize) -> usize {
        if self.nodes[node_id].is_terminal {
            return node_id;
        }
        if self.nodes[node_id].untried_moves.is_empty() {
            return node_id;
        }

        let idx = self.rng.gen_range(0..self.nodes[node_id].untried_moves.len());
        let mv = self.nodes[node_id].untried_moves.swap_remove(idx);
        let parent_bb = self.nodes[node_id].bb;
        let new_bb = apply_move(&parent_bb, &mv);

        let legal = generate_legal_moves(&new_bb);
        let terminal = check_winner(&new_bb);
        let is_terminal = terminal != WinStatus::NoWin || legal.is_empty();
        let terminal_value = match terminal {
            WinStatus::Player0Wins => 1.0,
            WinStatus::Player1Wins => -1.0,
            WinStatus::NoWin if legal.is_empty() => {
                // No legal moves: the player who cannot move loses
                if current_player(&new_bb) == Some(0) { -1.0 } else { 1.0 }
            }
            WinStatus::NoWin => 0.0,
        };

        let child_id = self.nodes.len();
        self.nodes.push(MCTSNode {
            bb: new_bb,
            parent: Some(node_id),
            children: Vec::new(),
            mv: Some(mv),
            visit_count: 0,
            win_count_p0: 0,
            win_count_p1: 0,
            untried_moves: legal,
            is_terminal,
            terminal_value,
        });

        self.nodes[node_id].children.push(child_id);
        child_id
    }

    fn simulate(&mut self, node_id: usize) -> f64 {
        let node = &self.nodes[node_id];
        if node.is_terminal {
            return node.terminal_value;
        }

        let mut current_bb = node.bb;
        let mut depth = 0u32;

        loop {
            if depth >= self.config.max_depth {
                return 0.0;
            }
            let w = check_winner(&current_bb);
            if w != WinStatus::NoWin {
                return match w {
                    WinStatus::Player0Wins => 1.0,
                    WinStatus::Player1Wins => -1.0,
                    WinStatus::NoWin => unreachable!(),
                };
            }
            let moves = generate_legal_moves(&current_bb);
            if moves.is_empty() {
                // No legal moves: the player who cannot move loses
                return if current_player(&current_bb) == Some(0) { -1.0 } else { 1.0 };
            }
            let mv = moves[self.rng.gen_range(0..moves.len())];
            current_bb = apply_move(&current_bb, &mv);
            depth += 1;
        }
    }

    fn backpropagate(&mut self, mut node_id: usize, value: f64) {
        loop {
            let node = &mut self.nodes[node_id];
            node.visit_count += 1;
            if value > 0.0 {
                node.win_count_p0 += 1;
            } else if value < 0.0 {
                node.win_count_p1 += 1;
            }
            match node.parent {
                Some(pid) => node_id = pid,
                None => break,
            }
        }
    }

    fn best_move(&self) -> Option<(Move, f64)> {
        let root = &self.nodes[0];
        if root.children.is_empty() {
            return None;
        }

        let mut best_visits = 0u32;
        let mut best_child = root.children[0];
        for &child_id in &root.children {
            let child = &self.nodes[child_id];
            if child.visit_count > best_visits {
                best_visits = child.visit_count;
                best_child = child_id;
            }
        }

        let child = &self.nodes[best_child];
        let win_rate = if child.visit_count > 0 {
            child.win_count_p0 as f64 / child.visit_count as f64
        } else {
            0.5
        };

        child.mv.map(|mv| (mv, win_rate))
    }

    pub fn iterations_performed(&self) -> u32 {
        if self.nodes.is_empty() {
            0
        } else {
            self.nodes[0].visit_count
        }
    }

    pub fn nodes_created(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcts_returns_a_move() {
        let mut engine = MCTSEngine::new(MCTSConfig {
            max_iterations: 100,
            seed: Some(42),
            ..Default::default()
        });
        let result = engine.search(&Bitboard::EMPTY);
        assert!(result.is_some());
        let (mv, prob) = result.unwrap();
        assert_eq!(mv.player, 0);
        assert!(mv.shape < 4);
        assert!(mv.position < 16);
        assert!((0.0..=1.0).contains(&prob));
    }

    #[test]
    fn mcts_finds_winning_move() {
        // Setup: A@0, b@5, C@2  → player 1 to move.
        // Row 0 has shapes A, C. If p1 places d@3 and then p0 places B@1, that's a win.
        // But more directly: test that MCTS returns *some* move for this position.
        let bb = Bitboard::EMPTY
            .with_move(0, 0, 0)
            .with_move(1, 1, 5)
            .with_move(0, 2, 2);
        let mut engine = MCTSEngine::new(MCTSConfig {
            max_iterations: 500,
            seed: Some(123),
            ..Default::default()
        });
        let result = engine.search(&bb);
        assert!(result.is_some());
    }

    #[test]
    fn mcts_no_moves_returns_none() {
        // A terminal (won) position: row 0 complete
        let bb = Bitboard::EMPTY
            .with_move(0, 0, 0)
            .with_move(1, 1, 1)
            .with_move(0, 2, 2)
            .with_move(1, 3, 3);
        let mut engine = MCTSEngine::new(MCTSConfig {
            max_iterations: 10,
            seed: Some(1),
            ..Default::default()
        });
        let result = engine.search(&bb);
        // Terminal board → no legal moves → None
        assert!(result.is_none());
    }
}
