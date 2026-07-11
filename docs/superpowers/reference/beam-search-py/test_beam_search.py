"""Tests for the parametrizable beam search engine."""

from typing import List

import pytest

from quantik_core import State, apply_move, generate_legal_moves
from quantik_core.beam_search import BeamSearchConfig, BeamSearchEngine, BeamLeaf
from quantik_core.mcts import MCTSEngine, MCTSConfig
from quantik_core.memory.compact_tree import (
    CompactGameTree,
    NODE_FLAG_TERMINAL,
    NODE_FLAG_WINNING_P0,
    NODE_FLAG_WINNING_P1,
)
from quantik_core.game_utils import check_game_winner, WinStatus

EMPTY_QFEN = "..../..../..../...."


class TestBeamSearchConfig:
    """Test beam search configuration defaults, custom values, validation."""

    def test_default_config(self):
        config = BeamSearchConfig()
        assert config.beam_width == 64
        assert config.max_depth == 16
        assert config.rollouts_per_candidate == 8
        assert config.random_seed is None
        assert config.evaluator is None
        assert config.initial_tree_capacity == 4096

    def test_custom_config(self):
        def evaluator(state: State) -> float:
            return 0.0

        config = BeamSearchConfig(
            beam_width=8,
            max_depth=4,
            rollouts_per_candidate=2,
            random_seed=7,
            evaluator=evaluator,
            initial_tree_capacity=128,
        )
        assert config.beam_width == 8
        assert config.max_depth == 4
        assert config.rollouts_per_candidate == 2
        assert config.random_seed == 7
        assert config.evaluator is evaluator
        assert config.initial_tree_capacity == 128

    def test_invalid_beam_width(self):
        with pytest.raises(ValueError):
            BeamSearchEngine(BeamSearchConfig(beam_width=0))

    def test_invalid_max_depth_too_low(self):
        with pytest.raises(ValueError):
            BeamSearchEngine(BeamSearchConfig(max_depth=0))

    def test_invalid_max_depth_too_high(self):
        with pytest.raises(ValueError):
            BeamSearchEngine(BeamSearchConfig(max_depth=17))

    def test_invalid_rollouts_per_candidate(self):
        with pytest.raises(ValueError):
            BeamSearchEngine(BeamSearchConfig(rollouts_per_candidate=0))


class TestBeamSearchEngine:
    """Test core beam search behavior."""

    def test_immediate_win_found(self):
        """Near-win fixture: best_leaf is the winning terminal at depth 1."""
        # P0: A at (0,0), B at (0,1); P1: c at (0,2), a at (3,3).
        # Row 0 has shapes A, B, C — P0 can win by placing D at (0,3), and
        # this is the *only* one-move win available from this position.
        config = BeamSearchConfig(
            beam_width=4, max_depth=2, rollouts_per_candidate=2, random_seed=1
        )
        engine = BeamSearchEngine(config)

        state = State.from_qfen("ABc./..../..../...a")
        result = engine.search(state)

        assert result.best_leaf is not None
        assert result.best_leaf.is_terminal
        assert result.best_leaf.depth == 1
        assert result.best_leaf.value == pytest.approx(1.0)
        assert len(result.best_leaf.moves) == 1
        winning_move = result.best_leaf.moves[0]
        assert winning_move.player == 0
        assert winning_move.shape == 3  # D
        assert winning_move.position == 3  # (0,3)

    def test_full_game_reachability(self):
        """From the empty board, a small seeded beam reaches true terminals."""
        config = BeamSearchConfig(
            beam_width=4, max_depth=16, rollouts_per_candidate=2, random_seed=42
        )
        engine = BeamSearchEngine(config)

        state = State.from_qfen(EMPTY_QFEN)
        result = engine.search(state)

        assert result.reached_terminal is True
        assert len(result.terminal_leaves) > 0

        for leaf in result.terminal_leaves:
            bb = state.bb
            for move in leaf.moves:
                bb = apply_move(bb, move)
            winner = check_game_winner(bb)
            if winner == WinStatus.NO_WIN:
                # Mover-blocked terminal: the player to move has no legal moves.
                from quantik_core import generate_legal_moves

                current_player, moves_by_shape = generate_legal_moves(bb)
                all_moves = [m for ms in moves_by_shape.values() for m in ms]
                assert not all_moves
            else:
                assert winner != WinStatus.NO_WIN

    def test_symmetry_dedup_depth_one(self):
        """From the empty board, 64 depth-1 moves collapse to 3 canonical states."""
        config = BeamSearchConfig(
            beam_width=64, max_depth=1, rollouts_per_candidate=1, random_seed=1
        )
        engine = BeamSearchEngine(config)

        state = State.from_qfen(EMPTY_QFEN)
        result = engine.search(state)

        assert result.stats["candidates_generated"] == 64
        # 64 candidates - 3 unique canonical states = 61 deduped
        assert result.stats["candidates_deduped"] == 61
        assert result.stats["nodes_inserted"] == 3

    def test_memory_bound(self):
        """nodes_inserted stays within the beam_width * depth + terminals bound."""
        state = State.from_qfen(EMPTY_QFEN)

        config_small = BeamSearchConfig(
            beam_width=2, max_depth=4, rollouts_per_candidate=1, random_seed=3
        )
        engine_small = BeamSearchEngine(config_small)
        result_small = engine_small.search(state)

        config_large = BeamSearchConfig(
            beam_width=16, max_depth=4, rollouts_per_candidate=1, random_seed=3
        )
        engine_large = BeamSearchEngine(config_large)
        result_large = engine_large.search(state)

        terminal_count_small = len(result_small.terminal_leaves)
        terminal_count_large = len(result_large.terminal_leaves)

        assert result_small.stats["nodes_inserted"] <= (
            config_small.beam_width * result_small.max_depth_reached
            + terminal_count_small
        )
        assert result_large.stats["nodes_inserted"] <= (
            config_large.beam_width * result_large.max_depth_reached
            + terminal_count_large
        )

    def test_determinism_same_seed(self):
        """Same seed produces an identical result."""
        state = State.from_qfen(EMPTY_QFEN)

        config1 = BeamSearchConfig(
            beam_width=4, max_depth=6, rollouts_per_candidate=2, random_seed=99
        )
        result1 = BeamSearchEngine(config1).search(state)

        config2 = BeamSearchConfig(
            beam_width=4, max_depth=6, rollouts_per_candidate=2, random_seed=99
        )
        result2 = BeamSearchEngine(config2).search(state)

        assert result1.stats == result2.stats
        assert result1.max_depth_reached == result2.max_depth_reached
        assert result1.reached_terminal == result2.reached_terminal
        assert [leaf.moves for leaf in result1.terminal_leaves] == [
            leaf.moves for leaf in result2.terminal_leaves
        ]
        assert result1.best_leaf is not None and result2.best_leaf is not None
        assert result1.best_leaf.moves == result2.best_leaf.moves
        assert result1.best_leaf.value == pytest.approx(result2.best_leaf.value)

    def test_determinism_different_seed_may_differ(self):
        """Different seeds are allowed to (and typically do) diverge."""
        state = State.from_qfen(EMPTY_QFEN)

        config1 = BeamSearchConfig(
            beam_width=3, max_depth=6, rollouts_per_candidate=3, random_seed=1
        )
        result1 = BeamSearchEngine(config1).search(state)

        config2 = BeamSearchConfig(
            beam_width=3, max_depth=6, rollouts_per_candidate=3, random_seed=2
        )
        result2 = BeamSearchEngine(config2).search(state)

        # Spec only guarantees same-seed determinism; different seeds are
        # merely allowed (not required) to diverge. Just assert both runs
        # complete and produce well-formed, independently valid results.
        assert result1.best_leaf is not None
        assert result2.best_leaf is not None
        assert result1.stats["evaluations"] > 0
        assert result2.stats["evaluations"] > 0

    def test_pluggable_evaluator_is_used(self):
        """A custom evaluator callable is invoked and biases the beam."""
        calls: List[State] = []

        def evaluator(state: State) -> float:
            calls.append(state)
            return 1.0  # always favor player 0

        config = BeamSearchConfig(
            beam_width=1, max_depth=2, evaluator=evaluator, random_seed=5
        )
        engine = BeamSearchEngine(config)

        state = State.from_qfen(EMPTY_QFEN)
        engine.search(state)

        assert len(calls) > 0

    def test_evaluator_clamping(self):
        """Evaluator values outside [-1, 1] are clamped."""

        def evaluator(state: State) -> float:
            return 5.0

        config = BeamSearchConfig(
            beam_width=2, max_depth=1, evaluator=evaluator, random_seed=1
        )
        engine = BeamSearchEngine(config)

        state = State.from_qfen(EMPTY_QFEN)
        result = engine.search(state)

        for node_id in engine.tree.get_children(engine.tree.root_id):
            node = engine.tree.get_node(node_id)
            if not (node.flags & NODE_FLAG_TERMINAL):
                assert float(node.best_value) <= 1.0

        assert result is not None

    def test_adversarial_perspective_p1_winning_reply(self):
        """When P1 is to move at depth 1 and has a winning reply, beam keeps it."""
        # P0 has placed A at (0,0). P1 to move.
        # Give P1 a winning reply: place B, C, D such that the row completes.
        # Row 0: A . . . ; P1 places b at 1, c at 2, then can win with d at 3
        # is too many moves; instead build a position where p1 has an
        # immediate one-move win: row 0 has A B C already (P0, P0, P1) and
        # it's P1's turn - wait shape D must be placed by whoever completes.
        # Use: P0 A at 0, B at 1; P1 c at 2. It's P1's turn (counts 2 vs 1).
        state = State.from_qfen("ABc./..../..../....")

        config = BeamSearchConfig(
            beam_width=2, max_depth=1, rollouts_per_candidate=1, random_seed=1
        )
        engine = BeamSearchEngine(config)
        result = engine.search(state)

        # P1 wins by placing shape D (index 3) at position 3.
        winning_moves = [
            leaf for leaf in result.terminal_leaves if leaf.value == pytest.approx(-1.0)
        ]
        assert len(winning_moves) > 0
        assert any(
            leaf.moves[0].player == 1
            and leaf.moves[0].shape == 3
            and leaf.moves[0].position == 3
            for leaf in winning_moves
        )

    def test_pruning_uses_mover_relative_score(self):
        """_score_and_prune must rank candidates mover-relative, not P0-fixed.

        Regression test: with only depth-1 TERMINAL wins exercised elsewhere,
        a mutant that drops the P1 sign flip (`score = raw_value` instead of
        `score = raw_value if mover == 0 else -raw_value`) still passes every
        other test in this file. This fixture forces every depth-1 reply to
        be NON-terminal, so the survivor must come from the mover-relative
        ranking in `_score_and_prune`, not from terminal-leaf handling.
        """
        # P0 has already placed A at (0,0); it is P1's turn. With only 2
        # pieces on the board after any reply, no line can be completed, so
        # every depth-1 candidate is non-terminal.
        root_state = State.from_qfen("A.../..../..../....")
        root_player, moves_by_shape = generate_legal_moves(root_state.bb)
        assert root_player == 1
        all_moves = [m for ms in moves_by_shape.values() for m in ms]
        assert len(all_moves) > 1

        # Assign each resulting state a distinct, known P0-perspective value.
        value_by_bb = {}
        for i, move in enumerate(all_moves):
            new_bb = apply_move(root_state.bb, move)
            value_by_bb[new_bb] = -1.0 + (2.0 * i) / len(all_moves)

        min_value = min(value_by_bb.values())
        max_value = max(value_by_bb.values())
        assert min_value != max_value  # sanity: evaluator actually discriminates

        def evaluator(state: State) -> float:
            return value_by_bb[state.bb]

        config = BeamSearchConfig(
            beam_width=1, max_depth=1, evaluator=evaluator, random_seed=1
        )
        engine = BeamSearchEngine(config)
        result = engine.search(root_state)

        # P1 is the mover; P1 wants the most negative P0-perspective value.
        # A correct mover-relative score keeps exactly that survivor. A
        # P0-fixed (unsigned) score would instead keep the candidate with
        # max_value, which is what this test guards against.
        assert result.best_leaf is not None
        assert not result.best_leaf.is_terminal
        assert result.best_leaf.value == pytest.approx(min_value)
        assert result.best_leaf.value != pytest.approx(max_value)

    def test_shared_tree_integration(self):
        """Passing an existing CompactGameTree writes terminal data into it."""
        mcts_config = MCTSConfig(random_seed=1)
        mcts_engine = MCTSEngine(mcts_config)

        config = BeamSearchConfig(
            beam_width=4, max_depth=2, rollouts_per_candidate=1, random_seed=1
        )
        engine = BeamSearchEngine(config, tree=mcts_engine.tree)
        assert engine.tree is mcts_engine.tree

        state = State.from_qfen("ABc./d.../..../....")
        result = engine.search(state)

        assert result.best_leaf is not None
        found_terminal_flag = False
        for node_id in range(engine.tree.storage.node_count):
            node = engine.tree.get_node(node_id)
            if node.flags & NODE_FLAG_TERMINAL:
                found_terminal_flag = True
                assert (node.flags & NODE_FLAG_WINNING_P0) or (
                    node.flags & NODE_FLAG_WINNING_P1
                )
        assert found_terminal_flag
        assert (
            engine.tree.storage.node_count
            <= config.beam_width * 2 + len(result.terminal_leaves) + 1
        )  # +1 for the root

    def test_shared_tree_with_fresh_compact_tree(self):
        """A fresh CompactGameTree instance can also be shared."""
        tree = CompactGameTree(initial_capacity=64)
        config = BeamSearchConfig(
            beam_width=2, max_depth=1, rollouts_per_candidate=1, random_seed=1
        )
        engine = BeamSearchEngine(config, tree=tree)

        state = State.from_qfen(EMPTY_QFEN)
        result = engine.search(state)

        assert engine.tree is tree
        assert result.stats["nodes_inserted"] > 0

    def test_root_already_terminal_raises(self):
        """Root state with a winning line raises ValueError."""
        config = BeamSearchConfig(random_seed=1)
        engine = BeamSearchEngine(config)

        state = State.from_qfen("ABCD/..../..../....")
        with pytest.raises(ValueError):
            engine.search(state)

    def test_root_no_legal_moves_raises(self):
        """Root state with no legal moves raises ValueError."""
        config = BeamSearchConfig(random_seed=1)
        engine = BeamSearchEngine(config)

        state = State.from_qfen(EMPTY_QFEN)
        from unittest.mock import patch

        with patch(
            "quantik_core.beam_search.generate_legal_moves", return_value=(0, {})
        ):
            with pytest.raises(ValueError):
                engine.search(state)

    def test_get_statistics(self):
        """get_statistics mirrors MCTSEngine's tree-delegated statistics."""
        config = BeamSearchConfig(
            beam_width=4, max_depth=2, rollouts_per_candidate=1, random_seed=1
        )
        engine = BeamSearchEngine(config)

        state = State.from_qfen(EMPTY_QFEN)
        engine.search(state)

        stats = engine.get_statistics()
        assert stats["nodes_created"] > 0
        assert stats["memory_usage"] > 0
        assert "tree_stats" in stats

    def test_beam_leaf_fields(self):
        """BeamLeaf exposes the documented fields."""
        leaf = BeamLeaf(moves=(), value=1.0, depth=0, is_terminal=True)
        assert leaf.moves == ()
        assert leaf.value == 1.0
        assert leaf.depth == 0
        assert leaf.is_terminal is True

    def test_stalemate_frontier_entry_marked_terminal(self):
        """A frontier state with no legal moves is terminal (mover loses)."""
        config = BeamSearchConfig(
            beam_width=2, max_depth=2, rollouts_per_candidate=1, random_seed=1
        )
        engine = BeamSearchEngine(config)

        state = State.from_qfen(EMPTY_QFEN)

        from unittest.mock import patch

        call_count = {"n": 0}
        real_generate = None

        import quantik_core.beam_search as beam_search_module

        real_generate = beam_search_module.generate_legal_moves

        def fake_generate(bb, player_id=None):
            call_count["n"] += 1
            if call_count["n"] == 2:
                # Force the second call (root expansion happened already via
                # validation, so this hits the first frontier-entry expansion)
                # to look like a stalemate.
                current_player, _ = real_generate(bb, player_id)
                return current_player, {0: [], 1: [], 2: [], 3: []}
            return real_generate(bb, player_id)

        with patch(
            "quantik_core.beam_search.generate_legal_moves", side_effect=fake_generate
        ):
            result = engine.search(state)

        assert any(leaf.depth == 0 for leaf in result.terminal_leaves)


class TestBeamSearchResultRanking:
    """Test result ranking/best_leaf selection semantics."""

    def test_best_leaf_prefers_root_player_perspective(self):
        """best_leaf is ranked from the root player's perspective, not P0-fixed."""
        # P1 to move; give P1 an immediate winning reply among the candidates.
        state = State.from_qfen("ABc./..../..../....")
        config = BeamSearchConfig(
            beam_width=4, max_depth=1, rollouts_per_candidate=1, random_seed=1
        )
        engine = BeamSearchEngine(config)

        result = engine.search(state)

        assert result.best_leaf is not None
        assert result.best_leaf.value == pytest.approx(
            -1.0
        )  # P1 (mover) wins -> P0-perspective -1

    def test_best_leaf_none_only_when_no_leaves(self):
        """best_leaf is derived from collected leaves; sanity check the invariant."""
        config = BeamSearchConfig(
            beam_width=1, max_depth=1, rollouts_per_candidate=1, random_seed=1
        )
        engine = BeamSearchEngine(config)
        state = State.from_qfen(EMPTY_QFEN)
        result = engine.search(state)
        assert result.best_leaf is not None
