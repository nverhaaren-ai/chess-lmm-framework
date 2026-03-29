"""Tests for recording module: GameRecorder, RecordingClient, board rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_lmm.mock_server import MockChessServer
from chess_lmm.recording import (
    GameRecorder,
    LlmInteractionLogger,
    RecordingClient,
    render_board,
)
from chess_lmm.types import McpError

from .conftest import INITIAL_FEN

# --- GameRecorder tests ---


class TestGameRecorder:
    def test_record_creates_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "game.jsonl"
        recorder = GameRecorder(log_file, "session-1")
        recorder.record("get_status", {}, {"game_status": "ongoing"})

        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["tool"] == "get_status"
        assert entry["session_id"] == "session-1"
        assert entry["result"]["game_status"] == "ongoing"
        assert "ts_ms" in entry
        assert entry["elapsed_ms"] == 0

    def test_record_error(self, tmp_path: Path) -> None:
        log_file = tmp_path / "game.jsonl"
        recorder = GameRecorder(log_file, "session-1")
        recorder.record(
            "make_move",
            {"move": "e4"},
            None,
            error={"error": "not_your_turn", "message": "It is Black's turn"},
        )

        entry = json.loads(log_file.read_text().strip())
        assert "error" in entry
        assert entry["error"]["error"] == "not_your_turn"
        assert "result" not in entry

    def test_record_multiple(self, tmp_path: Path) -> None:
        log_file = tmp_path / "game.jsonl"
        recorder = GameRecorder(log_file, "session-1")
        recorder.record("get_board", {}, {"fen": INITIAL_FEN})
        recorder.record("make_move", {"move": "e4"}, {"move_played": {"san": "e4"}})

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_write_marker(self, tmp_path: Path) -> None:
        log_file = tmp_path / "game.jsonl"
        recorder = GameRecorder(log_file, "session-1")
        recorder.write_marker(
            {"type": "game_start", "game_id": "game-1", "fen": INITIAL_FEN}
        )

        entry = json.loads(log_file.read_text().strip())
        assert entry["type"] == "game_start"
        assert "ts_ms" in entry

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        log_file = tmp_path / "deep" / "nested" / "game.jsonl"
        recorder = GameRecorder(log_file, "session-1")
        recorder.record("get_status", {}, {"game_status": "ongoing"})
        assert log_file.exists()

    def test_elapsed_ms(self, tmp_path: Path) -> None:
        log_file = tmp_path / "game.jsonl"
        recorder = GameRecorder(log_file, "session-1")
        recorder.record("make_move", {"move": "e4"}, {"ok": True}, elapsed_ms=42)

        entry = json.loads(log_file.read_text().strip())
        assert entry["elapsed_ms"] == 42


# --- RecordingClient tests ---


class TestRecordingClient:
    @pytest.fixture
    async def setup(
        self, tmp_path: Path
    ) -> tuple[RecordingClient, RecordingClient, Path]:
        """Set up a server with two recording clients."""
        server = MockChessServer()
        log_file = tmp_path / "game.jsonl"
        white_raw = await server.create_session()
        black_raw = await server.create_session()
        white_rec = GameRecorder(log_file, white_raw.session_id)
        black_rec = GameRecorder(log_file, black_raw.session_id)
        white = RecordingClient(white_raw, white_rec)
        black = RecordingClient(black_raw, black_rec)
        return white, black, log_file

    async def test_records_create_and_join(
        self, setup: tuple[RecordingClient, RecordingClient, Path]
    ) -> None:
        white, black, log_file = setup
        await white.create_game()
        await white.join_game("white")
        await black.join_game("black")

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 3

        entries = [json.loads(line) for line in lines]
        assert entries[0]["tool"] == "create_game"
        assert entries[1]["tool"] == "join_game"
        assert entries[2]["tool"] == "join_game"

    async def test_records_move(
        self, setup: tuple[RecordingClient, RecordingClient, Path]
    ) -> None:
        white, black, log_file = setup
        await white.create_game()
        await white.join_game("white")
        await black.join_game("black")
        await white.make_move("e4")

        lines = log_file.read_text().strip().split("\n")
        move_entry = json.loads(lines[-1])
        assert move_entry["tool"] == "make_move"
        assert move_entry["params"]["move"] == "e4"
        assert move_entry["result"]["move_played"]["san"] == "e4"

    async def test_records_error(
        self, setup: tuple[RecordingClient, RecordingClient, Path]
    ) -> None:
        white, black, log_file = setup
        await white.create_game()
        await white.join_game("white")
        await black.join_game("black")

        with pytest.raises(McpError):
            await black.make_move("e5")  # Not black's turn

        lines = log_file.read_text().strip().split("\n")
        error_entry = json.loads(lines[-1])
        assert "error" in error_entry
        assert error_entry["error"]["error"] == "not_your_turn"

    async def test_session_id_passthrough(
        self, setup: tuple[RecordingClient, RecordingClient, Path]
    ) -> None:
        white, black, _ = setup
        assert white.session_id.startswith("session-")

    async def test_records_query_tools(
        self, setup: tuple[RecordingClient, RecordingClient, Path]
    ) -> None:
        white, black, log_file = setup
        await white.create_game()
        await white.join_game("white")
        await black.join_game("black")

        await white.get_board()
        await white.get_status()
        await white.get_legal_moves(square="e2")
        await white.get_history()

        lines = log_file.read_text().strip().split("\n")
        tools = [json.loads(line)["tool"] for line in lines]
        assert "get_board" in tools
        assert "get_status" in tools
        assert "get_legal_moves" in tools
        assert "get_history" in tools

    async def test_records_draw_flow(
        self, setup: tuple[RecordingClient, RecordingClient, Path]
    ) -> None:
        white, black, log_file = setup
        await white.create_game()
        await white.join_game("white")
        await black.join_game("black")

        await white.offer_draw()
        await black.decline_draw()

        lines = log_file.read_text().strip().split("\n")
        tools = [json.loads(line)["tool"] for line in lines]
        assert "offer_draw" in tools
        assert "decline_draw" in tools

    async def test_records_resign(
        self, setup: tuple[RecordingClient, RecordingClient, Path]
    ) -> None:
        white, black, log_file = setup
        await white.create_game()
        await white.join_game("white")
        await black.join_game("black")
        await white.resign()

        lines = log_file.read_text().strip().split("\n")
        resign_entry = json.loads(lines[-1])
        assert resign_entry["tool"] == "resign"
        assert resign_entry["result"]["game_status"] == "resigned"

    async def test_records_done(
        self, setup: tuple[RecordingClient, RecordingClient, Path]
    ) -> None:
        white, black, log_file = setup
        await white.create_game()
        await white.join_game("white")
        await black.join_game("black")
        await white.resign()
        await white.done()

        lines = log_file.read_text().strip().split("\n")
        done_entry = json.loads(lines[-1])
        assert done_entry["tool"] == "done"

    async def test_records_preview_move(
        self, setup: tuple[RecordingClient, RecordingClient, Path]
    ) -> None:
        white, black, log_file = setup
        await white.create_game()
        await white.join_game("white")
        await black.join_game("black")

        result = await white.preview_move("e4")

        lines = log_file.read_text().strip().split("\n")
        preview_entry = json.loads(lines[-1])
        assert preview_entry["tool"] == "preview_move"
        assert preview_entry["params"]["move"] == "e4"
        assert preview_entry["result"]["move"]["san"] == "e4"
        assert "fen" in preview_entry["result"]
        assert result["move"]["san"] == "e4"


# --- LlmInteractionLogger tests ---


class TestLlmInteractionLogger:
    def test_log_entry(self, tmp_path: Path) -> None:
        log_file = tmp_path / "llm.jsonl"
        logger = LlmInteractionLogger(log_file)
        logger.log(
            {
                "type": "api_call",
                "model": "claude-sonnet-4-5-20250514",
                "messages": [{"role": "user", "content": "Make a move"}],
                "response": {"role": "assistant", "content": "e4"},
            }
        )

        entry = json.loads(log_file.read_text().strip())
        assert entry["type"] == "api_call"
        assert "ts_ms" in entry


# --- Board rendering tests ---


class TestRenderBoard:
    def test_initial_position(self) -> None:
        board = render_board(INITIAL_FEN)
        lines = board.split("\n")
        assert len(lines) == 9  # 8 ranks + file labels
        assert "8" in lines[0]
        assert "r  n  b  q  k  b  n  r" in lines[0]  # Black pieces
        assert "1" in lines[7]
        assert "R  N  B  Q  K  B  N  R" in lines[7]  # White pieces
        assert "a  b  c  d  e  f  g  h" in lines[8]

    def test_empty_board(self) -> None:
        fen = "8/8/8/8/8/8/8/8 w - - 0 1"
        board = render_board(fen)
        # All squares should be dots
        for line in board.split("\n")[:8]:
            assert "." in line

    def test_custom_position(self) -> None:
        # Scholar's mate position
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        board = render_board(fen)
        # Black queen on h4
        lines = board.split("\n")
        assert "q" in lines[4]  # rank 4 is line index 4

    def test_rank_numbers(self) -> None:
        board = render_board(INITIAL_FEN)
        lines = board.split("\n")
        for i in range(8):
            expected_rank = str(8 - i)
            assert lines[i].strip().startswith(expected_rank)

    def test_empty_fen_string(self) -> None:
        """render_board should handle empty FEN gracefully."""
        assert render_board("") == "(no board)"

    def test_fen_no_space(self) -> None:
        """render_board should handle FEN with no space separator."""
        assert render_board("garbage") == "(no board)"
