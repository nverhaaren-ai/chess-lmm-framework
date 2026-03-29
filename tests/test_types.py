"""Tests for chess_lmm.types — response types and McpError."""

from __future__ import annotations

import pytest

from chess_lmm.types import (
    BoardResult,
    CanClaimDraw,
    CreateGameResult,
    DeclineDrawResult,
    DoneResult,
    ExportResult,
    GameStatus,
    HistoryEntry,
    HistoryMoveHalf,
    HistoryResult,
    JoinGameResult,
    LegalMovesResult,
    MakeMoveResult,
    McpError,
    McpErrorContent,
    MessagesEntry,
    MessagesResult,
    MoveNotation,
    OfferDrawResult,
    PreviewMoveResult,
    SendMessageResult,
)


class TestMcpError:
    """Tests for the McpError exception class."""

    def test_basic_construction(self) -> None:
        err = McpError("illegal_move", "Pawn on e2 cannot reach e5")
        assert err.code == "illegal_move"
        assert err.message == "Pawn on e2 cannot reach e5"
        assert err.detail is None
        assert str(err) == "illegal_move: Pawn on e2 cannot reach e5"

    def test_with_detail(self) -> None:
        err = McpError(
            "internal_error",
            "An unexpected error occurred.",
            detail="IndexError at mock_server.py:42",
        )
        assert err.code == "internal_error"
        assert err.detail == "IndexError at mock_server.py:42"

    def test_to_dict_without_detail(self) -> None:
        err = McpError("not_your_turn", "It is Black's turn")
        d = err.to_dict()
        assert d == {"error": "not_your_turn", "message": "It is Black's turn"}
        assert "detail" not in d

    def test_to_dict_with_detail(self) -> None:
        err = McpError(
            "internal_error",
            "An unexpected error occurred.",
            detail="traceback...",
        )
        d = err.to_dict()
        assert d == {
            "error": "internal_error",
            "message": "An unexpected error occurred.",
            "detail": "traceback...",
        }

    def test_is_exception(self) -> None:
        err = McpError("wrong_state", "Server is not in no_game state")
        with pytest.raises(McpError) as exc_info:
            raise err
        assert exc_info.value.code == "wrong_state"


class TestResponseTypedDicts:
    """Verify that TypedDict response types can be constructed correctly.

    These are structural tests — TypedDicts are runtime dicts, so we verify
    that the expected keys are present and values have correct types.
    """

    def test_create_game_result(self) -> None:
        result: CreateGameResult = {
            "game_id": "game-1",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "game_status": "awaiting_players",
            "time_control": None,
        }
        assert result["game_status"] == "awaiting_players"
        assert result["time_control"] is None

    def test_join_game_result(self) -> None:
        result: JoinGameResult = {
            "assigned_color": "white",
            "game_status": "awaiting_players",
        }
        assert result["assigned_color"] == "white"

    def test_board_result(self) -> None:
        result: BoardResult = {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        }
        assert "fen" in result

    def test_game_status_full(self) -> None:
        result: GameStatus = {
            "game_id": "game-1",
            "server_state": "ongoing",
            "game_status": "ongoing",
            "turn": "white",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "fullmove_number": 1,
            "halfmove_clock": 0,
            "is_check": False,
            "can_claim_draw": {"fifty_move": False, "repetition": False},
            "insufficient_material": False,
            "draw_offered": False,
            "last_move": {"san": "e4", "lan": "e2e4"},
            "result": None,
            "termination_reason": None,
            "clock": None,
        }
        assert result["server_state"] == "ongoing"
        assert result["can_claim_draw"]["fifty_move"] is False

    def test_game_status_game_over(self) -> None:
        result: GameStatus = {
            "game_id": "game-1",
            "server_state": "game_over",
            "game_status": "checkmate",
            "turn": "black",
            "fen": "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
            "fullmove_number": 3,
            "halfmove_clock": 1,
            "is_check": True,
            "can_claim_draw": {"fifty_move": False, "repetition": False},
            "insufficient_material": False,
            "draw_offered": False,
            "last_move": {"san": "Qh4#", "lan": "d8h4"},
            "result": "0-1",
            "termination_reason": "Checkmate \u2014 Black wins",
            "clock": None,
        }
        assert result["result"] == "0-1"
        assert result["game_status"] == "checkmate"

    def test_legal_moves_result(self) -> None:
        result: LegalMovesResult = {
            "moves": [
                {"san": "e4", "lan": "e2e4"},
                {"san": "d4", "lan": "d2d4"},
            ],
            "count": 2,
        }
        assert result["count"] == 2
        assert len(result["moves"]) == 2

    def test_history_result(self) -> None:
        white_half: HistoryMoveHalf = {
            "san": "e4",
            "lan": "e2e4",
            "clock_ms": None,
        }
        black_half: HistoryMoveHalf = {
            "san": "e5",
            "lan": "e7e5",
            "clock_ms": None,
        }
        entry: HistoryEntry = {
            "move_number": 1,
            "white": white_half,
            "black": black_half,
        }
        result: HistoryResult = {
            "moves": [entry],
            "total_half_moves": 2,
        }
        assert result["total_half_moves"] == 2

    def test_history_result_incomplete_move(self) -> None:
        """Last entry may have black=None when White just moved."""
        entry: HistoryEntry = {
            "move_number": 1,
            "white": {"san": "e4", "lan": "e2e4"},
            "black": None,
        }
        result: HistoryResult = {
            "moves": [entry],
            "total_half_moves": 1,
        }
        assert result["moves"][0]["black"] is None

    def test_messages_result(self) -> None:
        msg: MessagesEntry = {
            "from_": "server",
            "text": "White offers a draw",
            "move_number": 5,
        }
        result: MessagesResult = {"messages": [msg]}
        assert len(result["messages"]) == 1

    def test_make_move_result(self) -> None:
        result: MakeMoveResult = {
            "move_played": {"san": "e4", "lan": "e2e4"},
            "game_id": "game-1",
            "server_state": "ongoing",
            "game_status": "ongoing",
            "turn": "black",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "fullmove_number": 1,
            "halfmove_clock": 0,
            "is_check": False,
            "can_claim_draw": {"fifty_move": False, "repetition": False},
            "insufficient_material": False,
            "draw_offered": False,
            "last_move": {"san": "e4", "lan": "e2e4"},
            "result": None,
            "termination_reason": None,
            "clock": None,
        }
        assert result["move_played"]["san"] == "e4"

    def test_offer_draw_result(self) -> None:
        result: OfferDrawResult = {"offered": True}
        assert result["offered"] is True

    def test_decline_draw_result(self) -> None:
        result: DeclineDrawResult = {"declined": True}
        assert result["declined"] is True

    def test_send_message_result(self) -> None:
        result: SendMessageResult = {"sent": True}
        assert result["sent"] is True

    def test_export_result(self) -> None:
        result: ExportResult = {"content": '[Event "?"]\n1. e4 e5 *'}
        assert "e4" in result["content"]

    def test_done_result(self) -> None:
        result: DoneResult = {"acknowledged": True, "clients_remaining": 1}
        assert result["clients_remaining"] == 1

    def test_move_notation_san_only(self) -> None:
        """MoveNotation with only SAN (when format='san')."""
        move: MoveNotation = {"san": "Nf3"}
        assert move["san"] == "Nf3"
        assert "lan" not in move

    def test_move_notation_both(self) -> None:
        move: MoveNotation = {"san": "Nf3", "lan": "g1f3"}
        assert move["san"] == "Nf3"
        assert move["lan"] == "g1f3"

    def test_can_claim_draw(self) -> None:
        draw: CanClaimDraw = {"fifty_move": True, "repetition": False}
        assert draw["fifty_move"] is True

    def test_preview_move_result(self) -> None:
        result: PreviewMoveResult = {
            "move": {"san": "e4", "lan": "e2e4"},
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "is_check": False,
            "is_checkmate": False,
            "is_stalemate": False,
            "legal_responses": [
                {"san": "e5", "lan": "e7e5"},
                {"san": "d5", "lan": "d7d5"},
            ],
            "legal_response_count": 20,
            "new_threats": [],
        }
        assert result["legal_response_count"] == 20
        assert result["is_check"] is False
        assert len(result["new_threats"]) == 0

    def test_mcp_error_content_minimal(self) -> None:
        content: McpErrorContent = {
            "error": "illegal_move",
            "message": "Not a legal move",
        }
        assert content["error"] == "illegal_move"

    def test_mcp_error_content_with_detail(self) -> None:
        content: McpErrorContent = {
            "error": "internal_error",
            "message": "An unexpected error occurred.",
            "detail": "stack trace here",
        }
        assert "detail" in content
