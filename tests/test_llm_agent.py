"""Tests for llm_agent module using a mock Anthropic client."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from chess_lmm.llm_agent import (
    CHESS_TOOLS,
    LlmTurnResult,
    ThinkingConfig,
    _build_position_context,
    _truncate_history,
    llm_turn,
    resolve_thinking,
)
from chess_lmm.mock_server import MockChessServer
from chess_lmm.recording import LlmInteractionLogger

# --- Mock Anthropic client ---


class MockContentBlock:
    """Mock for an Anthropic content block."""

    def __init__(self, block_type: str, **kwargs: Any) -> None:
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockResponse:
    """Mock for an Anthropic Messages response."""

    def __init__(self, content: list[MockContentBlock]) -> None:
        self.content = content

    def model_dump(self) -> dict[str, Any]:
        blocks: list[dict[str, Any]] = []
        for b in self.content:
            d: dict[str, Any] = {"type": b.type}
            if b.type == "thinking":
                d["thinking"] = getattr(b, "thinking", "")
                d["signature"] = getattr(b, "signature", "")
            elif b.type == "text":
                d["text"] = getattr(b, "text", "")
            elif b.type == "tool_use":
                d["id"] = getattr(b, "id", "")
                d["name"] = getattr(b, "name", "")
                d["input"] = getattr(b, "input", {})
            blocks.append(d)
        return {"content": blocks}


def make_tool_use_response(
    tool_name: str, tool_input: dict[str, Any], tool_id: str = "tool_1"
) -> MockResponse:
    """Create a mock response with a single tool_use block."""
    return MockResponse(
        [MockContentBlock("tool_use", id=tool_id, name=tool_name, input=tool_input)]
    )


def make_text_response(text: str) -> MockResponse:
    """Create a mock response with only text."""
    return MockResponse([MockContentBlock("text", text=text)])


def make_thinking_tool_response(
    thinking_text: str,
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    tool_id: str = "tool_1",
    signature: str = "test-signature-abc123",
) -> MockResponse:
    """Create a mock response with thinking + tool_use blocks."""
    return MockResponse(
        [
            MockContentBlock(
                "thinking",
                thinking=thinking_text,
                signature=signature,
            ),
            MockContentBlock(
                "tool_use",
                id=tool_id,
                name=tool_name,
                input=tool_input,
            ),
        ]
    )


# --- Fixtures ---


async def _setup_game(
    server: MockChessServer,
) -> tuple:
    """Create game and join two players."""
    white = await server.create_session()
    black = await server.create_session()
    await white.create_game()
    await white.join_game("white")
    await black.join_game("black")
    return white, black


@pytest.fixture
def server() -> MockChessServer:
    return MockChessServer()


# --- Tests ---


class TestBuildPositionContext:
    def test_basic_context(self) -> None:
        status: dict[str, Any] = {
            "turn": "white",
            "fullmove_number": 1,
            "is_check": False,
            "draw_offered": False,
            "can_claim_draw": {"fifty_move": False, "repetition": False},
            "insufficient_material": False,
        }
        moves: dict[str, Any] = {
            "moves": [{"san": "e4"}, {"san": "d4"}],
        }
        ctx = _build_position_context(status, "startpos", moves)
        assert "white" in ctx
        assert "e4" in ctx
        assert "d4" in ctx

    def test_check_context(self) -> None:
        status: dict[str, Any] = {
            "turn": "black",
            "fullmove_number": 5,
            "is_check": True,
            "draw_offered": False,
            "can_claim_draw": {"fifty_move": False, "repetition": False},
            "insufficient_material": False,
        }
        ctx = _build_position_context(status, "some_fen", {"moves": []})
        assert "CHECK" in ctx

    def test_draw_offered_context(self) -> None:
        status: dict[str, Any] = {
            "turn": "white",
            "fullmove_number": 10,
            "is_check": False,
            "draw_offered": True,
            "can_claim_draw": {"fifty_move": False, "repetition": False},
            "insufficient_material": False,
        }
        ctx = _build_position_context(status, "some_fen", {"moves": []})
        assert "accept_draw" in ctx


class TestLlmTurn:
    async def test_makes_move(self, server: MockChessServer) -> None:
        """LLM makes a valid move via tool_use."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "e4"}
        )

        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result.game_ongoing is True
        # Verify the move was made
        status = await white.get_status()
        assert status["turn"] == "black"

    async def test_handles_text_then_move(self, server: MockChessServer) -> None:
        """LLM first returns text, then makes a move on reprompt."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            make_text_response("Let me think..."),
            make_tool_use_response("make_move", {"move": "d4"}),
        ]

        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result.game_ongoing is True
        assert mock_anthropic.messages.create.call_count == 2

    async def test_handles_illegal_move_error(self, server: MockChessServer) -> None:
        """LLM tries an illegal move, gets error, then makes a valid move."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            make_tool_use_response("make_move", {"move": "e2e5"}),
            make_tool_use_response("make_move", {"move": "e4"}),
        ]

        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result.game_ongoing is True
        assert mock_anthropic.messages.create.call_count == 2

    async def test_handles_missing_move_parameter(
        self, server: MockChessServer
    ) -> None:
        """LLM calls make_move without 'move' key, gets error, then retries."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            make_tool_use_response("make_move", {}),  # missing 'move' key
            make_tool_use_response("make_move", {"move": "e4"}),
        ]

        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result.game_ongoing is True
        assert mock_anthropic.messages.create.call_count == 2
        # The error tool_result from the first call is at index 2 in history
        error_msg = result.messages[2]
        assert error_msg["content"][0]["is_error"] is True
        assert "move" in error_msg["content"][0]["content"].lower()

    async def test_handles_null_tool_input(self, server: MockChessServer) -> None:
        """LLM sends tool_use with null/None input, gets error, then retries."""
        white, black = await _setup_game(server)

        null_response = MockResponse(
            [MockContentBlock("tool_use", id="t1", name="make_move", input=None)]
        )
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            null_response,
            make_tool_use_response("make_move", {"move": "e4"}),
        ]

        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result.game_ongoing is True
        assert mock_anthropic.messages.create.call_count == 2
        error_msg = result.messages[2]
        assert error_msg["content"][0]["is_error"] is True

    async def test_handles_non_dict_tool_input(self, server: MockChessServer) -> None:
        """LLM sends tool_use with string input instead of object."""
        white, black = await _setup_game(server)

        bad_response = MockResponse(
            [MockContentBlock("tool_use", id="t1", name="make_move", input="e4")]
        )
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            bad_response,
            make_tool_use_response("make_move", {"move": "e4"}),
        ]

        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result.game_ongoing is True
        assert mock_anthropic.messages.create.call_count == 2
        error_msg = result.messages[2]
        assert error_msg["content"][0]["is_error"] is True

    async def test_handles_unexpected_params_on_no_param_tool(
        self, server: MockChessServer
    ) -> None:
        """LLM sends unexpected parameters to a no-param tool, gets error."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            # resign with spurious params
            make_tool_use_response("resign", {"move": "e4"}),
            # retry correctly
            make_tool_use_response("resign", {}),
        ]

        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result.game_ongoing is False
        assert mock_anthropic.messages.create.call_count == 2
        error_msg = result.messages[2]
        assert error_msg["content"][0]["is_error"] is True

    async def test_handles_non_string_move(self, server: MockChessServer) -> None:
        """LLM sends move as integer instead of string, gets error."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            make_tool_use_response("make_move", {"move": 4}),
            make_tool_use_response("make_move", {"move": "e4"}),
        ]

        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result.game_ongoing is True
        assert mock_anthropic.messages.create.call_count == 2
        error_msg = result.messages[2]
        assert error_msg["content"][0]["is_error"] is True

    async def test_handles_extra_keys_on_make_move(
        self, server: MockChessServer
    ) -> None:
        """LLM sends extra keys alongside 'move', gets error."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            make_tool_use_response("make_move", {"move": "e4", "reason": "central"}),
            make_tool_use_response("make_move", {"move": "e4"}),
        ]

        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result.game_ongoing is True
        assert mock_anthropic.messages.create.call_count == 2
        error_msg = result.messages[2]
        assert error_msg["content"][0]["is_error"] is True

    async def test_resign(self, server: MockChessServer) -> None:
        """LLM resigns the game."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "resign", {}
        )

        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result.game_ongoing is False

    async def test_game_already_over(self, server: MockChessServer) -> None:
        """Returns False when game is already over."""
        white, black = await _setup_game(server)
        await white.resign()

        mock_anthropic = MagicMock()
        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result.game_ongoing is False
        mock_anthropic.messages.create.assert_not_called()

    async def test_logs_to_llm_logger(
        self, server: MockChessServer, tmp_path: Path
    ) -> None:
        """Verify LLM interactions are logged."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "e4"}
        )

        llm_logger = LlmInteractionLogger(tmp_path / "llm.jsonl")
        await llm_turn(white, mock_anthropic, "test-model", llm_logger=llm_logger)

        log_content = (tmp_path / "llm.jsonl").read_text()
        assert "api_request" in log_content
        assert "api_response" in log_content


class TestThinking:
    """Tests for extended thinking support."""

    async def test_makes_move_with_thinking(self, server: MockChessServer) -> None:
        """Thinking block + tool_use: move succeeds normally."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_thinking_tool_response(
            "I should play e4 to control the center.",
            "make_move",
            {"move": "e4"},
        )

        result = await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            thinking={"type": "enabled", "budget_tokens": 2048},
            max_tokens=3072,
        )

        assert result.game_ongoing is True
        status = await white.get_status()
        assert status["turn"] == "black"

    async def test_manual_thinking_in_request(self, server: MockChessServer) -> None:
        """Verify manual thinking param and adjusted max_tokens in API request."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_thinking_tool_response(
            "Analysis...", "make_move", {"move": "e4"}
        )

        await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            thinking={"type": "enabled", "budget_tokens": 10000},
            max_tokens=11024,
        )

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs["thinking"] == {
            "type": "enabled",
            "budget_tokens": 10000,
        }
        assert call_kwargs["max_tokens"] == 11024

    async def test_adaptive_thinking_in_request(self, server: MockChessServer) -> None:
        """Adaptive thinking sends correct payload with output_config."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_thinking_tool_response(
            "Analysis...", "make_move", {"move": "e4"}
        )

        await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            thinking={"type": "adaptive"},
            max_tokens=16384,
            effort="high",
        )

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs["thinking"] == {"type": "adaptive"}
        assert call_kwargs["max_tokens"] == 16384
        assert call_kwargs["output_config"] == {"effort": "high"}

    async def test_resolve_thinking_round_trip(
        self, server: MockChessServer
    ) -> None:
        """resolve_thinking() output works directly with llm_turn()."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_thinking_tool_response(
            "Analysis...", "make_move", {"move": "e4"}
        )

        cfg = resolve_thinking("high")
        await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            thinking=cfg.thinking,
            max_tokens=cfg.max_tokens,
            effort=cfg.effort,
        )

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert call_kwargs["thinking"] == {"type": "adaptive"}
        assert call_kwargs["max_tokens"] == 16384
        assert call_kwargs["output_config"] == {"effort": "high"}

    async def test_thinking_disabled_by_default(self, server: MockChessServer) -> None:
        """Without thinking, no thinking param and max_tokens=1024."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "e4"}
        )

        await llm_turn(white, mock_anthropic, "test-model")

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        assert "thinking" not in call_kwargs
        assert "output_config" not in call_kwargs
        assert call_kwargs["max_tokens"] == 1024

    async def test_thinking_blocks_preserved_in_history(
        self, server: MockChessServer
    ) -> None:
        """Thinking blocks (with signature) are preserved in messages
        passed to the API on tool-result continuation."""
        white, black = await _setup_game(server)

        # First response: thinking + illegal move (triggers retry)
        # Second response: thinking + valid move
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            make_thinking_tool_response(
                "Let me try e5.",
                "make_move",
                {"move": "e2e5"},
                signature="sig-first",
            ),
            make_thinking_tool_response(
                "e5 was illegal, try e4.",
                "make_move",
                {"move": "e4"},
                signature="sig-second",
            ),
        ]

        result = await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            thinking={"type": "enabled", "budget_tokens": 2048},
            max_tokens=3072,
        )

        assert result.game_ongoing is True
        assert mock_anthropic.messages.create.call_count == 2

        # Check the second API call's messages contain the thinking
        # block from the first response
        second_call_kwargs = mock_anthropic.messages.create.call_args_list[1][1]
        second_messages = second_call_kwargs["messages"]

        # Find the assistant message with the first thinking block
        assistant_msgs = [m for m in second_messages if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
        first_assistant = assistant_msgs[0]["content"]

        # The thinking block should be the SDK object with signature
        thinking_blocks = [
            b for b in first_assistant if getattr(b, "type", None) == "thinking"
        ]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].signature == "sig-first"

    async def test_thinking_with_illegal_move_retry(
        self, server: MockChessServer
    ) -> None:
        """Thinking enabled, illegal move then valid move: both succeed."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            make_thinking_tool_response(
                "Try Ke9.",
                "make_move",
                {"move": "Ke9"},
            ),
            make_thinking_tool_response(
                "That was wrong. Play d4.",
                "make_move",
                {"move": "d4"},
            ),
        ]

        result = await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            thinking={"type": "enabled", "budget_tokens": 2048},
            max_tokens=3072,
        )

        assert result.game_ongoing is True
        assert mock_anthropic.messages.create.call_count == 2
        status = await white.get_status()
        assert status["turn"] == "black"

    async def test_thinking_budget_too_small(self, server: MockChessServer) -> None:
        """Manual budget < 1024 raises ValueError."""
        white, black = await _setup_game(server)
        mock_anthropic = MagicMock()

        with pytest.raises(ValueError, match="budget_tokens must be"):
            await llm_turn(
                white,
                mock_anthropic,
                "test-model",
                thinking={"type": "enabled", "budget_tokens": 500},
            )

    async def test_thinking_logged(
        self, server: MockChessServer, tmp_path: Path
    ) -> None:
        """Thinking content appears in LLM interaction log."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_thinking_tool_response(
            "Deep analysis of the position.",
            "make_move",
            {"move": "e4"},
        )

        llm_logger = LlmInteractionLogger(tmp_path / "llm.jsonl")
        await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            llm_logger=llm_logger,
            thinking={"type": "enabled", "budget_tokens": 2048},
            max_tokens=3072,
        )

        log_content = (tmp_path / "llm.jsonl").read_text()
        assert "api_response" in log_content
        assert "Deep analysis of the position" in log_content

    async def test_thinking_logged_on_retry(
        self, server: MockChessServer, tmp_path: Path
    ) -> None:
        """Logging doesn't crash when thinking blocks are in retry messages.

        Thinking blocks are SDK objects (not dicts) in the messages list.
        The logger must handle serializing them when logging the request
        payload on the second iteration of the tool-use loop.
        """
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.side_effect = [
            make_thinking_tool_response("Try e5.", "make_move", {"move": "e2e5"}),
            make_thinking_tool_response("Try e4.", "make_move", {"move": "e4"}),
        ]

        llm_logger = LlmInteractionLogger(tmp_path / "llm.jsonl")
        result = await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            llm_logger=llm_logger,
            thinking={"type": "enabled", "budget_tokens": 2048},
            max_tokens=3072,
        )

        assert result.game_ongoing is True
        # Should have logged 4 entries: request, response, request, response
        log_lines = (tmp_path / "llm.jsonl").read_text().strip().split("\n")
        assert len(log_lines) == 4


class TestResolveThinking:
    """Tests for resolve_thinking() helper."""

    def test_off(self) -> None:
        cfg = resolve_thinking("off")
        assert cfg == ThinkingConfig(thinking=None, max_tokens=1024)

    def test_low(self) -> None:
        cfg = resolve_thinking("low")
        assert cfg.thinking == {"type": "adaptive"}
        assert cfg.max_tokens == 4096
        assert cfg.effort == "low"

    def test_medium(self) -> None:
        cfg = resolve_thinking("medium")
        assert cfg.thinking == {"type": "adaptive"}
        assert cfg.max_tokens == 8192
        assert cfg.effort == "medium"

    def test_high(self) -> None:
        cfg = resolve_thinking("high")
        assert cfg.thinking == {"type": "adaptive"}
        assert cfg.max_tokens == 16384
        assert cfg.effort == "high"

    def test_max(self) -> None:
        cfg = resolve_thinking("max")
        assert cfg.thinking == {"type": "adaptive"}
        assert cfg.max_tokens == 32768
        assert cfg.effort == "max"

    def test_integer_string(self) -> None:
        cfg = resolve_thinking("10000")
        assert cfg.thinking == {"type": "enabled", "budget_tokens": 10000}
        assert cfg.max_tokens == 11024
        assert cfg.effort is None

    def test_integer_string_too_small(self) -> None:
        with pytest.raises(ValueError, match="budget_tokens must be"):
            resolve_thinking("500")

    def test_invalid_string(self) -> None:
        with pytest.raises(ValueError, match="Invalid thinking"):
            resolve_thinking("turbo")

    def test_case_insensitive(self) -> None:
        cfg = resolve_thinking("HIGH")
        assert cfg.thinking == {"type": "adaptive"}
        assert cfg.effort == "high"
        off_cfg = resolve_thinking("Off")
        assert off_cfg.thinking is None

    def test_whitespace_stripped(self) -> None:
        cfg = resolve_thinking("  high  ")
        assert cfg.thinking == {"type": "adaptive"}
        assert cfg.max_tokens == 16384
        cfg2 = resolve_thinking(" 10000 ")
        assert cfg2.thinking == {"type": "enabled", "budget_tokens": 10000}
        assert cfg2.max_tokens == 11024

    def test_minimum_valid_budget(self) -> None:
        cfg = resolve_thinking("1024")
        assert cfg.thinking == {"type": "enabled", "budget_tokens": 1024}
        assert cfg.max_tokens == 2048

    def test_just_below_minimum_budget(self) -> None:
        with pytest.raises(ValueError, match="budget_tokens must be"):
            resolve_thinking("1023")

    def test_negative_budget(self) -> None:
        with pytest.raises(ValueError, match="budget_tokens must be"):
            resolve_thinking("-5")


class TestHistory:
    """Tests for persistent conversation history."""

    async def test_history_passed_through(self, server: MockChessServer) -> None:
        """conversation_history is included in the API call messages."""
        white, black = await _setup_game(server)

        prior = [
            {"role": "user", "content": "prior position context"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "prior response"}],
            },
        ]

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "e4"}
        )

        await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            conversation_history=prior,
            enable_cache=False,
        )

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        msgs = call_kwargs["messages"]
        # First two messages are from history, third is position context
        assert msgs[0]["content"] == "prior position context"
        assert msgs[1]["role"] == "assistant"

    async def test_history_returned(self, server: MockChessServer) -> None:
        """result.messages includes position context, assistant, and tool_result."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "e4"}
        )

        result = await llm_turn(white, mock_anthropic, "test-model", enable_cache=False)

        assert isinstance(result, LlmTurnResult)
        msgs = result.messages
        # user (position), assistant (tool_use), user (tool_result)
        assert len(msgs) >= 3
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"

    async def test_history_accumulates(self, server: MockChessServer) -> None:
        """History from first turn appears in second turn's API call."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "e4"}
        )

        r1 = await llm_turn(white, mock_anthropic, "test-model", enable_cache=False)

        # Black plays
        await black.make_move("e5")

        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "Nf3"}
        )

        r2 = await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            conversation_history=r1.messages,
            enable_cache=False,
        )

        # Second turn should have more messages than first
        assert len(r2.messages) > len(r1.messages)
        # Second API call should have history from first turn
        second_call_msgs = mock_anthropic.messages.create.call_args[1]["messages"]
        assert len(second_call_msgs) > 3  # more than just this turn

    async def test_game_already_over_returns_history(
        self, server: MockChessServer
    ) -> None:
        """Early return when game is over returns incoming history."""
        white, black = await _setup_game(server)
        await white.resign()

        prior = [{"role": "user", "content": "old context"}]
        mock_anthropic = MagicMock()

        result = await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            conversation_history=prior,
        )

        assert result.game_ongoing is False
        assert result.messages == prior
        mock_anthropic.messages.create.assert_not_called()

    async def test_game_already_over_no_history(self, server: MockChessServer) -> None:
        """Early return with no history returns empty list."""
        white, black = await _setup_game(server)
        await white.resign()

        mock_anthropic = MagicMock()
        result = await llm_turn(white, mock_anthropic, "test-model")

        assert result.game_ongoing is False
        assert result.messages == []

    async def test_max_history_too_small(self, server: MockChessServer) -> None:
        """max_history < 2 raises ValueError."""
        white, black = await _setup_game(server)
        mock_anthropic = MagicMock()

        with pytest.raises(ValueError, match="max_history must be"):
            await llm_turn(
                white,
                mock_anthropic,
                "test-model",
                max_history=1,
            )

    async def test_history_truncated_through_llm_turn(
        self, server: MockChessServer
    ) -> None:
        """History exceeding max_history is truncated with a note prepended."""
        white, black = await _setup_game(server)

        # Build a history with 10 turn pairs (20 messages)
        prior: list[dict[str, Any]] = []
        for i in range(10):
            prior.append({"role": "user", "content": f"position {i}"})
            prior.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"analysis {i}"}],
                }
            )

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "e4"}
        )

        await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            conversation_history=prior,
            max_history=6,
            enable_cache=False,
        )

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        msgs = call_kwargs["messages"]
        # First message should be the truncation note
        assert "omitted" in msgs[0]["content"].lower()
        # Messages list is mutated after the API call (assistant + tool_result
        # appended), so total = note + kept + position + assistant + tool_result.
        # Key check: significantly fewer than the original 20 + extras.
        assert len(msgs) < 20


class TestTruncateHistory:
    """Tests for _truncate_history."""

    def test_no_truncation_needed(self) -> None:
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        result = _truncate_history(msgs, 10)
        assert result is msgs  # same object, no copy

    def test_truncation_prepends_note(self) -> None:
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": f"turn {i}"} for i in range(20)
        ]
        result = _truncate_history(msgs, 5)
        assert len(result) <= 6  # note + up to 5
        assert "omitted" in result[0]["content"].lower()
        assert result[0]["role"] == "user"

    def test_truncation_respects_turn_boundaries(self) -> None:
        """Don't orphan a tool_result without its tool_use."""
        msgs: list[dict[str, Any]] = [
            # Turn 1: position context (string)
            {"role": "user", "content": "position 1"},
            # Turn 1: assistant with tool_use
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "make_move",
                        "input": {"move": "e4"},
                    }
                ],
            },
            # Turn 1: tool_result (list content)
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "{}",
                    }
                ],
            },
            # Turn 2: position context (string)
            {"role": "user", "content": "position 2"},
            # Turn 2: assistant
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": "make_move",
                        "input": {"move": "d4"},
                    }
                ],
            },
            # Turn 2: tool_result
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t2",
                        "content": "{}",
                    }
                ],
            },
        ]
        # max_messages=4 — naive cut at index 2 lands on tool_result
        # Should walk forward to index 3 (position context string)
        result = _truncate_history(msgs, 4)
        # First message is the note
        assert "omitted" in result[0]["content"].lower()
        # Second message should be a position context, not a tool_result
        assert isinstance(result[1]["content"], str)

    def test_truncation_exactly_at_limit(self) -> None:
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": f"turn {i}"} for i in range(5)
        ]
        result = _truncate_history(msgs, 5)
        assert result is msgs


class TestCaching:
    """Tests for prompt caching support."""

    async def test_cache_control_on_system(self, server: MockChessServer) -> None:
        """System prompt is a list with cache_control when caching enabled."""
        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "e4"}
        )

        await llm_turn(white, mock_anthropic, "test-model", enable_cache=True)

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        system = call_kwargs["system"]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    async def test_cache_control_on_tools(self, server: MockChessServer) -> None:
        """Last tool has cache_control; CHESS_TOOLS is not mutated."""
        tools_before = copy.deepcopy(CHESS_TOOLS)

        white, black = await _setup_game(server)

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "e4"}
        )

        await llm_turn(white, mock_anthropic, "test-model", enable_cache=True)

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        tools = call_kwargs["tools"]
        assert "cache_control" in tools[-1]
        # Original constant not mutated
        assert tools_before == CHESS_TOOLS

    async def test_cache_control_on_history_frontier(
        self, server: MockChessServer
    ) -> None:
        """Last user message in history gets cache_control."""
        white, black = await _setup_game(server)

        prior = [
            {"role": "user", "content": "old position"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "response"}],
            },
        ]

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "e4"}
        )

        await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            conversation_history=prior,
            enable_cache=True,
        )

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        msgs = call_kwargs["messages"]
        # The first user message (from history) should have cache_control
        first_user = msgs[0]
        content = first_user["content"]
        if isinstance(content, list):
            assert content[-1].get("cache_control") == {"type": "ephemeral"}
        else:
            # Should not happen — string content gets converted to list
            pytest.fail("Expected list content with cache_control")

    async def test_stale_cache_control_stripped(self, server: MockChessServer) -> None:
        """Old cache_control markers are stripped before marking new frontier."""
        white, black = await _setup_game(server)

        prior = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "old",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "resp"}],
            },
            {"role": "user", "content": "newer position"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "resp2"}],
            },
        ]

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "e4"}
        )

        await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            conversation_history=prior,
            enable_cache=True,
        )

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        msgs = call_kwargs["messages"]

        # First user message should have cache_control stripped
        first_user_content = msgs[0]["content"]
        if isinstance(first_user_content, list):
            assert "cache_control" not in first_user_content[0]

        # The frontier (last user msg before new position) should have it
        # That's msgs[2] ("newer position"), which becomes a list with cc
        frontier = msgs[2]
        fc = frontier["content"]
        if isinstance(fc, list):
            assert fc[-1].get("cache_control") == {"type": "ephemeral"}

    async def test_no_cache(self, server: MockChessServer) -> None:
        """No cache_control anywhere when enable_cache=False."""
        white, black = await _setup_game(server)

        prior = [
            {"role": "user", "content": "old position"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "resp"}],
            },
        ]

        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = make_tool_use_response(
            "make_move", {"move": "e4"}
        )

        await llm_turn(
            white,
            mock_anthropic,
            "test-model",
            conversation_history=prior,
            enable_cache=False,
        )

        call_kwargs = mock_anthropic.messages.create.call_args[1]
        # System is a plain string
        assert isinstance(call_kwargs["system"], str)
        # No cache_control on tools
        for tool in call_kwargs["tools"]:
            assert "cache_control" not in tool
        # No cache_control on messages
        for msg in call_kwargs["messages"]:
            content = msg["content"]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        assert "cache_control" not in block


class TestChessTools:
    def test_tool_definitions_structure(self) -> None:
        """Verify tool definitions have required fields."""
        for tool in CHESS_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_make_move_has_required_param(self) -> None:
        make_move = next(t for t in CHESS_TOOLS if t["name"] == "make_move")
        assert "move" in make_move["input_schema"]["properties"]
        assert "move" in make_move["input_schema"]["required"]
