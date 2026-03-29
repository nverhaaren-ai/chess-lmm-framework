"""Tests for orchestrator module."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from chess_lmm.orchestrator import parse_args, run_game


class MockContentBlock:
    def __init__(self, block_type: str, **kwargs: Any) -> None:
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockResponse:
    def __init__(self, content: list[MockContentBlock]) -> None:
        self.content = content

    def model_dump(self) -> dict[str, Any]:
        return {"content": []}


class TestParseArgs:
    def test_defaults(self) -> None:
        args = parse_args([])
        assert args.color == "white"
        assert args.model == "claude-sonnet-4-6"
        assert args.fen is None
        assert args.log_dir == Path("./game-logs")
        assert args.thinking == "off"
        assert args.max_history == 40
        assert args.no_cache is False

    def test_custom_args(self) -> None:
        args = parse_args(
            [
                "--color",
                "black",
                "--model",
                "claude-opus-4-6",
                "--fen",
                "8/8/8/8/8/8/8/8 w - - 0 1",
                "--log-dir",
                "/tmp/logs",
            ]
        )
        assert args.color == "black"
        assert args.model == "claude-opus-4-6"
        assert args.fen == "8/8/8/8/8/8/8/8 w - - 0 1"
        assert args.log_dir == Path("/tmp/logs")

    def test_thinking_adaptive(self) -> None:
        args = parse_args(["--thinking", "high"])
        assert args.thinking == "high"

    def test_thinking_manual_budget(self) -> None:
        args = parse_args(["--thinking", "10000"])
        assert args.thinking == "10000"

    def test_thinking_default_off(self) -> None:
        args = parse_args([])
        assert args.thinking == "off"

    def test_max_history(self) -> None:
        args = parse_args(["--max-history", "20"])
        assert args.max_history == 20

    def test_no_cache(self) -> None:
        args = parse_args(["--no-cache"])
        assert args.no_cache is True


class TestRunGame:
    async def test_human_resigns_immediately(self, tmp_path: Path) -> None:
        """Integration test: human resigns on first move."""
        args = parse_args(["--color", "white", "--log-dir", str(tmp_path)])

        mock_anthropic = MagicMock()
        stdin = io.StringIO("/resign\n")
        stdout = io.StringIO()

        await run_game(
            args,
            anthropic_client=mock_anthropic,
            input_stream=stdin,
            output_stream=stdout,
        )

        output = stdout.getvalue()
        assert "white" in output.lower()
        assert "logs saved" in output.lower() or "Game logs" in output

        # Check that log files were created
        assert (tmp_path / "mcp_recording.jsonl").exists()

    async def test_llm_moves_then_human_resigns(self, tmp_path: Path) -> None:
        """Integration test: LLM plays as white, human as black, human resigns."""
        args = parse_args(["--color", "black", "--log-dir", str(tmp_path)])

        # LLM plays e4
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = MockResponse(
            [
                MockContentBlock(
                    "tool_use", id="t1", name="make_move", input={"move": "e4"}
                )
            ]
        )

        stdin = io.StringIO("/resign\n")
        stdout = io.StringIO()

        await run_game(
            args,
            anthropic_client=mock_anthropic,
            input_stream=stdin,
            output_stream=stdout,
        )

        output = stdout.getvalue()
        assert "Claude played: e4" in output

    async def test_game_start_marker_in_logs(self, tmp_path: Path) -> None:
        """Game-start marker appears in both MCP and LLM log files."""
        args = parse_args(["--color", "white", "--log-dir", str(tmp_path)])

        mock_anthropic = MagicMock()
        stdin = io.StringIO("/resign\n")
        stdout = io.StringIO()

        await run_game(
            args,
            anthropic_client=mock_anthropic,
            input_stream=stdin,
            output_stream=stdout,
        )

        # Check MCP recording log
        mcp_lines = (tmp_path / "mcp_recording.jsonl").read_text().splitlines()
        mcp_markers = [json.loads(ln) for ln in mcp_lines if '"game_start"' in ln]
        assert len(mcp_markers) == 1
        marker = mcp_markers[0]
        assert marker["type"] == "game_start"
        assert marker["model"] == "claude-sonnet-4-6"
        assert marker["human_color"] == "white"
        assert marker["llm_color"] == "black"
        assert "game_id" in marker
        assert "ts_ms" in marker

        # Check LLM interaction log
        llm_lines = (tmp_path / "llm_interactions.jsonl").read_text().splitlines()
        llm_markers = [json.loads(ln) for ln in llm_lines if '"game_start"' in ln]
        assert len(llm_markers) == 1
        assert llm_markers[0]["type"] == "game_start"
        assert llm_markers[0]["model"] == "claude-sonnet-4-6"

    async def test_game_start_marker_includes_thinking(self, tmp_path: Path) -> None:
        """Game-start marker includes thinking config when enabled."""
        args = parse_args(
            ["--color", "white", "--thinking", "high", "--log-dir", str(tmp_path)]
        )

        mock_anthropic = MagicMock()
        stdin = io.StringIO("/resign\n")
        stdout = io.StringIO()

        await run_game(
            args,
            anthropic_client=mock_anthropic,
            input_stream=stdin,
            output_stream=stdout,
        )

        mcp_lines = (tmp_path / "mcp_recording.jsonl").read_text().splitlines()
        mcp_markers = [json.loads(ln) for ln in mcp_lines if '"game_start"' in ln]
        marker = mcp_markers[0]
        assert marker["thinking"] == {"type": "adaptive"}
        assert marker["effort"] == "high"

    async def test_game_start_marker_no_thinking(self, tmp_path: Path) -> None:
        """Game-start marker omits thinking fields when disabled."""
        args = parse_args(["--color", "white", "--log-dir", str(tmp_path)])

        mock_anthropic = MagicMock()
        stdin = io.StringIO("/resign\n")
        stdout = io.StringIO()

        await run_game(
            args,
            anthropic_client=mock_anthropic,
            input_stream=stdin,
            output_stream=stdout,
        )

        mcp_lines = (tmp_path / "mcp_recording.jsonl").read_text().splitlines()
        mcp_markers = [json.loads(ln) for ln in mcp_lines if '"game_start"' in ln]
        marker = mcp_markers[0]
        assert "thinking" not in marker
        assert "effort" not in marker
        assert "fen" not in marker
