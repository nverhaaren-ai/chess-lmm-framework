"""Game lifecycle management and CLI entry point.

Creates a chess server (mock or real MCP), wraps clients with recording,
launches players as async tasks, and manages game flow.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from chess_lmm.human_player import human_turn
from chess_lmm.llm_agent import llm_turn, resolve_thinking
from chess_lmm.recording import (
    GameRecorder,
    LlmInteractionLogger,
    RecordingClient,
    render_board,
)
from chess_lmm.types import McpError

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="chess-lmm",
        description="Play chess against Claude",
    )
    parser.add_argument(
        "--color",
        choices=["white", "black", "random"],
        default="white",
        help="Color to play as (default: white)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--fen",
        default=None,
        help="Starting position in FEN notation",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("./game-logs"),
        help="Directory for game logs (default: ./game-logs)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--thinking",
        default="off",
        help=(
            "Thinking mode: 'off', 'low', 'medium', 'high', 'max' "
            "(adaptive), or an integer token budget (manual). "
            "Default: off"
        ),
    )

    parser.add_argument(
        "--max-history",
        type=int,
        default=40,
        help="Maximum conversation messages to keep (default: 40)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable prompt caching",
    )

    server_group = parser.add_mutually_exclusive_group()
    server_group.add_argument(
        "--server-url",
        default=None,
        help="URL of a running MCP chess server (SSE endpoint)",
    )
    server_group.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Use the built-in mock server (default when --server-url is not given)",
    )

    return parser.parse_args(argv)


async def run_game(
    args: argparse.Namespace,
    *,
    anthropic_client: Any = None,
    input_stream: Any = None,
    output_stream: Any = None,
) -> None:
    """Run a complete chess game."""
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Resolve thinking config early so invalid values fail fast
    thinking_cfg = resolve_thinking(args.thinking)

    out = output_stream or sys.stdout

    def write(text: str) -> None:
        out.write(text + "\n")
        out.flush()

    # Create server
    mcp_connection = None
    if args.server_url:
        from chess_lmm.mcp_client import McpServerConnection

        mcp_connection = McpServerConnection(args.server_url)
        server: Any = mcp_connection
        write(f"Connecting to MCP server at {args.server_url}...")
    else:
        from chess_lmm.mock_server import MockChessServer

        server = MockChessServer()

    try:
        human_raw = await server.create_session()
        llm_raw = await server.create_session()

        # Set up recording
        log_dir: Path = args.log_dir
        mcp_log = log_dir / "mcp_recording.jsonl"
        llm_log = log_dir / "llm_interactions.jsonl"

        human_recorder = GameRecorder(mcp_log, human_raw.session_id)
        llm_recorder = GameRecorder(mcp_log, llm_raw.session_id)
        llm_interaction_logger = LlmInteractionLogger(llm_log)

        human_client = RecordingClient(human_raw, human_recorder)
        llm_client = RecordingClient(llm_raw, llm_recorder)

        # Create game
        create_result = await human_client.create_game(fen=args.fen)
        logger.info("Game created: %s", create_result["game_id"])

        # Determine colors
        human_color = args.color
        if human_color == "random":
            import random

            human_color = random.choice(["white", "black"])

        llm_color = "black" if human_color == "white" else "white"

        # Write game-start markers to both log files
        game_start_info: dict[str, Any] = {
            "type": "game_start",
            "game_id": create_result["game_id"],
            "model": args.model,
            "human_color": human_color,
            "llm_color": llm_color,
        }
        if args.fen:
            game_start_info["fen"] = args.fen
        if thinking_cfg.thinking:
            game_start_info["thinking"] = thinking_cfg.thinking
        if thinking_cfg.effort:
            game_start_info["effort"] = thinking_cfg.effort
        human_recorder.write_marker(game_start_info)
        llm_interaction_logger.log(dict(game_start_info))

        # Join game
        await human_client.join_game(human_color)
        await llm_client.join_game(llm_color)

        write(f"\nYou are playing as {human_color}.")
        write(f"Claude ({args.model}) is playing as {llm_color}.\n")

        # Display initial board
        board = await human_client.get_board()
        write(render_board(board["fen"]))
        write("")

        # Determine who goes first
        human_goes_first = human_color == "white"

        # Lazy-import anthropic to avoid import errors when not installed
        if anthropic_client is None:
            try:
                import anthropic

                anthropic_client = anthropic.Anthropic()
            except ImportError:
                write("Error: anthropic package not installed.")
                write("Install with: pip install anthropic")
                return

        in_stream = input_stream or sys.stdin

        # Game loop
        game_ongoing = True
        is_human_turn = human_goes_first
        conversation_history: list[dict[str, Any]] = []

        while game_ongoing:
            # Check game status
            try:
                status = await human_client.get_status()
            except McpError as e:
                logger.error("Failed to get game status: %s", e.message)
                break

            if status.get("server_state") == "game_over":
                reason = status.get("termination_reason", "Game over")
                result = status.get("result", "")
                write(f"\n{reason}")
                if result:
                    write(f"Result: {result}")
                break

            if is_human_turn:
                write(f"\nYour turn ({human_color}):")

                # Show draw offer if pending
                if status.get("draw_offered"):
                    write(
                        "Your opponent has offered a draw. "
                        "Use /draw accept or /draw decline."
                    )

                game_ongoing = await human_turn(
                    human_client,
                    input_stream=in_stream,
                    output_stream=out,
                )
            else:
                write("\nClaude is thinking...")
                llm_result = await llm_turn(
                    llm_client,
                    anthropic_client,
                    args.model,
                    llm_logger=llm_interaction_logger,
                    thinking=thinking_cfg.thinking,
                    max_tokens=thinking_cfg.max_tokens,
                    effort=thinking_cfg.effort,
                    conversation_history=conversation_history,
                    enable_cache=not args.no_cache,
                    max_history=args.max_history,
                )
                game_ongoing = llm_result.game_ongoing
                conversation_history = llm_result.messages

                if game_ongoing:
                    # Show Claude's move
                    llm_status = await human_client.get_status()
                    last_move = llm_status.get("last_move")
                    if last_move:
                        write(f"Claude played: {last_move.get('san', '?')}")
                    board = await human_client.get_board()
                    write(render_board(board["fen"]))

            is_human_turn = not is_human_turn

        # Game over — send done
        try:
            await human_client.done()
            await llm_client.done()
        except McpError as e:
            logger.debug("Error during done cleanup: %s", e.message)

        write(f"\nGame logs saved to {log_dir}/")

    finally:
        if mcp_connection is not None:
            await mcp_connection.close_all()


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    asyncio.run(run_game(args))
