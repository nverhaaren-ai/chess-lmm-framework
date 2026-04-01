"""Claude agentic loop for playing chess.

Uses the Anthropic SDK directly with a manual loop:
1. Get status + board + legal moves via MCP (query calls)
2. Build Claude API request with tool definitions for actions only
3. Claude responds with tool_use block
4. Execute the tool via MCP client
5. Feed tool_result back, repeat if needed
6. Log everything

Query tools are NOT exposed as Claude tools — board state and legal moves
are injected into the user message to reduce API calls.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from chess_lmm.mcp_interface import ChessSessionClient
from chess_lmm.recording import LlmInteractionLogger, render_board
from chess_lmm.types import McpError, PreviewMoveResult

logger = logging.getLogger(__name__)

# Tool definitions exposed to Claude (action tools only)
CHESS_TOOLS: list[dict[str, Any]] = [
    {
        "name": "make_move",
        "description": (
            "Submit a chess move in SAN (e.g. 'e4', 'Nf3', 'O-O') "
            "or LAN (e.g. 'e2e4') notation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "description": "The move to play.",
                }
            },
            "required": ["move"],
        },
    },
    {
        "name": "offer_draw",
        "description": "Offer a draw to the opponent.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "claim_draw",
        "description": (
            "Claim a draw under the fifty-move rule or threefold "
            "repetition. Only valid when can_claim_draw is true."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "accept_draw",
        "description": "Accept a pending draw offer from the opponent.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "decline_draw",
        "description": "Decline a pending draw offer from the opponent.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "resign",
        "description": "Resign the game. The opponent wins.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


def _build_position_context(
    status: dict[str, Any],
    board_fen: str,
    legal_moves: dict[str, Any],
) -> str:
    """Build a context string describing the current position for Claude."""
    parts: list[str] = []

    parts.append(f"You are playing as {status.get('turn', '?')}.")
    parts.append(f"Current FEN: {board_fen}")
    parts.append(f"Move number: {status.get('fullmove_number', '?')}")

    if status.get("is_check"):
        parts.append("You are in CHECK! You must get out of check.")

    if status.get("draw_offered"):
        parts.append(
            "Your opponent has offered a draw. "
            "You must accept_draw or decline_draw before making a move."
        )

    claim = status.get("can_claim_draw", {})
    if claim.get("fifty_move"):
        parts.append("You can claim a draw under the fifty-move rule.")
    if claim.get("repetition"):
        parts.append("You can claim a draw by threefold repetition.")

    if status.get("insufficient_material"):
        parts.append("Insufficient material — the position is a theoretical draw.")

    # Legal moves
    moves = legal_moves.get("moves", [])
    move_strs = [m.get("san", m.get("lan", "?")) for m in moves]
    parts.append(f"\nLegal moves ({len(move_strs)}): {', '.join(move_strs)}")

    return "\n".join(parts)


@dataclass
class LlmTurnResult:
    """Return value from llm_turn()."""

    game_ongoing: bool
    messages: list[dict[str, Any]]


_ADAPTIVE_MAX_TOKENS: dict[str, int] = {
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    "max": 32768,
}


@dataclass
class ThinkingConfig:
    """Resolved thinking configuration for the API."""

    thinking: dict[str, Any] | None
    max_tokens: int
    effort: str | None = None


def resolve_thinking(value: str) -> ThinkingConfig:
    """Resolve a CLI thinking value to API parameters.

    Accepts: "off", "low", "medium", "high", "max", or an integer string
    (manual budget_tokens). Case-insensitive, whitespace-trimmed.

    Returns a ThinkingConfig with thinking dict, max_tokens, and optional
    effort level (placed in output_config at the API call site).
    """
    normalized = value.strip().lower()
    if normalized == "off":
        return ThinkingConfig(thinking=None, max_tokens=1024)
    if normalized in _ADAPTIVE_MAX_TOKENS:
        return ThinkingConfig(
            thinking={"type": "adaptive"},
            max_tokens=_ADAPTIVE_MAX_TOKENS[normalized],
            effort=normalized,
        )
    try:
        budget = int(value)
    except ValueError:
        raise ValueError(
            f"Invalid thinking value: {value!r}. "
            f"Use 'off', 'low', 'medium', 'high', 'max', or an integer."
        ) from None
    if budget < 1024:
        raise ValueError("budget_tokens must be >= 1024")
    return ThinkingConfig(
        thinking={"type": "enabled", "budget_tokens": budget},
        max_tokens=budget + 1024,
    )


def _strip_cache_control(messages: list[dict[str, Any]]) -> None:
    """Remove cache_control from all user messages."""
    for msg in messages:
        if msg["role"] != "user":
            continue
        content = msg["content"]
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)


def _add_cache_control(message: dict[str, Any]) -> None:
    """Add cache_control to a user message's last content block."""
    content = message["content"]
    if isinstance(content, str):
        message["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    elif isinstance(content, list) and content:
        last_block = content[-1]
        if isinstance(last_block, dict):
            last_block["cache_control"] = {"type": "ephemeral"}


def _truncate_history(
    messages: list[dict[str, Any]],
    max_messages: int,
) -> list[dict[str, Any]]:
    """Truncate history to at most max_messages, respecting turn boundaries.

    Walks forward from the naive cut point to find a position-context
    user message (string content) to avoid orphaning tool_result messages.
    """
    if len(messages) <= max_messages:
        return messages
    # Find a safe cut point
    cut = len(messages) - max_messages
    while cut < len(messages):
        msg = messages[cut]
        if msg["role"] == "user" and isinstance(msg["content"], str):
            break
        cut += 1
    if cut >= len(messages):
        # Can't find a safe boundary — keep everything
        return messages
    truncated = messages[cut:]
    note: dict[str, Any] = {
        "role": "user",
        "content": (
            "[Earlier moves and analysis have been omitted. "
            "The current position and legal moves are "
            "provided below.]"
        ),
    }
    return [note, *truncated]


async def llm_turn(
    client: ChessSessionClient,
    anthropic_client: Any,
    model: str,
    *,
    llm_logger: LlmInteractionLogger | None = None,
    system_prompt: str | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    thinking: dict[str, Any] | None = None,
    max_tokens: int = 1024,
    effort: str | None = None,
    enable_cache: bool = True,
    max_history: int = 40,
    verify_moves: bool = False,
) -> LlmTurnResult:
    """Handle one turn for the LLM agent.

    Returns LlmTurnResult with game_ongoing flag and updated messages.

    Args:
        thinking: Thinking config dict for the API, e.g.
            {"type": "enabled", "budget_tokens": N} for manual or
            {"type": "adaptive"} for adaptive. None disables thinking.
        max_tokens: Maximum tokens for the API response (thinking +
            output). Use resolve_thinking() to compute both thinking
            and max_tokens from a CLI string.
        effort: Effort level for output_config (e.g. "low", "medium",
            "high", "max"). Only used with adaptive thinking.
        enable_cache: Add cache_control breakpoints to system prompt,
            tools, and history frontier for prompt caching.
        max_history: Maximum messages to keep in history. Must be >= 2.
        verify_moves: When True, preview moves before executing and
            ask Claude to confirm or change. Helps reduce blunders.
    """
    if thinking is not None:
        budget = thinking.get("budget_tokens")
        if budget is not None and budget < 1024:
            raise ValueError("budget_tokens must be >= 1024")
    if max_history < 2:
        raise ValueError("max_history must be >= 2")

    # 1. Query current state
    status = await client.get_status()
    board = await client.get_board()
    legal_moves = await client.get_legal_moves(format="san")

    # Check if game already ended
    if status.get("server_state") == "game_over":
        return LlmTurnResult(
            game_ongoing=False,
            messages=conversation_history or [],
        )

    # 2. Build the context message
    position_context = _build_position_context(
        dict(status), board["fen"], dict(legal_moves)
    )

    if system_prompt is None:
        system_prompt = (
            "You are playing a game of chess. Analyze the position and make "
            "your best move. Use the make_move tool to submit your move in "
            "SAN notation. Think carefully about tactics and strategy."
        )

    # 3. Build messages: history → truncate → cache → new position
    messages: list[dict[str, Any]] = []
    if conversation_history:
        messages.extend(conversation_history)
    messages = _truncate_history(messages, max_history)

    if enable_cache and messages:
        _strip_cache_control(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                _add_cache_control(messages[i])
                break

    messages.append({"role": "user", "content": position_context})

    # 4. Build system prompt and tools (with optional caching)
    system_value: Any
    if enable_cache:
        system_value = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_value = system_prompt

    tools = [dict(t) for t in CHESS_TOOLS]
    if enable_cache and tools:
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}

    # 5. Call Claude with tools
    max_iterations = 5  # Safety limit for tool-use loop
    for _ in range(max_iterations):
        request_payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_value,
            "messages": messages,
            "tools": tools,
        }
        if thinking is not None:
            request_payload["thinking"] = thinking
        if effort is not None:
            request_payload["output_config"] = {"effort": effort}

        if llm_logger:
            llm_logger.log({"type": "api_request", "payload": request_payload})

        response = anthropic_client.messages.create(**request_payload)

        if llm_logger:
            llm_logger.log(
                {
                    "type": "api_response",
                    "response": _serialize_response(response),
                }
            )

        # Process response
        assistant_content: list[Any] = []
        tool_use_blocks = []

        for block in response.content:
            if block.type == "thinking":
                # Pass the SDK object directly to preserve the
                # signature field required by the API.
                assistant_content.append(block)
                logger.debug(
                    "LLM thinking: %s",
                    block.thinking[:200] if block.thinking else "",
                )
            elif block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                logger.info("LLM says: %s", block.text)
            elif block.type == "tool_use":
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
                tool_use_blocks.append(block)

        messages.append({"role": "assistant", "content": assistant_content})

        if not tool_use_blocks:
            # No tool call — Claude just talked. Prompt again.
            messages.append(
                {
                    "role": "user",
                    "content": "Please use one of the tools to take an action.",
                }
            )
            continue

        # Blunder-check: intercept make_move for verification
        if (
            verify_moves
            and len(tool_use_blocks) == 1
            and tool_use_blocks[0].name == "make_move"
        ):
            block = tool_use_blocks[0]
            validation_error = _validate_tool_input("make_move", block.input)
            if validation_error is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "is_error": True,
                                "content": json.dumps(
                                    validation_error.get("error", {})
                                ),
                            }
                        ],
                    }
                )
                continue

            move_str = block.input["move"]
            try:
                preview = await client.preview_move(move_str)
            except McpError as e:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "is_error": True,
                                "content": json.dumps(e.to_dict()),
                            }
                        ],
                    }
                )
                continue

            return await _run_verification_loop(
                client,
                anthropic_client,
                model,
                move_str,
                block.id,
                preview,
                messages,
                system_value=system_value,
                tools=tools,
                thinking=thinking,
                max_tokens=max_tokens,
                effort=effort,
                llm_logger=llm_logger,
            )

        # Execute tool calls (normal path)
        tool_results: list[dict[str, Any]] = []
        game_ended = False

        for tool_block in tool_use_blocks:
            tool_result = await _execute_tool(client, tool_block.name, tool_block.input)

            if tool_result.get("is_error"):
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "is_error": True,
                        "content": json.dumps(tool_result.get("error", {})),
                    }
                )
            else:
                result_data = tool_result.get("result", {})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": json.dumps(result_data),
                    }
                )
                # Check if game ended
                if result_data.get("server_state") == "game_over":
                    game_ended = True

        messages.append({"role": "user", "content": tool_results})

        if game_ended:
            return LlmTurnResult(game_ongoing=False, messages=messages)

        # If a move was successfully made, we're done
        for tr in tool_results:
            if not tr.get("is_error"):
                return LlmTurnResult(game_ongoing=True, messages=messages)

    logger.warning("LLM turn exceeded max iterations")
    return LlmTurnResult(game_ongoing=True, messages=messages)


def _build_verification_prompt(
    preview: PreviewMoveResult,
    move_str: str,
) -> str:
    """Build the verification tool_result content for a previewed move."""
    move_info = preview["move"]
    san = move_info.get("san", move_str)
    lan = move_info.get("lan", move_str)
    fen = preview["fen"]
    board_diagram = render_board(fen)
    response_count = preview["legal_response_count"]

    parts = [
        f"MOVE PREVIEW — {san} ({lan}) has not been played yet.",
        "",
        f"Position after {san}:",
        board_diagram,
        "",
        f"FEN: {fen}",
        f"Opponent's legal responses: {response_count} moves",
    ]

    if preview.get("is_checkmate"):
        parts.append("\nThis move delivers CHECKMATE!")
    elif preview.get("is_check"):
        parts.append("\nThis move gives check.")
    elif preview.get("is_stalemate"):
        parts.append("\nWARNING: This move results in STALEMATE (draw)!")

    new_threats = preview.get("new_threats", [])
    if new_threats:
        parts.append("")
        parts.append(
            "⚠ WARNING — This move gives your opponent new options "
            "that were not available before:"
        )
        for threat in new_threats:
            threat_san = threat.get("san", threat.get("lan", "?"))
            parts.append(f"  {threat_san}")
        parts.append("Check whether any of these are dangerous before confirming.")

    parts.append("")
    parts.append(
        f'Review the position. To confirm, call make_move("{move_str}"). '
        "To choose a different move, call make_move with another move."
    )
    return "\n".join(parts)


async def _run_verification_loop(
    client: ChessSessionClient,
    anthropic_client: Any,
    model: str,
    initial_move: str,
    initial_tool_use_id: str,
    initial_preview: PreviewMoveResult,
    messages: list[dict[str, Any]],
    *,
    system_value: Any,
    tools: list[dict[str, Any]],
    thinking: dict[str, Any] | None,
    max_tokens: int,
    effort: str | None,
    llm_logger: LlmInteractionLogger | None,
) -> LlmTurnResult:
    """Run the blunder-check verification sub-loop.

    Previews the proposed move, asks Claude to confirm or change,
    and loops until confirmed, a different action is taken, or
    4 distinct moves have been proposed (auto-executes the last).
    """
    max_distinct_moves = 4
    max_verify_iterations = 10
    seen_moves: set[str] = {initial_move}
    pending_move = initial_move
    pending_preview = initial_preview
    pending_tool_use_id = initial_tool_use_id

    for _verify_iter in range(max_verify_iterations):  # safety cap
        # Build and append verification tool_result
        prompt = _build_verification_prompt(pending_preview, pending_move)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": pending_tool_use_id,
                        "content": prompt,
                    }
                ],
            }
        )

        # Check if we've hit the distinct-moves cap
        if len(seen_moves) >= max_distinct_moves:
            logger.info(
                "Verification cap reached (%d moves), auto-executing %s",
                max_distinct_moves,
                pending_move,
            )
            exec_result = await _execute_tool(
                client, "make_move", {"move": pending_move}
            )
            result_data = exec_result.get("result", {})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": pending_tool_use_id,
                            "content": json.dumps(result_data),
                        }
                    ],
                }
            )
            game_over = result_data.get("server_state") == "game_over"
            return LlmTurnResult(game_ongoing=not game_over, messages=messages)

        # Call Claude for confirmation
        request_payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_value,
            "messages": messages,
            "tools": tools,
        }
        if thinking is not None:
            request_payload["thinking"] = thinking
        if effort is not None:
            request_payload["output_config"] = {"effort": effort}

        if llm_logger:
            llm_logger.log({"type": "api_request", "payload": request_payload})

        response = anthropic_client.messages.create(**request_payload)

        if llm_logger:
            llm_logger.log(
                {
                    "type": "api_response",
                    "response": _serialize_response(response),
                }
            )

        # Process response
        assistant_content: list[Any] = []
        tool_use_block = None

        for block in response.content:
            if block.type == "thinking":
                assistant_content.append(block)
            elif block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
                tool_use_block = block

        messages.append({"role": "assistant", "content": assistant_content})

        if tool_use_block is None:
            # Claude didn't call a tool — prompt again
            messages.append(
                {
                    "role": "user",
                    "content": "Please use make_move to confirm or change your move.",
                }
            )
            continue

        # Non-make_move tool — execute directly (resign, draw, etc.)
        if tool_use_block.name != "make_move":
            exec_result = await _execute_tool(
                client, tool_use_block.name, tool_use_block.input
            )
            if exec_result.get("is_error"):
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_block.id,
                                "is_error": True,
                                "content": json.dumps(exec_result.get("error", {})),
                            }
                        ],
                    }
                )
                continue
            result_data = exec_result.get("result", {})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": json.dumps(result_data),
                        }
                    ],
                }
            )
            game_over = result_data.get("server_state") == "game_over"
            return LlmTurnResult(game_ongoing=not game_over, messages=messages)

        # make_move — check if confirmation or change
        validation_error = _validate_tool_input("make_move", tool_use_block.input)
        if validation_error is not None:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "is_error": True,
                            "content": json.dumps(validation_error.get("error", {})),
                        }
                    ],
                }
            )
            continue

        new_move = tool_use_block.input["move"]

        if new_move == pending_move:
            # Confirmed — execute the move
            exec_result = await _execute_tool(client, "make_move", {"move": new_move})
            if exec_result.get("is_error"):
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_block.id,
                                "is_error": True,
                                "content": json.dumps(exec_result.get("error", {})),
                            }
                        ],
                    }
                )
                continue
            result_data = exec_result.get("result", {})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": json.dumps(result_data),
                        }
                    ],
                }
            )
            game_over = result_data.get("server_state") == "game_over"
            return LlmTurnResult(game_ongoing=not game_over, messages=messages)

        # Different move — preview it
        try:
            new_preview = await client.preview_move(new_move)
        except McpError as e:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "is_error": True,
                            "content": json.dumps(e.to_dict()),
                        }
                    ],
                }
            )
            continue

        seen_moves.add(new_move)
        pending_move = new_move
        pending_preview = new_preview
        pending_tool_use_id = tool_use_block.id

    # Safety cap reached — auto-execute last pending move
    logger.warning(
        "Verification iteration cap (%d) reached, auto-executing %s",
        max_verify_iterations,
        pending_move,
    )
    exec_result = await _execute_tool(client, "make_move", {"move": pending_move})
    result_data = exec_result.get("result", {})
    game_over = result_data.get("server_state") == "game_over"
    return LlmTurnResult(game_ongoing=not game_over, messages=messages)


_NO_PARAM_TOOLS = frozenset(
    {"offer_draw", "claim_draw", "accept_draw", "decline_draw", "resign"}
)


def _validate_tool_input(tool_name: str, tool_input: Any) -> dict[str, Any] | None:
    """Validate tool input, returning an error dict if invalid, or None if OK."""
    if not isinstance(tool_input, dict):
        return {
            "is_error": True,
            "error": {
                "error": "invalid_params",
                "message": (
                    f"Expected object for {tool_name} input, "
                    f"got {type(tool_input).__name__}"
                ),
            },
        }
    if tool_name == "make_move":
        if "move" not in tool_input:
            return {
                "is_error": True,
                "error": {
                    "error": "invalid_params",
                    "message": "make_move requires a 'move' parameter",
                },
            }
        if not isinstance(tool_input["move"], str):
            return {
                "is_error": True,
                "error": {
                    "error": "invalid_params",
                    "message": (
                        f"'move' must be a string, "
                        f"got {type(tool_input['move']).__name__}"
                    ),
                },
            }
        extra = set(tool_input.keys()) - {"move"}
        if extra:
            return {
                "is_error": True,
                "error": {
                    "error": "invalid_params",
                    "message": (
                        f"make_move only accepts 'move', "
                        f"got extra: {', '.join(sorted(extra))}"
                    ),
                },
            }
    if tool_name in _NO_PARAM_TOOLS and tool_input:
        return {
            "is_error": True,
            "error": {
                "error": "invalid_params",
                "message": (
                    f"{tool_name} takes no parameters, "
                    f"got: {', '.join(tool_input.keys())}"
                ),
            },
        }
    return None


async def _execute_tool(
    client: ChessSessionClient,
    tool_name: str,
    tool_input: Any,
) -> dict[str, Any]:
    """Execute a chess tool call via the MCP client."""
    validation_error = _validate_tool_input(tool_name, tool_input)
    if validation_error is not None:
        return validation_error

    try:
        tool_result: Any
        if tool_name == "make_move":
            tool_result = await client.make_move(tool_input["move"])
        elif tool_name == "offer_draw":
            tool_result = await client.offer_draw()
        elif tool_name == "claim_draw":
            tool_result = await client.claim_draw()
        elif tool_name == "accept_draw":
            tool_result = await client.accept_draw()
        elif tool_name == "decline_draw":
            tool_result = await client.decline_draw()
        elif tool_name == "resign":
            tool_result = await client.resign()
        else:
            return {
                "is_error": True,
                "error": {
                    "error": "unknown_tool",
                    "message": f"Unknown tool: {tool_name}",
                },
            }
        return {"result": dict(tool_result)}
    except McpError as e:
        return {"is_error": True, "error": e.to_dict()}


def _serialize_response(response: Any) -> dict[str, Any]:
    """Serialize an Anthropic API response for logging."""
    try:
        return response.model_dump()  # type: ignore[no-any-return]
    except AttributeError:
        return {"raw": str(response)}
