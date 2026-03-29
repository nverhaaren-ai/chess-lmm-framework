"""JSON-lines recording and RecordingClient decorator.

Provides three levels of observability:
1. MCP Recording: every tool call/response logged to JSON-lines
2. LLM Interaction Log: Anthropic API request/response pairs (separate file)
3. Orchestrator Log: Python logging for high-level events

This module handles level 1 (MCP Recording) and provides the
LlmInteractionLogger for level 2.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, TypeVar, cast

from chess_lmm.types import (
    BoardResult,
    CreateGameResult,
    DeclineDrawResult,
    DoneResult,
    ExportResult,
    GameStatus,
    HistoryResult,
    JoinGameResult,
    LegalMovesResult,
    MakeMoveResult,
    McpError,
    MessagesResult,
    OfferDrawResult,
    PreviewMoveResult,
    SendMessageResult,
)

_T = TypeVar("_T")


class GameRecorder:
    """Writes MCP tool call records to a JSON-lines file.

    Each line matches the recording format from SPEC.md Section 8.8.
    """

    def __init__(self, path: Path, session_id: str) -> None:
        self._path = path
        self._session_id = session_id
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        tool: str,
        params: dict[str, Any],
        result: dict[str, Any] | None,
        *,
        error: dict[str, Any] | None = None,
        elapsed_ms: int = 0,
    ) -> None:
        """Write a single tool invocation record."""
        entry: dict[str, Any] = {
            "ts_ms": int(time.time() * 1000),
            "session_id": self._session_id,
            "tool": tool,
            "params": params,
            "elapsed_ms": elapsed_ms,
        }
        if error is not None:
            entry["error"] = error
        else:
            entry["result"] = result

        with open(self._path, "a") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def write_marker(self, marker: dict[str, Any]) -> None:
        """Write a boundary marker (game_start, game_end)."""
        marker.setdefault("ts_ms", int(time.time() * 1000))
        with open(self._path, "a") as f:
            f.write(json.dumps(marker, separators=(",", ":")) + "\n")


class RecordingClient:
    """Decorator that wraps any ChessSessionClient and records all calls.

    Satisfies the ChessSessionClient protocol by delegating all calls
    to the wrapped client and logging via GameRecorder.
    """

    def __init__(self, client: Any, recorder: GameRecorder) -> None:
        self._client = client
        self._recorder = recorder

    @property
    def session_id(self) -> str:
        return self._client.session_id  # type: ignore[no-any-return]

    async def _call(self, tool: str, params: dict[str, Any], coro: Awaitable[_T]) -> _T:
        """Execute a tool call, record result or error, and return."""
        start = time.monotonic()
        try:
            result = await coro
            elapsed = int((time.monotonic() - start) * 1000)
            # TypedDicts are dicts at runtime; cast for the recorder
            result_dict = cast(dict[str, Any], result)
            self._recorder.record(tool, params, result_dict, elapsed_ms=elapsed)
            return result
        except McpError as e:
            elapsed = int((time.monotonic() - start) * 1000)
            self._recorder.record(
                tool, params, None, error=dict(e.to_dict()), elapsed_ms=elapsed
            )
            raise

    # --- Game management ---

    async def create_game(
        self,
        *,
        fen: str | None = None,
        history: list[str] | None = None,
    ) -> CreateGameResult:
        params: dict[str, Any] = {}
        if fen is not None:
            params["fen"] = fen
        if history is not None:
            params["history"] = history
        return await self._call(
            "create_game", params, self._client.create_game(fen=fen, history=history)
        )

    async def join_game(
        self,
        color: str,
        *,
        name: str | None = None,
    ) -> JoinGameResult:
        params: dict[str, Any] = {"color": color}
        if name is not None:
            params["name"] = name
        return await self._call(
            "join_game", params, self._client.join_game(color, name=name)
        )

    async def export_game(
        self,
        *,
        format: str = "pgn",
    ) -> ExportResult:
        return await self._call(
            "export_game", {"format": format}, self._client.export_game(format=format)
        )

    async def done(self) -> DoneResult:
        return await self._call("done", {}, self._client.done())

    # --- Query tools ---

    async def get_board(self) -> BoardResult:
        return await self._call("get_board", {}, self._client.get_board())

    async def get_status(self) -> GameStatus:
        return await self._call("get_status", {}, self._client.get_status())

    async def get_legal_moves(
        self,
        *,
        square: str | None = None,
        format: str = "both",
    ) -> LegalMovesResult:
        params: dict[str, Any] = {"format": format}
        if square is not None:
            params["square"] = square
        return await self._call(
            "get_legal_moves",
            params,
            self._client.get_legal_moves(square=square, format=format),
        )

    async def get_history(
        self,
        *,
        format: str = "san",
    ) -> HistoryResult:
        return await self._call(
            "get_history",
            {"format": format},
            self._client.get_history(format=format),
        )

    async def get_messages(
        self,
        *,
        clear: bool = True,
    ) -> MessagesResult:
        return await self._call(
            "get_messages",
            {"clear": clear},
            self._client.get_messages(clear=clear),
        )

    # --- Action tools ---

    async def make_move(self, move: str) -> MakeMoveResult:
        return await self._call(
            "make_move", {"move": move}, self._client.make_move(move)
        )

    async def preview_move(self, move: str) -> PreviewMoveResult:
        return await self._call(
            "preview_move", {"move": move}, self._client.preview_move(move)
        )

    async def claim_draw(self) -> GameStatus:
        return await self._call("claim_draw", {}, self._client.claim_draw())

    async def offer_draw(self) -> OfferDrawResult:
        return await self._call("offer_draw", {}, self._client.offer_draw())

    async def accept_draw(self) -> GameStatus:
        return await self._call("accept_draw", {}, self._client.accept_draw())

    async def decline_draw(self) -> DeclineDrawResult:
        return await self._call("decline_draw", {}, self._client.decline_draw())

    async def resign(self) -> GameStatus:
        return await self._call("resign", {}, self._client.resign())

    async def send_message(self, text: str) -> SendMessageResult:
        return await self._call(
            "send_message", {"text": text}, self._client.send_message(text)
        )


def _json_default(obj: Any) -> Any:
    """JSON fallback for SDK objects (e.g. Anthropic ThinkingBlock).

    Tries model_dump() (Pydantic), then falls back to str().
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


class LlmInteractionLogger:
    """Logs full Anthropic API request/response pairs to JSON-lines.

    Each line contains the messages sent to Claude, the response
    (including tool_use blocks), and usage statistics.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: dict[str, Any]) -> None:
        """Write a single interaction record."""
        entry.setdefault("ts_ms", int(time.time() * 1000))
        with open(self._path, "a") as f:
            f.write(
                json.dumps(
                    entry,
                    separators=(",", ":"),
                    default=_json_default,
                )
                + "\n"
            )


def render_board(fen: str) -> str:
    """Render a FEN position as an ASCII board diagram.

    Returns a string like:
      8  r  n  b  q  k  b  n  r
      7  p  p  p  p  p  p  p  p
      6  .  .  .  .  .  .  .  .
      5  .  .  .  .  .  .  .  .
      4  .  .  .  .  .  .  .  .
      3  .  .  .  .  .  .  .  .
      2  P  P  P  P  P  P  P  P
      1  R  N  B  Q  K  B  N  R
         a  b  c  d  e  f  g  h
    """
    if not fen or " " not in fen:
        return "(no board)"
    ranks_str = fen.split()[0]
    lines: list[str] = []

    for rank_idx, rank_fen in enumerate(ranks_str.split("/")):
        rank_num = 8 - rank_idx
        squares: list[str] = []
        for ch in rank_fen:
            if ch.isdigit():
                squares.extend(["."] * int(ch))
            else:
                squares.append(ch)
        line = f"  {rank_num}  " + "  ".join(squares)
        lines.append(line)

    lines.append("     a  b  c  d  e  f  g  h")
    return "\n".join(lines)
