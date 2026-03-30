"""Protocol definitions for the chess MCP session client and server factory.

These protocols define the abstraction boundary between the framework
(orchestrator, players) and the chess server (mock or real MCP).
"""

from __future__ import annotations

from typing import Protocol

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
    MessagesResult,
    OfferDrawResult,
    PreviewMoveResult,
    SendMessageResult,
)


class ChessSessionClient(Protocol):
    """Per-session interface to a chess MCP server.

    Each method corresponds to an MCP tool from SPEC.md Section 8.5.
    Implementations raise McpError on failure.
    """

    # --- Game management (Section 8.5.1) ---

    async def create_game(
        self,
        *,
        fen: str | None = None,
        history: list[str] | None = None,
    ) -> CreateGameResult:
        """Create a new game. Only valid in `no_game` state."""
        ...

    async def join_game(
        self,
        color: str,
        *,
        name: str | None = None,
    ) -> JoinGameResult:
        """Claim a seat in the active game."""
        ...

    async def export_game(
        self,
        *,
        format: str = "pgn",
    ) -> ExportResult:
        """Export the game in the requested format."""
        ...

    async def done(self) -> DoneResult:
        """Signal that this client is finished with the game."""
        ...

    # --- Query tools (Section 8.5.2) ---

    async def get_board(self) -> BoardResult:
        """Return the current board position as FEN."""
        ...

    async def get_status(self) -> GameStatus:
        """Return the current game status."""
        ...

    async def get_legal_moves(
        self,
        *,
        square: str | None = None,
        format: str = "both",
    ) -> LegalMovesResult:
        """Return all legal moves for the side to move."""
        ...

    async def get_history(
        self,
        *,
        format: str = "san",
    ) -> HistoryResult:
        """Return the move history."""
        ...

    async def get_messages(
        self,
        *,
        clear: bool = True,
    ) -> MessagesResult:
        """Return messages sent to this player."""
        ...

    async def preview_move(self, move: str) -> PreviewMoveResult:
        """Preview a move without applying it. Turn-gated, read-only."""
        ...

    # --- Action tools (Section 8.5.3) ---

    async def make_move(self, move: str) -> MakeMoveResult:
        """Submit a move. Turn-gated."""
        ...

    async def claim_draw(self) -> GameStatus:
        """Claim a draw (fifty-move or threefold repetition). Turn-gated."""
        ...

    async def offer_draw(self) -> OfferDrawResult:
        """Offer a draw to the opponent."""
        ...

    async def accept_draw(self) -> GameStatus:
        """Accept a pending draw offer."""
        ...

    async def decline_draw(self) -> DeclineDrawResult:
        """Decline a pending draw offer."""
        ...

    async def resign(self) -> GameStatus:
        """Resign the game."""
        ...

    async def send_message(self, text: str) -> SendMessageResult:
        """Send a message to the opponent."""
        ...


class ChessServerFactory(Protocol):
    """Factory that creates per-session clients for a chess server.

    The mock implementation creates clients backed by a shared in-memory game.
    The real MCP implementation would create clients that connect via HTTP.
    """

    async def create_session(self) -> ChessSessionClient:
        """Create a new session client."""
        ...
