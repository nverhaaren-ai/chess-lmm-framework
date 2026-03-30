"""TypedDict response types matching the MCP tool interface spec (Section 8)."""

from __future__ import annotations

from typing import Literal, TypedDict

# --- Shared sub-types ---


class MoveNotation(TypedDict, total=False):
    """A move in SAN and/or LAN notation."""

    san: str
    lan: str


class CanClaimDraw(TypedDict):
    """Draw claim availability flags."""

    fifty_move: bool
    repetition: bool


class ClockState(TypedDict, total=False):
    """Chess clock state (Section 8.7.5)."""

    white_time_ms: int
    black_time_ms: int
    running_for: str | None  # "white" | "black" | None
    current_period: int
    moves_until_next_period: int | None
    time_control: dict[str, object]


# --- Game status values ---

GameStatusValue = Literal[
    "no_game",
    "awaiting_players",
    "ongoing",
    "check",
    "checkmate",
    "stalemate",
    "draw_fifty_move",
    "draw_repetition",
    "draw_automatic_fifty",
    "draw_automatic_repetition",
    "draw_agreement",
    "draw_inactivity",
    "resigned",
    "flag",
    "flag_draw",
]

ServerState = Literal["no_game", "awaiting_players", "ongoing", "game_over"]

Color = Literal["white", "black"]

ResultValue = Literal["1-0", "0-1", "1/2-1/2"]


# --- Tool response types ---


class CreateGameResult(TypedDict):
    """Response from create_game (Section 8.5.1)."""

    game_id: str
    fen: str
    game_status: Literal["awaiting_players"]
    time_control: dict[str, object] | None


class JoinGameResult(TypedDict):
    """Response from join_game (Section 8.5.1)."""

    assigned_color: str  # "white" | "black" | "spectator"
    game_status: Literal["awaiting_players", "ongoing"]


class BoardResult(TypedDict):
    """Response from get_board (Section 8.5.2)."""

    fen: str


class GameStatus(TypedDict, total=False):
    """Response from get_status and other status-returning tools (Section 8.5.2).

    Also the base shape for make_move, claim_draw, accept_draw, resign responses.
    All fields are always present in server responses; total=False is used here
    because some fields are nullable in the spec.
    """

    game_id: str | None
    server_state: ServerState
    game_status: GameStatusValue
    turn: Color | None
    fen: str | None
    fullmove_number: int
    halfmove_clock: int
    is_check: bool
    can_claim_draw: CanClaimDraw
    insufficient_material: bool
    draw_offered: bool
    last_move: MoveNotation | None
    result: ResultValue | None
    termination_reason: str | None
    clock: ClockState | None


class LegalMovesResult(TypedDict):
    """Response from get_legal_moves (Section 8.5.2)."""

    moves: list[MoveNotation]
    count: int


class HistoryMoveHalf(TypedDict, total=False):
    """One half-move in the history (white or black's move)."""

    san: str
    lan: str
    clock_ms: int | None


class HistoryEntry(TypedDict, total=False):
    """One full move in the history."""

    move_number: int
    white: HistoryMoveHalf | None
    black: HistoryMoveHalf | None


class HistoryResult(TypedDict):
    """Response from get_history (Section 8.5.2)."""

    moves: list[HistoryEntry]
    total_half_moves: int


class MessagesEntry(TypedDict):
    """A single message in the messages list."""

    from_: str  # "white" | "black" | "server" — field is "from" in JSON
    text: str
    move_number: int


class MessagesResult(TypedDict):
    """Response from get_messages (Section 8.5.2)."""

    messages: list[MessagesEntry]


class MakeMoveResult(GameStatus):
    """Response from make_move (Section 8.5.3).

    Same shape as GameStatus plus move_played.
    """

    move_played: MoveNotation


class PreviewMoveResult(TypedDict):
    """Response from preview_move (Section 8.5.2).

    Hypothetical position after a move, without applying it.
    """

    move: MoveNotation
    fen: str
    is_check: bool
    is_checkmate: bool
    is_stalemate: bool
    legal_responses: list[MoveNotation]
    legal_response_count: int
    new_threats: list[MoveNotation]


class OfferDrawResult(TypedDict):
    """Response from offer_draw (Section 8.5.3)."""

    offered: bool


class DeclineDrawResult(TypedDict):
    """Response from decline_draw (Section 8.5.3)."""

    declined: bool


class SendMessageResult(TypedDict):
    """Response from send_message (Section 8.5.3)."""

    sent: bool


class ExportResult(TypedDict):
    """Response from export_game (Section 8.5.1)."""

    content: str


class DoneResult(TypedDict):
    """Response from done (Section 8.5.1)."""

    acknowledged: bool
    clients_remaining: int


# --- Error type ---


class McpErrorContent(TypedDict, total=False):
    """Structured error content (Section 8.6)."""

    error: str
    message: str
    detail: str


class McpError(Exception):
    """Exception representing an MCP tool error (Section 8.6).

    Attributes:
        code: The error code (e.g. "illegal_move", "not_your_turn").
        message: Human-readable description of the error.
        detail: Optional implementation-defined context (for internal_error).
    """

    def __init__(self, code: str, message: str, *, detail: str | None = None) -> None:
        self.code = code
        self.message = message
        self.detail = detail
        super().__init__(f"{code}: {message}")

    def to_dict(self) -> McpErrorContent:
        """Serialize to the spec error format."""
        result: McpErrorContent = {"error": self.code, "message": self.message}
        if self.detail is not None:
            result["detail"] = self.detail
        return result
