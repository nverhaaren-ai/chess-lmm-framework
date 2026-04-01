"""Live LLM tests for verification prompt effectiveness.

These tests call the real Anthropic API to check that verification
prompts actually cause Claude to reconsider bad moves. They are
expensive and non-deterministic, so they are marked with `@pytest.mark.live`
and deselected by default.

Run with:
    uv run pytest tests/test_live_verification.py -m live -v

Each test sends a recorded conversation (from game logs) to the API and
checks whether Claude avoids the blunder. Since LLM responses are
stochastic, each case is run N times and checked against a threshold
(e.g. "should avoid the blunder >= 80% of the time").

To add a new test case:
1. Find the blunder in game logs (llm_interactions.jsonl)
2. Extract the api_request payload at the verification prompt
3. Save as tests/fixtures/<name>.json
4. Update the fixture to use current prompt wording if needed
5. Write a test that loads the fixture, calls the API, and checks the
   response
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _require_api_key() -> str:
    """Return the Anthropic API key or skip the test."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


def _load_fixture(name: str) -> dict[str, Any]:
    """Load a conversation fixture from the fixtures directory."""
    path = FIXTURES_DIR / name
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _call_api(payload: dict[str, Any], api_key: str) -> dict[str, Any] | None:
    """Call the Anthropic Messages API and return the response.

    Returns None on transient API errors (rate limits, timeouts).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(**payload)
    except (
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
    ) as e:
        print(f"  API error (skipping trial): {e}")
        return None
    return response.model_dump()  # type: ignore[no-any-return]


def _extract_move(response: dict[str, Any]) -> str | None:
    """Extract the move from a make_move tool call in the response."""
    for block in response.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "make_move":
            return block.get("input", {}).get("move")  # type: ignore[no-any-return]
    return None


def _extract_tool_name(response: dict[str, Any]) -> str | None:
    """Extract the tool name from the first tool call in the response."""
    for block in response.get("content", []):
        if block.get("type") == "tool_use":
            return block.get("name")  # type: ignore[no-any-return]
    return None


@pytest.mark.live
class TestRd5Blunder:
    """Rd5 blunder: rook moves to a square where the queen can take it.

    Position: 2k4r/ppp2ppp/8/2Nrp3/8/3P1Q2/PPP2P1P/R3R1K1 w - - 1 15
    The LLM played Rd5, but the verification prompt clearly showed
    Qxd5 as a new opponent response. Claude should change its move
    to avoid hanging the rook.
    """

    FIXTURE = "rd5_blunder_verification.json"
    # The blunder move — if Claude confirms this, it's a failure.
    # Check both SAN and LAN forms since the tool accepts either.
    BLUNDER_MOVES = frozenset({"Rd5", "d8d5"})
    # Number of trials to run
    TRIALS = 5
    # Minimum success rate (fraction of trials where blunder is avoided)
    MIN_SUCCESS_RATE = 0.8

    def test_avoids_rd5_blunder(self) -> None:
        """Claude should NOT confirm Rd5 when shown Qxd5 as a threat."""
        api_key = _require_api_key()
        payload = _load_fixture(self.FIXTURE)

        successes = 0
        completed = 0
        results: list[dict[str, str | None]] = []

        for trial in range(self.TRIALS):
            response = _call_api(payload, api_key)
            if response is None:
                results.append(
                    {
                        "trial": str(trial + 1),
                        "move": None,
                        "tool": None,
                        "status": "API_ERROR",
                    }
                )
                continue

            completed += 1
            move = _extract_move(response)
            tool = _extract_tool_name(response)
            # A non-make_move response (resign, text-only) or a
            # different move all count as "avoided the blunder".
            avoided = move not in self.BLUNDER_MOVES
            if avoided:
                successes += 1
            results.append(
                {
                    "trial": str(trial + 1),
                    "move": move,
                    "tool": tool,
                    "status": "AVOIDED" if avoided else "BLUNDERED",
                }
            )

        if completed == 0:
            pytest.skip("All API calls failed")

        success_rate = successes / completed
        # Print results for manual inspection
        print(f"\n{'=' * 60}")
        print(f"Rd5 blunder avoidance: {successes}/{completed} ({success_rate:.0%})")
        print(f"{'=' * 60}")
        for r in results:
            print(
                f"  Trial {r['trial']}: {r['status']} — "
                f"tool={r['tool']}, move={r['move']}"
            )
        print()

        assert success_rate >= self.MIN_SUCCESS_RATE, (
            f"Blunder avoidance rate {success_rate:.0%} "
            f"< threshold {self.MIN_SUCCESS_RATE:.0%}. "
            f"Claude confirmed Rd5 too often."
        )
