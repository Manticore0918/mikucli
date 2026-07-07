from __future__ import annotations

from pathlib import Path

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only by live MCP benchmark setup.
    raise SystemExit("The MCP benchmark fixture server requires the 'mcp' Python package.") from exc


mcp = FastMCP("mikucli-bench-fixture")


@mcp.tool()
def read_fixture_note() -> str:
    """Read the benchmark fixture note from the current workspace."""
    return Path("fixture_note.txt").read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run()
