from __future__ import annotations

from .types import SearchResult


def format_search_results(results: list[SearchResult], *, max_content_chars: int = 1200) -> str:
    if not results:
        return "No codebase results matched."
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        symbol = f" {result.symbol}" if result.symbol else ""
        blocks.append(
            "\n".join(
                [
                    f"{index}. {result.path}:{result.start_line}-{result.end_line} [{result.kind}{symbol}]",
                    "   "
                    + _score_summary(
                        hybrid=result.hybrid_score,
                        semantic=result.semantic_score,
                        lexical=result.lexical_score,
                    ),
                    _indent(_truncate(result.content.strip(), max_content_chars)),
                ]
            )
        )
    return "\n\n".join(blocks)


def _score_summary(*, hybrid: float, semantic: float | None, lexical: float | None) -> str:
    parts = [f"hybrid={hybrid:.4f}"]
    if semantic is not None:
        parts.append(f"semantic={semantic:.4f}")
    if lexical is not None:
        parts.append(f"lexical={lexical:.4f}")
    return " ".join(parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... truncated ..."


def _indent(text: str) -> str:
    return "\n".join(f"   {line}" if line else "" for line in text.splitlines())
