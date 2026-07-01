from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .types import CodeChunk


CODE_SUFFIXES = {".py", ".java"}
TEXT_CHUNK_CHARS = 2000
MAX_STRUCTURAL_CHUNK_CHARS = 8000


class ChunkingError(RuntimeError):
    pass


def chunk_file(path: str, content: str) -> list[CodeChunk]:
    suffix = Path(path).suffix.lower()
    if suffix in CODE_SUFFIXES:
        return _tree_sitter_chunks(path, content, suffix)
    return _line_chunks(path, content, kind="text", max_chars=TEXT_CHUNK_CHARS)


def _tree_sitter_chunks(path: str, content: str, suffix: str) -> list[CodeChunk]:
    parser = _parser_for_suffix(suffix)
    tree = parser.parse(content.encode("utf-8"))
    root = tree.root_node
    nodes = _structural_nodes(root, suffix)
    chunks: list[CodeChunk] = []
    for node in nodes:
        kind = _kind_for_node(node.type)
        symbol = _node_symbol(node)
        text = _node_text(content, node)
        start_line = int(node.start_point[0]) + 1
        end_line = int(node.end_point[0]) + 1
        if len(text) > MAX_STRUCTURAL_CHUNK_CHARS:
            chunks.extend(_line_chunks(path, text, kind=kind, start_line=start_line, max_chars=TEXT_CHUNK_CHARS))
            continue
        chunks.append(_make_chunk(path, start_line, end_line, kind, symbol, text))

    if chunks:
        return chunks
    return _line_chunks(path, content, kind="code", max_chars=TEXT_CHUNK_CHARS)


def _parser_for_suffix(suffix: str) -> Any:
    try:
        from tree_sitter import Language, Parser
    except ImportError as exc:  # pragma: no cover - exercised only when optional runtime is absent.
        raise ChunkingError(
            "tree-sitter is required for Codebase Retrieval. Install tree-sitter, "
            "tree-sitter-python, and tree-sitter-java."
        ) from exc

    try:
        if suffix == ".py":
            import tree_sitter_python as language_module
        elif suffix == ".java":
            import tree_sitter_java as language_module
        else:
            raise ChunkingError(f"unsupported tree-sitter suffix: {suffix}")
    except ImportError as exc:  # pragma: no cover - exercised only when optional runtime is absent.
        raise ChunkingError(f"tree-sitter parser package is missing for {suffix} files.") from exc

    raw_language = language_module.language()
    try:
        language = Language(raw_language)
    except TypeError:
        language = raw_language

    parser = Parser()
    try:
        parser.language = language
    except AttributeError:
        parser.set_language(language)
    return parser


def _structural_nodes(root: Any, suffix: str) -> list[Any]:
    target_types = (
        {"class_definition", "function_definition"}
        if suffix == ".py"
        else {
            "annotation_type_declaration",
            "class_declaration",
            "constructor_declaration",
            "enum_declaration",
            "interface_declaration",
            "method_declaration",
            "record_declaration",
        }
    )
    nodes: list[Any] = []

    def walk(node: Any) -> None:
        if node.type in target_types:
            nodes.append(node)
        for child in node.children:
            walk(child)

    walk(root)
    return sorted(nodes, key=lambda node: (node.start_byte, node.end_byte))


def _kind_for_node(node_type: str) -> str:
    if "class" in node_type:
        return "class"
    if "method" in node_type:
        return "method"
    if "constructor" in node_type:
        return "constructor"
    if "interface" in node_type:
        return "interface"
    if "enum" in node_type:
        return "enum"
    if "record" in node_type:
        return "record"
    if "function" in node_type:
        return "function"
    return "code"


def _node_symbol(node: Any) -> str:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return ""
    return bytes(name_node.text).decode("utf-8", errors="replace")


def _node_text(content: str, node: Any) -> str:
    data = content.encode("utf-8")
    return data[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _line_chunks(
    path: str,
    content: str,
    *,
    kind: str,
    max_chars: int,
    start_line: int = 1,
) -> list[CodeChunk]:
    lines = content.splitlines(keepends=True)
    chunks: list[CodeChunk] = []
    current: list[str] = []
    current_start = start_line
    line_no = start_line

    for line in lines:
        if current and sum(len(part) for part in current) + len(line) > max_chars:
            text = "".join(current)
            chunks.append(_make_chunk(path, current_start, line_no - 1, kind, "", text))
            current = []
            current_start = line_no
        current.append(line)
        line_no += 1

    if current:
        text = "".join(current)
        chunks.append(_make_chunk(path, current_start, line_no - 1, kind, "", text))
    elif not content:
        chunks.append(_make_chunk(path, start_line, start_line, kind, "", ""))
    return chunks


def _make_chunk(
    path: str,
    start_line: int,
    end_line: int,
    kind: str,
    symbol: str,
    content: str,
) -> CodeChunk:
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    chunk_key = f"{path}:{start_line}:{end_line}:{kind}:{symbol}:{content_hash}"
    return CodeChunk(
        path=path,
        start_line=start_line,
        end_line=end_line,
        kind=kind,
        symbol=symbol,
        content=content,
        content_hash=content_hash,
        chunk_hash=hashlib.sha256(chunk_key.encode("utf-8")).hexdigest(),
    )
