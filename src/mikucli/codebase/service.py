from __future__ import annotations

from pathlib import Path
from typing import Callable

from .chunking import chunk_file
from .embeddings import (
    EmbeddingClient,
    OllamaEmbeddingClient,
    document_inputs,
    query_inputs,
    retrieval_profile,
)
from .files import select_index_files
from .index import ActiveIndex, CodebaseIndexReader, CodebaseIndexWriter
from .types import IndexStats, SearchResult


Progress = Callable[[str], None]


class CodebaseService:
    def __init__(
        self,
        *,
        workspace: Path,
        embedding_provider: str,
        embedding_model: str,
        ollama_base_url: str,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        embedding_provider = embedding_provider.casefold()
        if embedding_provider != "ollama":
            raise ValueError("Codebase Retrieval v1 supports only MIKUCLI_EMBEDDING_PROVIDER=ollama")
        self.workspace = workspace.resolve()
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_profile = retrieval_profile(embedding_model)
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.active_index = ActiveIndex(self.workspace / ".mikucli" / "codebase_index")
        self.embedding_client = embedding_client or OllamaEmbeddingClient(
            model=embedding_model,
            base_url=self.ollama_base_url,
        )

    @property
    def index_path(self) -> Path:
        return self.active_index.db_path

    def rebuild_index(self, progress: Progress | None = None) -> IndexStats:
        progress = progress or (lambda _message: None)
        progress(
            "Codebase Retrieval: "
            f"provider={self.embedding_provider} model={self.embedding_model} "
            f"base_url={self.ollama_base_url} index={self.index_path}"
        )
        selection = select_index_files(self.workspace)
        stats = IndexStats(
            files_scanned=selection.scanned,
            files_skipped=len(selection.skips),
            embedding_model=self.embedding_model,
            index_path=self.index_path,
            skips=selection.skips,
        )
        progress(f"Indexing: scanned={stats.files_scanned} selected={len(selection.files)} skipped={stats.files_skipped}")

        with CodebaseIndexWriter(
            self.active_index,
            self.embedding_model,
            self.embedding_profile,
        ) as writer:
            for index, file in enumerate(selection.files, start=1):
                content = file.absolute_path.read_text(encoding="utf-8")
                chunks = chunk_file(file.path, content)
                inputs = document_inputs(
                    self.embedding_model,
                    [chunk.content for chunk in chunks],
                )
                embeddings = self.embedding_client.embed(inputs)
                writer.add_file(file, chunks, embeddings)
                stats.files_indexed += 1
                stats.chunks_embedded += len(chunks)
                if index == len(selection.files) or index % 10 == 0:
                    progress(
                        "Indexing: "
                        f"files_indexed={stats.files_indexed}/{len(selection.files)} "
                        f"chunks_embedded={stats.chunks_embedded}"
                    )
            writer.commit_active()

        progress(
            "Codebase Index ready: "
            f"files={stats.files_indexed} chunks={stats.chunks_embedded} "
            f"skipped={stats.files_skipped} model={stats.embedding_model} path={stats.index_path}"
        )
        return stats

    def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        embeddings = self.embedding_client.embed(query_inputs(self.embedding_model, [query]))
        return CodebaseIndexReader(
            self.active_index,
            embedding_model=self.embedding_model,
            embedding_profile=self.embedding_profile,
        ).search(query, embeddings[0], limit=limit)
