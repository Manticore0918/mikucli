from __future__ import annotations

import asyncio
import math
from typing import Protocol
import httpx
from mikucli.http_requests import ModelResponseError, ProviderRequestError, RequestPolicy, request_json


PLAIN_RETRIEVAL_PROFILE = "plain-v1"
NOMIC_RETRIEVAL_PROFILE = "nomic-search-prefixes-v1"
NOMIC_DOCUMENT_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "


class EmbeddingError(RuntimeError):
    reason = "dependency_unavailable"


class EmbeddingClient(Protocol):
    model: str

    def embed(self, inputs: list[str]) -> list[list[float]]: ...


def retrieval_profile(model: str) -> str:
    model_name = model.casefold().rsplit("/", 1)[-1].split(":", 1)[0]
    if model_name.startswith("nomic-embed-text"):
        return NOMIC_RETRIEVAL_PROFILE
    return PLAIN_RETRIEVAL_PROFILE


def document_inputs(model: str, inputs: list[str]) -> list[str]:
    if retrieval_profile(model) == NOMIC_RETRIEVAL_PROFILE:
        return [f"{NOMIC_DOCUMENT_PREFIX}{value}" for value in inputs]
    return inputs


def query_inputs(model: str, inputs: list[str]) -> list[str]:
    if retrieval_profile(model) == NOMIC_RETRIEVAL_PROFILE:
        return [f"{NOMIC_QUERY_PREFIX}{value}" for value in inputs]
    return inputs


class OllamaEmbeddingClient:
    def __init__(self, *, model: str, base_url: str, request_policy: RequestPolicy | None = None,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.request_policy = request_policy or RequestPolicy()
        self.transport = transport

    def embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        try:
            raw_response, _ = asyncio.run(request_json(
                f"{self.base_url}/api/embed", policy=self.request_policy,
                payload={"model": self.model, "input": inputs}, transport=self.transport,
            ))
        except (ProviderRequestError, ModelResponseError) as exc:
            raise EmbeddingError(
                f"Ollama embedding service failed: {exc} Start Ollama, run 'ollama pull {self.model}', "
                "or correct MIKUCLI_OLLAMA_BASE_URL. Run mikucli --check to verify dependencies."
            ) from exc
        embeddings = raw_response.get("embeddings") if isinstance(raw_response, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise EmbeddingError("Ollama returned an invalid embedding response.")
        try:
            vectors = [[float(value) for value in embedding] for embedding in embeddings]
            if any(not vector or any(not math.isfinite(value) for value in vector) for vector in vectors):
                raise ValueError("empty or non-finite vector")
            return vectors
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Ollama returned invalid embedding vectors.") from exc
