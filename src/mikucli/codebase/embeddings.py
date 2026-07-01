from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol


class EmbeddingError(RuntimeError):
    pass


class EmbeddingClient(Protocol):
    model: str

    def embed(self, inputs: list[str]) -> list[list[float]]: ...


class OllamaEmbeddingClient:
    def __init__(self, *, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        payload = {"model": self.model, "input": inputs}
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            if "not found" in raw.casefold() or exc.code == 404:
                raise EmbeddingError(
                    f"Ollama model '{self.model}' is unavailable. Run: ollama pull {self.model}"
                ) from exc
            raise EmbeddingError(f"Ollama embedding request failed with HTTP {exc.code}: {raw}") from exc
        except urllib.error.URLError as exc:
            raise EmbeddingError(
                f"Ollama is not reachable at {self.base_url}. Start Ollama or set MIKUCLI_OLLAMA_BASE_URL."
            ) from exc

        raw_response = json.loads(body)
        embeddings = raw_response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise EmbeddingError("Ollama returned an invalid embedding response.")
        return [[float(value) for value in embedding] for embedding in embeddings]
