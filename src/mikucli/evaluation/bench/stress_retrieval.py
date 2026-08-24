from __future__ import annotations

from pathlib import Path

from mikucli.codebase.service import CodebaseService

from .models import TaskSetup
from .task_support import write_fixture


class StressEmbeddingClient:
    model = "stress-fake-embed"

    def embed(self, inputs: list[str]) -> list[list[float]]:
        return [_stress_vector(text) for text in inputs]


def setup_retrieval_with_false_matches(workspace: Path) -> TaskSetup:
    write_fixture(
        workspace / "README.md",
        (
            "# Export Dispatch\n\n"
            "Premium export routing has active, archived, test, and migration references. "
            "Use the active production implementation as the authority.\n"
        ),
    )
    write_fixture(workspace / "src" / "dispatch" / "__init__.py", "")
    write_fixture(
        workspace / "src" / "dispatch" / "premium_export.py",
        (
            '"""Active production policy for premium export routing."""\n\n'
            'PREMIUM_EXPORT_SENTINEL = "KESTREL-731"\n'
            'PREMIUM_EXPORT_QUEUE = "violet-fastlane"\n\n\n'
            "def premium_export_route() -> tuple[str, str]:\n"
            "    return PREMIUM_EXPORT_SENTINEL, PREMIUM_EXPORT_QUEUE\n"
        ),
    )
    write_fixture(
        workspace / "src" / "dispatch" / "router.py",
        (
            "from .premium_export import (\n"
            "    PREMIUM_EXPORT_QUEUE,\n"
            "    PREMIUM_EXPORT_SENTINEL,\n"
            ")\n\n\n"
            "def active_premium_export_route() -> tuple[str, str]:\n"
            "    return PREMIUM_EXPORT_SENTINEL, PREMIUM_EXPORT_QUEUE\n"
        ),
    )
    write_fixture(workspace / "src" / "archive" / "__init__.py", "")
    write_fixture(
        workspace / "src" / "archive" / "premium_export_v1.py",
        (
            '"""Retired legacy premium export routing policy. Do not use in production."""\n\n'
            'PREMIUM_EXPORT_SENTINEL = "KESTREL-318"\n'
            'PREMIUM_EXPORT_QUEUE = "violet-fastlane"\n'
        ),
    )
    write_fixture(
        workspace / "docs" / "premium-export-migration.md",
        (
            "# Premium export routing migration\n\n"
            "During migration, sentinel `KESTREL-731` temporarily routed to the "
            "`indigo-transition` queue. This historical route is no longer active.\n"
        ),
    )
    write_fixture(
        workspace / "src" / "migrations" / "premium_export_transition.py",
        (
            '"""Historical migration note; never use as the active route."""\n\n'
            'PREMIUM_EXPORT_SENTINEL = "KESTREL-731"\n'
            'PREMIUM_EXPORT_QUEUE = "indigo-transition"\n'
        ),
    )
    write_fixture(
        workspace / "tests" / "fixtures" / "premium_export_route.txt",
        (
            "Premium export routing test fixture\n"
            "sentinel=KESTREL-731\n"
            "queue=violet-sandbox\n"
            "environment=test-only\n"
        ),
    )
    write_fixture(
        workspace / "config" / "premium-export.example.toml",
        (
            "# Example only; not active production configuration.\n"
            'sentinel = "KESTREL-000"\n'
            'queue = "violet-fastlane"\n'
        ),
    )
    write_fixture(
        workspace / "src" / "examples" / "premium_export_example.py",
        (
            '"""Example premium export route; not active production policy."""\n\n'
            'PREMIUM_EXPORT_SENTINEL = "KESTREL-000"\n'
            'PREMIUM_EXPORT_QUEUE = "violet-fastlane"\n'
        ),
    )
    service = CodebaseService(
        workspace=workspace,
        embedding_provider="ollama",
        embedding_model="stress-fake-embed",
        ollama_base_url="http://localhost:11434",
        embedding_client=StressEmbeddingClient(),
    )
    service.rebuild_index()
    return TaskSetup(
        codebase_service=service,
        metadata={
            "authoritative_path": "src/dispatch/premium_export.py",
            "false_match_paths": [
                "src/archive/premium_export_v1.py",
                "src/migrations/premium_export_transition.py",
                "tests/fixtures/premium_export_route.txt",
                "src/examples/premium_export_example.py",
            ],
        },
    )


def _stress_vector(text: str) -> list[float]:
    lowered = text.casefold()
    return [
        float("premium" in lowered and "export" in lowered),
        float("routing" in lowered or "route" in lowered),
        float("sentinel" in lowered),
        float("queue" in lowered),
        float("active" in lowered),
        float("production" in lowered),
        -float(
            any(
                marker in lowered
                for marker in ("legacy", "retired", "historical", "fixture", "example")
            )
        ),
    ]
