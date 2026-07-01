from __future__ import annotations

from pathlib import Path


class WorkspaceError(ValueError):
    pass


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def resolve(self, raw_path: str | Path = ".") -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            candidate = path.expanduser().resolve()
        else:
            candidate = (self.root / path).resolve()

        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError(f"path is outside workspace: {raw_path}")

        return candidate

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()
