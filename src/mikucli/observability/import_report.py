from __future__ import annotations

import argparse
from pathlib import Path

from .store import LocalTraceStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mikucli.observability.import_report")
    parser.add_argument("reports", nargs="+", help="Benchmark JSON report path(s) to import.")
    parser.add_argument("--store-root", default=str(Path.cwd() / ".mikucli" / "observability"))
    args = parser.parse_args(argv)
    store = LocalTraceStore(Path(args.store_root), mode="sqlite")
    for report in args.reports:
        path = Path(report)
        store.import_eval_report(path)
        print(f"imported {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
