"""Writing a result out: JSON for the whole thing, CSV for one table, "-" for stdout.

The CLIs share this so a `--json` file is the same shape the Dune-compatible API returns, and a
`--csv` file has the same columns in the same order.
"""
from __future__ import annotations

import csv
import json
import sys
from typing import Iterable, Sequence


def write_json(payload: object, path: str) -> None:
    """Write `payload` as indented JSON to `path`, or to stdout when `path` is "-"."""
    if path == "-":
        json.dump(payload, sys.stdout, indent=1, default=str)
        sys.stdout.write("\n")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)


def write_csv(rows: Iterable[dict], path: str, columns: Sequence[str]) -> None:
    """Write `rows` as CSV in `columns` order, "-" for stdout.

    A value that is a dict or a list (a row's `origins`, `by_exchange`) is written as JSON, so
    nothing a row carries is dropped.
    """
    columns = list(columns)
    out = sys.stdout if path == "-" else open(path, "w", newline="", encoding="utf-8")
    try:
        w = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: json.dumps(r[c], sort_keys=True) if isinstance(r.get(c), (dict, list)) else r.get(c)
                        for c in columns})
    finally:
        if out is not sys.stdout:
            out.close()
