"""Exchange hot wallets per chain, from labels/hot_wallets.csv (chain,address,exchange,source).

The labels carry eth-labels' MIT notice (`labels/LICENSE-eth-labels`), so they stay in the
repository instead of inside the package: a plain `pip install` of the wheel has no label file
and needs CEXFLOWS_LABELS pointing at the CSV.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, Optional

DEFAULT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "labels", "hot_wallets.csv")


def label_file(path: Optional[str] = None) -> str:
    """`path`, else CEXFLOWS_LABELS, else the repository's labels/hot_wallets.csv."""
    return path or os.environ.get("CEXFLOWS_LABELS") or DEFAULT_FILE


def hot_wallets(chain: str, path: Optional[str] = None) -> Dict[str, str]:
    """address (lowercase) -> exchange name, for one chain."""
    resolved = label_file(path)
    out: Dict[str, str] = {}
    try:
        f = open(resolved, newline="", encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"no hot-wallet labels at {resolved}: set CEXFLOWS_LABELS to labels/hot_wallets.csv "
            "from the repository (the labels are not packaged with the wheel)") from None
    with f:
        for row in csv.DictReader(f):
            if row["chain"].strip().lower() == chain.lower():
                out[row["address"].strip().lower()] = row["exchange"].strip()
    return out
