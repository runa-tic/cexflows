"""SQLite cache of transfers per (chain, token, wallet key), with the block range covered.

The wallet key is a hot wallet's address for its inbound transfers, or "out:<address>" for a
tracked wallet's outbound ones.

A wallet's history is downloaded once; later queries fetch only the blocks on either side of
what is already stored. Default location ~/.cexflows/cache.sqlite, or CEXFLOWS_CACHE.

Prices are cached per day alongside the spans that were requested from CoinGecko, so a token
minted years ago is priced once rather than on every run.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Tuple

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transfers (
    chain TEXT, token TEXT, wallet TEXT,
    block INTEGER, ts INTEGER, tx TEXT, log_index INTEGER, sender TEXT, recipient TEXT, value TEXT,
    PRIMARY KEY (chain, token, wallet, tx, log_index)
);
CREATE INDEX IF NOT EXISTS transfers_ts ON transfers (chain, token, wallet, ts);
CREATE TABLE IF NOT EXISTS coverage (
    chain TEXT, token TEXT, wallet TEXT, from_block INTEGER, to_block INTEGER,
    PRIMARY KEY (chain, token, wallet)
);
CREATE TABLE IF NOT EXISTS prices (
    chain TEXT, token TEXT, day TEXT, usd REAL,
    PRIMARY KEY (chain, token, day)
);
CREATE TABLE IF NOT EXISTS price_coverage (
    chain TEXT, token TEXT, from_day TEXT, to_day TEXT
);
CREATE TABLE IF NOT EXISTS token_logs (
    chain TEXT, token TEXT, block INTEGER, ts INTEGER, tx TEXT, log_index INTEGER,
    sender TEXT, recipient TEXT, value TEXT,
    PRIMARY KEY (chain, token, tx, log_index)
);
CREATE INDEX IF NOT EXISTS token_logs_ts ON token_logs (chain, token, ts);
CREATE TABLE IF NOT EXISTS token_log_coverage (
    chain TEXT, token TEXT, from_block INTEGER, to_block INTEGER
);
"""


def default_path() -> str:
    return os.environ.get("CEXFLOWS_CACHE") or os.path.join(os.path.expanduser("~"), ".cexflows", "cache.sqlite")


def _adjacent_day(day: str, step: int) -> str:
    return (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=step)).strftime("%Y-%m-%d")


def _merge_ranges(ranges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Overlapping or touching [from, to] block ranges as the fewest ranges."""
    out: List[Tuple[int, int]] = []
    for lo, hi in sorted(ranges):
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _merge_days(ranges: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Overlapping or consecutive [from_day, to_day] day ranges, as the fewest ranges."""
    out: List[Tuple[str, str]] = []
    for lo, hi in sorted(ranges):
        if out and lo <= _adjacent_day(out[-1][1], 1):
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


class Cache:
    def __init__(self, path: Optional[str] = None):
        self.path = path or default_path()
        if self.path != ":memory:":
            # A bare filename (CEXFLOWS_CACHE=cache.sqlite) lives in the working directory, whose
            # dirname is empty: makedirs("") raises FileNotFoundError.
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(_SCHEMA)

    def coverage(self, chain: str, token: str, wallet: str) -> Optional[Tuple[int, int]]:
        with self.lock:
            row = self.db.execute("SELECT from_block, to_block FROM coverage WHERE chain=? AND token=? AND wallet=?",
                                  (chain, token, wallet)).fetchone()
        return (row[0], row[1]) if row else None

    def store(self, chain: str, token: str, wallet: str, rows: Iterable[dict], from_block: int, to_block: int) -> int:
        """Insert transfers fetched for [from_block, to_block] and widen the recorded coverage.

        Only call with a range that was fetched completely, and only ranges that touch the
        existing coverage: coverage is one contiguous interval per wallet. An inverted range is
        refused rather than recorded: it would claim every block up to the larger bound as cached
        and the blocks a later query needs would never be fetched.
        """
        if from_block > to_block:
            raise ValueError(f"inverted block range for {chain} {wallet}: from {from_block} > to {to_block}")
        rows = list(rows)
        with self.lock, self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO transfers VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(chain, token, wallet, r["block"], r["ts"], r["tx"], r["log_index"], r["from"].lower(),
                  (r.get("to") or wallet).lower(), str(r["value"])) for r in rows],
            )
            cur = self.db.execute("SELECT from_block, to_block FROM coverage WHERE chain=? AND token=? AND wallet=?",
                                  (chain, token, wallet)).fetchone()
            lo, hi = (min(cur[0], from_block), max(cur[1], to_block)) if cur else (from_block, to_block)
            self.db.execute("INSERT OR REPLACE INTO coverage VALUES (?,?,?,?,?)", (chain, token, wallet, lo, hi))
        return len(rows)

    def transfers(self, chain: str, token: str, wallet: str, ts_from: int, ts_to: int) -> List[dict]:
        with self.lock:
            cur = self.db.execute(
                "SELECT block, ts, tx, log_index, sender, recipient, value FROM transfers "
                "WHERE chain=? AND token=? AND wallet=? AND ts>=? AND ts<? ORDER BY block, log_index",
                (chain, token, wallet, ts_from, ts_to))
            return [{"block": b, "ts": t, "tx": tx, "log_index": li, "from": s, "to": rc, "value": int(v)}
                    for b, t, tx, li, s, rc, v in cur.fetchall()]

    def prices(self, chain: str, token: str, day_from: str, day_to: str) -> Dict[str, float]:
        with self.lock:
            cur = self.db.execute("SELECT day, usd FROM prices WHERE chain=? AND token=? AND day>=? AND day<=?",
                                  (chain, token, day_from, day_to))
            return dict(cur.fetchall())

    def store_prices(self, chain: str, token: str, points: Dict[str, float]) -> None:
        with self.lock, self.db:
            self.db.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?,?)",
                                [(chain, token, d, p) for d, p in points.items()])

    def price_ranges(self, chain: str, token: str) -> List[Tuple[str, str]]:
        """The day spans that have been requested from CoinGecko, merged."""
        with self.lock:
            return self.db.execute("SELECT from_day, to_day FROM price_coverage WHERE chain=? AND token=? "
                                   "ORDER BY from_day", (chain, token)).fetchall()

    def prices_covered(self, chain: str, token: str, day_from: str, day_to: str) -> bool:
        """True when [day_from, day_to] was already requested, so no day in it is missing untried."""
        return any(a <= day_from and day_to <= b for a, b in self.price_ranges(chain, token))

    def store_price_range(self, chain: str, token: str, day_from: str, day_to: str) -> None:
        """Record that [day_from, day_to] was requested, merging it into the stored spans.

        A span means asked for, not necessarily served: CoinGecko has no price for a token before
        it listed, and a free key only serves a year of history.
        """
        if day_from > day_to:
            raise ValueError(f"inverted day range for {chain} {token}: {day_from} > {day_to}")
        with self.lock, self.db:
            own = list(self.db.execute("SELECT from_day, to_day FROM price_coverage WHERE chain=? AND token=?",
                                       (chain, token)))
            self.db.execute("DELETE FROM price_coverage WHERE chain=? AND token=?", (chain, token))
            self.db.executemany("INSERT INTO price_coverage VALUES (?,?,?,?)",
                                [(chain, token, a, b) for a, b in _merge_days([*own, (day_from, day_to)])])

    # Every transfer of a token in a block window (not just into labelled wallets), for cohort
    # tracking. Windows are stored as separate covered ranges, since cliffs are months apart.
    def token_log_ranges(self, chain: str, token: str) -> List[Tuple[int, int]]:
        with self.lock:
            return self.db.execute("SELECT from_block, to_block FROM token_log_coverage WHERE chain=? AND token=? "
                                   "ORDER BY from_block", (chain, token)).fetchall()

    def store_token_logs(self, chain: str, token: str, rows: Iterable[dict], from_block: int, to_block: int) -> int:
        """Insert every transfer of the token in [from_block, to_block] and record the range.

        Ranges are merged as they are written: one row per fetch would grow the table without
        bound and would make a window that spans two adjacent fetched ranges look uncoverable.
        """
        if from_block > to_block:
            raise ValueError(f"inverted block range for {chain} {token}: from {from_block} > to {to_block}")
        rows = list(rows)
        with self.lock, self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO token_logs VALUES (?,?,?,?,?,?,?,?,?)",
                [(chain, token, r["block"], r["ts"], r["tx"], r["log_index"], r["from"].lower(), r["to"].lower(),
                  str(r["value"])) for r in rows])
            own = list(self.db.execute("SELECT from_block, to_block FROM token_log_coverage "
                                       "WHERE chain=? AND token=?", (chain, token)))
            self.db.execute("DELETE FROM token_log_coverage WHERE chain=? AND token=?", (chain, token))
            self.db.executemany("INSERT INTO token_log_coverage VALUES (?,?,?,?)",
                                [(chain, token, a, b) for a, b in _merge_ranges([*own, (from_block, to_block)])])
        return len(rows)

    def token_logs(self, chain: str, token: str, ts_from: int, ts_to: int) -> List[dict]:
        with self.lock:
            cur = self.db.execute(
                "SELECT block, ts, tx, log_index, sender, recipient, value FROM token_logs "
                "WHERE chain=? AND token=? AND ts>=? AND ts<? ORDER BY block, log_index", (chain, token, ts_from, ts_to))
            return [{"block": b, "ts": t, "tx": tx, "log_index": li, "from": s, "to": rc, "value": int(v)}
                    for b, t, tx, li, s, rc, v in cur.fetchall()]

    def ensure(self, chain: str, token: str, wallet: str, from_block: int, to_block: int,
               fetch: Callable[[str, str, str, int, int], Iterable[dict]]) -> int:
        """Make sure [from_block, to_block] is cached for the wallet; returns rows fetched.

        An inverted range means the caller asked for a window that does not exist (a negative
        `days`, a zero-length release window). It is refused here rather than fetched and stored,
        because storing it would mark blocks up to the larger bound as covered.
        """
        if from_block > to_block:
            raise ValueError(f"inverted block range for {chain} {wallet}: from {from_block} > to {to_block}; "
                             "check the date window")
        have = self.coverage(chain, token, wallet)
        gaps = []
        if have is None:
            gaps.append((from_block, to_block))
        else:
            lo, hi = have
            if from_block < lo:
                gaps.append((from_block, lo - 1))
            if to_block > hi:
                gaps.append((hi + 1, to_block))
        fetched = 0
        for a, b in gaps:
            fetched += self.store(chain, token, wallet, fetch(chain, token, wallet, a, b), a, b)
        return fetched
