"""Daily token inflows into exchange hot wallets: a realized-selling proxy.

Users deposit into per-user deposit addresses, which no public label set covers, and exchanges
sweep those into a few hot wallets. So the inflow counted here is every transfer of the token
into a labelled hot wallet from an address outside the labelled set: a sweep of deposits (hours
late), or a direct deposit. Transfers between two labelled wallets are internal and skipped.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional

from . import etherscan
from .cache import Cache
from .labels import hot_wallets
from .prices import daily_prices
from .report import write_csv, write_json

# Column order of a cex_inflows row, as the Dune-shaped API and the CLI both report it.
FLOW_COLUMNS = ["day", "tokens_to_cex", "usd_to_cex", "usd_price", "n_transfers", "by_exchange"]


def token_decimals(chain: str, token: str) -> int:
    """ERC-20 decimals() through Etherscan's eth_call proxy."""
    res = etherscan.call(chain, module="proxy", action="eth_call", to=token, data="0x313ce567", tag="latest")
    return int(res, 16) if res and res != "0x" else 18


def aggregate_daily(transfers: Iterable[dict], wallets: Dict[str, str], decimals: int,
                    start_day: str, end_day: str) -> List[dict]:
    """One row per UTC day in [start_day, end_day]: tokens_to_cex, n_transfers, by_exchange.

    `transfers` carry `wallet` (the receiving hot wallet), `from`, `ts` and raw `value`.
    Days with no inflow are included with zeros, so a quiet day reads as 0, not as missing.
    """
    scale = 10 ** decimals
    days: Dict[str, dict] = {}
    d = datetime.strptime(start_day, "%Y-%m-%d").date()
    last = datetime.strptime(end_day, "%Y-%m-%d").date()
    while d <= last:
        days[d.isoformat()] = {"day": d.isoformat(), "tokens_to_cex": 0.0, "n_transfers": 0, "by_exchange": {}}
        d += timedelta(days=1)
    for t in transfers:
        if t["from"].lower() in wallets:
            continue
        day = datetime.fromtimestamp(t["ts"], timezone.utc).strftime("%Y-%m-%d")
        if day not in days:
            continue
        row = days[day]
        amount = t["value"] / scale
        row["tokens_to_cex"] += amount
        row["n_transfers"] += 1
        ex = wallets[t["wallet"]]
        row["by_exchange"][ex] = row["by_exchange"].get(ex, 0.0) + amount
    return list(days.values())


def daily_inflows(chain: str, token: str, days: int = 365, end_ts: Optional[float] = None,
                  cache: Optional[Cache] = None, labels_path: Optional[str] = None,
                  progress: Optional[Callable[[str], None]] = None) -> List[dict]:
    """Daily inflows of `token` into labelled hot wallets over the last `days` UTC days."""
    chain = chain.lower()
    token = token.lower()
    if days < 1:
        # days <= 0 puts start_day after end_day: the block window inverts, the daily loop runs
        # zero times, and the inverted window used to be stored as cached coverage.
        raise ValueError(f"days must be at least 1, got {days}")
    wallets = hot_wallets(chain, labels_path)
    if not wallets:
        raise ValueError(f"no labelled hot wallets for chain {chain!r}")
    cache = cache or Cache()
    end = datetime.fromtimestamp(end_ts, timezone.utc) if end_ts else datetime.now(timezone.utc)
    end_day = end.date()
    start_day = end_day - timedelta(days=days - 1)
    ts_from = int(datetime(start_day.year, start_day.month, start_day.day, tzinfo=timezone.utc).timestamp())
    ts_to = int(end.timestamp())
    from_block = etherscan.block_at(chain, ts_from, "after")
    to_block = etherscan.block_at(chain, ts_to, "before")
    if from_block > to_block:
        raise ValueError(f"no blocks for {start_day}..{end_day} on {chain} "
                         f"(block {from_block} is after block {to_block})")
    decimals = token_decimals(chain, token)
    rows: List[dict] = []
    for i, (wallet, name) in enumerate(sorted(wallets.items()), 1):
        n = cache.ensure(chain, token, wallet, from_block, to_block, etherscan.inbound_transfers)
        if progress:
            progress(f"[{i}/{len(wallets)}] {name} {wallet}: {n} new transfers")
        for t in cache.transfers(chain, token, wallet, ts_from, ts_to + 1):
            t["wallet"] = wallet
            rows.append(t)
    daily = aggregate_daily(rows, wallets, decimals, start_day.isoformat(), end_day.isoformat())
    add_usd(daily, chain, token, ts_from, ts_to, cache, progress)
    return daily


def add_usd(daily: List[dict], chain: str, token: str, ts_from: int, ts_to: int,
            cache: Cache, progress: Optional[Callable[[str], None]] = None) -> None:
    """Set usd_to_cex (and usd_price) on each day from CoinGecko's daily price; None where no
    price is available, so a missing price never reads as $0."""
    try:
        prices = daily_prices(chain, token, ts_from, ts_to, cache)
    except Exception as e:
        if progress:
            progress(f"no USD prices: {e}")
        prices = {}
    for row in daily:
        price = prices.get(row["day"])
        row["usd_price"] = price
        row["usd_to_cex"] = row["tokens_to_cex"] * price if price is not None else None


def _num(value: Optional[float], digits: int = 0) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def main(argv: Optional[List[str]] = None) -> None:
    """Daily inflows into labelled hot wallets from the command line (API query 1)."""
    ap = argparse.ArgumentParser(description="Daily token inflows into exchange hot wallets.")
    ap.add_argument("--chain", default="ethereum")
    ap.add_argument("--token", required=True)
    ap.add_argument("--days", type=int, default=365, help="UTC days to report, today included")
    ap.add_argument("--end", help="YYYY-MM-DD: the last day reported (default today)")
    ap.add_argument("--json", metavar="PATH", help="write the rows as JSON ('-' for stdout)")
    ap.add_argument("--csv", metavar="PATH", help="write the rows as CSV ('-' for stdout)")
    ap.add_argument("--quiet", action="store_true", help="no progress on stderr")
    a = ap.parse_args(argv)
    end_ts = None
    if a.end:
        try:
            end_ts = datetime.strptime(a.end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() + 86399
        except ValueError:
            raise SystemExit(f"cexflows flows: --end must be YYYY-MM-DD, got {a.end!r}")
    progress = None if a.quiet else (lambda m: print(m, file=sys.stderr, flush=True))
    try:
        rows = daily_inflows(a.chain, a.token, days=a.days, end_ts=end_ts, progress=progress)
    except ValueError as e:
        raise SystemExit(f"cexflows flows: {e}")
    if a.json:
        write_json(rows, a.json)
    if a.csv:
        write_csv(rows, a.csv, FLOW_COLUMNS)
    if "-" in (a.json, a.csv):
        return                                   # stdout is carrying the data, not a table

    print(f"{'day':>10}  {'tokens to cex':>20}  {'USD':>18}  {'price':>12}  {'transfers':>9}  top exchanges")
    for r in rows:
        top = ", ".join(f"{k} {_num(v)}" for k, v in
                        sorted(r["by_exchange"].items(), key=lambda kv: -kv[1])[:3])
        print(f"{r['day']:>10}  {_num(r['tokens_to_cex'], 2):>20}  {_num(r.get('usd_to_cex'), 2):>18}  "
              f"{_num(r.get('usd_price'), 6):>12}  {r['n_transfers']:>9}  {top}")
    tokens = sum(r["tokens_to_cex"] for r in rows)
    usd = [r.get("usd_to_cex") for r in rows]
    total_usd = "-" if any(u is None for u in usd) else f"{sum(u or 0.0 for u in usd):,.2f}"
    print(f"\n{tokens:,.2f} tokens over {len(rows)} days ({sum(1 for r in rows if r['n_transfers'])} with inflow), "
          f"USD {total_usd}")


if __name__ == "__main__":
    main()
