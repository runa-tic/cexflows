"""Daily USD prices for a token by contract address, from CoinGecko, cached per day.

Key: COINGECKO_PRO_API_KEY or ~/.config/cexflows/coingecko_pro_key (pro host), else
COINGECKO_API_KEY or ~/.config/cexflows/coingecko_key (demo key, public host), else the public
host without a key. Pro keys only work on pro-api.coingecko.com and demo keys only on the public
host; mixing them is an error, not a downgrade. The free and demo tiers serve 365 days of history.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from .cache import Cache

PLATFORMS = {"ethereum": "ethereum", "arbitrum": "arbitrum-one"}
_CFG = os.path.join(os.path.expanduser("~"), ".config", "cexflows")


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def _day_ts(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _shift_day(day: str, step: int) -> str:
    return (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=step)).strftime("%Y-%m-%d")


def missing_days(ranges: List[Tuple[str, str]], day_from: str, day_to: str) -> List[Tuple[str, str]]:
    """The parts of [day_from, day_to] no stored span covers, as day spans."""
    out: List[Tuple[str, str]] = []
    cur = day_from
    for a, b in sorted(ranges):
        if b < cur or a > day_to:
            continue
        if a > cur:
            out.append((cur, _shift_day(a, -1)))
        cur = max(cur, _shift_day(b, 1))
        if cur > day_to:
            break
    if cur <= day_to:
        out.append((cur, day_to))
    return out


def _read(path: str) -> str:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def endpoint() -> Tuple[str, Dict[str, str]]:
    pro = os.environ.get("COINGECKO_PRO_API_KEY", "").strip() or _read(os.path.join(_CFG, "coingecko_pro_key"))
    if pro:
        return "https://pro-api.coingecko.com/api/v3", {"x-cg-pro-api-key": pro}
    demo = os.environ.get("COINGECKO_API_KEY", "").strip() or _read(os.path.join(_CFG, "coingecko_key"))
    return "https://api.coingecko.com/api/v3", ({"x-cg-demo-api-key": demo} if demo else {})


def _get(url: str, headers: Dict[str, str], retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cexflows", "accept": "application/json", **headers})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"CoinGecko HTTP {e.code}: {e.read()[:200]!r}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"CoinGecko: {e}") from e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("CoinGecko: retries exhausted")


def daily_prices(chain: str, token: str, start_ts: int, end_ts: int,
                 cache: Optional[Cache] = None) -> Dict[str, float]:
    """UTC day -> USD price for every day in [start_ts, end_ts] that CoinGecko has.

    Each day's value is CoinGecko's daily point for that date (its 00:00 UTC price). Only the day
    spans that were never requested are fetched: a token minted years ago used to be re-priced on
    every run, because a free key serves a year of history and the day count never matched the
    range asked for. A range that reaches today still re-reads today, whose price is moving, and
    only for that day.
    """
    chain, token = chain.lower(), token.lower()
    platform = PLATFORMS.get(chain)
    if not platform:
        raise ValueError(f"no CoinGecko platform for chain {chain!r}")
    if start_ts > end_ts:
        raise ValueError(f"start_ts {start_ts} is after end_ts {end_ts}")
    cache = cache or Cache()
    day_from, day_to = _day(start_ts), _day(end_ts)
    today = _day(time.time())
    spans: List[Tuple[str, str]] = []
    for lo, hi in missing_days(cache.price_ranges(chain, token), day_from, day_to):
        if day_to < today or hi < today:
            spans.append((lo, hi))
        elif lo < today:
            # Only the part before today: today, whose price is still moving, is re-read below.
            spans.append((lo, _shift_day(today, -1)))
    if day_to >= today:
        spans.append((today, day_to))
    for lo, hi in spans:
        base, headers = endpoint()
        q = urllib.parse.urlencode({"vs_currency": "usd", "from": _day_ts(lo) - 86400, "to": _day_ts(hi) + 86400})
        data = _get(f"{base}/coins/{platform}/contract/{token}/market_chart/range?{q}", headers)
        points: Dict[str, float] = {}
        for ms, price in data.get("prices", []):
            d = _day(ms / 1000)
            # Keep the earliest point of each day: CoinGecko's daily series stamps 00:00 UTC.
            if d not in points:
                points[d] = float(price)
        cache.store_prices(chain, token, points)
        cache.store_price_range(chain, token, lo, hi)
    return cache.prices(chain, token, day_from, day_to)


def daily_market(chain: str, token: str, start_ts: int, end_ts: int) -> Dict[str, Dict[str, float]]:
    """UTC day -> {"price": 00:00 UTC price, "volume": that day's 24h USD trading volume}.

    CoinGecko stamps each volume with the end of its 24 hours, so a day's volume is the first
    point of the following day. Not cached: one call covers a whole study.
    """
    platform = PLATFORMS.get(chain.lower())
    if not platform:
        raise ValueError(f"no CoinGecko platform for chain {chain!r}")
    base, headers = endpoint()
    q = urllib.parse.urlencode({"vs_currency": "usd", "from": int(start_ts) - 86400, "to": int(end_ts) + 2 * 86400})
    data = _get(f"{base}/coins/{platform}/contract/{token.lower()}/market_chart/range?{q}", headers)
    day = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")
    first_price: Dict[str, float] = {}
    first_volume: Dict[str, float] = {}
    for ms, p in data.get("prices", []):
        first_price.setdefault(day(ms), float(p))
    for ms, v in data.get("total_volumes", []):
        first_volume.setdefault(day(ms), float(v))
    out = {}
    for d, p in first_price.items():
        nxt = datetime.strftime(datetime.strptime(d, "%Y-%m-%d") + timedelta(days=1), "%Y-%m-%d")
        if nxt in first_volume:
            out[d] = {"price": p, "volume": first_volume[nxt]}
    return out
