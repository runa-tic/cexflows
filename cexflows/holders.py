"""Holder positions: every wallet's cost basis and sale history, rebuilt from the token's transfers.

Tokens are followed as cost lots from the moment they are minted:

    mint                               lots enter at zero cost (origin "mint")
    wallet -> wallet                   lots move with their cost, oldest first
    exchange wallet or pool -> wallet  a purchase: the lot is repriced at that day's price
    wallet -> exchange wallet or pool  a sale: recorded for the sender with the multiple of cost
    wallet -> exchange deposit address a sale (the address only forwards to a hot wallet)

So vesting and treasury contracts hand zero-cost lots to insiders without being labelled, and
buyers carry what they paid. Exchange-side purchases are priced at the day's price: the DEX pool
price where a pool was deep enough is not yet used (ALT, the first token, has no deep pool), so
every price is CoinGecko's daily close for now.

Not modelled as holders, though their transfers still move lots: exchange wallets (many users
each), pools, contracts other than Safe multisigs, and unlabelled services (wallets with
thousands of counterparties).
"""
from __future__ import annotations

import bisect
import json
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple

from . import etherscan, sellcurve
from .cache import Cache
from .flows import token_decimals
from .labels import hot_wallets
from .prices import daily_prices

ZERO = "0x" + "0" * 40
GECKOTERMINAL = {"ethereum": "eth", "arbitrum": "arbitrum"}
SAFE_GET_OWNERS = "0xa0e67e2b"     # getOwners(): answers only on a Safe multisig
SERVICE_COUNTERPARTIES = 2000      # more distinct counterparties than this: a service, not a holder
TRADER_SALES = 100                 # this many sales or more: a market maker or bot, not a holder
# A wallet holding this share of supply or more, all at zero cost and never sold, is a treasury
# the project-wallet rule missed (APEX: one wallet with 50% of supply, another 15.7%).
TREASURY_SHARE = 0.05
HISTORY_CHUNK = 250_000            # blocks per cached chunk of a token's history


def dex_pools(chain: str, token: str) -> Dict[str, str]:
    """Pool address -> dex id for the token's pools, from GeckoTerminal's free API."""
    url = f"https://api.geckoterminal.com/api/v2/networks/{GECKOTERMINAL[chain]}/tokens/{token}/pools?page=1"
    req = urllib.request.Request(url, headers={"accept": "application/json", "User-Agent": "cexflows"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    out = {}
    for p in data.get("data", []):
        addr = p["attributes"]["address"].lower()
        if addr.startswith("0x") and len(addr) == 42:   # v4 pools are ids, not addresses
            out[addr] = p["relationships"]["dex"]["data"]["id"]
    return out


def is_safe(chain: str, address: str) -> bool:
    try:
        r = etherscan.call(chain, module="proxy", action="eth_call", to=address, data=SAFE_GET_OWNERS, tag="latest")
    except etherscan.EtherscanError:
        return False
    return isinstance(r, str) and len(r) > 130


def services(transfers: Iterable[dict], limit: int = SERVICE_COUNTERPARTIES) -> Set[str]:
    """Wallets with more than `limit` distinct counterparties: exchanges and other services that
    the labels miss. They act for many users, so they are treated like exchange wallets."""
    peers: Dict[str, Set[str]] = defaultdict(set)
    for t in transfers:
        for a, b in ((t["from"], t["to"]), (t["to"], t["from"])):
            if a != ZERO and len(peers[a]) <= limit:
                peers[a].add(b)
    return {a for a, p in peers.items() if len(p) > limit}


def deposit_addresses(transfers: Iterable[dict], market: Set[str], min_share: float = 0.9) -> Set[str]:
    """Exchange deposit addresses: they pass on what they receive, almost all of it into market
    wallets. Both conditions matter: a holder who once sold to an exchange sends only there too,
    but keeps most of what it received."""
    in_total: Dict[str, int] = defaultdict(int)
    out_total: Dict[str, int] = defaultdict(int)
    out_market: Dict[str, int] = defaultdict(int)
    for t in transfers:
        in_total[t["to"]] += t["value"]
        if t["from"] in market or t["from"] == ZERO:
            continue
        out_total[t["from"]] += t["value"]
        if t["to"] in market:
            out_market[t["from"]] += t["value"]
    return {a for a, v in out_market.items()
            if v >= min_share * out_total[a] and v >= min_share * in_total[a]}


def project_wallets(transfers: List[dict], market: Set[str], is_contract: Callable[[str], bool],
                    min_share: float = 0.9, min_recipients: int = 3) -> Set[str]:
    """The project's own wallets, found from the token's flow: whoever received the mint, plus
    any contract (Safes included) that got nearly all its tokens from a project wallet and pays
    them on to several addresses itself: treasuries, distributors, vesting vaults. They release
    on the unlock schedule rather than at a price."""
    inflow: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    recipients: Dict[str, Set[str]] = defaultdict(set)
    for t in transfers:
        inflow[t["to"]][t["from"]] += t["value"]
        if t["to"] not in market and len(recipients[t["from"]]) < min_recipients:
            recipients[t["from"]].add(t["to"])
    project = {t["to"] for t in transfers if t["from"] == ZERO and t["to"] not in market}
    grew = True
    while grew:
        grew = False
        for w, by in inflow.items():
            if w in project or w in market or w == ZERO or len(recipients[w]) < min_recipients:
                continue
            if sum(v for f, v in by.items() if f in project) >= min_share * sum(by.values()) and is_contract(w):
                project.add(w)
                grew = True
    return project


class Book:
    """Cost lots per wallet, oldest first: [amount, unit cost in USD, acquired ts, origin]."""

    def __init__(self):
        self.lots: Dict[str, Deque[list]] = defaultdict(deque)
        self.held: Dict[str, int] = defaultdict(int)

    def add(self, wallet: str, amount: int, cost: float, ts: int, origin: str) -> None:
        if amount > 0:
            self.lots[wallet].append([amount, cost, ts, origin])
            self.held[wallet] += amount

    def take(self, wallet: str, amount: int) -> List[list]:
        """Remove `amount` oldest first; returns the pieces taken. A shortfall (a sender with
        tokens from before the history, which shouldn't happen from mint) comes back as an
        "unknown" piece at no cost information."""
        taken, q = [], self.lots[wallet]
        while amount > 0 and q:
            lot = q[0]
            part = min(lot[0], amount)
            taken.append([part, lot[1], lot[2], lot[3]])
            lot[0] -= part
            amount -= part
            self.held[wallet] -= part
            if lot[0] == 0:
                q.popleft()
        if amount > 0:
            taken.append([amount, None, None, "unknown"])
        return taken

    def balance(self, wallet: str) -> int:
        return self.held[wallet]


def build_positions(transfers: List[dict], hot: Set[str], pools: Set[str], deposits: Set[str],
                    price_on: Callable[[int], Optional[float]]) -> Tuple[Book, Dict[str, List[dict]]]:
    """Replay transfers (oldest first). Returns the lot book and each wallet's sales."""
    book = Book()
    sales: Dict[str, List[dict]] = defaultdict(list)
    market = hot | pools
    for t in transfers:
        src, dst, v, ts = t["from"], t["to"], t["value"], t["ts"]
        if v == 0 or src == dst:
            continue
        price = price_on(ts)
        if src == ZERO:
            book.add(dst, v, 0.0, ts, "mint")
            continue
        held = book.balance(src)
        pieces = book.take(src, v)
        if src in market and dst not in market:
            # A purchase: whatever the market wallet held is replaced by what the buyer paid.
            book.add(dst, v, price if price is not None else float("nan"), ts, "pool" if src in pools else "exchange")
            continue
        if dst in market or dst in deposits:
            if src not in market:
                known = [p for p in pieces if p[1] is not None]
                qty = sum(p[0] for p in known)
                cost = sum(p[0] * p[1] for p in known) / qty if qty else None
                sales[src].append({"ts": ts, "tokens": v, "held": held, "price": price, "cost": cost,
                                   "venue": "pool" if dst in pools else "exchange"})
            if dst in deposits:
                book.add(dst, v, price if price is not None else float("nan"), ts, "deposit")
            continue
        for amount, cost, acq, origin in pieces:
            book.add(dst, amount, cost if cost is not None else (price or float("nan")), acq or ts, origin)
    return book, sales


def _median(xs: List[float]) -> Optional[float]:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def sale_fraction(sale: dict) -> Optional[float]:
    return min(sale["tokens"] / sale["held"], 1.0) if sale.get("held") else None


def profit_multiple(sale: dict) -> Optional[float]:
    """Sale price over cost, for sales made in profit with a known, non-zero cost."""
    if sale["price"] and sale["cost"] and sale["price"] > sale["cost"]:
        return sale["price"] / sale["cost"]
    return None


# Holder size bands by share of supply at the time of a sale. Small wallets dump everything at
# once while large ones sell in slices (ALT: median 100% below 0.001% of supply, 13% above 0.1%),
# so a holder's priors come from sales by wallets of its own size.
SIZE_BANDS = [0.001, 0.0001, 0.00001, 0.0]


def size_band(share: float) -> float:
    return next(b for b in SIZE_BANDS if share >= b)


def peer_priors(sales: Dict[str, List[dict]], exclude: Set[str], supply: int) -> Dict[float, dict]:
    """What holders of this token typically do, per size band, from every wallet's sales: the
    share of its balance a holder sells at once, and the multiple of cost at which it takes
    profit. Market wallets, services, deposit addresses, project wallets and traders are left
    out. A band with fewer than 20 sales borrows from the next band down."""
    fractions: Dict[float, List[float]] = defaultdict(list)
    multiples: Dict[float, List[float]] = defaultdict(list)
    for w, ss in sales.items():
        if w in exclude or len(ss) >= TRADER_SALES:
            continue
        for x in ss:
            band = size_band(x["held"] / supply) if supply else 0.0
            f, m = sale_fraction(x), profit_multiple(x)
            if f is not None:
                fractions[band].append(f)
            if m is not None:
                multiples[band].append(m)
    out: Dict[float, dict] = {}
    for i, band in enumerate(SIZE_BANDS):
        fr, mu = list(fractions[band]), list(multiples[band])
        for lower in SIZE_BANDS[i + 1:]:
            if len(fr) >= 20 and len(mu) >= 20:
                break
            fr, mu = fr + fractions[lower], mu + multiples[lower]
        out[band] = {"sell_fraction": _median(fr), "take_profit_multiple": _median(mu),
                     "n_sales": len(fr), "n_profitable": len(mu)}
    return out


def sell_levels(kind: str, balance: float, cost: Optional[float], own_sales: List[dict],
                priors: dict, price_now: Optional[float]) -> dict:
    """A price-triggered holder's two levels, each with the tokens it would sell there:

      break-even   only when underwater: price gets back to cost, sell the usual fraction
      take-profit  cost x the take-profit multiple, sell the usual fraction of what is left;
                   a holder already past it sells at today's price

    The fraction and multiple are the wallet's own medians with two or more sales of that
    kind, otherwise the peer priors for its size band. Other kinds have no price levels."""
    none = {"breakeven_price": None, "breakeven_tokens": 0.0, "takeprofit_price": None, "takeprofit_tokens": 0.0,
            "sell_fraction": None, "take_profit_multiple": None, "confidence": None}
    if kind != "price" or not cost or not price_now:
        return none
    fracs = [f for f in map(sale_fraction, own_sales) if f is not None]
    mults = [m for m in map(profit_multiple, own_sales) if m is not None]
    f = _median(fracs) if len(fracs) >= 2 else priors["sell_fraction"]
    m = _median(mults) if len(mults) >= 2 else priors["take_profit_multiple"]
    if f is None or m is None:
        return none
    out = dict(none, sell_fraction=f, take_profit_multiple=m,
               confidence="high" if len(fracs) >= 5 else "medium" if len(fracs) >= 2 else "low")
    left = balance
    if cost > price_now:
        out.update(breakeven_price=cost, breakeven_tokens=f * left)
        left -= f * left
    out.update(takeprofit_price=max(cost * m, price_now), takeprofit_tokens=f * left)
    return out


def supply_curve(holders: List[dict], price_now: float, max_multiple: float = 3.0,
                 step_pct: float = 5.0, target_price: Optional[float] = None,
                 curves: Optional[dict] = None) -> List[dict]:
    """Tokens price-triggered holders would sell, by price step from today's price up to
    `max_multiple` times it: each step's own tokens and the running total, split by level.
    `target_price` adds a step edge at exactly that price, so a buyback priced to it stops there
    instead of at the top of the step it falls in. With `curves` (sellcurve.load()), each step
    also has `curve_tokens` and `trader_tokens`: the net selling the fitted sell-through curves
    expect from the holders and the market makers as the price runs through it. The backtest
    found the levels under-predict rally selling 2-32x; the curves came within 1.2-2.7x on three
    of four held-out rallies and over-predicted the fourth 4x."""
    edges = [price_now]
    while edges[-1] < price_now * max_multiple:
        edges.append(edges[-1] * (1 + step_pct / 100))
    if target_price and price_now < target_price < edges[-1] and min(abs(e / target_price - 1) for e in edges) > 1e-9:
        bisect.insort(edges, target_price)
    rows, cum = [], 0.0
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        # Steps are (lo, hi]; the first also takes levels sitting exactly at today's price.
        inside = lambda p: p is not None and (lo <= p <= hi if i == 0 else lo < p <= hi)
        be = [(h["breakeven_tokens"], h) for h in holders if inside(h["breakeven_price"])]
        tp = [(h["takeprofit_tokens"], h) for h in holders if inside(h["takeprofit_price"])]
        step = sum(t for t, _ in be) + sum(t for t, _ in tp)
        cum += step
        layers = sellcurve.step_tokens(holders, curves, price_now, lo, hi) if curves else None
        rows.append({"price_from": lo, "price_to": hi, "breakeven_tokens": sum(t for t, _ in be),
                     "takeprofit_tokens": sum(t for t, _ in tp), "step_tokens": step, "cumulative_tokens": cum,
                     "step_usd": step * (lo + hi) / 2, "wallets": len({h["address"] for _, h in be + tp}),
                     "curve_tokens": layers["holders"] if layers else None,
                     "trader_tokens": layers["traders"] if layers else None})
    return rows


def ensure_history(chain: str, token: str, cache: Cache, progress: Callable[[str], None] = lambda m: None) -> None:
    """Cache every transfer of the token since its contract was created, in resumable chunks.
    The first pull of a small cap takes about an hour (ALT: 1.5M transfers, 1,500 calls)."""
    created = etherscan.call(chain, module="contract", action="getcontractcreation", contractaddresses=token)
    start = int(created[0]["blockNumber"])
    end = etherscan.latest_block(chain)
    covered = cache.token_log_ranges(chain, token)
    b = start
    while b <= end:
        z = min(b + HISTORY_CHUNK - 1, end)
        gaps, lo = [], b
        for c_lo, c_hi in covered:      # fetch only the parts of [b, z] not already cached
            if c_hi < lo or c_lo > z:
                continue
            if c_lo > lo:
                gaps.append((lo, c_lo - 1))
            lo = max(lo, c_hi + 1)
        if lo <= z:
            gaps.append((lo, z))
        for g_lo, g_hi in gaps:
            n = cache.store_token_logs(chain, token, etherscan.transfers(chain, token, g_lo, g_hi), g_lo, g_hi)
            progress(f"history blocks {g_lo}-{g_hi}: {n} transfers")
        b = z + 1


def market_view(chain: str, token: str, end_ts: int, cache: Optional[Cache] = None,
                labels_path: Optional[str] = None, progress: Callable[[str], None] = lambda m: None) -> dict:
    """The token's transfers before `end_ts`, the market wallets they reveal (exchanges, services,
    pools, deposit addresses, project wallets), daily prices, and the lot book and sales replayed
    from the mint. The history is fetched into the cache first if missing."""
    cache = cache or Cache()
    ensure_history(chain, token, cache, progress)
    transfers = cache.token_logs(chain, token, 0, end_ts)
    if not transfers or transfers[0]["from"] != ZERO:
        raise ValueError("the cached history doesn't start at the token's first mint; pull it first")
    progress(f"{len(transfers)} transfers")
    scale = 10 ** token_decimals(chain, token)
    hot = set(hot_wallets(chain, labels_path))
    pools = set(dex_pools(chain, token))
    service = services(transfers) - hot - pools
    hot |= service
    deposits = deposit_addresses(transfers, hot | pools) - pools
    prices = daily_prices(chain, token, transfers[0]["ts"], end_ts - 1, cache)
    day = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
    price_on = lambda ts: prices.get(day(ts))
    progress(f"{len(hot) - len(service)} hot wallets, {len(service)} services, {len(pools)} pools, {len(deposits)} deposit addresses, {len(prices)} price days")
    contract_memo: Dict[str, bool] = {}

    def contract(a: str) -> bool:
        if a not in contract_memo:
            contract_memo[a] = etherscan.is_contract(chain, a)
        return contract_memo[a]

    project = project_wallets(transfers, hot | pools, contract)
    progress(f"{len(project)} project wallets")
    book, sales = build_positions(transfers, hot, pools, deposits, price_on)
    return {"transfers": transfers, "scale": scale, "hot": hot, "pools": pools, "service": service,
            "deposits": deposits, "project": project, "prices": prices, "day": day, "contract": contract,
            "book": book, "sales": sales}


def holder_positions(chain: str, token: str, top_n: int = 100, as_of: Optional[str] = None,
                     cache: Optional[Cache] = None, labels_path: Optional[str] = None,
                     progress: Callable[[str], None] = lambda m: None) -> dict:
    """The top N modelled holders with balance, cost basis and sale history, from the token's
    full transfer history (fetched into the cache first if missing)."""
    chain, token = chain.lower(), token.lower()
    end_ts = (int(datetime.strptime(as_of, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) + 86400
              if as_of else int(datetime.now(timezone.utc).timestamp()) + 1)
    v = market_view(chain, token, end_ts, cache, labels_path, progress)
    transfers, scale, hot, pools, service, deposits, project, prices, day, contract, book, sales = (
        v[k] for k in ("transfers", "scale", "hot", "pools", "service", "deposits", "project", "prices", "day",
                       "contract", "book", "sales"))

    price_now = prices.get(day(end_ts - 1))
    raw: Dict[str, int] = defaultdict(int)
    for t in transfers:
        raw[t["from"]] -= t["value"]
        raw[t["to"]] += t["value"]
    supply = sum(v for a, v in raw.items() if a != ZERO and v > 0)
    ranked = sorted(((v, a) for a, v in raw.items() if a != ZERO and v > 0), reverse=True)
    priors = peer_priors(sales, hot | pools | deposits | project, supply)
    for band, p in priors.items():
        progress(f"priors for holders of {band:.3%}+ of supply: sell {p['sell_fraction'] or 0:.0%} at once "
                 f"({p['n_sales']} sales), take profit at {p['take_profit_multiple'] or 0:.2f}x cost "
                 f"({p['n_profitable']} profitable)")
    mismatched = 0
    holders, excluded = [], []
    for bal, w in ranked:
        if len(holders) >= top_n or bal <= 0:
            break
        why = None
        if w in service:
            why = "service"
        elif w in hot:
            why = "exchange"
        elif w in pools:
            why = "pool"
        elif w in deposits:
            why = "deposit address"
        elif w in project:
            why = "project"
        elif (bal / supply >= TREASURY_SHARE and not sales.get(w)
              and sum(l[0] for l in book.lots[w] if l[1] == 0) >= 0.5 * bal):
            why = "treasury"
        elif contract(w) and not is_safe(chain, w):
            why = "contract"
        if why:
            excluded.append({"address": w, "balance": bal / scale, "share_of_supply": bal / supply, "reason": why})
            continue
        lots = list(book.lots[w])
        if book.balance(w) != bal:
            mismatched += 1
        priced = [l for l in lots if l[1] == l[1]]          # drop NaN costs
        qty = sum(l[0] for l in priced)
        zero = sum(l[0] for l in lots if l[1] == 0)
        s = sales[w]
        multiples = sorted(x["price"] / x["cost"] for x in s if x["price"] and x["cost"])
        zero_share = zero / bal
        kind = ("trader" if len(s) >= TRADER_SALES else "time" if zero_share >= 0.5
                else "price" if qty else "unknown")
        cost_basis = (sum(l[0] * l[1] for l in priced) / qty) if qty else None
        holders.append({
            "address": w,
            "kind": kind,
            "balance": bal / scale,
            "share_of_supply": bal / supply,
            "cost_basis_usd": cost_basis,
            "zero_cost_share": zero_share,
            "origins": json.dumps({o: round(sum(l[0] for l in lots if l[3] == o) / bal, 4)
                                   for o in sorted({l[3] for l in lots})}),
            "oldest_lot": day(min(l[2] for l in lots)) if lots else None,
            "n_sales": len(s),
            "sold_tokens": sum(x["tokens"] for x in s) / scale,
            "median_sale_multiple": multiples[len(multiples) // 2] if multiples else None,
            "last_sale": day(s[-1]["ts"]) if s else None,
            **sell_levels(kind, bal / scale, cost_basis, s, priors[size_band(bal / supply)], price_now),
        })
    progress(f"{len(holders)} holders modelled, {len(excluded)} large wallets set aside, "
             f"{mismatched} with lots not matching their balance")
    return {"holders": holders, "excluded": excluded, "supply": supply / scale, "lot_mismatches": mismatched,
            "price_now": price_now, "priors": {f"{b:.5f}": p for b, p in priors.items()},
            "time_triggered_tokens": sum(h["balance"] for h in holders if h["kind"] == "time"),
            "params": {"chain": chain, "token": token, "top_n": top_n, "as_of": as_of}}
