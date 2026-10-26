"""Backtest the holder sell levels on a past rally.

Positions and levels are built as of the rally's start (the trough), from data up to that day
only. The same wallets' actual sales are then read over the rally and a short tail after the
peak, and compared with what the levels said:

  volume   tokens the price-triggered holders sold vs the tokens their levels put at or below
           the highest price reached in the window
  who      precision and recall of "this wallet sells during the rally"
  where    each seller's first sale price vs its lowest predicted level, as a log error, against
           a constant baseline: every wallet sells at the same multiple of the start price, the
           median one observed (in-sample, so it favours the baseline)
  curve    tokens sold by price band (multiples of the start price), predicted vs realized
  outside  what the other kinds (time, trader, unknown) sold, which the levels don't price

Stdlib only. python -m cexflows.backtest --token 0x... --start 2026-02-21 --peak 2026-04-20
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from . import holders
from .cache import Cache
from .sellcurve import position_group

BANDS = [1.25, 1.5, 2.0, 3.0, 5.0, math.inf]


def _ts(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _median(xs: List[float]) -> Optional[float]:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def lowest_level(h: dict) -> Optional[float]:
    levels = [p for p in (h.get("breakeven_price"), h.get("takeprofit_price")) if p]
    return min(levels) if levels else None


def predicted_tokens(h: dict, up_to: float) -> float:
    """Tokens a holder's levels sell at prices up to `up_to`."""
    out = 0.0
    for p, t in ((h.get("breakeven_price"), h.get("breakeven_tokens")), (h.get("takeprofit_price"), h.get("takeprofit_tokens"))):
        if p and p <= up_to:
            out += t or 0.0
    return out


def score(modelled: List[dict], realized: Dict[str, List[dict]], p0: float, peak_price: float) -> dict:
    """Compare levels with realized sales. `realized` maps address -> sales in the window, each
    {tokens, price} in whole tokens and USD."""
    price_kind = [h for h in modelled if h["kind"] == "price"]
    pred_sellers = {h["address"] for h in price_kind if lowest_level(h) and lowest_level(h) <= peak_price}
    real_sellers = {h["address"] for h in price_kind if realized.get(h["address"])}
    both = pred_sellers & real_sellers

    sold = lambda a: sum(s["tokens"] for s in realized.get(a, []))
    sold_usd = lambda a: sum(s["tokens"] * (s["price"] or 0) for s in realized.get(a, []))
    pred_tok = sum(predicted_tokens(h, peak_price) for h in price_kind)
    pred_usd = sum((h["breakeven_tokens"] or 0) * h["breakeven_price"] for h in price_kind
                   if h.get("breakeven_price") and h["breakeven_price"] <= peak_price)
    pred_usd += sum((h["takeprofit_tokens"] or 0) * h["takeprofit_price"] for h in price_kind
                    if h.get("takeprofit_price") and h["takeprofit_price"] <= peak_price)
    real_tok = sum(sold(h["address"]) for h in price_kind)
    # Wallets that buy during the rally can sell more than they started with; capped, each counts
    # only up to its starting balance, which is what the levels are a share of.
    real_capped = sum(min(sold(h["address"]), h["balance"]) for h in price_kind)
    real_usd = sum(sold_usd(h["address"]) for h in price_kind)

    # Where: first sale price vs the wallet's lowest level, and vs one constant multiple.
    firsts = {a: next((s["price"] for s in realized[a] if s["price"]), None) for a in real_sellers}
    by_addr = {h["address"]: h for h in price_kind}
    pairs = [(firsts[a], lowest_level(by_addr[a])) for a in real_sellers if firsts.get(a) and lowest_level(by_addr[a])]
    const = _median([f / p0 for f, _ in pairs])
    model_err = [abs(math.log(f / lvl)) for f, lvl in pairs]
    base_err = [abs(math.log(f / (p0 * const))) for f, _ in pairs] if const else []
    at_or_past = sum(1 for f, lvl in pairs if f >= lvl * 0.95)

    # Bands of the start price; levels above the highest price reached never trigger.
    levels = [(p, t) for h in price_kind for p, t in ((h.get("breakeven_price"), h.get("breakeven_tokens")),
                                                      (h.get("takeprofit_price"), h.get("takeprofit_tokens")))
              if p and t and p <= peak_price]
    sales = [s for h in price_kind for s in realized.get(h["address"], []) if s["price"]]
    curve, lo = [], 0.0
    for hi in BANDS:
        inside = lambda p: p0 * lo < p <= p0 * hi
        curve.append({"band": f"{lo:g}-{hi:g}x" if hi != math.inf else f"{lo:g}x+",
                      "predicted_tokens": sum(t for p, t in levels if inside(p)),
                      "realized_tokens": sum(s["tokens"] for s in sales if inside(s["price"]))})
        lo = hi
        if p0 * hi >= peak_price:
            break

    others: Dict[str, dict] = {}
    for h in modelled:
        if h["kind"] == "price":
            continue
        o = others.setdefault(h["kind"], {"wallets": 0, "balance": 0.0, "sellers": 0, "sold_tokens": 0.0})
        o["wallets"] += 1
        o["balance"] += h["balance"]
        o["sellers"] += 1 if realized.get(h["address"]) else 0
        o["sold_tokens"] += sold(h["address"])

    return {
        "price_kind_wallets": len(price_kind),
        "price_kind_balance": sum(h["balance"] for h in price_kind),
        "volume": {"predicted_tokens": pred_tok, "realized_tokens": real_tok,
                   "predicted_usd": pred_usd, "realized_usd": real_usd,
                   "realized_tokens_capped": real_capped,
                   "realized_over_predicted": real_tok / pred_tok if pred_tok else None,
                   "capped_over_predicted": real_capped / pred_tok if pred_tok else None,
                   "capped_share_of_balance": real_capped / sum(h["balance"] for h in price_kind) if price_kind else None},
        "who": {"predicted_sellers": len(pred_sellers), "realized_sellers": len(real_sellers), "both": len(both),
                "precision": len(both) / len(pred_sellers) if pred_sellers else None,
                "recall": len(both) / len(real_sellers) if real_sellers else None,
                "base_rate": len(real_sellers) / len(price_kind) if price_kind else None},
        "where": {"wallets": len(pairs), "first_sale_at_or_past_level": at_or_past,
                  "median_log_error_model": _median(model_err), "median_log_error_constant": _median(base_err),
                  "constant_multiple": const},
        "curve": curve,
        "outside_model": others,
    }


def backtest(chain: str, token: str, start: str, peak: str, after_days: int = 14, top_n: int = 100,
             cache: Optional[Cache] = None, progress: Callable[[str], None] = lambda m: None) -> dict:
    cache = cache or Cache()
    pred = holders.holder_positions(chain, token, top_n=top_n, as_of=start, cache=cache, progress=progress)
    start_end = _ts(start) + 86400
    window_end = _ts(peak) + 86400 * (after_days + 1)
    view = holders.market_view(chain.lower(), token.lower(), window_end, cache, progress=progress)
    scale, prices = view["scale"], view["prices"]
    realized = {h["address"]: [{"ts": s["ts"], "tokens": s["tokens"] / scale, "price": s["price"]}
                               for s in view["sales"].get(h["address"], []) if start_end <= s["ts"] < window_end]
                for h in pred["holders"]}
    last_day = datetime.fromtimestamp(window_end - 1, timezone.utc).strftime("%Y-%m-%d")
    window_prices = [p for d, p in prices.items() if start <= d <= last_day and p]
    p0 = pred["price_now"]
    peak_price = max(window_prices) if window_prices else p0
    out = score(pred["holders"], realized, p0, peak_price)
    out["params"] = {"chain": chain, "token": token, "start": start, "peak": peak, "after_days": after_days,
                     "top_n": top_n, "start_price": p0, "peak_price": peak_price, "rally": peak_price / p0 if p0 else None}
    out["priors"] = pred["priors"]
    out["excluded_share"] = sum(e["share_of_supply"] for e in pred["excluded"])
    return out


def rally_records(chain: str, token: str, start: str, peak: str, after_days: int = 14, top_n: int = 100,
                  cache: Optional[Cache] = None, progress: Callable[[str], None] = lambda m: None) -> dict:
    """Each top holder's net selling over a rally, keyed by how far the price had run.

    Holders and their groups are fixed at the start (the trough), from data up to that day. Over
    the rally and `after_days` after the peak, each of their sales (to an exchange, pool or
    deposit address) adds and each purchase from an exchange or pool subtracts; every event is
    tagged with the highest daily price so far as a multiple of the start price. Tokens are
    whole units."""
    cache = cache or Cache()
    pred = holders.holder_positions(chain, token, top_n=top_n, as_of=start, cache=cache, progress=progress)
    start_end = _ts(start) + 86400
    window_end = _ts(peak) + 86400 * (after_days + 1)
    view = holders.market_view(chain.lower(), token.lower(), window_end, cache, progress=progress)
    scale, prices, day = view["scale"], view["prices"], view["day"]
    p0 = pred["price_now"]
    run_max, best = {}, p0
    for d in sorted(prices):
        if start < d and prices[d]:
            best = max(best, prices[d])
        run_max[d] = best
    k_at = lambda ts: run_max.get(day(ts), p0) / p0
    who = {h["address"]: h for h in pred["holders"]}
    events: Dict[str, List[list]] = {a: [] for a in who}
    for a in who:
        for x in view["sales"].get(a, []):
            if start_end <= x["ts"] < window_end:
                events[a].append([x["ts"], k_at(x["ts"]), x["tokens"] / scale])
    market = view["hot"] | view["pools"]
    for t in view["transfers"]:
        if start_end <= t["ts"] < window_end and t["to"] in who and t["from"] in market:
            events[t["to"]].append([t["ts"], k_at(t["ts"]), -t["value"] / scale])
    last_day = datetime.fromtimestamp(window_end - 1, timezone.utc).strftime("%Y-%m-%d")
    return {"token": token, "chain": chain, "start": start, "peak": peak, "start_price": p0,
            "max_multiple": max((m for d, m in run_max.items() if d <= last_day), default=p0) / p0,
            "holders": [{"address": a, "kind": h["kind"], "group": position_group(h, p0), "balance": h["balance"],
                         "cost_basis_usd": h["cost_basis_usd"], "n_sales_before": h["n_sales"],
                         "share_of_supply": h["share_of_supply"], "zero_cost_share": h["zero_cost_share"],
                         "events": sorted(([k, v] for _, k, v in sorted(events[a])), key=lambda e: e[0])}
                        for a, h in who.items()]}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", default="ethereum")
    ap.add_argument("--token", required=True)
    ap.add_argument("--start", required=True, help="rally start (trough), YYYY-MM-DD")
    ap.add_argument("--peak", required=True, help="rally peak, YYYY-MM-DD")
    ap.add_argument("--after-days", type=int, default=14)
    ap.add_argument("--top-n", type=int, default=100)
    a = ap.parse_args(argv)
    res = backtest(a.chain, a.token, a.start, a.peak, a.after_days, a.top_n,
                   progress=lambda m: print(m, file=sys.stderr))
    json.dump(res, sys.stdout, indent=1, default=str)
    print()


if __name__ == "__main__":
    main()
