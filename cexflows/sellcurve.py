"""Sell-through curves: how much of their holdings top holders have sold, net, by the time a rally
has taken the price to k times where it started.

The break-even and take-profit levels give each holder one or two slices. On real rallies (DEXE,
LSK, APEX, BIO) large holders kept selling as the price climbed and sold 2-32x what those slices
said. A curve per holder group, fitted on past rallies, replaces them:

  S_g(k) = sum of net tokens sold by group-g holders once the price has run to k x start
           / sum of their starting balances

Net is sales to exchanges, pools and deposit addresses less purchases from exchanges and pools,
clamped to [0, starting balance] per holder and never decreasing in k. Groups are fixed at the
rally's start: underwater (cost above the start price), in_profit, zero_cost (mostly vested or
airdropped), unknown (no priced lots), and trader (market-maker-like). A rally only informs the
multiples it reached. Within a rally a group's share is pooled by tokens, so large holders weigh
most; across rallies each rally counts once, since token units of different tokens don't compare,
and the shipped curves take the median (market makers either sit or dump; a mean fits neither).
A group with fewer than MIN_HOLDERS holders in a rally at some k leaves that rally out of it; with
no rally left it uses all groups.

The median is the typical rally, and for market makers it hides the other kind. In 7 of 12 rallies
they sold 0-5% by 1.25x; in the other 5 they sold 19-84%, and nothing at the trough told the two
apart. So --save also fits a dump case: the market-maker curve over only the rallies where they had
sold at least DUMP_SHARE of their balance by DUMP_K, saved as curves["trader_dump"]. Quotes show it
as a second hold cost beside the typical one, with the share of rallies it came from.

Records come from backtest.rally_records. python -m cexflows.sellcurve records/*.json runs the
leave-one-rally-out check: fit on the others, predict the held-out rally's top holders.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import sys
from typing import Dict, List, Optional

GRID = [1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 12.0]
GROUPS = ["underwater", "in_profit", "zero_cost", "unknown", "trader"]
MIN_HOLDERS = 3
# A dump rally: market makers had sold this share of their balance, net, by this multiple. The
# 13-rally check put the two populations at 0-5% and 19-84% by 1.25x, so 10% sits in the gap.
DUMP_K, DUMP_SHARE = 1.25, 0.10
# Curves fitted on the backtest rallies, shipped with the package; refit with `--save`.
CURVES_PATH = os.path.join(os.path.dirname(__file__), "sell_curves.json")


def position_group(h: dict, p0: float) -> str:
    """A holder's group at the start of a rally (or today, when predicting)."""
    if h["kind"] == "trader":
        return "trader"
    if h["kind"] == "time":
        return "zero_cost"
    if h.get("cost_basis_usd") is None:
        return "unknown"
    return "underwater" if h["cost_basis_usd"] > p0 else "in_profit"


def load(path: str = CURVES_PATH) -> Optional[dict]:
    """{"grid", "curves", "rallies", ...} as saved by --save, or None when there is no file."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def step_tokens(holders: List[dict], fitted: dict, p0: float, lo: float, hi: float) -> Dict[str, float]:
    """Net tokens holders would sell while the price runs from lo to hi, split into the
    "holders" layer and the "traders" (market makers) layer, plus "traders_dump", the market
    makers on the dump-case curve (None when the fitted curves have none)."""
    grid, curves = fitted["grid"], fitted["curves"]
    dump = curves.get("trader_dump")
    out = {"holders": 0.0, "traders": 0.0, "traders_dump": 0.0 if dump else None}
    step = lambda c: share_at(c, hi / p0, grid) - share_at(c, lo / p0, grid)
    for h in holders:
        g = position_group(h, p0)
        out["traders" if g == "trader" else "holders"] += h["balance"] * step(curves[g])
        if g == "trader" and dump:
            out["traders_dump"] += h["balance"] * step(dump)
    return out


def sold_by(events: List[list], balance: float, k: float) -> float:
    """Net tokens a holder had sold once the run reached k: events are [k, tokens] in time order,
    sales positive and purchases negative; clamped to [0, balance], never decreasing."""
    net, best = 0.0, 0.0
    for ek, tokens in events:
        if ek > k + 1e-12:
            break
        net += tokens
        best = max(best, min(max(net, 0.0), balance))
    return best


def fit(rallies: List[dict], grid: List[float] = GRID, stat: str = "mean") -> Dict[str, List[Optional[float]]]:
    """Group -> share sold at each grid multiple (None where no rally reached it). `stat` combines
    the rallies: "mean", or "median", which one outlier rally can't drag."""
    curves: Dict[str, List[Optional[float]]] = {}
    for g in GROUPS + ["all"]:
        row: List[Optional[float]] = []
        for k in grid:
            shares = [s for s in (group_share(r, g, k) for r in rallies) if s is not None]
            row.append(_combine(shares, stat) if shares else None)
        curves[g] = row
    for g in GROUPS:                       # thin groups borrow the pooled curve
        curves[g] = [v if v is not None else a for v, a in zip(curves[g], curves["all"])]
    for g, row in curves.items():          # monotone in k; flat past the last multiple seen
        last = 0.0
        for i, v in enumerate(row):
            last = max(last, v) if v is not None else last
            row[i] = last
    return curves


def group_share(r: dict, g: str, k: float) -> Optional[float]:
    """Net share of group g's starting balance sold by k in rally r; None when the rally didn't
    reach k or the group has too few holders in it ("all" takes every holder)."""
    hs = [h for h in r["holders"] if (g == "all" or h["group"] == g) and h["balance"] > 0]
    if r["max_multiple"] < k or len(hs) < (1 if g == "all" else MIN_HOLDERS):
        return None
    return sum(sold_by(h["events"], h["balance"], k) for h in hs) / sum(h["balance"] for h in hs)


def dump_case(rallies: List[dict], curves: Dict[str, List[float]], grid: List[float] = GRID,
              stat: str = "median") -> Optional[dict]:
    """The market-maker curve over the dump rallies only, never below the typical curve, and which
    rallies those were out of the ones with a market-maker reading at DUMP_K. None with no dump
    rally."""
    read = [r for r in rallies if group_share(r, "trader", DUMP_K) is not None]
    dumped = [r for r in read if group_share(r, "trader", DUMP_K) >= DUMP_SHARE]
    if not dumped:
        return None
    curve = [max(d, t) for d, t in zip(fit(dumped, grid, stat)["trader"], curves["trader"])]
    return {"curve": curve, "k": DUMP_K, "share": DUMP_SHARE,
            "rallies": [r.get("name") or r["token"] for r in dumped], "of": len(read)}


def _combine(xs: List[float], stat: str) -> float:
    if stat == "median":
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    return sum(xs) / len(xs)


def share_at(curve: List[float], k: float, grid: List[float] = GRID) -> float:
    """Interpolated in log k between grid points; 0 at k = 1, flat past the grid."""
    if k <= 1.0:
        return 0.0
    xs, ys = [1.0] + grid, [0.0] + curve
    i = bisect.bisect_right(xs, k)
    if i >= len(xs):
        return ys[-1]
    x0, x1 = math.log(xs[i - 1]), math.log(xs[i])
    return ys[i - 1] + (ys[i] - ys[i - 1]) * (math.log(k) - x0) / (x1 - x0)


def predict(holders: List[dict], curves: Dict[str, List[float]], k: float) -> Dict[str, float]:
    """Net tokens each group of `holders` (dicts with group and balance) would have sold by k."""
    out: Dict[str, float] = {}
    for h in holders:
        out[h["group"]] = out.get(h["group"], 0.0) + h["balance"] * share_at(curves[h["group"]], k)
    return out


def leave_one_out(rallies: List[dict], ks=(1.1, 1.25, 1.5, 2.0, 3.0, 5.0), stat: str = "mean") -> List[dict]:
    """For each rally, fit on the others and compare predicted with realized net selling by its
    top holders at a few multiples and at its own peak."""
    rows = []
    for i, r in enumerate(rallies):
        curves = fit(rallies[:i] + rallies[i + 1:], stat=stat)
        for k in [k for k in ks if k <= r["max_multiple"]] + [r["max_multiple"]]:
            for layer, keep in (("holders", lambda g: g != "trader"), ("traders", lambda g: g == "trader")):
                hs = [h for h in r["holders"] if keep(h["group"])]
                pred = sum(predict(hs, curves, k).values())
                real = sum(sold_by(h["events"], h["balance"], k) for h in hs)
                rows.append({"token": r["token"], "name": r.get("name"), "k": round(k, 2),
                             "at_peak": k == r["max_multiple"], "layer": layer,
                             "balance": sum(h["balance"] for h in hs), "predicted": pred, "realized": real,
                             "realized_over_predicted": real / pred if pred else None})
    return rows


def summarize(rows: List[dict]) -> List[dict]:
    """Per lift and layer: median |log(realized / predicted)| and the share of rallies within 2x.
    Rows with nothing realized and nothing predicted are skipped; one side zero counts as a miss."""
    out = []
    for layer in ("holders", "traders"):
        for label, pick in [(f"{k:g}x", lambda r, k=k: r["k"] == k and not r["at_peak"]) for k in (1.1, 1.25, 1.5, 2.0)] + [
                ("peak", lambda r: r["at_peak"])]:
            rs = [r for r in rows if r["layer"] == layer and pick(r) and (r["predicted"] > 0 or r["realized"] > 0)]
            if not rs:
                continue
            errs = [abs(math.log(r["realized"] / r["predicted"])) if r["predicted"] > 0 and r["realized"] > 0 else math.inf
                    for r in rs]
            errs.sort()
            # At small lifts realized selling is a few percent of balance, so the factor explodes on
            # tiny numbers; the miss in percentage points of balance is what moves a dollar cost.
            pts = sorted(abs(r["realized"] - r["predicted"]) / r["balance"] for r in rs if r["balance"])
            out.append({"layer": layer, "lift": label, "rallies": len(rs), "median_factor": math.exp(errs[len(errs) // 2]),
                        "within_2x": sum(e <= math.log(2) for e in errs) / len(errs),
                        "median_points": pts[len(pts) // 2] if pts else None, "max_points": pts[-1] if pts else None})
    return out


def main(argv: Optional[List[str]] = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    save = "--save" in args
    stat = "median" if "--median" in args else "mean"
    paths = [a for a in args if not a.startswith("--")]
    rallies = [json.load(open(p, encoding="utf-8")) for p in paths]
    curves = fit(rallies, stat=stat)
    loo = leave_one_out(rallies, stat=stat)
    names = {r["token"]: r.get("name") or r["token"] for r in rallies}
    dump = dump_case(rallies, curves, stat=stat)
    if save:
        peaks = [r for r in loo if r["layer"] == "holders" and r["k"] == round(
            next(x["max_multiple"] for x in rallies if x["token"] == r["token"]), 2)]
        with open(CURVES_PATH, "w", encoding="utf-8") as f:
            saved = dict(curves, trader_dump=dump["curve"]) if dump else curves
            json.dump({"grid": GRID, "stat": stat, "curves": saved,
                       "dump_case": {k: v for k, v in dump.items() if k != "curve"} if dump else None,
                       "rallies": [{"name": r.get("name") or r["token"], "start": r["start"], "peak": r["peak"],
                                    "max_multiple": round(r["max_multiple"], 2)} for r in rallies],
                       "leave_one_out_at_peak": {names[r["token"]]: round(r["realized_over_predicted"], 2) for r in peaks
                                                 if r["realized_over_predicted"] is not None}}, f, indent=1)
        print(f"saved {CURVES_PATH}")
    print("pooled curves (share of starting balance sold, net, by price multiple):")
    print("  " + "".join(f"{k:>7g}x" for k in GRID))
    for g in GROUPS + ["all"]:
        print(f"  {g:<11}" + "".join(f"{v:>8.0%}" for v in curves[g]))
    print(f"\nmarket makers' net share sold by {DUMP_K:g}x per rally (dump case at {DUMP_SHARE:.0%} or more):")
    for r in rallies:
        v = group_share(r, "trader", DUMP_K)
        if v is not None:
            print(f"  {names[r['token']]:<10} {v:>6.1%}" + ("  dump" if v >= DUMP_SHARE else ""))
    if dump:
        print("  dump curve " + "".join(f"{v:>8.0%}" for v in dump["curve"]) +
              f"  ({len(dump['rallies'])} of {dump['of']} rallies)")
    print("\nleave one rally out, summary (median error as a factor, share of rallies within 2x):")
    for x in summarize(loo):
        print(f"  {x['layer']:<8} {x['lift']:<6} {x['rallies']:>3} rallies  median off by {x['median_factor']:.2f}x  "
              f"within 2x: {x['within_2x']:.0%}  off by {x['median_points']:.1%} of balance (worst {x['max_points']:.1%})")
    if "--detail" not in args:
        return
    for row in loo:
        r = row["realized_over_predicted"]
        print(f"  {row['token'][:10]} k={row['k']:<6} {row['layer']:<8} predicted {row['predicted']:>14,.0f}  "
              f"realized {row['realized']:>14,.0f}  ratio {r:.2f}" if r is not None else
              f"  {row['token'][:10]} k={row['k']:<6} {row['layer']:<8} predicted 0  realized {row['realized']:,.0f}")


if __name__ == "__main__":
    main()
