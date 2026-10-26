"""Cliff event study: do exchange inflows rise after unlock cliffs, beyond what the market explains?

Daily exchange inflow swings by a factor of three or more on its own, so a raw before/after
comparison around a cliff is noise. Instead, for each post-cliff window length W:

    log(inflow_d) = a + b1 log(volume_d) + b2 |log return_d| + b3 trend
                    + c_e [d within W days of other event e] + g [d within W days of a cliff]

volume and returns are CoinGecko's daily USD volume and price. exp(g) - 1 is the cliff uplift per
day. Placebo: the same fit with every cliff moved by a shift whose windows overlap no real cliff
window; where the real g ranks among the shifted ones says whether it stands out from ordinary
month-to-month variation.

In token terms, uplift x median daily inflow x W, over the release size, is the share of each
release that reached the tracked exchanges within W days. Inflows are those into labelled hot
wallets only, so the share is relative to label coverage, not an absolute sell-through.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import flows
from .cache import Cache
from .prices import daily_market

# Column order of a cliff_study row, as the Dune-shaped API reports it.
STUDY_COLUMNS = ["window_days", "uplift", "uplift_se", "t_stat", "placebo_n", "placebo_median", "placebo_min",
                 "placebo_max", "placebo_share_above", "excess_tokens", "release_share", "release_share_low",
                 "release_share_high", "events", "r2"]


def ols(X: List[List[float]], y: List[float]) -> Tuple[List[float], List[float], float]:
    """Least squares by the normal equations: coefficients, standard errors, R^2."""
    k = len(X[0])
    xtx = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    xty = [sum(r[i] * v for r, v in zip(X, y)) for i in range(k)]
    # Invert X'X by Gauss-Jordan with partial pivoting.
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(k)] for i, row in enumerate(xtx)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(a[r][c]))
        if abs(a[p][c]) < 1e-12:
            raise ValueError("singular design: a regressor is constant or duplicated")
        a[c], a[p] = a[p], a[c]
        piv = a[c][c]
        a[c] = [v / piv for v in a[c]]
        for r in range(k):
            if r != c and a[r][c]:
                f = a[r][c]
                a[r] = [v - f * w for v, w in zip(a[r], a[c])]
    inv = [row[k:] for row in a]
    b = [sum(inv[i][j] * xty[j] for j in range(k)) for i in range(k)]
    resid = [v - sum(bi * xi for bi, xi in zip(b, r)) for r, v in zip(X, y)]
    s2 = sum(e * e for e in resid) / max(len(y) - k, 1)
    mean = sum(y) / len(y)
    tss = sum((v - mean) ** 2 for v in y)
    return b, [math.sqrt(max(s2 * inv[i][i], 0.0)) for i in range(k)], 1 - sum(e * e for e in resid) / tss if tss else 0.0


def _in_window(d: date, starts: Iterable[date], w: int) -> float:
    return 1.0 if any(0 <= (d - s).days < w for s in starts) else 0.0


def placebo_shifts(cliffs: Sequence[date], w: int, max_shift: int = 25) -> List[int]:
    """Shifts whose moved windows overlap no real cliff window."""
    out = []
    for s in list(range(-max_shift, 0)) + list(range(1, max_shift + 1)):
        if all(abs((c + timedelta(days=s) - c2).days) >= w for c in cliffs for c2 in cliffs):
            out.append(s)
    return out


def fit_study(days: List[dict], cliffs: Sequence[date], window: int,
              other_events: Sequence[date] = ()) -> Dict[str, float]:
    """One window's regression on prepared days ({day, inflow, volume, abs_return})."""
    n = len(days)
    X, y = [], []
    for i, r in enumerate(days):
        d = date.fromisoformat(r["day"])
        X.append([1.0, math.log(r["volume"]), r["abs_return"], i / n]
                 + [_in_window(d, [e], window) for e in other_events] + [_in_window(d, cliffs, window)])
        y.append(math.log(r["inflow"]))
    b, se, r2 = ols(X, y)
    return {"g": b[-1], "se": se[-1], "r2": r2,
            "events": {e.isoformat(): (b[4 + k], se[4 + k]) for k, e in enumerate(other_events)}}


def prepare_days(inflows: List[dict], market: Dict[str, Dict[str, float]]) -> List[dict]:
    out = []
    for r in inflows:
        d = r["day"]
        nxt = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
        if r["tokens_to_cex"] > 0 and d in market and nxt in market and market[d]["volume"] > 0:
            out.append({"day": d, "inflow": r["tokens_to_cex"], "volume": market[d]["volume"],
                        "abs_return": abs(math.log(market[nxt]["price"] / market[d]["price"]))})
    return out


def cliff_study(chain: str, token: str, cliff_dates: Iterable[str], release_tokens: Optional[float] = None,
                windows: Sequence[int] = (3, 7, 10, 15), other_events: Iterable[str] = (),
                lead_days: int = 20, cache: Optional[Cache] = None,
                progress: Callable[[str], None] = lambda m: None) -> dict:
    cliffs = sorted(date.fromisoformat(c) for c in set(cliff_dates))
    events = sorted(date.fromisoformat(e) for e in set(other_events))
    if len(cliffs) < 3:
        raise ValueError("an event study needs at least three cliff dates")
    today = datetime.now(timezone.utc).date()
    first = min(cliffs + events) - timedelta(days=lead_days)
    inflows = flows.daily_inflows(chain, token, days=(today - first).days + 1, cache=cache,
                                  progress=progress)
    ts = lambda d: int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
    market = daily_market(chain, token, ts(first), ts(today))
    days = prepare_days(inflows, market)
    if len(days) < 60:
        raise ValueError(f"only {len(days)} days with both inflows and market data")
    med = sorted(r["inflow"] for r in days)[len(days) // 2]
    rows = []
    for w in windows:
        real = fit_study(days, cliffs, w, events)
        shifts = placebo_shifts(cliffs, w)
        plac = sorted(fit_study(days, [c + timedelta(days=s) for c in cliffs], w, events)["g"] for s in shifts)
        up = math.expm1(real["g"])
        tokens = up * med * w
        share = lambda g: math.expm1(g) * med * w / release_tokens if release_tokens else None
        rows.append({
            "window_days": w,
            "uplift": up,
            "uplift_se": real["se"],
            "t_stat": real["g"] / real["se"] if real["se"] else None,
            "placebo_n": len(plac),
            "placebo_median": math.expm1(plac[len(plac) // 2]) if plac else None,
            "placebo_min": math.expm1(plac[0]) if plac else None,
            "placebo_max": math.expm1(plac[-1]) if plac else None,
            "placebo_share_above": sum(1 for p in plac if p >= real["g"]) / len(plac) if plac else None,
            "excess_tokens": tokens,
            "release_share": share(real["g"]),
            "release_share_low": share(real["g"] - 1.96 * real["se"]),
            "release_share_high": share(real["g"] + 1.96 * real["se"]),
            "events": {e: math.expm1(g) for e, (g, _) in real["events"].items()},
            "r2": real["r2"],
        })
        progress(f"window {w}d: uplift {up:+.1%}/day, placebo n={len(plac)}")
    return {"rows": rows, "median_daily_inflow": med, "n_days": len(days),
            "params": {"chain": chain, "token": token.lower(), "cliffs": [c.isoformat() for c in cliffs],
                       "release_tokens": release_tokens, "other_events": [e.isoformat() for e in events]}}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Excess exchange inflow after unlock cliffs, controlled for market activity.")
    p.add_argument("--chain", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--cliffs", required=True, help="comma-separated YYYY-MM-DD dates")
    p.add_argument("--release", type=float, help="tokens released per cliff, for release shares")
    p.add_argument("--windows", default="3,7,10,15")
    p.add_argument("--events", default="", help="other dated events to control for, comma-separated")
    p.add_argument("--out", help="write the full result as JSON")
    a = p.parse_args(argv)
    res = cliff_study(a.chain, a.token, a.cliffs.split(","), a.release,
                      [int(w) for w in a.windows.split(",")], [e for e in a.events.split(",") if e],
                      progress=lambda m: print(m, file=sys.stderr, flush=True))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    fmt = lambda v: "-" if v is None else f"{v:+.1%}"
    print(f"{res['n_days']} days, median inflow {res['median_daily_inflow']:,.0f}/day")
    for r in res["rows"]:
        print(f"{r['window_days']:>3}d  uplift {fmt(r['uplift'])}/day (t {r['t_stat']:+.2f})  "
              f"placebo n={r['placebo_n']} {fmt(r['placebo_min'])}..{fmt(r['placebo_max'])}  "
              f"of release {fmt(r['release_share'])} [{fmt(r['release_share_low'])}, {fmt(r['release_share_high'])}]")


if __name__ == "__main__":
    main()
