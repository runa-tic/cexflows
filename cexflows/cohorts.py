"""Cohort wallet tracking: who receives tokens at unlock cliffs, and where the tokens go next.

1. Releases. For each cliff date, read every transfer of the token in a short window from that
   date, and in a control window of the same length `control_offset_days` earlier. A vesting
   payout is a (sender, recipient) pair that recurs in the cliff windows of at least
   `min_cliffs` cliffs, never appears in a control window, never runs the other way, and pays
   steady amounts (at least two within 10% of their median): a vesting contract or distributor
   paying its beneficiary on schedule. Exchange hot wallets and mints are excluded as senders.
2. Follow-through. Each payout is followed as a lot tagged with its cliff, for `follow_days`.
   The beneficiaries' outbound transfers spend their lots oldest first, are credited to the
   cliff that paid them, and are classified by destination:
     exchange   a labelled hot wallet
     deposit    a wallet that passes the tokens on to a hot wallet within `deposit_days`
                (an exchange deposit address)
     contract   any contract: DEX pools and routers, multisigs, staking, bridges
     wallet     any other account
     internal   another beneficiary or a vesting source
   What isn't sent anywhere in the window is retained.

Exchange plus deposit over released is the cohort's measured sell-through to exchanges. DEX
selling lands in "contract", mixed with everything else that is a contract, so it is reported
separately rather than counted as selling.

Released is what left vesting: a payout from one beneficiary to another, such as a distributor
passing tokens on, moves lots rather than counting twice. Outflows beyond a beneficiary's lots
are not counted, so a wallet that also trades its own holdings can't inflate the shares, and no
cliff's row can send more than it released.

Beneficiaries who claim later than the release window, or whose contracts stream rather than
pay on the cliff, are missed, so released amounts are a
sample of the scheduled unlock, not all of it; the shares are what this sample did.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from . import etherscan
from .cache import Cache
from .flows import token_decimals
from .labels import hot_wallets
from .prices import daily_prices

ZERO = "0x" + "0" * 40
DAY = 86400

# Column order of a cohort_flows row, as the Dune-shaped API and the CLI both report it.
COHORT_COLUMNS = ["cliff_date", "released_tokens", "released_usd", "to_exchange_tokens", "to_deposit_tokens",
                  "to_contract_tokens", "to_wallet_tokens", "to_internal_tokens", "to_unclassified_tokens",
                  "retained_tokens", "exchange_share", "contract_share"]


def _day_ts(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def window_transfers(chain: str, token: str, ts_from: int, ts_to: int, cache: Cache) -> List[dict]:
    """Every transfer of the token in [ts_from, ts_to), cached by block range."""
    if ts_from >= ts_to:
        raise ValueError(f"empty window: {ts_from} >= {ts_to} (release_days and follow_days must be positive)")
    from_block = etherscan.block_at(chain, ts_from, "after")
    to_block = etherscan.block_at(chain, ts_to - 1, "before")
    if from_block > to_block:
        raise ValueError(f"window {ts_from}..{ts_to} maps to an empty block range "
                         f"({from_block} > {to_block}); check release_days")
    if not any(a <= from_block and b >= to_block for a, b in cache.token_log_ranges(chain, token)):
        cache.store_token_logs(chain, token, etherscan.transfers(chain, token, from_block, to_block),
                               from_block, to_block)
    return cache.token_logs(chain, token, ts_from, ts_to)


def steady(amounts: Iterable[int], tolerance: float = 0.1) -> bool:
    """At least two payouts within `tolerance` of their median: a schedule pays the same each
    period. One outlier is allowed, since a first claim often carries accrued periods."""
    vals = sorted(amounts)
    med = vals[len(vals) // 2]
    return med > 0 and sum(abs(v - med) <= tolerance * med for v in vals) >= 2


def find_vesting_pairs(cliff_windows: List[List[dict]], control_windows: List[List[dict]],
                       excluded_senders: Set[str], min_cliffs: int,
                       tolerance: float = 0.1) -> Dict[Tuple[str, str], Dict[int, int]]:
    """(sender, recipient) -> {cliff index: raw amount} for pairs that recur on cliff dates only,
    in steady amounts. Pairs that also run the other way are dropped: neither side is a source."""
    control_pairs = {(t["from"], t["to"]) for w in control_windows for t in w}
    seen: Dict[Tuple[str, str], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for i, w in enumerate(cliff_windows):
        for t in w:
            if t["from"] in excluded_senders or t["from"] == ZERO or t["from"] == t["to"]:
                continue
            seen[(t["from"], t["to"])][i] += t["value"]
    return {pair: dict(amounts) for pair, amounts in seen.items()
            if len(amounts) >= min_cliffs and pair not in control_pairs
            and (pair[1], pair[0]) not in seen and steady(amounts.values(), tolerance)}


def classify(dest: str, hot: Set[str], internal: Set[str], is_contract: Callable[[str], bool],
             forwards_to_hot: Callable[[str], bool]) -> str:
    if dest in hot:
        return "exchange"
    if dest in internal:
        return "internal"
    if is_contract(dest):
        return "contract"
    if forwards_to_hot(dest):
        return "deposit"
    return "wallet"


def cohort_flows(chain: str, token: str, cliff_dates: Iterable[str], release_days: int = 2,
                 follow_days: int = 30, control_offset_days: int = 10, min_cliffs: Optional[int] = None,
                 deposit_days: int = 7, max_destinations: int = 150, cache: Optional[Cache] = None,
                 labels_path: Optional[str] = None,
                 progress: Callable[[str], None] = lambda m: None) -> dict:
    chain, token = chain.lower(), token.lower()
    cache = cache or Cache()
    cliffs = sorted(set(cliff_dates))
    if len(cliffs) < 2:
        raise ValueError("cohort tracking needs at least two cliff dates")
    min_cliffs = min_cliffs or max(2, math.ceil(len(cliffs) / 2))
    hot = set(hot_wallets(chain, labels_path))
    scale = 10 ** token_decimals(chain, token)

    starts = [_day_ts(d) for d in cliffs]
    cliff_windows, control_windows = [], []
    for d, s in zip(cliffs, starts):
        cliff_windows.append(window_transfers(chain, token, s, s + release_days * DAY, cache))
        c = s - control_offset_days * DAY
        control_windows.append(window_transfers(chain, token, c, c + release_days * DAY, cache))
        progress(f"{d}: {len(cliff_windows[-1])} transfers in the release window, "
                 f"{len(control_windows[-1])} in its control window")

    pairs = find_vesting_pairs(cliff_windows, control_windows, hot, min_cliffs)
    sources = {s for s, _ in pairs}
    beneficiaries = {r for _, r in pairs}
    internal = sources | beneficiaries
    progress(f"{len(pairs)} vesting payout pairs: {len(sources)} sources, {len(beneficiaries)} beneficiaries")

    # Tokens are followed as lots tagged with the cliff that paid them. A payout from a vesting
    # source opens a lot and counts as released at that cliff. A payout from one beneficiary to
    # another (a distributor hub passing tokens on) moves the sender's lots instead, so it isn't
    # a second release; only what the sender can't cover from its own lots opens a new one.
    # Every other outflow of a beneficiary spends its lots oldest first, within `follow_days` of
    # the lot's cliff, and is credited to that cliff. A wallet that also trades its own tokens
    # spends nothing beyond what vesting paid it, and no cliff can send more than it released.
    pay_keys = set()
    events: List[Tuple[int, int, str, int, dict]] = []
    for i, w in enumerate(cliff_windows):
        for t in w:
            if (t["from"], t["to"]) in pairs:
                pay_keys.add((t["tx"], t["log_index"]))
                events.append((t["block"], t["log_index"], "pay", i, t))
    now = int(datetime.now(timezone.utc).timestamp())
    if beneficiaries:
        lo = etherscan.block_at(chain, starts[0], "after")
        hi = etherscan.block_at(chain, min(starts[-1] + follow_days * DAY, now) - 1, "before")
        for n, b in enumerate(sorted(beneficiaries), 1):
            cache.ensure(chain, token, "out:" + b, lo, hi,
                         lambda c, tk, w, a, z, b=b: etherscan.transfers(c, tk, a, z, sender=b))
            for t in cache.transfers(chain, token, "out:" + b, starts[0], starts[-1] + follow_days * DAY):
                if (t["tx"], t["log_index"]) not in pay_keys:
                    events.append((t["block"], t["log_index"], "out", -1, t))
            if n % 25 == 0:
                progress(f"followed {n}/{len(beneficiaries)} beneficiaries")
    events.sort(key=lambda e: (e[0], e[1]))

    lots: Dict[str, List[List[int]]] = defaultdict(list)   # wallet -> [[cliff index, amount], ...]
    released_raw: Dict[int, int] = defaultdict(int)
    outflows: List[Tuple[int, dict]] = []

    def take(wallet: str, amount: int, ts: int) -> List[Tuple[int, int]]:
        taken = []
        for lot in sorted(lots[wallet], key=lambda l: l[0]):
            if amount <= 0:
                break
            if lot[1] <= 0 or ts >= starts[lot[0]] + follow_days * DAY:
                continue
            part = min(lot[1], amount)
            lot[1] -= part
            amount -= part
            taken.append((lot[0], part))
        lots[wallet] = [l for l in lots[wallet] if l[1] > 0]
        return taken

    for _, _, kind_, i, t in events:
        if kind_ == "pay":
            moved = take(t["from"], t["value"], t["ts"]) if t["from"] in beneficiaries else []
            lots[t["to"]].extend([k, part] for k, part in moved)
            fresh = t["value"] - sum(part for _, part in moved)
            if fresh > 0:
                lots[t["to"]].append([i, fresh])
                released_raw[i] += fresh
            continue
        for k, part in take(t["from"], t["value"], t["ts"]):
            outflows.append((k, {**t, "value": part}))
            if t["to"] in beneficiaries:
                lots[t["to"]].append([k, part])

    # Classify the largest destinations; the rest are reported as unclassified.
    by_dest: Dict[str, int] = defaultdict(int)
    first_seen: Dict[str, int] = {}
    for _, t in outflows:
        by_dest[t["to"]] += t["value"]
        first_seen[t["to"]] = min(first_seen.get(t["to"], t["ts"]), t["ts"])
    ranked = sorted(by_dest, key=lambda d: -by_dest[d])
    contract_memo: Dict[str, bool] = {}

    def is_contract(a: str) -> bool:
        if a not in contract_memo:
            contract_memo[a] = etherscan.is_contract(chain, a)
        return contract_memo[a]

    def forwards_to_hot(a: str) -> bool:
        lo = etherscan.block_at(chain, first_seen[a], "after")
        hi = etherscan.block_at(chain, min(first_seen[a] + deposit_days * DAY,
                                           int(datetime.now(timezone.utc).timestamp()) - 60), "before")
        for t in etherscan.transfers(chain, token, lo, hi, sender=a):
            if t["to"] in hot:
                return True
        return False

    kind: Dict[str, str] = {}
    for n, d in enumerate(ranked):
        kind[d] = classify(d, hot, internal, is_contract, forwards_to_hot) if n < max_destinations else "unclassified"
    progress(f"classified {min(len(ranked), max_destinations)} of {len(ranked)} destinations")

    try:
        prices = daily_prices(chain, token, starts[0], starts[-1] + DAY, cache)
    except Exception as e:
        progress(f"no USD prices: {e}")
        prices = {}

    cats = ["exchange", "deposit", "contract", "wallet", "internal", "unclassified"]
    rows = []
    total = {c: 0.0 for c in cats + ["released"]}
    for i, d in enumerate(cliffs):
        released = released_raw.get(i, 0) / scale
        sent = {c: 0.0 for c in cats}
        for k, t in outflows:
            if k == i:
                sent[kind[t["to"]]] += t["value"] / scale
        external = sum(sent[c] for c in cats if c != "internal")
        row = {"cliff_date": d, "released_tokens": released,
               "released_usd": released * prices[d] if d in prices else None,
               **{f"to_{c}_tokens": sent[c] for c in cats},
               "retained_tokens": max(released - external, 0.0),
               "exchange_share": (sent["exchange"] + sent["deposit"]) / released if released else None,
               "contract_share": sent["contract"] / released if released else None}
        rows.append(row)
        total["released"] += released
        for c in cats:
            total[c] += sent[c]
    rel = total["released"]
    usd = [r["released_usd"] for r in rows]
    summary = {"cliff_date": "all", "released_tokens": rel,
               "released_usd": sum(usd) if None not in usd else None,
               **{f"to_{c}_tokens": total[c] for c in cats},
               "retained_tokens": max(rel - sum(total[c] for c in cats if c != "internal"), 0.0),
               "exchange_share": (total["exchange"] + total["deposit"]) / rel if rel else None,
               "contract_share": total["contract"] / rel if rel else None}
    return {
        "rows": rows + [summary],
        "sources": sorted(sources),
        "beneficiaries": sorted(beneficiaries),
        "destinations": [{"address": d, "kind": kind[d], "tokens": by_dest[d] / scale} for d in ranked[:max_destinations]],
        "params": {"chain": chain, "token": token, "cliffs": cliffs, "release_days": release_days,
                   "follow_days": follow_days, "control_offset_days": control_offset_days,
                   "min_cliffs": min_cliffs, "deposit_days": deposit_days},
    }


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Track who receives tokens at unlock cliffs and where they go.")
    p.add_argument("--chain", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--cliffs", required=True, help="comma-separated YYYY-MM-DD dates")
    p.add_argument("--release-days", type=int, default=2)
    p.add_argument("--follow-days", type=int, default=30)
    p.add_argument("--min-cliffs", type=int, default=None)
    p.add_argument("--out", help="write the full result as JSON")
    a = p.parse_args(argv)
    res = cohort_flows(a.chain, a.token, a.cliffs.split(","), a.release_days, a.follow_days,
                       min_cliffs=a.min_cliffs, progress=lambda m: print(m, file=sys.stderr, flush=True))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    for r in res["rows"]:
        share = "-" if r["exchange_share"] is None else f"{r['exchange_share']:.1%}"
        print(f"{r['cliff_date']:>10}  released {r['released_tokens']:>16,.0f}  to exchanges {share:>6}  "
              f"exchange {r['to_exchange_tokens']:>14,.0f}  deposit {r['to_deposit_tokens']:>14,.0f}  "
              f"contract {r['to_contract_tokens']:>14,.0f}  wallet {r['to_wallet_tokens']:>14,.0f}  "
              f"retained {r['retained_tokens']:>16,.0f}")


if __name__ == "__main__":
    main()
