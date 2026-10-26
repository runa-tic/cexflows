import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cexflows import etherscan, server
from cexflows.cache import Cache
from cexflows.flows import aggregate_daily

DAY = 86400
T0 = 1_767_225_600  # 2026-01-01 00:00 UTC
HOT = {"0x" + "a" * 40: "Alpha", "0x" + "b" * 40: "Beta"}


def tr(ts, sender, wallet, value):
    return {"ts": ts, "from": sender, "wallet": wallet, "value": value}


def test_aggregate_skips_internal_transfers_and_fills_quiet_days():
    a, b = list(HOT)
    rows = aggregate_daily(
        [tr(T0 + 10, "0x" + "1" * 40, a, 5 * 10**18),
         tr(T0 + 20, "0x" + "2" * 40, b, 2 * 10**18),
         tr(T0 + 30, a, b, 100 * 10**18),          # hot wallet to hot wallet: internal
         tr(T0 + 2 * DAY + 5, "0x" + "1" * 40, a, 10**18)],
        HOT, 18, "2026-01-01", "2026-01-03")
    assert [r["day"] for r in rows] == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert rows[0]["tokens_to_cex"] == 7 and rows[0]["n_transfers"] == 2
    assert rows[0]["by_exchange"] == {"Alpha": 5, "Beta": 2}
    assert rows[1]["tokens_to_cex"] == 0 and rows[1]["n_transfers"] == 0
    assert rows[2]["tokens_to_cex"] == 1


def test_cache_fetches_only_missing_blocks():
    cache = Cache(":memory:")
    calls = []

    def fetch(chain, token, wallet, a, b):
        calls.append((a, b))
        return [{"block": n, "ts": T0 + n, "tx": f"0x{n:064x}", "log_index": 0, "from": "0x" + "1" * 40, "value": n}
                for n in range(a, b + 1, 10)]

    w = "0x" + "a" * 40
    cache.ensure("arbitrum", "0xt", w, 100, 200, fetch)
    cache.ensure("arbitrum", "0xt", w, 150, 180, fetch)   # inside coverage: no call
    cache.ensure("arbitrum", "0xt", w, 50, 250, fetch)    # both sides
    assert calls == [(100, 200), (50, 99), (201, 250)]
    assert cache.coverage("arbitrum", "0xt", w) == (50, 250)
    got = cache.transfers("arbitrum", "0xt", w, T0, T0 + 1000)
    expected = list(range(50, 100, 10)) + list(range(100, 201, 10)) + list(range(201, 251, 10))
    assert [t["block"] for t in got] == expected and len({t["tx"] for t in got}) == len(got)


def test_inbound_transfers_rereads_last_block_without_duplicates(monkeypatch):
    def log(block, idx):
        return {"blockNumber": hex(block), "timeStamp": hex(T0 + block), "transactionHash": f"0x{block:062x}{idx:02x}",
                "logIndex": hex(idx), "topics": ["0x", "0x" + "0" * 24 + "1" * 40, "0x"], "data": hex(1)}

    monkeypatch.setattr(etherscan, "PAGE", 3)
    # Pages keyed by fromBlock: each full page ends mid-block, so the next one re-reads it.
    pages = {1: [log(1, 0), log(2, 0), log(2, 1)],
             2: [log(2, 0), log(2, 1), log(3, 0)],
             3: [log(3, 0), log(4, 0)]}
    monkeypatch.setattr(etherscan, "call", lambda chain, **p: pages.get(p["fromBlock"], []))
    got = list(etherscan.inbound_transfers("arbitrum", "0xt", "0x" + "a" * 40, 1, 10))
    assert [(t["block"], t["log_index"]) for t in got] == [(1, 0), (2, 0), (2, 1), (3, 0), (4, 0)]


def _serve(runner):
    key = "test-key"
    qmap = {1: "cex_inflows", 4242424: "cex_inflows"}
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(
        key, server.allowed_hosts(0) | {"127.0.0.1:%d" % 0}, server.Executions(runner), qmap))
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = server.make_handler(key, server.allowed_hosts(port), server.Executions(runner), qmap)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port, key


def _req(port, method, path, key, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"X-Dune-API-Key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_speaks_dunes_execute_status_results():
    seen = []

    def runner(name, params):
        seen.append((name, params))
        return [{"day": "2026-01-01", "tokens_to_cex": 7.0, "usd_to_cex": None, "n_transfers": 2, "by_exchange": "{}"}]

    httpd, port, key = _serve(runner)
    try:
        assert _req(port, "POST", "/api/v1/query/1/execute", "wrong", {})[0] == 401
        code, body = _req(port, "POST", "/api/v1/query/99/execute", key, {"query_parameters": {}})
        assert code == 404 and "not implemented" in body["error"]
        code, body = _req(port, "POST", "/api/v1/query/4242424/execute", key,
                          {"query_parameters": {"chain": "arbitrum", "days": 30}})
        assert code == 200 and body["execution_id"]
        eid = body["execution_id"]
        for _ in range(50):
            st = _req(port, "GET", f"/api/v1/execution/{eid}/status", key)[1]
            if st["is_execution_finished"]:
                break
            time.sleep(0.05)
        assert st["state"] == "QUERY_STATE_COMPLETED"
        res = _req(port, "GET", f"/api/v1/execution/{eid}/results", key)[1]
        assert res["result"]["rows"][0]["tokens_to_cex"] == 7.0
        assert res["result"]["metadata"]["column_names"][0] == "day"
        assert _req(port, "GET", "/api/v1/query/4242424/results", key)[1]["execution_id"] == eid
        assert seen == [("cex_inflows", {"chain": "arbitrum", "days": 30})]
    finally:
        httpd.shutdown()


def test_server_rejects_foreign_host():
    httpd, port, key = _serve(lambda n, p: [])
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/query/1/results",
                                     headers={"X-Dune-API-Key": key, "Host": "evil.example"})
        try:
            urllib.request.urlopen(req, timeout=10)
            assert False
        except urllib.error.HTTPError as e:
            assert e.code == 403
    finally:
        httpd.shutdown()


def test_inbound_transfers_refuses_a_block_with_more_than_a_page(monkeypatch):
    def log(idx):
        return {"blockNumber": hex(5), "timeStamp": hex(T0), "transactionHash": f"0x{idx:064x}",
                "logIndex": hex(idx), "topics": ["0x", "0x" + "0" * 24 + "1" * 40, "0x"], "data": hex(1)}

    monkeypatch.setattr(etherscan, "PAGE", 2)
    monkeypatch.setattr(etherscan, "call", lambda chain, **p: [log(0), log(1)])
    try:
        list(etherscan.inbound_transfers("arbitrum", "0xt", "0x" + "a" * 40, 5, 10))
        assert False, "should refuse rather than loop"
    except etherscan.EtherscanError as e:
        assert "block 5" in str(e)


def test_vesting_pairs_recur_on_cliffs_and_never_in_controls():
    from cexflows.cohorts import find_vesting_pairs

    vest, ben, trader, pool, hot = ("0x" + c * 40 for c in "12345")
    def t(f, to, v=10):
        return {"from": f, "to": to, "value": v}
    cliffs = [[t(vest, ben, 100), t(trader, pool)],
              [t(vest, ben, 100), t(trader, pool), t(hot, ben)],
              [t(vest, ben, 120)]]
    controls = [[t(trader, pool)], [], []]
    pairs = find_vesting_pairs(cliffs, controls, {hot}, min_cliffs=2)
    assert pairs == {(vest, ben): {0: 100, 1: 100, 2: 120}}


def test_vesting_pairs_need_steady_one_way_payouts():
    from cexflows.cohorts import find_vesting_pairs

    vest, ben, bot, mm, a, b = ("0x" + c * 40 for c in "123456")
    def t(f, to, v):
        return {"from": f, "to": to, "value": v}
    cliffs = [[t(vest, ben, 900), t(bot, mm, 31), t(a, b, 50), t(b, a, 50)],
              [t(vest, ben, 100), t(bot, mm, 71), t(a, b, 50), t(b, a, 50)],
              [t(vest, ben, 100), t(bot, mm, 172)]]
    # The first claim carries accrued periods; the bot's amounts never repeat; a and b trade both ways.
    assert find_vesting_pairs(cliffs, [[], [], []], set(), min_cliffs=2) == {(vest, ben): {0: 900, 1: 100, 2: 100}}


def test_cohort_outflows_are_capped_at_vesting_payouts(tmp_path, monkeypatch):
    from cexflows import cohorts
    from cexflows.cache import Cache

    vest, ben, hot, dex = "0x" + "1" * 40, "0x" + "2" * 40, "0x" + "3" * 40, "0x" + "4" * 40
    starts = [cohorts._day_ts(d) for d in ("2026-01-15", "2026-02-15")]
    D = cohorts.DAY

    def logs(ts_from, ts_to):
        rows = []
        for i, s in enumerate(starts):
            if ts_from <= s + 3600 < ts_to:
                rows.append({"block": s + 3600, "ts": s + 3600, "tx": f"0xv{i}", "log_index": 0,
                             "from": vest, "to": ben, "value": 100})
        return rows

    monkeypatch.setattr(cohorts, "window_transfers", lambda c, tk, a, z, cache: logs(a, z))
    monkeypatch.setattr(cohorts, "hot_wallets", lambda chain, path=None: {hot: "Ex"})
    monkeypatch.setattr(cohorts, "token_decimals", lambda c, tk: 0)
    monkeypatch.setattr(cohorts, "daily_prices", lambda *a: {"2026-01-15": 2.0, "2026-02-15": 3.0})
    monkeypatch.setattr(cohorts.etherscan, "block_at", lambda c, ts, closest="before": ts)
    monkeypatch.setattr(cohorts.etherscan, "is_contract", lambda c, a: a == dex)
    # The beneficiary also trades its own holdings: 1,000 to the exchange after the first cliff.
    out = [(starts[0] + 2 * D, hot, 60), (starts[0] + 3 * D, hot, 1000), (starts[1] + 2 * D, dex, 30)]
    monkeypatch.setattr(cohorts.etherscan, "transfers",
                        lambda c, tk, a, z, sender=None, recipient=None:
                        [{"block": ts, "ts": ts, "tx": f"0xo{n}", "log_index": 0, "from": sender, "to": to, "value": v}
                         for n, (ts, to, v) in enumerate(out) if a <= ts <= z])
    res = cohorts.cohort_flows("arbitrum", "0x" + "9" * 40, ["2026-01-15", "2026-02-15"],
                               cache=Cache(str(tmp_path / "c.sqlite")))
    jan, feb, total = res["rows"]
    assert (jan["released_tokens"], jan["to_exchange_tokens"], jan["exchange_share"]) == (100, 100, 1.0)
    assert (feb["to_contract_tokens"], feb["retained_tokens"]) == (30, 70)
    assert total["released_usd"] == 100 * 2.0 + 100 * 3.0


def test_classify_destinations():
    from cexflows.cohorts import classify

    hot, internal = {"0xhot"}, {"0xben2"}
    contracts, forwarders = {"0xpool"}, {"0xdep"}
    kind = lambda d: classify(d, hot, internal, lambda a: a in contracts, lambda a: a in forwarders)
    assert [kind(d) for d in ("0xhot", "0xben2", "0xpool", "0xdep", "0xfriend")] == \
        ["exchange", "internal", "contract", "deposit", "wallet"]


def test_ols_recovers_known_coefficients():
    from cexflows.eventstudy import ols

    X = [[1.0, x, (x * 7) % 5] for x in range(40)]
    y = [2.0 + 0.5 * r[1] - 1.5 * r[2] for r in X]
    b, se, r2 = ols(X, y)
    assert [round(v, 6) for v in b] == [2.0, 0.5, -1.5] and r2 > 0.999999


def test_placebo_shifts_avoid_real_cliff_windows():
    from datetime import date
    from cexflows.eventstudy import placebo_shifts

    cliffs = [date(2026, m, 15) for m in range(1, 7)]
    shifts = placebo_shifts(cliffs, 10)
    assert 0 not in shifts and 5 not in shifts and -5 not in shifts
    assert 10 in shifts and -10 in shifts
    # 18 days after a cliff is 10-13 days before the next one: clear of both windows.
    assert 18 in shifts


def test_cliff_study_finds_a_planted_uplift(monkeypatch):
    import math
    from datetime import date, timedelta
    from cexflows import eventstudy

    start = date(2025, 9, 1)
    cliffs = [date(2025, 9, 20) + timedelta(days=30 * k) for k in range(8)]
    inflows, market = [], {}
    for i in range(300):
        d = start + timedelta(days=i)
        vol = 1e6 * (1.5 + math.sin(i / 7.0))
        bump = 1.5 if any(0 <= (d - c).days < 7 for c in cliffs) else 1.0
        inflows.append({"day": d.isoformat(), "tokens_to_cex": 100 * (vol / 1e6) ** 0.5 * bump * (1 + 0.05 * math.sin(i * 1.3))})
        market[d.isoformat()] = {"price": 1 + 0.01 * math.sin(i / 3.0), "volume": vol}
    monkeypatch.setattr(eventstudy.flows, "daily_inflows", lambda *a, **k: inflows)
    monkeypatch.setattr(eventstudy, "daily_market", lambda *a, **k: market)
    res = eventstudy.cliff_study("ethereum", "0x" + "1" * 40, [c.isoformat() for c in cliffs],
                                 release_tokens=1000, windows=[7])
    row = res["rows"][0]
    assert 0.4 < row["uplift"] < 0.6
    assert row["placebo_share_above"] == 0.0


def test_positions_follow_cost_from_mint_through_vesting_and_exchanges():
    from cexflows.holders import ZERO, build_positions, deposit_addresses

    vest, insider, cex, buyer, dep, pool = ("0x" + c * 40 for c in "123456")
    day = 86400
    price = {0: 1.0, 1: 2.0, 2: 3.0, 3: 6.0}
    def t(ts_day, f, to, v):
        return {"ts": ts_day * day, "from": f, "to": to, "value": v}
    transfers = [
        t(0, ZERO, vest, 1000),        # mint to the vesting contract
        t(1, vest, insider, 100),      # insider vests zero-cost tokens
        t(1, cex, buyer, 50),          # buyer withdraws from an exchange at $2
        t(2, buyer, dep, 20),          # deposits 20 at $3: a sale at 1.5x cost
        t(2, dep, cex, 20),            # the deposit address sweeps
        t(3, insider, pool, 40),       # insider sells 40 into a pool at $6, cost 0
    ]
    deposits = deposit_addresses(transfers, {cex, pool})
    assert deposits == {dep}
    book, sales = build_positions(transfers, {cex}, {pool}, deposits, lambda ts: price[ts // day])
    assert book.balance(insider) == 60 and all(l[1] == 0.0 and l[3] == "mint" for l in book.lots[insider])
    assert book.balance(buyer) == 30 and book.lots[buyer][0][1] == 2.0
    assert [(s["tokens"], s["price"], s["cost"], s["venue"]) for s in sales[buyer]] == [(20, 3.0, 2.0, "exchange")]
    assert [(s["tokens"], s["cost"], s["venue"]) for s in sales[insider]] == [(40, 0.0, "pool")]


def test_proxy_rate_limit_is_retried_not_returned(monkeypatch):
    import io, json
    from cexflows import etherscan

    answers = [{"status": "0", "message": "NOTOK", "result": "Max calls per sec rate limit reached (5/sec)"},
               {"jsonrpc": "2.0", "id": 1, "result": "0x"}]
    monkeypatch.setattr(etherscan, "api_key", lambda: "k")
    monkeypatch.setattr(etherscan.time, "sleep", lambda s: None)
    monkeypatch.setattr(etherscan.urllib.request, "urlopen",
                        lambda req, timeout=None: io.BytesIO(json.dumps(answers.pop(0)).encode()))
    assert etherscan.is_contract("ethereum", "0x" + "1" * 40) is False
    assert answers == []


def test_sell_levels_break_even_then_take_profit():
    from cexflows.holders import sell_levels

    priors = {"sell_fraction": 0.5, "take_profit_multiple": 2.0}
    # Underwater at cost 1.0, price 0.5, no history: half at break-even, half of the rest at 2x.
    lv = sell_levels("price", 1000, 1.0, [], priors, 0.5)
    assert (lv["breakeven_price"], lv["breakeven_tokens"]) == (1.0, 500)
    assert (lv["takeprofit_price"], lv["takeprofit_tokens"], lv["confidence"]) == (2.0, 250, "low")
    # Own history wins with two or more sales: sold a quarter at a time, took profit at 3x.
    own = [{"tokens": 25, "held": 100, "price": 3.0, "cost": 1.0}, {"tokens": 25, "held": 100, "price": 3.0, "cost": 1.0}]
    lv = sell_levels("price", 1000, 1.0, own, priors, 1.5)
    assert lv["breakeven_price"] is None and (lv["takeprofit_price"], lv["takeprofit_tokens"]) == (3.0, 250)
    # Already past take-profit: sells at today's price. Zero-cost and traders get no levels.
    assert sell_levels("price", 100, 0.1, [], priors, 1.0)["takeprofit_price"] == 1.0
    assert sell_levels("time", 100, 0.0, [], priors, 1.0)["takeprofit_price"] is None


def test_supply_curve_buckets_levels_by_price():
    from cexflows.holders import supply_curve

    holders = [{"address": "a", "breakeven_price": 1.08, "breakeven_tokens": 100, "takeprofit_price": 2.0, "takeprofit_tokens": 50},
               {"address": "b", "breakeven_price": None, "breakeven_tokens": 0, "takeprofit_price": 1.0, "takeprofit_tokens": 10}]
    rows = supply_curve(holders, 1.0, max_multiple=2.0, step_pct=10)
    assert rows[0]["takeprofit_tokens"] == 10 and rows[0]["breakeven_tokens"] == 100   # (1.00, 1.10]
    assert rows[-1]["cumulative_tokens"] == 160
    assert sum(r["step_tokens"] for r in rows) == 160
    split = supply_curve(holders, 1.0, max_multiple=2.0, step_pct=10, target_price=1.05)
    assert split[0]["price_to"] == 1.05 and split[0]["breakeven_tokens"] == 0     # 1.08 is past the target
    assert split[1]["breakeven_tokens"] == 100 and len(split) == len(rows) + 1


def test_peer_priors_come_from_wallets_of_the_same_size():
    from cexflows.holders import peer_priors

    supply = 10_000_000
    whale = [{"tokens": 1_000, "held": 10_000, "price": 2.0, "cost": 1.0} for _ in range(25)]     # 0.1%: sells 10%
    small = [{"tokens": 50, "held": 50, "price": 1.1, "cost": 1.0} for _ in range(25)]            # tiny: dumps all
    priors = peer_priors({"0xw": whale[:12], "0xv": whale[12:], "0xs": small}, set(), supply)
    assert priors[0.001]["sell_fraction"] == 0.1 and priors[0.001]["take_profit_multiple"] == 2.0
    assert priors[0.0]["sell_fraction"] == 1.0


def test_backtest_score_compares_levels_with_realized_sales():
    from cexflows.backtest import score

    modelled = [
        {"address": "a", "kind": "price", "balance": 100, "breakeven_price": 1.2, "breakeven_tokens": 10,
         "takeprofit_price": 2.0, "takeprofit_tokens": 9},
        {"address": "b", "kind": "price", "balance": 100, "breakeven_price": None, "breakeven_tokens": 0,
         "takeprofit_price": 4.0, "takeprofit_tokens": 20},            # never reached
        {"address": "c", "kind": "price", "balance": 50, "breakeven_price": None, "breakeven_tokens": 0,
         "takeprofit_price": 1.5, "takeprofit_tokens": 5},             # predicted, didn't sell
        {"address": "t", "kind": "time", "balance": 500, "breakeven_price": None, "breakeven_tokens": 0,
         "takeprofit_price": None, "takeprofit_tokens": 0},
    ]
    realized = {"a": [{"tokens": 12, "price": 1.3}, {"tokens": 8, "price": 2.1}],
                "b": [{"tokens": 30, "price": 2.5}], "t": [{"tokens": 100, "price": 2.0}]}
    r = score(modelled, realized, p0=1.0, peak_price=3.0)
    assert r["volume"]["predicted_tokens"] == 10 + 9 + 5 and r["volume"]["realized_tokens"] == 12 + 8 + 30
    realized["c"] = [{"tokens": 80, "price": 1.6}]                     # sold more than it held
    assert score(modelled, realized, p0=1.0, peak_price=3.0)["volume"]["realized_tokens_capped"] == 20 + 30 + 50
    del realized["c"]
    assert r["who"] == {"predicted_sellers": 2, "realized_sellers": 2, "both": 1, "precision": 0.5,
                        "recall": 0.5, "base_rate": 2 / 3}
    assert r["where"]["wallets"] == 2 and r["where"]["first_sale_at_or_past_level"] == 1   # a at 1.3 >= 1.2; b at 2.5 < 4
    assert r["outside_model"]["time"]["sold_tokens"] == 100
    bands = {c["band"]: c for c in r["curve"]}
    assert bands["1.25-1.5x"]["realized_tokens"] == 12 and bands["0-1.25x"]["predicted_tokens"] == 10
    assert bands["2-3x"]["realized_tokens"] == 38 and "3-5x" not in bands


def test_sell_through_is_net_clamped_and_never_decreasing():
    from cexflows.sellcurve import sold_by

    events = [[1.2, 40], [1.5, -30], [2.0, 200]]          # sell 40, buy 30 back, then dump
    assert sold_by(events, 100, 1.3) == 40
    assert sold_by(events, 100, 1.6) == 40                 # buying back doesn't un-sell
    assert sold_by(events, 100, 2.0) == 100                # capped at the starting balance


def test_curves_use_only_rallies_that_reached_each_multiple():
    from cexflows.sellcurve import GRID, fit, predict, share_at

    def rally(top, sold_at_2x):
        hs = [{"group": "in_profit", "balance": 100, "events": [[2.0, sold_at_2x]]} for _ in range(3)]
        return {"token": "t", "max_multiple": top, "holders": hs}

    curves = fit([rally(3.0, 50), rally(1.6, 0)])
    i15, i2 = GRID.index(1.5), GRID.index(2.0)
    assert curves["in_profit"][i15] == 0 and curves["in_profit"][i2] == 0.5     # 2x only from the first rally
    assert curves["underwater"][i2] == curves["all"][i2]                          # empty group borrows the pool
    assert share_at(curves["in_profit"], 1.0) == 0 and share_at(curves["in_profit"], 50) == curves["in_profit"][-1]
    assert abs(predict([{"group": "in_profit", "balance": 10}], curves, 2.0)["in_profit"] - 5) < 1e-9


def test_supply_curve_adds_curve_layers_split_by_market_makers():
    from cexflows.holders import supply_curve

    fitted = {"grid": [1.1, 1.25, 1.5, 2.0], "curves": {
        "underwater": [0.1, 0.2, 0.3, 0.4], "in_profit": [0, 0, 0, 0], "zero_cost": [0, 0, 0, 0.1],
        "unknown": [0, 0, 0, 0], "trader": [0.5, 0.5, 0.5, 0.5], "all": [0, 0, 0, 0]}}
    hs = [{"address": "u", "kind": "price", "balance": 100, "cost_basis_usd": 3.0, "breakeven_price": None,
           "breakeven_tokens": 0, "takeprofit_price": None, "takeprofit_tokens": 0},
          {"address": "m", "kind": "trader", "balance": 10, "cost_basis_usd": 1.0, "breakeven_price": None,
           "breakeven_tokens": 0, "takeprofit_price": None, "takeprofit_tokens": 0}]
    rows = supply_curve(hs, 1.0, max_multiple=2.0, step_pct=10, curves=fitted)
    assert abs(rows[0]["curve_tokens"] - 10) < 1e-9 and abs(rows[0]["trader_tokens"] - 5) < 1e-9   # 1.0 -> 1.1
    assert abs(sum(r["curve_tokens"] for r in rows) - 40) < 1e-6                 # flat past the grid
    assert supply_curve(hs, 1.0, max_multiple=2.0, step_pct=10)[0]["curve_tokens"] is None
