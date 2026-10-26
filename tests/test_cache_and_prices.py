"""Cache windows and price fetches: the ranges that are fetched, and those that are not.

The bugs these pin down were silent: a stored range that covers blocks no chain has, a price
fetch that re-downloads a whole history on every run, a curve that never terminates.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from cexflows import cohorts, holders, prices, server
from cexflows.cache import Cache

DAY = 86400
WALLET = "0x" + "a" * 40


def test_cache_accepts_a_bare_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = Cache("cache.sqlite")          # CEXFLOWS_CACHE=cache.sqlite: dirname is ""
    cache.store("ethereum", "0xt", WALLET, [], 10, 20)
    assert (tmp_path / "cache.sqlite").is_file()
    assert cache.coverage("ethereum", "0xt", WALLET) == (10, 20)


def test_cache_refuses_an_inverted_wallet_range():
    cache = Cache(":memory:")
    with pytest.raises(ValueError, match="inverted block range"):
        cache.store("ethereum", "0xt", WALLET, [], 2000, 1000)
    assert cache.coverage("ethereum", "0xt", WALLET) is None


def test_cache_ensure_refuses_an_inverted_range():
    cache = Cache(":memory:")
    asked = []
    with pytest.raises(ValueError, match="inverted block range"):
        cache.ensure("ethereum", "0xt", WALLET, 2000, 1000, lambda *a: asked.append(a) or [])
    assert asked == []                     # nothing is fetched for a window that cannot exist


def test_cache_refuses_an_inverted_token_log_range():
    cache = Cache(":memory:")
    with pytest.raises(ValueError, match="inverted block range"):
        cache.store_token_logs("ethereum", "0xt", [], 2000, 1000)
    assert cache.token_log_ranges("ethereum", "0xt") == []


def test_cache_refuses_an_inverted_price_range():
    cache = Cache(":memory:")
    with pytest.raises(ValueError, match="inverted day range"):
        cache.store_price_range("ethereum", "0xt", "2026-02-01", "2026-01-01")
    assert cache.price_ranges("ethereum", "0xt") == []


def test_wallet_coverage_merges_touching_ranges():
    cache = Cache(":memory:")
    cache.store("ethereum", "0xt", WALLET, [], 100, 200)
    cache.store("ethereum", "0xt", WALLET, [], 201, 300)
    cache.store("ethereum", "0xt", WALLET, [], 50, 99)
    assert cache.coverage("ethereum", "0xt", WALLET) == (50, 300)


def test_token_log_coverage_merges_adjacent_ranges_and_keeps_a_gap():
    cache = Cache(":memory:")
    cache.store_token_logs("ethereum", "0xt", [], 100, 200)
    cache.store_token_logs("ethereum", "0xt", [], 201, 300)      # touches: merges
    cache.store_token_logs("ethereum", "0xt", [], 500, 600)      # apart: stays a second range
    cache.store_token_logs("ethereum", "0xt", [], 100, 200)      # again: no new row
    assert cache.token_log_ranges("ethereum", "0xt") == [(100, 300), (500, 600)]


def test_price_coverage_merges_consecutive_days():
    cache = Cache(":memory:")
    cache.store_price_range("ethereum", "0xt", "2026-01-01", "2026-01-10")
    cache.store_price_range("ethereum", "0xt", "2026-01-11", "2026-01-20")
    assert cache.price_ranges("ethereum", "0xt") == [("2026-01-01", "2026-01-20")]
    assert cache.prices_covered("ethereum", "0xt", "2026-01-05", "2026-01-15")
    assert not cache.prices_covered("ethereum", "0xt", "2026-01-05", "2026-01-21")


def test_missing_days_splits_around_covered_spans():
    ranges = [("2026-01-05", "2026-01-10"), ("2026-01-20", "2026-01-25")]
    assert prices.missing_days(ranges, "2026-01-01", "2026-01-30") == [
        ("2026-01-01", "2026-01-04"), ("2026-01-11", "2026-01-19"), ("2026-01-26", "2026-01-30")]
    assert prices.missing_days(ranges, "2026-01-06", "2026-01-09") == []
    assert prices.missing_days([], "2026-01-01", "2026-01-02") == [("2026-01-01", "2026-01-02")]


class Fetch:
    """A stand-in for CoinGecko: counts calls, and answers with `days` daily points ending now."""

    def __init__(self, days: int = 400):
        self.calls = []
        self.days = days
        self.now = int(time.time())

    def __call__(self, url, headers, retries=4):
        self.calls.append(url)
        return {"prices": [[(self.now - d * DAY) * 1000, 1.0 + d / 10000] for d in range(self.days)][::-1],
                "total_volumes": []}


def test_daily_prices_fetches_a_past_window_once(tmp_path, monkeypatch):
    fetch = Fetch()
    monkeypatch.setattr(prices, "_get", fetch)
    cache = Cache(str(tmp_path / "c.sqlite"))
    mint, end = fetch.now - 3 * 365 * DAY, fetch.now - 3 * DAY
    first = prices.daily_prices("ethereum", "0xt", mint, end, cache)
    second = prices.daily_prices("ethereum", "0xt", mint, end, cache)
    assert len(fetch.calls) == 1
    assert first == second and len(first) == 397
    assert 1.0 < first[prices._day(fetch.now - 3 * DAY)] < 1.05     # the day the window ends on


def test_daily_prices_refreshes_only_today(monkeypatch):
    fetch = Fetch()
    monkeypatch.setattr(prices, "_get", fetch)
    cache = Cache(":memory:")
    mint = fetch.now - 2 * 365 * DAY
    prices.daily_prices("ethereum", "0xt", mint, fetch.now, cache)
    assert len(fetch.calls) == 2            # the history, then today on its own
    prices.daily_prices("ethereum", "0xt", mint, fetch.now, cache)
    prices.daily_prices("ethereum", "0xt", mint, fetch.now, cache)
    assert len(fetch.calls) == 4           # one call per run after that, and it is one day long
    midnight = (fetch.now // DAY) * DAY
    assert f"from={midnight - DAY}" in fetch.calls[-1] and f"to={midnight + DAY}" in fetch.calls[-1]


def test_daily_prices_fills_the_gap_and_keeps_the_points(monkeypatch):
    fetch = Fetch()
    monkeypatch.setattr(prices, "_get", fetch)
    cache = Cache(":memory:")
    end = fetch.now - 3 * DAY
    prices.daily_prices("ethereum", "0xt", fetch.now - 30 * DAY, end - 10 * DAY, cache)
    rows = prices.daily_prices("ethereum", "0xt", fetch.now - 30 * DAY, end, cache)
    assert len(fetch.calls) == 2            # the first span, then only the ten days after it
    assert len(rows) == 28                 # now-30d .. now-3d inclusive
    assert cache.price_ranges("ethereum", "0xt") == [
        (prices._day(fetch.now - 30 * DAY), prices._day(end))]


def test_daily_prices_keeps_the_earliest_point_of_each_day(monkeypatch):
    now = int(time.time())
    monkeypatch.setattr(prices, "_get", lambda url, headers, retries=4: {
        "prices": [[(now - 2 * DAY) * 1000, 5.0], [(now - 2 * DAY + 3600) * 1000, 9.0],
                   [(now - DAY) * 1000, 6.0]], "total_volumes": []})
    got = prices.daily_prices("ethereum", "0xt", now - 2 * DAY, now - DAY, Cache(":memory:"))
    assert got[prices._day(now - 2 * DAY)] == 5.0 and got[prices._day(now - DAY)] == 6.0


def test_daily_prices_rejects_bad_input():
    with pytest.raises(ValueError, match="no CoinGecko platform"):
        prices.daily_prices("solana", "0xt", 0, 86400, Cache(":memory:"))
    with pytest.raises(ValueError, match="after"):
        prices.daily_prices("ethereum", "0xt", 86400, 0, Cache(":memory:"))


def test_daily_inflows_rejects_a_window_that_cannot_exist(monkeypatch):
    from cexflows import flows

    monkeypatch.setattr(flows, "hot_wallets", lambda chain, path=None: {WALLET: "Alpha"})
    monkeypatch.setattr(flows.etherscan, "block_at", lambda chain, ts, closest="before": int(ts))
    for days in (0, -5):
        with pytest.raises(ValueError, match="days must be at least 1"):
            flows.daily_inflows("ethereum", "0xt", days=days, end_ts=1_790_000_000, cache=Cache(":memory:"))


def test_daily_inflows_rejects_an_empty_block_window(monkeypatch):
    from cexflows import flows

    monkeypatch.setattr(flows, "hot_wallets", lambda chain, path=None: {WALLET: "Alpha"})
    # The chain head is behind the requested day: no block range exists for it.
    monkeypatch.setattr(flows.etherscan, "block_at", lambda chain, ts, closest="before": 100 if closest == "after" else 50)
    with pytest.raises(ValueError, match="no blocks for"):
        flows.daily_inflows("ethereum", "0xt", days=7, end_ts=1_790_000_000, cache=Cache(":memory:"))


def test_supply_curve_rejects_inputs_that_cannot_terminate():
    for kwargs in ({"step_pct": 0}, {"step_pct": -5}, {"max_multiple": 1.0}, {"max_multiple": 0.5},
                   {"target_price": float("inf")}):
        with pytest.raises(ValueError):
            holders.supply_curve([], 1.0, **kwargs)
    with pytest.raises(ValueError, match="today's price"):
        holders.supply_curve([], None)


def test_supply_curve_step_count_is_capped():
    rows = holders.supply_curve([], 1.0, max_multiple=1e6, step_pct=0.001)
    assert len(rows) <= holders.MAX_STEPS                    # a tiny step cannot run away


def test_window_transfers_rejects_an_empty_window(monkeypatch):
    cache = Cache(":memory:")
    monkeypatch.setattr(cohorts.etherscan, "block_at", lambda chain, ts, closest="before": int(ts))
    with pytest.raises(ValueError, match="empty window"):
        cohorts.window_transfers("ethereum", "0xt", 1000, 1000, cache)
    # A release window shorter than a block interval: the head block is behind the tail block.
    monkeypatch.setattr(cohorts.etherscan, "block_at",
                        lambda chain, ts, closest="before": 900 if closest == "after" else 800)
    with pytest.raises(ValueError, match="empty block range"):
        cohorts.window_transfers("ethereum", "0xt", 1000, 2000, cache)


def test_query_map_explains_a_bad_entry(monkeypatch):
    monkeypatch.setenv("CEXFLOWS_QUERY_MAP", "abc=cex_inflows")
    with pytest.raises(SystemExit, match="not a numeric Dune query id"):
        server.query_map()
    monkeypatch.setenv("CEXFLOWS_QUERY_MAP", "4242424")
    with pytest.raises(SystemExit, match="expected dune_id=query_name"):
        server.query_map()
    monkeypatch.setenv("CEXFLOWS_QUERY_MAP", "4242424=nonsense")
    with pytest.raises(SystemExit, match="unknown query"):
        server.query_map()
    monkeypatch.setenv("CEXFLOWS_QUERY_MAP", "4242424=cex_inflows, 7 = supply_curve")
    assert server.query_map()[4242424] == "cex_inflows"
    assert server.query_map()[7] == "supply_curve"


def test_int_param_reads_json_numbers_and_strings():
    assert server._int_param({}, "days", 365) == 365
    assert server._int_param({"days": 30}, "days", 365) == 30
    assert server._int_param({"days": "30"}, "days", 365) == 30
    assert server._int_param({"days": 30.7}, "days", 365) == 30
    assert server._int_param({"days": None}, "days", 365) == 365
    with pytest.raises(ValueError, match="at least 1"):
        server._int_param({"days": -1}, "days", 365, minimum=1)


def test_handlers_refuse_parameters_that_invert_a_window(monkeypatch):
    token = "0x" + "1" * 40
    with pytest.raises(ValueError, match="days must be at least 1"):
        server._cex_inflows({"token_address": token, "chain": "ethereum", "days": -5})
    with pytest.raises(ValueError, match="top_n must be at least 1"):
        server._holder_behaviour({"token_address": token, "chain": "ethereum", "top_n": 0})
    with pytest.raises(ValueError, match="release_days must be at least 1"):
        server._cohort_flows({"token_address": token, "chain": "ethereum", "cliff_dates": "2026-01-15,2026-02-15",
                              "release_days": 0})


def test_executions_prune_keeps_memory_bounded():
    done = []

    def runner(name, params):
        done.append(name)
        return [{"day": "2026-01-01"}]

    executions = server.Executions(runner, max_items=3)
    ids = [executions.start(1, "cex_inflows", {})["execution_id"] for _ in range(10)]
    for _ in range(100):
        if all(executions.items[eid]["state"] == "QUERY_STATE_COMPLETED" for eid in ids if eid in executions.items):
            break
        time.sleep(0.02)
    assert len(executions.items) <= 3
    assert ids[-1] in executions.items                     # the newest survives
    assert ids[0] not in executions.items                  # the oldest finished ones are gone


def test_column_orders_match_the_rows_they_describe():
    assert server.QUERIES["holder_behaviour"][2] == holders.HOLDER_COLUMNS
    assert server.QUERIES["supply_curve"][2] == holders.CURVE_COLUMNS
    assert server.QUERIES["cohort_flows"][2] == cohorts.COHORT_COLUMNS
    from cexflows.eventstudy import STUDY_COLUMNS
    assert server.QUERIES["cliff_study"][2] == STUDY_COLUMNS
    assert server.QUERIES["cex_inflows"][2] == ["day", "tokens_to_cex", "usd_to_cex", "n_transfers", "by_exchange"]


def test_a_warm_cache_costs_no_history_request(monkeypatch):
    """The point of the price coverage table: a rerun of a past as-of date makes no HTTP call."""
    fetch = Fetch()
    monkeypatch.setattr(prices, "_get", fetch)
    from cexflows import flows

    monkeypatch.setattr(flows, "hot_wallets", lambda chain, path=None: {WALLET: "Alpha"})
    monkeypatch.setattr(flows, "token_decimals", lambda chain, token: 18)
    monkeypatch.setattr(flows.etherscan, "block_at", lambda chain, ts, closest="before": int(ts))
    monkeypatch.setattr(flows.etherscan, "inbound_transfers", lambda *a: [])
    cache = Cache(":memory:")
    end = fetch.now - 3 * DAY
    flows.daily_inflows("ethereum", "0xt", days=30, end_ts=end, cache=cache)
    calls = len(fetch.calls)
    assert calls == 1
    flows.daily_inflows("ethereum", "0xt", days=30, end_ts=end, cache=cache)
    assert len(fetch.calls) == calls         # the price history is not downloaded twice


def test_hot_wallets_names_the_labels_file_it_could_not_find(tmp_path):
    from cexflows import labels

    with pytest.raises(FileNotFoundError, match="CEXFLOWS_LABELS"):
        labels.hot_wallets("ethereum", str(tmp_path / "missing.csv"))
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("chain,address,exchange,source\nethereum,0xABCDEF,Alpha,x\narbitrum,0xabc,Arb,y\n")
    assert labels.hot_wallets("ethereum", str(csv_path)) == {"0xabcdef": "Alpha"}
    assert labels.hot_wallets("ARBITRUM", str(csv_path)) == {"0xabc": "Arb"}
