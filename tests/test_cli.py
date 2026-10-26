"""The two CLIs that reach a query without the HTTP API, and the writers they share."""
import csv
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from cexflows import flows, holders, report

HOLDER = {
    "address": "0x" + "a" * 40, "kind": "price", "balance": 1_000_000.0, "share_of_supply": 0.0123,
    "cost_basis_usd": 0.01, "zero_cost_share": 0.0, "origins": json.dumps({"exchange": 1.0}),
    "oldest_lot": "2025-01-05", "n_sales": 12, "sold_tokens": 20_000.0, "median_sale_multiple": 1.8,
    "last_sale": "2026-03-01", "breakeven_price": 0.03, "breakeven_tokens": 400_000.0,
    "takeprofit_price": 0.02, "takeprofit_tokens": 300_000.0, "sell_fraction": 0.4,
    "take_profit_multiple": 2.0, "confidence": "high",
}

TIME_HOLDER = dict(HOLDER, address="0x" + "b" * 40, kind="time", cost_basis_usd=None,
                   breakeven_price=None, breakeven_tokens=0.0, takeprofit_price=None,
                   takeprofit_tokens=0.0, confidence=None)

POSITIONS = {
    "holders": [HOLDER, TIME_HOLDER],
    "excluded": [{"address": "0x" + "c" * 40, "balance": 5e6, "share_of_supply": 0.4, "reason": "exchange"}],
    "supply": 81_000_000.0, "lot_mismatches": 0, "price_now": 0.0123,
    "priors": {"0.00100": {"sell_fraction": 0.1, "take_profit_multiple": 2.5}},
    "time_triggered_tokens": 4_000_000.0,
    "params": {"chain": "ethereum", "token": "0x" + "1" * 40, "top_n": 100, "as_of": None},
}

FLOW_ROWS = [{"day": "2026-01-01", "tokens_to_cex": 1500.0, "n_transfers": 3,
              "by_exchange": {"Alpha": 1000.0, "Beta": 500.0}, "usd_price": 0.5, "usd_to_cex": 750.0},
             {"day": "2026-01-02", "tokens_to_cex": 0.0, "n_transfers": 0, "by_exchange": {},
              "usd_price": None, "usd_to_cex": None}]


class FakeRun:
    def __init__(self):
        self.calls = []

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return POSITIONS


def test_holders_cli_prints_positions_and_levels(monkeypatch, capsys):
    run = FakeRun()
    monkeypatch.setattr(holders, "holder_positions", run)
    holders.main(["--token", "0x" + "1" * 40, "--top-n", "5", "--quiet"])
    out = capsys.readouterr().out
    assert "price 0.012300 USD" in out and "81,000,000 tokens" in out
    assert "2 holders modelled" in out and "1 large wallets set aside" in out
    assert HOLDER["address"] in out and "0.03" in out and "400,000" in out
    assert "supply curve:" in out and "price from" in out
    assert run.calls[0][1]["top_n"] == 5


def test_holders_cli_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr(holders, "holder_positions", lambda *a, **k: POSITIONS)
    out = tmp_path / "positions.json"
    holders.main(["--token", "0x" + "1" * 40, "--json", str(out), "--quiet"])
    payload = json.loads(out.read_text())
    assert payload["holders"][0]["address"] == HOLDER["address"]
    assert payload["curve"] and payload["curve"][0]["price_from"] == 0.0123
    assert payload["excluded"][0]["reason"] == "exchange" and payload["priors"]
    assert payload["curve_params"]["max_multiple"] == 3.0


def test_holders_cli_writes_json_to_stdout(monkeypatch, capsys):
    monkeypatch.setattr(holders, "holder_positions", lambda *a, **k: POSITIONS)
    holders.main(["--token", "0x" + "1" * 40, "--json", "-", "--no-curve", "--quiet"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["curve"] == [] and payload["holders"][0]["kind"] == "price"


def test_holders_cli_writes_the_holders_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(holders, "holder_positions", lambda *a, **k: POSITIONS)
    out = tmp_path / "holders.csv"
    holders.main(["--token", "0x" + "1" * 40, "--csv", str(out), "--quiet"])
    rows = list(csv.DictReader(io.StringIO(out.read_text())))
    assert list(rows[0]) == holders.HOLDER_COLUMNS
    assert rows[0]["address"] == HOLDER["address"] and rows[0]["origins"] == HOLDER["origins"]
    assert rows[1]["cost_basis_usd"] == "" or rows[1]["cost_basis_usd"] == "None"


def test_holders_cli_writes_the_curve_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(holders, "holder_positions", lambda *a, **k: POSITIONS)
    out = tmp_path / "curve.csv"
    holders.main(["--token", "0x" + "1" * 40, "--curve-csv", str(out), "--quiet"])
    rows = list(csv.DictReader(io.StringIO(out.read_text())))
    assert list(rows[0]) == holders.CURVE_COLUMNS
    # Break-even 400k at 0.03 and take-profit 300k at 0.02 both sit inside a 3x curve.
    assert float(rows[-1]["cumulative_tokens"]) == pytest.approx(700_000.0)
    assert float(rows[-1]["cumulative_tokens"]) == pytest.approx(
        sum(float(r["step_tokens"]) for r in rows))


def test_holders_cli_no_curve_skips_it(monkeypatch, capsys):
    monkeypatch.setattr(holders, "holder_positions", lambda *a, **k: POSITIONS)
    holders.main(["--token", "0x" + "1" * 40, "--no-curve", "--quiet"])
    out = capsys.readouterr().out
    assert "holders modelled" in out and "supply curve:" not in out


def test_holders_cli_reports_a_value_error_as_an_exit(monkeypatch):
    monkeypatch.setattr(holders, "holder_positions", lambda *a, **k: dict(POSITIONS, price_now=None))
    with pytest.raises(SystemExit) as e:
        holders.main(["--token", "0x" + "1" * 40, "--quiet"])
    assert "no price" in str(e.value) or "today's price" in str(e.value)


def test_holders_cli_passes_the_curve_options(monkeypatch):
    seen = {}

    def fake_curve(holders_rows, price_now, max_multiple=3.0, step_pct=5.0, target_price=None):
        seen.update(price_now=price_now, max_multiple=max_multiple, step_pct=step_pct, target_price=target_price)
        return []

    monkeypatch.setattr(holders, "holder_positions", lambda *a, **k: POSITIONS)
    monkeypatch.setattr(holders, "supply_curve", fake_curve)
    holders.main(["--token", "0x" + "1" * 40, "--max-multiple", "4", "--step-pct", "2.5",
                  "--target-price", "0.02", "--quiet"])
    assert seen == {"price_now": 0.0123, "max_multiple": 4.0, "step_pct": 2.5, "target_price": 0.02}


def test_flows_cli_prints_the_daily_table(monkeypatch, capsys):
    monkeypatch.setattr(flows, "daily_inflows", lambda *a, **k: FLOW_ROWS)
    flows.main(["--token", "0x" + "1" * 40, "--days", "2", "--quiet"])
    out = capsys.readouterr().out
    assert "2026-01-01" in out and "1,500.00" in out and "Alpha 1,000, Beta 500" in out
    assert "750.00" in out                       # usd
    assert "2 days" in out and "1 with inflow" in out


def test_flows_cli_writes_csv_and_json(tmp_path, monkeypatch):
    monkeypatch.setattr(flows, "daily_inflows", lambda *a, **k: FLOW_ROWS)
    csv_path, json_path = tmp_path / "flows.csv", tmp_path / "flows.json"
    flows.main(["--token", "0x" + "1" * 40, "--csv", str(csv_path), "--json", str(json_path), "--quiet"])
    rows = list(csv.DictReader(io.StringIO(csv_path.read_text())))
    assert list(rows[0]) == flows.FLOW_COLUMNS
    assert rows[0]["by_exchange"] == '{"Alpha": 1000.0, "Beta": 500.0}'
    assert rows[1]["usd_to_cex"] == "" or rows[1]["usd_to_cex"] == "None"
    assert json.loads(json_path.read_text())[0]["day"] == "2026-01-01"


def test_flows_cli_rejects_a_bad_end_date(monkeypatch):
    monkeypatch.setattr(flows, "daily_inflows", lambda *a, **k: FLOW_ROWS)
    with pytest.raises(SystemExit, match="--end must be YYYY-MM-DD"):
        flows.main(["--token", "0x" + "1" * 40, "--end", "01/01/2026"])


def test_flows_cli_reports_a_value_error_as_an_exit(monkeypatch):
    def boom(*a, **k):
        raise ValueError("days must be at least 1, got 0")

    monkeypatch.setattr(flows, "daily_inflows", boom)
    with pytest.raises(SystemExit, match="days must be at least 1"):
        flows.main(["--token", "0x" + "1" * 40])


def test_flows_cli_end_date_reaches_the_engine(monkeypatch):
    seen = {}

    def fake(chain, token, days=365, end_ts=None, **k):
        seen.update(chain=chain, token=token, days=days, end_ts=end_ts)
        return []

    monkeypatch.setattr(flows, "daily_inflows", fake)
    flows.main(["--token", "0x" + "1" * 40, "--days", "7", "--end", "2026-03-01", "--quiet"])
    last_second = datetime(2026, 3, 1, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    assert seen["days"] == 7 and seen["end_ts"] == last_second     # the last day is reported whole


def test_report_writes_nested_values_as_json(tmp_path):
    path = tmp_path / "t.json"
    report.write_json({"when": object(), "rows": [{"a": 1}]}, str(path))
    assert json.loads(path.read_text())["rows"] == [{"a": 1}]


def test_report_csv_encodes_a_nested_cell_and_ignores_extra_keys(tmp_path):
    path = tmp_path / "t.csv"
    report.write_csv([{"a": 1, "b": {"x": 2}, "z": 9}], str(path), ["a", "b"])
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    assert rows == [{"a": "1", "b": '{"x": 2}'}]


def test_report_csv_writes_to_stdout(capsys):
    report.write_csv([{"a": 1}], "-", ["a"])
    assert capsys.readouterr().out.splitlines() == ["a", "1"]
