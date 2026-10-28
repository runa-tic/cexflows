"""A local stand-in for part of Dune's API, answered from Etherscan data instead of Dune's warehouse.

Serves the calls a saved-query client makes, with Dune's paths and JSON shapes:

    POST /api/v1/query/{query_id}/execute      {"query_parameters": {...}} -> execution_id
    GET  /api/v1/execution/{execution_id}/status
    GET  /api/v1/execution/{execution_id}/results
    GET  /api/v1/query/{query_id}/results      latest finished execution of that query

It runs no SQL. Each supported query is implemented natively and registered under a local id;
CEXFLOWS_QUERY_MAP ("4242424=cex_inflows,...") also maps real Dune query ids to them, so one id
works against both. Anything else is a 404 naming what is supported, never empty rows.

Auth works like Dune's: the X-Dune-API-Key header, here holding this server's own key
(CEXFLOWS_API_KEY, else ~/.cexflows-key, generated on first run, mode 600). Requests must name a
loopback Host; CEXFLOWS_HOSTS allows more, comma-separated. Port: CEXFLOWS_PORT (8766).
"""
from __future__ import annotations

import json
import os
import re
import secrets
import sys
import threading
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List, Optional

from . import flows

# Each query: name -> (local id, handler(params) -> rows, column order).
def _cex_inflows(params: dict) -> List[dict]:
    token = str(params.get("token_address") or "").strip()
    chain = str(params.get("chain") or "").strip()
    if not re.fullmatch(r"0x[0-9a-fA-F]{40}", token) or not chain:
        raise ValueError("cex_inflows needs token_address (0x + 40 hex) and chain")
    days = int(float(params.get("days") or 365))
    rows = flows.daily_inflows(chain, token, days=days, progress=lambda m: print(m, file=sys.stderr))
    return [{"day": r["day"], "tokens_to_cex": r["tokens_to_cex"], "usd_to_cex": r.get("usd_to_cex"),
             "n_transfers": r["n_transfers"], "by_exchange": json.dumps(r["by_exchange"], sort_keys=True)}
            for r in rows]


def _cohort_flows(params: dict) -> List[dict]:
    from . import cohorts

    token = str(params.get("token_address") or "").strip()
    chain = str(params.get("chain") or "").strip()
    raw = params.get("cliff_dates") or ""
    dates = [d.strip() for d in (raw if isinstance(raw, list) else str(raw).split(",")) if d.strip()]
    if not re.fullmatch(r"0x[0-9a-fA-F]{40}", token) or not chain or len(dates) < 2:
        raise ValueError("cohort_flows needs token_address, chain and at least two cliff_dates (YYYY-MM-DD, comma-separated)")
    res = cohorts.cohort_flows(chain, token, dates,
                               release_days=int(float(params.get("release_days") or 2)),
                               follow_days=int(float(params.get("follow_days") or 30)),
                               progress=lambda m: print(m, file=sys.stderr))
    return res["rows"]


def _cliff_study(params: dict) -> List[dict]:
    from . import eventstudy

    token = str(params.get("token_address") or "").strip()
    chain = str(params.get("chain") or "").strip()
    split = lambda v: [x.strip() for x in (v if isinstance(v, list) else str(v or "").split(",")) if str(x).strip()]
    dates = split(params.get("cliff_dates"))
    if not re.fullmatch(r"0x[0-9a-fA-F]{40}", token) or not chain or len(dates) < 3:
        raise ValueError("cliff_study needs token_address, chain and at least three cliff_dates (YYYY-MM-DD, comma-separated)")
    release = params.get("release_tokens")
    res = eventstudy.cliff_study(chain, token, dates, float(release) if release not in (None, "") else None,
                                 [int(float(w)) for w in split(params.get("windows")) or [3, 7, 10, 15]],
                                 split(params.get("other_events")),
                                 progress=lambda m: print(m, file=sys.stderr))
    return [{**{k: v for k, v in r.items() if k != "events"}, "events": json.dumps(r["events"], sort_keys=True)}
            for r in res["rows"]]


def _holder_behaviour(params: dict) -> List[dict]:
    from . import holders

    token = str(params.get("token_address") or "").strip()
    chain = str(params.get("chain") or "").strip()
    if not re.fullmatch(r"0x[0-9a-fA-F]{40}", token) or not chain:
        raise ValueError("holder_behaviour needs token_address (0x + 40 hex) and chain")
    as_of = str(params.get("as_of") or "").strip() or None
    res = holders.holder_positions(chain, token, top_n=int(float(params.get("top_n") or 100)), as_of=as_of,
                                   progress=lambda m: print(m, file=sys.stderr))
    return res["holders"]


def _supply_curve(params: dict) -> List[dict]:
    from . import holders, sellcurve

    token = str(params.get("token_address") or "").strip()
    chain = str(params.get("chain") or "").strip()
    if not re.fullmatch(r"0x[0-9a-fA-F]{40}", token) or not chain:
        raise ValueError("supply_curve needs token_address (0x + 40 hex) and chain")
    as_of = str(params.get("as_of") or "").strip() or None
    res = holders.holder_positions(chain, token, top_n=int(float(params.get("top_n") or 100)), as_of=as_of,
                                   progress=lambda m: print(m, file=sys.stderr))
    rows = holders.supply_curve(res["holders"], res["price_now"],
                                max_multiple=float(params.get("max_multiple") or 3),
                                step_pct=float(params.get("step_pct") or 5),
                                target_price=float(params.get("target_price") or 0) or None,
                                curves=sellcurve.load())
    # Zero-cost holders release on a schedule, not at a price: one constant column, not a level.
    return [{**r, "time_triggered_tokens": res["time_triggered_tokens"]} for r in rows]


CURVE_COLUMNS = ["price_from", "price_to", "breakeven_tokens", "takeprofit_tokens", "step_tokens",
                 "cumulative_tokens", "step_usd", "wallets", "time_triggered_tokens", "curve_tokens",
                 "trader_tokens", "trader_dump_tokens"]

HOLDER_COLUMNS = ["address", "kind", "balance", "share_of_supply", "cost_basis_usd", "zero_cost_share", "origins",
                  "oldest_lot", "n_sales", "sold_tokens", "median_sale_multiple", "last_sale",
                  "breakeven_price", "breakeven_tokens", "takeprofit_price", "takeprofit_tokens",
                  "sell_fraction", "take_profit_multiple", "confidence"]

STUDY_COLUMNS = ["window_days", "uplift", "uplift_se", "t_stat", "placebo_n", "placebo_median", "placebo_min",
                 "placebo_max", "placebo_share_above", "excess_tokens", "release_share", "release_share_low",
                 "release_share_high", "events", "r2"]

COHORT_COLUMNS = ["cliff_date", "released_tokens", "released_usd", "to_exchange_tokens", "to_deposit_tokens",
                  "to_contract_tokens", "to_wallet_tokens", "to_internal_tokens", "to_unclassified_tokens",
                  "retained_tokens", "exchange_share", "contract_share"]

QUERIES: Dict[str, tuple] = {
    "cex_inflows": (1, _cex_inflows, ["day", "tokens_to_cex", "usd_to_cex", "n_transfers", "by_exchange"]),
    "cohort_flows": (2, _cohort_flows, COHORT_COLUMNS),
    "cliff_study": (3, _cliff_study, STUDY_COLUMNS),
    "holder_behaviour": (4, _holder_behaviour, HOLDER_COLUMNS),
    "supply_curve": (5, _supply_curve, CURVE_COLUMNS),
}


def query_map() -> Dict[int, str]:
    out = {local_id: name for name, (local_id, _, _) in QUERIES.items()}
    for pair in filter(None, (p.strip() for p in os.environ.get("CEXFLOWS_QUERY_MAP", "").split(","))):
        qid, _, name = pair.partition("=")
        if name.strip() not in QUERIES:
            raise SystemExit(f"CEXFLOWS_QUERY_MAP: unknown query {name.strip()!r} (supported: {', '.join(QUERIES)})")
        out[int(qid)] = name.strip()
    return out


def server_key() -> str:
    key = os.environ.get("CEXFLOWS_API_KEY", "").strip()
    if key:
        return key
    path = os.path.join(os.path.expanduser("~"), ".cexflows-key")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    key = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(key + "\n")
    return key


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Executions:
    def __init__(self, runner: Optional[Callable[[str, dict], List[dict]]] = None):
        self.lock = threading.Lock()
        self.items: Dict[str, dict] = {}
        self.latest: Dict[int, str] = {}
        self.runner = runner or (lambda name, params: QUERIES[name][1](params))

    def start(self, query_id: int, name: str, params: dict) -> dict:
        eid = "01" + uuid.uuid4().hex.upper()[:24]
        item = {"execution_id": eid, "query_id": query_id, "name": name, "state": "QUERY_STATE_PENDING",
                "submitted_at": _now(), "params": params}
        with self.lock:
            self.items[eid] = item
        threading.Thread(target=self._run, args=(eid,), daemon=True).start()
        return item

    def _run(self, eid: str) -> None:
        item = self.items[eid]
        item["state"] = "QUERY_STATE_EXECUTING"
        item["execution_started_at"] = _now()
        try:
            rows = self.runner(item["name"], item["params"])
            item.update(rows=rows, state="QUERY_STATE_COMPLETED")
            with self.lock:
                self.latest[item["query_id"]] = eid
        except Exception as e:
            traceback.print_exc()
            item.update(state="QUERY_STATE_FAILED", error={"type": type(e).__name__, "message": str(e)})
        item["execution_ended_at"] = _now()

    def status(self, item: dict) -> dict:
        out = {k: item[k] for k in ("execution_id", "query_id", "state", "submitted_at") if k in item}
        out["is_execution_finished"] = item["state"] in ("QUERY_STATE_COMPLETED", "QUERY_STATE_FAILED")
        for k in ("execution_started_at", "execution_ended_at", "error"):
            if k in item:
                out[k] = item[k]
        return out

    def results(self, item: dict) -> dict:
        out = self.status(item)
        if item["state"] == "QUERY_STATE_COMPLETED":
            cols = QUERIES[item["name"]][2]
            out["result"] = {"rows": item["rows"],
                             "metadata": {"column_names": cols, "row_count": len(item["rows"]),
                                          "total_row_count": len(item["rows"])}}
        return out


def make_handler(key: str, hosts: set, executions: Executions, qmap: Dict[int, str]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "cexflows"

        def log_message(self, fmt, *args):
            print("%s %s" % (self.command, self.path), file=sys.stderr)

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _guard(self) -> bool:
            if (self.headers.get("Host") or "") not in hosts:
                self._send(403, {"error": "host not allowed"})
                return False
            if not secrets.compare_digest(self.headers.get("X-Dune-API-Key") or "", key):
                self._send(401, {"error": "invalid API Key"})
                return False
            return True

        def _query(self, qid: str) -> Optional[str]:
            name = qmap.get(int(qid))
            if name is None:
                supported = ", ".join(f"{i}={n}" for i, n in sorted(qmap.items()))
                self._send(404, {"error": f"query {qid} is not implemented by cexflows (supported: {supported})"})
            return name

        def do_POST(self):
            if not self._guard():
                return
            m = re.fullmatch(r"/api/v1/query/(\d+)/execute", self.path.split("?")[0])
            if not m:
                return self._send(404, {"error": "not found"})
            name = self._query(m.group(1))
            if name is None:
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                params = body.get("query_parameters") or {}
                if not isinstance(params, dict):
                    raise ValueError
            except (ValueError, json.JSONDecodeError):
                return self._send(400, {"error": "body must be JSON with an object query_parameters"})
            item = executions.start(int(m.group(1)), name, params)
            self._send(200, {"execution_id": item["execution_id"], "state": item["state"]})

        def do_GET(self):
            if not self._guard():
                return
            path = self.path.split("?")[0]
            m = re.fullmatch(r"/api/v1/execution/([0-9A-Z]+)/(status|results)", path)
            if m:
                item = executions.items.get(m.group(1))
                if item is None:
                    return self._send(404, {"error": "execution not found"})
                return self._send(200, executions.status(item) if m.group(2) == "status" else executions.results(item))
            m = re.fullmatch(r"/api/v1/query/(\d+)/results", path)
            if m:
                if self._query(m.group(1)) is None:
                    return
                eid = executions.latest.get(int(m.group(1)))
                if eid is None:
                    return self._send(404, {"error": "not found: No execution found for the latest version of the given query"})
                return self._send(200, executions.results(executions.items[eid]))
            self._send(404, {"error": "not found"})

    return Handler


def allowed_hosts(port: int) -> set:
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}", "127.0.0.1", "localhost"}
    hosts |= {h.strip() for h in os.environ.get("CEXFLOWS_HOSTS", "").split(",") if h.strip()}
    return hosts


def main() -> None:
    port = int(os.environ.get("CEXFLOWS_PORT", "8766"))
    qmap = query_map()
    httpd = ThreadingHTTPServer(("127.0.0.1", port),
                                make_handler(server_key(), allowed_hosts(port), Executions(), qmap))
    print(f"cexflows: Dune-compatible API on http://127.0.0.1:{port}/api/v1 "
          f"(queries: {', '.join(f'{i}={n}' for i, n in sorted(qmap.items()))})", file=sys.stderr)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
