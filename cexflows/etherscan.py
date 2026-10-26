"""Etherscan V2 client: inbound ERC-20 transfers to a wallet, and block lookup by time.

Inbound transfers come from the logs endpoint filtered on the Transfer event's `to` topic, so a
hot wallet's withdrawals (its outbound traffic, often the larger half) are never downloaded. The
free tier returns at most 1,000 logs per call, so ranges are walked forward by block. Key:
ETHERSCAN_API_KEY, else ~/.config/cexflows/etherscan_key, else ~/.config/aml-check/etherscan_key.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterator, List, Optional

API = "https://api.etherscan.io/v2/api"
CHAIN_IDS = {"ethereum": 1, "arbitrum": 42161}
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
PAGE = 1000

_KEY_FILES = [
    os.path.join(os.path.expanduser("~"), ".config", "cexflows", "etherscan_key"),
    os.path.join(os.path.expanduser("~"), ".config", "aml-check", "etherscan_key"),
]


class EtherscanError(Exception):
    pass


def api_key() -> str:
    key = os.environ.get("ETHERSCAN_API_KEY", "").strip()
    for path in _KEY_FILES:
        if key:
            break
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                key = f.read().strip()
    if not key:
        raise EtherscanError("no Etherscan key: set ETHERSCAN_API_KEY or create ~/.config/cexflows/etherscan_key")
    return key


class _RateLimit:
    """Free tier: 5 calls per second per key, shared by every thread in this process."""

    def __init__(self, per_second: float = 4.5):
        self.gap = 1.0 / per_second
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.next - now
            self.next = max(now, self.next) + self.gap
        if delay > 0:
            time.sleep(delay)


_limit = _RateLimit()


def chain_id(chain: str) -> int:
    try:
        return CHAIN_IDS[chain.lower()]
    except KeyError:
        raise EtherscanError(f"unsupported chain {chain!r} (supported: {', '.join(CHAIN_IDS)})") from None


def call(chain: str, retries: int = 4, **params) -> object:
    params.update(chainid=chain_id(chain), apikey=api_key())
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        _limit.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cexflows"})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                raise EtherscanError(f"{params.get('module')}/{params.get('action')}: {e}") from e
            time.sleep(2 * (attempt + 1))
            continue
        result = body.get("result")
        if params.get("module") == "proxy":
            # JSON-RPC answers carry "jsonrpc". Etherscan's own errors, a rate limit included,
            # come back in the status/message/result shape with the error text as the result.
            err = body.get("error")
            if "jsonrpc" in body and not err:
                return result
            if isinstance(err, dict):   # a JSON-RPC error such as a revert: retrying won't help
                raise EtherscanError(f"proxy/{params.get('action')}: {str(err.get('message'))[:200]}")
            text = str(result or body.get("message") or "")
        elif body.get("status") == "1":
            return result
        else:
            text = str(result or body.get("message") or "")
        if "no records found" in text.lower() or "no transactions found" in text.lower():
            return []
        if "rate limit" in text.lower() and attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))
            continue
        raise EtherscanError(f"{params.get('module')}/{params.get('action')}: {text[:200]}")
    raise EtherscanError("retries exhausted")


def is_contract(chain: str, address: str) -> bool:
    code = call(chain, module="proxy", action="eth_getCode", address=address, tag="latest")
    return bool(code) and code not in ("0x", "0x0")


def latest_block(chain: str) -> int:
    return int(call(chain, module="proxy", action="eth_blockNumber"), 16)


def block_at(chain: str, ts: int, closest: str = "before") -> int:
    return int(call(chain, module="block", action="getblocknobytime", timestamp=int(ts), closest=closest))


def _topic(address: str) -> str:
    return "0x" + address.lower()[2:].rjust(64, "0")


def inbound_transfers(chain: str, token: str, wallet: str, from_block: int, to_block: int) -> Iterator[dict]:
    """Every Transfer of `token` into `wallet` in [from_block, to_block], oldest first."""
    return transfers(chain, token, from_block, to_block, recipient=wallet)


def _transfer(lg: dict) -> dict:
    return {
        "block": int(lg["blockNumber"], 16),
        "ts": int(lg["timeStamp"], 16),
        "tx": lg["transactionHash"],
        "log_index": int(lg["logIndex"], 16) if lg["logIndex"] != "0x" else 0,
        "from": "0x" + lg["topics"][1][-40:],
        "to": "0x" + lg["topics"][2][-40:],
        "value": int(lg["data"], 16) if lg["data"] not in ("0x", "") else 0,
    }


def transfers(chain: str, token: str, from_block: int, to_block: int,
              sender: Optional[str] = None, recipient: Optional[str] = None) -> Iterator[dict]:
    """Transfers of `token` in [from_block, to_block], oldest first, optionally only those from
    `sender` and/or to `recipient`.

    Yields {block, ts, tx, log_index, from, to, value} with `value` in raw token units.
    """
    filters = {"topic0": TRANSFER_TOPIC}
    if sender:
        filters.update(topic1=_topic(sender), topic0_1_opr="and")
    if recipient:
        filters.update(topic2=_topic(recipient), topic0_2_opr="and")
    if sender and recipient:
        filters["topic1_2_opr"] = "and"
    start = from_block
    seen = set()
    while start <= to_block:
        logs: List[dict] = call(chain, module="logs", action="getLogs", address=token, **filters,
                                fromBlock=start, toBlock=to_block, page=1, offset=PAGE) or []
        if not logs:
            return
        last_block = int(logs[-1]["blockNumber"], 16)
        page_last_block = set()
        for lg in logs:
            t = _transfer(lg)
            key = (t["tx"], t["log_index"])
            if t["block"] == last_block:
                page_last_block.add(key)
            if key not in seen:
                yield t
        if len(logs) < PAGE:
            return
        if last_block == start:
            # One block holds more than a page (an airdrop): page through that block alone.
            page = 2
            while True:
                more: List[dict] = call(chain, module="logs", action="getLogs", address=token, **filters,
                                        fromBlock=start, toBlock=start, page=page, offset=PAGE) or []
                for lg in more:
                    t = _transfer(lg)
                    if (t["tx"], t["log_index"]) not in seen:
                        yield t
                if len(more) < PAGE:
                    break
                page += 1
            seen, start = set(), start + 1
            continue
        # A full page may stop mid-block, so the next call re-reads the last block and `seen`
        # (that block's keys only) drops the repeats.
        seen = page_last_block
        start = last_block
