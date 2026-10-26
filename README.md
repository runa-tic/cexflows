# cexflows

Daily token inflows into exchange hot wallets, measured from Etherscan data and served through a
local, Dune-compatible API. It answers the questions a Dune query about exchange deposits would,
for EVM tokens on Ethereum and Arbitrum, with no Dune plan.

```
python -m cexflows.server        # serves http://127.0.0.1:8766/api/v1
```

Any client that runs saved Dune queries can point at it: same paths, same JSON, same
`X-Dune-API-Key` header, holding this server's key instead of a Dune one.

```
export DUNE_API_BASE=http://127.0.0.1:8766/api/v1
export DUNE_API_KEY=$(cat ~/.cexflows-key)
```

The queries also run from the shell, without a server or a Dune client:

```
cexflows-holders --token 0x... --top-n 100        # positions, sell levels, supply curve
cexflows-flows   --token 0x... --days 90          # daily inflow, with the top exchanges
cexflows-cohorts --chain arbitrum --token 0x... --cliffs 2026-04-15,2026-05-16
cexflows-cliff   --chain ethereum --token 0x... --cliffs 2026-01-15,2026-02-15,2026-03-15 --release 1.5e8
cexflows-backtest --token 0x... --start 2026-02-21 --peak 2026-04-20
```

`cexflows-holders` and `cexflows-flows` take `--json PATH` for the whole result and `--csv PATH`
for the table (`-` is stdout). The server and every command are one entry point each, so
`pip install .` is enough; the same modules also run as `python -m cexflows.holders` and so on.

Stdlib only, Python 3.8+.

## What it measures

Users deposit into per-user deposit addresses, which no public label set covers. Exchanges then
sweep those into a handful of hot wallets. cexflows counts every transfer of the token into a
labelled hot wallet from an address outside the labelled set: a sweep of user deposits, usually
hours after the deposit, or a direct deposit. Transfers between two labelled wallets are internal
and skipped. Only inbound transfers are downloaded (the logs endpoint filtered on the Transfer
event's recipient), so a hot wallet's withdrawals never cost a call.

Each wallet's history is cached in SQLite (`~/.cexflows/cache.sqlite`, or `CEXFLOWS_CACHE`) with
the block range it covers, so a rerun only fetches new blocks. Prices are cached the same way,
with the day spans that were requested, so a token minted years ago is priced once: a free key
serves a year of history, and the count of cached days never matched the range asked for, so every
run used to download the whole series again. A range that reaches today still re-reads today, one
day at a time.

## Queries

The server runs no SQL. Each supported query is implemented natively under a local id:

| id | name | parameters | columns |
| - | - | - | - |
| 1 | `cex_inflows` | `token_address`, `chain` (`ethereum`, `arbitrum`), `days` (default 365) | `day`, `tokens_to_cex`, `usd_to_cex`, `n_transfers`, `by_exchange` (JSON) |
| 2 | `cohort_flows` | `token_address`, `chain`, `cliff_dates` (comma-separated), `release_days` (2), `follow_days` (30) | one row per cliff plus an `all` row: `released_tokens`, `released_usd`, `to_exchange_tokens`, `to_deposit_tokens`, `to_contract_tokens`, `to_wallet_tokens`, `to_internal_tokens`, `to_unclassified_tokens`, `retained_tokens`, `exchange_share`, `contract_share` |
| 3 | `cliff_study` | `token_address`, `chain`, `cliff_dates` (at least 3), `release_tokens` (per cliff, optional), `windows` (`3,7,10,15`), `other_events` (dates to control for) | one row per window: `uplift`, `uplift_se`, `t_stat`, `placebo_n`, `placebo_median`, `placebo_min`, `placebo_max`, `placebo_share_above`, `excess_tokens`, `release_share` with `release_share_low`/`_high` (95%), `events` (JSON), `r2` |
| 4 | `holder_behaviour` | `token_address`, `chain`, `top_n` (100), `as_of` (date, for backtests) | one row per modelled holder: `address`, `kind` (`price`, `time`, `trader`, `unknown`), `balance`, `share_of_supply`, `cost_basis_usd`, `zero_cost_share`, `origins` (JSON), `oldest_lot`, `n_sales`, `sold_tokens`, `median_sale_multiple`, `last_sale` |
| 5 | `supply_curve` | as query 4, plus `max_multiple` (3), `step_pct` (5) and optional `target_price` (adds a step edge there) | one row per price step from today's price: `price_from`, `price_to`, `breakeven_tokens`, `takeprofit_tokens`, `step_tokens`, `cumulative_tokens`, `step_usd`, `wallets`, `time_triggered_tokens` |

`days`, `top_n`, `release_days`, `follow_days` and `windows` are validated rather than coerced: a
`0` or a negative number used to invert the date and block windows, which read back as a token
with no inflow.

`usd_to_cex` is each day's token amount at CoinGecko's daily price for that date, null where
CoinGecko has no price.

### Cohort tracking

`cohort_flows` follows the wallets that receive tokens at unlock cliffs. For each cliff it reads
every transfer of the token in a short window from the cliff date, and in a control window of the
same length ten days earlier. A vesting payout is a sender and recipient pair that recurs in the
cliff windows of at least half the cliffs, never in a control window, never the other way round,
and in steady amounts (at least two within 10% of their median; a first claim carrying accrued
periods is allowed): a vesting contract or distributor paying its beneficiary on schedule.
`released_tokens` counts only the first hop out of vesting, so a distributor passing tokens to
beneficiaries isn't counted twice.

For the following 30 days (stopping at the next cliff), everything the beneficiaries send is
classified: a labelled hot wallet (`exchange`), a wallet that passes the tokens on to one within 7
days (`deposit`, an exchange deposit address), any contract (`contract`: DEX pools, multisigs,
staking, bridges), another beneficiary (`internal`), or any other wallet. A beneficiary's outflows
count only up to what vesting has paid it so far, so a wallet that also trades its own holdings
can't inflate the shares. `exchange_share` is exchange plus deposit over released, the cohort's
measured sell-through to exchanges. DEX selling sits inside `contract`, so it is reported apart
rather than counted as selling. Beneficiaries who claim after the release window, or whose
contracts stream continuously rather than paying on the cliff, are missed, so `released_tokens`
is a sample of the scheduled unlock. Also runnable directly:
`python -m cexflows.cohorts --chain arbitrum --token 0x... --cliffs 2026-04-15,2026-05-16,...`.

### Cliff event study

`cliff_study` asks whether exchange inflows rise after unlock cliffs beyond what the market
explains. Daily inflow swings threefold on its own, so a before/after comparison is noise.
Instead it regresses log daily inflow on log CoinGecko volume, the day's absolute log return, a
trend, a flag for each `other_events` window, and a flag for the W days after any cliff.
`uplift` is the flag's effect per day. The placebo refits with every cliff moved by shifts whose
windows miss all real cliff windows; `placebo_share_above` is the fraction of those that beat the
real estimate, so 0 means the cliffs stand out from ordinary month-to-month variation.
`release_share` is uplift times median daily inflow times W over `release_tokens`: the share of
each release that reached the labelled exchanges within W days, a floor on exchange
sell-through because label coverage is partial. Long windows leave few valid shifts when cliffs
are monthly, so read the placebo at 10 days or less. Also runnable directly:
`python -m cexflows.eventstudy --chain ethereum --token 0x... --cliffs ... --release 1.5e8`.

### Holder positions

`holder_behaviour` rebuilds every wallet's cost basis and sales from the token's full transfer
history. A first run fetches that history, about an hour for a small cap (1.5M transfers); after
that it comes from the cache. Tokens are followed as cost lots from the mint. A transfer between
wallets moves lots with their cost. Out of an exchange wallet or pool is a purchase, priced at
that day's price. Into one, or into an exchange deposit address, is a sale, recorded with its
multiple of cost.

Set aside, not modelled:
- **Exchange wallets.** Labelled hot wallets, plus unlabelled services: anything with more than
  2,000 counterparties.
- **Pools.** From GeckoTerminal's free API.
- **Deposit addresses.** Addresses that pass on nearly all they receive into those.
- **Project wallets.** The mint recipient, plus contracts it funds that distribute on.
- **Other contracts.** Safe multisigs are kept as holders.

`kind` says how a holder is likely to sell, and the level columns (`breakeven_price` and
`_tokens`, `takeprofit_price` and `_tokens`, `sell_fraction`, `take_profit_multiple`,
`confidence`) say where. `price`: bought with a real cost. `time`: mostly
zero-cost tokens from vesting or airdrops, released on a schedule. `trader`: 100 or more sales,
a market maker or bot. Every price is CoinGecko's daily close for now.

Each price-triggered holder gets up to two levels. *Break-even*, only when underwater: the
price gets back to its cost and it sells its usual fraction. *Take-profit*: cost times its
take-profit multiple, where it sells its usual fraction of what is left. The fraction and
multiple are the wallet's own medians when it has two or more such sales, otherwise the medians
of sales by wallets of the same size (bands by share of supply). Small wallets dump everything
at once and large ones sell in slices, so size matters. `supply_curve` sums the levels by
price step. Zero-cost holders are a separate constant, since they release on a schedule, not at
a price. The multiples are learned from the token's own past, so a token that has mostly fallen
teaches small take-profit multiples. Also runnable directly, levels and curve together:
`cexflows-holders --token 0x... --top-n 100 --as-of 2026-04-01 --curve-csv curve.csv` (or
`python -m cexflows.holders`), with `--json` for the whole result and `--csv` for the holdings.

To make a real Dune query id work against both, map it: `CEXFLOWS_QUERY_MAP=4242424=cex_inflows`.
A query that isn't implemented gets a 404 that lists what is, never empty rows.

Endpoints: `POST /api/v1/query/{id}/execute`, `GET /api/v1/execution/{id}/status`,
`GET /api/v1/execution/{id}/results`, `GET /api/v1/query/{id}/results`. Executions live in memory,
bounded to the 200 most recent: the oldest finished ones are dropped first, so a long-running
server doesn't keep every result it ever produced.

## Backtesting the levels

`python -m cexflows.backtest --token 0x... --start <trough> --peak <peak>` builds positions and
levels as of the rally's start, from data up to that day only, then reads the same wallets'
sales over the rally and 14 days after the peak. It reports the volume they sold against what
their levels predicted (raw, and capped at each wallet's starting balance), the precision and
recall of "this wallet sells", each seller's first sale price against its lowest level (scored
against a constant-multiple baseline), tokens by price band, and what the unpriced kinds sold.

First run, four rallies (top 100 holders, capped volume over predicted):

| token | rally | predicted share of balance sold | realized | realized / predicted |
| - | - | - | - | - |
| DEXE | 5.9x, Feb-Apr 2026 | 2% | 74% | 32x |
| LSK | 11.2x, Aug-Sep 2026 | 31% | 65% | 2.1x |
| APEX | 10.7x, Sep-Oct 2025 | 23% | 17% | 0.74x |
| BIO | 5.3x, Jun-Aug 2025 | 16% | 85% | 5.4x |

The levels under-predict selling on three of four, because each holder gets at most two slices
(break-even, take-profit) while on a real rally large holders keep selling as the price climbs.
Recall of sellers is high (88-96%) but precision only matches the base rate, since a 5-11x move
crosses nearly every level. Per-wallet sale prices beat the constant baseline on DEXE, tie on
BIO and lose on LSK and APEX.

## Configuration

| variable | meaning |
| - | - |
| `ETHERSCAN_API_KEY` | Etherscan V2 key; else `~/.config/cexflows/etherscan_key` |
| `COINGECKO_PRO_API_KEY` / `COINGECKO_API_KEY` | CoinGecko key for USD prices (pro or demo); else `~/.config/cexflows/coingecko_pro_key` / `coingecko_key`, else the keyless public API |
| `CEXFLOWS_API_KEY` | this server's key; else `~/.cexflows-key`, generated on first run |
| `CEXFLOWS_PORT` | port, default 8766 |
| `CEXFLOWS_HOSTS` | extra allowed `Host` values, comma-separated (loopback is always allowed) |
| `CEXFLOWS_QUERY_MAP` | `dune_id=query_name` pairs; a pair without `=` or with a non-numeric id is refused at start-up |
| `CEXFLOWS_LABELS` | a different hot-wallet CSV |

A free Etherscan key covers both chains: 5 calls a second and 1,000 logs a call. A busy hot
wallet's first year of history takes a few hundred calls; the cache makes reruns cheap.

## Labels

`labels/hot_wallets.csv` lists exchange hot wallets per chain (`chain,address,exchange,source`).
The seed set comes from [eth-labels](https://github.com/dawsbot/eth-labels) (MIT, notice in `labels/LICENSE-eth-labels`). Exchanges
reuse hot-wallet addresses across EVM chains, so a wallet named on one chain and active with the
token on another is included for that chain, with `source` saying where the name came from. The
labels stay in the repository rather than inside the wheel, so a `pip install` of it needs
`CEXFLOWS_LABELS` pointing at this file; every command says so if it can't find one.

Coverage is the main limit. A missing exchange lowers the totals, so compare inflows against
their own baseline (a cliff week against normal weeks) rather than reading them as an absolute
figure.

## Limits

- EVM chains on Etherscan only, and only Ethereum and Arbitrum so far.
- Deposits are counted when swept, typically hours late; aggregate to weeks for timing work.
- Not affiliated with Dune. It implements the shape of part of Dune's public API so the same
  client code can use either; it doesn't reproduce Dune's data.

## License

MIT
