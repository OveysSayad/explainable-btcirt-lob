# Dense Data Collection Design (Study D)

**Status:** design + safe scaffolding only. No live API credentials assumed.
Do **not** invent WebSocket payloads.

## Goals

Collect dense Nobitex BTCIRT data sufficient for true fixed-horizon and
event-level microstructure research:

1. Incremental order-book events (add/cancel/modify/trade)
2. Periodic full snapshots (resync)
3. Trade prints with exchange timestamps
4. Local receive timestamps and sequence numbers
5. Heartbeats, reconnect logs, dropped-message detection
6. Daily Parquet partitions + data-quality monitors

## Recommended schema (logical)

| Field | Description |
|-------|-------------|
| `exchange_ts` | Exchange event time (UTC) |
| `local_recv_ts` | Local receive time (UTC) |
| `seq` | Monotonic sequence / update id |
| `msg_type` | snapshot / delta / trade / heartbeat / gap |
| `symbol` | BTCIRT |
| `bids` / `asks` | Level arrays or delta ops |
| `trade_price`, `trade_qty`, `trade_side` | When applicable |
| `reconnect_id`, `gap_from_seq`, `gap_to_seq` | Continuity |

## Partitioning

```text
data/dense/exchange=nobitex/symbol=BTCIRT/date=YYYY-MM-DD/*.parquet
```

## Quality monitors

- Sequence gaps per day
- Snapshot-vs-delta consistency checks
- Crossed-book rate
- Latency (`local_recv_ts - exchange_ts`) distribution
- Observation rate (events/sec) vs target

## Collector scaffolding

Place optional stubs under `scripts/` / `src/collectors/` only if they:

- Read credentials from environment variables
- Log clearly when credentials are missing
- Never write fabricated market events

## Relation to current project

The present CSV is **sparse snapshots**. Study C underpowered results motivate
this collector for a future redesign with honest 10/30/60s horizons and true OFI.
