# quantik-api

An Axum HTTP gateway around the engines in the sibling `quantik-core` crate.
It is an independently runnable workspace package and is not published to
crates.io.

## Run

```sh
cargo run --release -p quantik-api
```

The server listens on `127.0.0.1:8000` by default. Override it with:

```sh
QUANTIK_API_ADDR=0.0.0.0:9000 cargo run --release -p quantik-api
```

Set `RUST_LOG=quantik_api=debug,tower_http=debug` for more request logging.

## Endpoints

- `GET /health`
- `GET /v1/engines`
- `POST /v1/move/minimax`
- `POST /v1/move/mcts`
- `POST /v1/move/beam`

The move endpoints accept the visualizer's portable request:

```json
{
  "schema": "quantik.engine-request.v1",
  "qfen": "AbC./..../..../....",
  "side_to_move": 1,
  "legal_action_indices": [6, 7, 9, 10, 11, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 36, 37, 40, 41, 43, 44, 45, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
  "config": {
    "max_depth": 6,
    "time_limit_ms": 1000,
    "iterations": 1500,
    "beam_width": 64,
    "rollouts": 8,
    "seed": 1
  }
}
```

When present, `value` is normalized to `[-1, 1]` from the root player's
perspective. Minimax currently omits it rather than exposing its heuristic score
under incompatible semantics.

All configuration fields are optional. Fields irrelevant to the selected
engine are ignored. The server recalculates legality with `quantik-core` and
rejects a request if its legal-action set differs.

Responses use the shared action-index convention `shape * 16 + position`:

```json
{
  "schema": "quantik.engine-response.v1",
  "action_index": 51,
  "engine_kind": "minimax",
  "engine_version": "1.1.0",
  "elapsed_ms": 3
}
```

For local visualizer use, serve `quantik-qfen-visualizer` over HTTP and enter
one of the move endpoint URLs in a player's remote endpoint field. Development
CORS currently allows any origin; production deployments should restrict it at
the proxy or replace the permissive layer with an explicit origin list.
