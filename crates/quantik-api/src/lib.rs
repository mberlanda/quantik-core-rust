use std::time::Instant;

use axum::{
    extract::Path,
    http::{header, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use quantik_core::{
    beam_search::{BeamSearchConfig, BeamSearchEngine},
    mcts::{MCTSConfig, MCTSEngine},
    minimax::{MinimaxConfig, MinimaxEngine},
    moves::{generate_legal_moves, Move},
    state::State,
};
use serde::{Deserialize, Serialize};
use tower_http::{cors::CorsLayer, trace::TraceLayer};

pub const API_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const REQUEST_SCHEMA: &str = "quantik.engine-request.v1";
pub const RESPONSE_SCHEMA: &str = "quantik.engine-response.v1";

pub fn app() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/engines", get(list_engines))
        .route("/v1/move/{engine}", post(choose_move))
        .layer(
            CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([header::CONTENT_TYPE]),
        )
        .layer(TraceLayer::new_for_http())
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
    version: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "quantik-api",
        version: API_VERSION,
    })
}

#[derive(Debug, Serialize)]
struct EngineDescriptor {
    kind: &'static str,
    version: &'static str,
}

async fn list_engines() -> Json<Vec<EngineDescriptor>> {
    Json(
        ["minimax", "mcts", "beam"]
            .into_iter()
            .map(|kind| EngineDescriptor {
                kind,
                version: quantik_core::constants::PACKAGE_VERSION,
            })
            .collect(),
    )
}

#[derive(Clone, Debug, Deserialize)]
pub struct MoveRequest {
    pub schema: String,
    pub qfen: String,
    pub side_to_move: u8,
    pub legal_action_indices: Vec<u8>,
    #[serde(default)]
    pub config: SearchConfig,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct SearchConfig {
    pub max_depth: Option<u32>,
    pub time_limit_ms: Option<u64>,
    pub iterations: Option<u32>,
    pub beam_width: Option<usize>,
    pub rollouts: Option<u32>,
    pub seed: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct MoveResponse {
    pub schema: &'static str,
    pub action_index: u8,
    pub engine_kind: String,
    pub engine_version: &'static str,
    pub elapsed_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
}

async fn choose_move(
    Path(engine): Path<String>,
    Json(request): Json<MoveRequest>,
) -> Result<Json<MoveResponse>, ApiError> {
    validate_request(&request)?;
    let engine_for_search = engine.clone();
    let result = tokio::task::spawn_blocking(move || run_search(&engine_for_search, request))
        .await
        .map_err(|error| ApiError::internal(format!("engine task failed: {error}")))??;
    Ok(Json(result))
}

fn validate_request(request: &MoveRequest) -> Result<(), ApiError> {
    if request.schema != REQUEST_SCHEMA {
        return Err(ApiError::bad_request(format!(
            "schema must be {REQUEST_SCHEMA}"
        )));
    }
    if request.side_to_move > 1 {
        return Err(ApiError::bad_request("side_to_move must be 0 or 1"));
    }
    if request.legal_action_indices.iter().any(|index| *index > 63) {
        return Err(ApiError::bad_request(
            "legal_action_indices must contain values from 0 through 63",
        ));
    }
    Ok(())
}

fn run_search(engine: &str, request: MoveRequest) -> Result<MoveResponse, ApiError> {
    let state = State::from_qfen(&request.qfen).map_err(ApiError::bad_request)?;
    let legal_moves = generate_legal_moves(&state.bb);
    let current_player = legal_moves
        .first()
        .map(|mv| mv.player)
        .ok_or_else(|| ApiError::unprocessable("position is terminal or has no legal moves"))?;
    if current_player != request.side_to_move {
        return Err(ApiError::unprocessable(format!(
            "side_to_move is {}, but core calculated {current_player}",
            request.side_to_move
        )));
    }

    let core_actions: Vec<u8> = legal_moves.iter().map(action_index).collect();
    let mut requested_actions = request.legal_action_indices.clone();
    requested_actions.sort_unstable();
    requested_actions.dedup();
    if core_actions != requested_actions {
        return Err(ApiError::unprocessable(
            "legal_action_indices do not exactly match quantik-core",
        ));
    }

    let started = Instant::now();
    let (best_move, value) = match engine {
        "minimax" => search_minimax(&state, &request.config)?,
        "mcts" => search_mcts(&state, &request.config)?,
        "beam" => search_beam(&state, &request.config)?,
        _ => return Err(ApiError::not_found(format!("unknown engine {engine:?}"))),
    };
    let selected_action = action_index(&best_move);
    if !core_actions.contains(&selected_action) {
        return Err(ApiError::internal(
            "engine returned an action that quantik-core considers illegal",
        ));
    }

    Ok(MoveResponse {
        schema: RESPONSE_SCHEMA,
        action_index: selected_action,
        engine_kind: engine.to_owned(),
        engine_version: quantik_core::constants::PACKAGE_VERSION,
        elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        value,
    })
}

fn search_minimax(state: &State, input: &SearchConfig) -> Result<(Move, Option<f64>), ApiError> {
    let config = MinimaxConfig {
        max_depth: input.max_depth.unwrap_or(6).clamp(1, 16),
        time_limit_s: seconds(input.time_limit_ms),
        random_seed: input.seed,
        ..MinimaxConfig::default()
    };
    let result = MinimaxEngine::new(config)
        .search(state)
        .map_err(ApiError::unprocessable)?;
    Ok((result.best_move, None))
}

fn search_mcts(state: &State, input: &SearchConfig) -> Result<(Move, Option<f64>), ApiError> {
    let config = MCTSConfig {
        max_iterations: input.iterations.unwrap_or(1_500).clamp(1, 1_000_000),
        max_depth: input.max_depth.unwrap_or(16).clamp(1, 16),
        time_limit_s: seconds(input.time_limit_ms),
        seed: input.seed,
        ..MCTSConfig::default()
    };
    MCTSEngine::new(config)
        .search(&state.bb)
        .map(|(mv, win_probability)| (mv, Some(2.0 * win_probability - 1.0)))
        .ok_or_else(|| ApiError::unprocessable("MCTS found no move"))
}

fn search_beam(state: &State, input: &SearchConfig) -> Result<(Move, Option<f64>), ApiError> {
    let config = BeamSearchConfig {
        beam_width: input.beam_width.unwrap_or(64).clamp(1, 100_000),
        max_depth: input.max_depth.unwrap_or(8).clamp(1, 16),
        rollouts_per_candidate: input.rollouts.unwrap_or(8).clamp(1, 100_000),
        random_seed: input.seed,
        time_limit_s: seconds(input.time_limit_ms),
        ..BeamSearchConfig::default()
    };
    let result = BeamSearchEngine::new(config)
        .map_err(ApiError::bad_request)?
        .search(state)
        .map_err(ApiError::unprocessable)?;
    if let Some(leaf) = result
        .best_leaf
        .as_ref()
        .filter(|leaf| !leaf.moves.is_empty())
    {
        let root_value = if result.root_player == 0 {
            leaf.value
        } else {
            -leaf.value
        };
        return Ok((leaf.moves[0], Some(root_value)));
    }
    result
        .ranked_root_moves(Some(1))
        .first()
        .map(|ranked| (ranked.mv, Some(ranked.best_value)))
        .ok_or_else(|| ApiError::unprocessable("beam search found no move"))
}

fn seconds(milliseconds: Option<u64>) -> Option<f64> {
    milliseconds.map(|value| value.clamp(1, 300_000) as f64 / 1_000.0)
}

fn action_index(mv: &Move) -> u8 {
    mv.shape * 16 + mv.position
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }

    fn unprocessable(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, message)
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, message)
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }

    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorBody {
                error: self.message,
            }),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    async fn json_response(response: Response) -> Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn health_identifies_the_service() {
        let response = app()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(json_response(response).await["service"], "quantik-api");
    }

    #[tokio::test]
    async fn minimax_returns_a_legal_portable_action() {
        let request = json!({
            "schema": REQUEST_SCHEMA,
            "qfen": "AbC./..../..../....",
            "side_to_move": 1,
            "legal_action_indices": generate_legal_moves(&State::from_qfen("AbC./..../..../....").unwrap().bb)
                .iter().map(action_index).collect::<Vec<_>>(),
            "config": { "max_depth": 2, "seed": 7 }
        });
        let response = app()
            .oneshot(
                Request::post("/v1/move/minimax")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_response(response).await;
        assert_eq!(body["schema"], RESPONSE_SCHEMA);
        assert_eq!(body["action_index"], 51);
    }

    #[tokio::test]
    async fn request_legality_must_match_core_exactly() {
        let request = json!({
            "schema": REQUEST_SCHEMA,
            "qfen": "..../..../..../....",
            "side_to_move": 0,
            "legal_action_indices": [0]
        });
        let response = app()
            .oneshot(
                Request::post("/v1/move/mcts")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }
}
