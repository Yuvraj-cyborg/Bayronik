use anyhow::{Context, Result};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path as AxPath, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

mod dataset;
mod inference;

use registry::ModelRegistry;
use dataset::MapsDataset;
use inference::Model;

#[derive(Parser, Debug)]
#[command(version, about = "Pure-Rust inference server for the Bayronik baryonic emulator")]
struct Args {
    #[arg(long, default_value = "0.0.0.0:8000")]
    bind: SocketAddr,
    #[arg(
        long,
        default_value = "model/weights/traced_ufno_cond_LH_IllustrisTNG.pt"
    )]
    weights: PathBuf,
    #[arg(long, default_value = "model/weights/model_registry.json")]
    registry: PathBuf,
    #[arg(long, default_value = "model/data")]
    data_dir: PathBuf,
    #[arg(
        long,
        default_value = "model/data/params_LH_IllustrisTNG.txt"
    )]
    params_path: PathBuf,
}

#[derive(Clone)]
struct AppState {
    model: Arc<Model>,
    registry: Arc<ModelRegistry>,
    started_at: Instant,
    dataset: Option<Arc<DatasetBundle>>,
}

struct DatasetBundle {
    name: &'static str,
    dm: MapsDataset,
    mtot: Option<MapsDataset>,
    params: Option<Vec<[f32; 6]>>,
    maps_per_sim: usize,
}

#[derive(Deserialize)]
struct InferRequest {
    input_map: Vec<Vec<f32>>,
    #[serde(default = "default_omega_m")]
    omega_m: f32,
    #[serde(default = "default_sigma_8")]
    sigma_8: f32,
    #[serde(default = "default_one")]
    a_sn1: f32,
    #[serde(default = "default_one")]
    a_agn1: f32,
    #[serde(default = "default_one")]
    a_sn2: f32,
    #[serde(default = "default_one")]
    a_agn2: f32,
}

fn default_omega_m() -> f32 {
    0.3
}
fn default_sigma_8() -> f32 {
    0.8
}
fn default_one() -> f32 {
    1.0
}

#[derive(Serialize)]
struct InferResponse {
    output_map: Vec<Vec<f32>>,
    input_shape: [usize; 2],
    output_shape: [usize; 2],
}

#[derive(Serialize)]
struct VersionResponse<'a> {
    name: &'a str,
    backend: &'a str,
    weights_sha256: &'a str,
    weights_path: String,
    device: String,
    uptime_secs: u64,
    registry: &'a ModelRegistry,
}

#[derive(Serialize)]
struct DatasetInfo<'a> {
    dataset_type: &'a str,
    n_samples: usize,
    resolution: usize,
    has_ground_truth: bool,
}

#[derive(Serialize)]
struct SampleResponse {
    input_map: Vec<Vec<f32>>,
    ground_truth: Vec<Vec<f32>>,
    params: [f32; 6],
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=info".into()),
        )
        .init();

    let args = Args::parse();

    tracing::info!("loading model: {}", args.weights.display());
    let model = Model::load(&args.weights)?;

    let registry = match ModelRegistry::load(&args.registry) {
        Ok(r) => {
            tracing::info!(version = %r.version, "registry loaded");
            r
        }
        Err(err) => {
            tracing::warn!(?err, "registry not found, using placeholder");
            ModelRegistry::placeholder()
        }
    };

    let dataset = build_dataset_bundle(&args.data_dir, &args.params_path).await?;

    let state = AppState {
        model,
        registry: Arc::new(registry),
        started_at: Instant::now(),
        dataset: dataset.map(Arc::new),
    };

    let cors = CorsLayer::new()
        .allow_methods(Any)
        .allow_origin(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health))
        .route("/version", get(version))
        .route("/metrics", get(metrics))
        .route("/dataset/info", get(dataset_info))
        .route("/sample/{idx}", get(sample))
        .route("/infer", post(infer))
        .route("/infer_npy", post(infer_npy))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    tracing::info!(addr = %args.bind, "Bayronik inference server listening");
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install ctrl-c handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("shutting down");
}

async fn build_dataset_bundle(
    data_dir: &PathBuf,
    params_path: &PathBuf,
) -> Result<Option<DatasetBundle>> {
    for (name, dm_name, mt_name) in [
        (
            "LH",
            "Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy",
            "Maps_Mtot_IllustrisTNG_LH_z=0.00.npy",
        ),
        (
            "CV",
            "Maps_Mcdm_IllustrisTNG_CV_z=0.00.npy",
            "Maps_Mtot_IllustrisTNG_CV_z=0.00.npy",
        ),
    ] {
        let dm_path = data_dir.join(dm_name);
        if dm_path.exists() {
            let dm = MapsDataset::open(&dm_path)?;
            let mtot = MapsDataset::open(data_dir.join(mt_name)).ok();
            let params = if params_path.exists() {
                Some(parse_params_file(params_path)?)
            } else {
                None
            };
            let maps_per_sim = match params.as_ref() {
                Some(p) if !p.is_empty() => (dm.n_samples / p.len()).max(1),
                _ => 1,
            };
            tracing::info!(
                dataset = name,
                n = dm.n_samples,
                params = params.as_ref().map(|p| p.len()).unwrap_or(0),
                "dataset bundle"
            );
            return Ok(Some(DatasetBundle {
                name,
                dm,
                mtot,
                params,
                maps_per_sim,
            }));
        }
    }
    tracing::warn!("no CAMELS data found; /sample endpoints disabled");
    Ok(None)
}

fn parse_params_file(path: &PathBuf) -> Result<Vec<[f32; 6]>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    let mut rows = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if parts.len() >= 6 {
            rows.push([parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]]);
        }
    }
    Ok(rows)
}

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "backend": "rust-axum-tch",
        "model_loaded": true,
        "weights_sha256": state.model.weights_sha(),
    }))
}

async fn version(State(state): State<AppState>) -> Json<serde_json::Value> {
    let resp = VersionResponse {
        name: "bayronik",
        backend: "rust-axum-tch",
        weights_sha256: state.model.weights_sha(),
        weights_path: state.model.weights_path().display().to_string(),
        device: format!("{:?}", state.model.device()),
        uptime_secs: state.started_at.elapsed().as_secs(),
        registry: &state.registry,
    };
    Json(serde_json::to_value(&resp).unwrap_or(serde_json::Value::Null))
}

async fn metrics(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(
        serde_json::to_value(&state.registry.frozen_metrics)
            .unwrap_or(serde_json::Value::Null),
    )
}

async fn dataset_info(State(state): State<AppState>) -> Result<Json<serde_json::Value>, ApiError> {
    let bundle = state
        .dataset
        .as_ref()
        .ok_or_else(|| ApiError::not_found("no CAMELS data available"))?;
    Ok(Json(serde_json::to_value(DatasetInfo {
        dataset_type: bundle.name,
        n_samples: bundle.dm.n_samples,
        resolution: bundle.dm.resolution,
        has_ground_truth: bundle.mtot.is_some(),
    })?))
}

async fn sample(
    State(state): State<AppState>,
    AxPath(idx): AxPath<usize>,
) -> Result<Json<SampleResponse>, ApiError> {
    let bundle = state
        .dataset
        .as_ref()
        .ok_or_else(|| ApiError::not_found("no CAMELS data available"))?;
    if idx >= bundle.dm.n_samples {
        return Err(ApiError::bad_request(format!(
            "index {idx} out of range [0,{})",
            bundle.dm.n_samples
        )));
    }
    let dm = bundle
        .dm
        .get(idx)
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let gt = match &bundle.mtot {
        Some(m) => m.get(idx).map_err(|e| ApiError::internal(e.to_string()))?,
        None => vec![0.0; bundle.dm.resolution * bundle.dm.resolution],
    };
    let params = match &bundle.params {
        Some(p) if !p.is_empty() => {
            let sim_idx = (idx / bundle.maps_per_sim).min(p.len() - 1);
            p[sim_idx]
        }
        _ => [0.3, 0.8, 1.0, 1.0, 1.0, 1.0],
    };
    Ok(Json(SampleResponse {
        input_map: rows(&dm, bundle.dm.resolution),
        ground_truth: rows(&gt, bundle.dm.resolution),
        params,
    }))
}

async fn infer(
    State(state): State<AppState>,
    Json(req): Json<InferRequest>,
) -> Result<Json<InferResponse>, ApiError> {
    let h = req.input_map.len();
    if h != 256 {
        return Err(ApiError::bad_request(format!(
            "expected 256 rows, got {h}"
        )));
    }
    let mut flat = Vec::with_capacity(256 * 256);
    for row in &req.input_map {
        if row.len() != 256 {
            return Err(ApiError::bad_request(format!(
                "expected 256 cols, got {}",
                row.len()
            )));
        }
        flat.extend_from_slice(row);
    }
    let cond = [
        req.omega_m,
        req.sigma_8,
        req.a_sn1,
        req.a_agn1,
        req.a_sn2,
        req.a_agn2,
    ];
    let out = state
        .model
        .infer_single(flat, cond)
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(InferResponse {
        output_map: rows(&out, 256),
        input_shape: [256, 256],
        output_shape: [256, 256],
    }))
}

async fn infer_npy(State(state): State<AppState>, body: Bytes) -> Result<Response, ApiError> {
    let arr = parse_npy_payload(&body).map_err(|e| ApiError::bad_request(e.to_string()))?;
    if arr.shape != [256, 256] && arr.shape != [1, 256, 256] {
        return Err(ApiError::bad_request(format!(
            "expected shape (256,256) or (1,256,256), got {:?}",
            arr.shape
        )));
    }
    let out = state
        .model
        .infer_single(arr.data, [0.3, 0.8, 1.0, 1.0, 1.0, 1.0])
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?;
    let bytes = encode_npy(&out, &[256, 256]);
    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "application/octet-stream"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=output.npy",
            ),
        ],
        bytes,
    )
        .into_response())
}

fn rows(flat: &[f32], n: usize) -> Vec<Vec<f32>> {
    flat.chunks(n).map(|r| r.to_vec()).collect()
}

struct ParsedNpy {
    data: Vec<f32>,
    shape: Vec<usize>,
}

fn parse_npy_payload(bytes: &[u8]) -> Result<ParsedNpy> {
    let reader = npyz::NpyFile::new(bytes).context("parse npy")?;
    let shape = reader.shape().iter().map(|&n| n as usize).collect::<Vec<_>>();
    let data: Vec<f32> = reader
        .into_vec::<f32>()
        .context("npy must contain float32 data")?;
    Ok(ParsedNpy { data, shape })
}

fn encode_npy(data: &[f32], shape: &[usize]) -> Vec<u8> {
    let shape_str = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
        shape_str
    );
    // Pad header so total len (10 + header_len) % 64 == 0 and ends with '\n'.
    let total_pre_header = 10 + header.len() + 1;
    let pad = (64 - (total_pre_header % 64)) % 64;
    let mut header = header;
    header.push_str(&" ".repeat(pad));
    header.push('\n');

    let mut out = Vec::with_capacity(10 + header.len() + data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    let header_len = header.len() as u16;
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn not_found(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: msg.into(),
        }
    }
    fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }
    fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(serde_json::json!({"detail": self.message}))).into_response()
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(value: serde_json::Error) -> Self {
        Self::internal(value.to_string())
    }
}
