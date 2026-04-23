//! `ModelRegistry`: machine-readable metadata about a deployed Bayronik model.
//!
//! Read by the HTTP server (`/version`, `/metrics`) and by the regression
//! tests. Written by the `registry` CLI from the validation report.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Per-split aggregated metrics. Mirrors `validation_report.json`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SplitMetrics {
    pub n: u32,
    pub log_mse: f64,
    pub log_mae: f64,
    pub baseline_log_mse: f64,
    pub mse_improvement_x: f64,
    pub pk_log_mae: f64,
    pub pk_improvement_x: f64,
    pub cross_corr: f64,
    pub suppression_err: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FrozenMetrics {
    pub lh: SplitMetrics,
    pub cv: SplitMetrics,
    pub weights_sha256: String,
    pub env: serde_json::Value,
    pub validation_args: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub model_id: String,
    pub version: String,
    pub architecture: String,
    pub training_dataset: String,
    pub training_steps: Option<u64>,
    pub training_epochs: Option<u32>,
    pub trained_on: String,
    pub trained_at: String,
    pub git_revision: Option<String>,
    pub conditions: Vec<String>,
    pub conditions_units: Option<Vec<String>>,
    pub input_shape: Vec<u32>,
    pub output_shape: Vec<u32>,
    pub weights_pth_sha256: Option<String>,
    pub weights_pt_sha256: Option<String>,
    pub paper_reference: Option<String>,
    pub license: String,
    pub frozen_metrics: FrozenMetrics,
    pub limitations: Vec<String>,
    pub citation: Option<String>,
}

impl ModelRegistry {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        let parsed: ModelRegistry = serde_json::from_str(&data)
            .with_context(|| format!("parse {}", path.display()))?;
        Ok(parsed)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json).with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    pub fn placeholder() -> Self {
        Self {
            model_id: "ufno-cond-illustristng".into(),
            version: "0.0.0-dev".into(),
            architecture: "UFNO2dConditional".into(),
            training_dataset: "CAMELS-LH/IllustrisTNG (z=0)".into(),
            training_steps: None,
            training_epochs: None,
            trained_on: "unknown".into(),
            trained_at: "unknown".into(),
            git_revision: None,
            conditions: default_conditions(),
            conditions_units: Some(default_conditions_units()),
            input_shape: vec![1, 256, 256],
            output_shape: vec![1, 256, 256],
            weights_pth_sha256: None,
            weights_pt_sha256: None,
            paper_reference: None,
            license: "MIT".into(),
            frozen_metrics: FrozenMetrics::default(),
            limitations: default_limitations(),
            citation: None,
        }
    }
}

pub fn default_conditions() -> Vec<String> {
    vec![
        "Omega_m".into(),
        "sigma_8".into(),
        "A_SN1".into(),
        "A_AGN1".into(),
        "A_SN2".into(),
        "A_AGN2".into(),
    ]
}

pub fn default_conditions_units() -> Vec<String> {
    vec![
        "dimensionless".into(),
        "dimensionless".into(),
        "× CAMELS fiducial".into(),
        "× CAMELS fiducial".into(),
        "× CAMELS fiducial".into(),
        "× CAMELS fiducial".into(),
    ]
}

pub fn default_limitations() -> Vec<String> {
    vec![
        "trained only on IllustrisTNG sub-grid; SIMBA / TNG-extreme require retraining".into(),
        "z=0 only; no redshift conditioning".into(),
        "fixed 256x256 resolution; super-resolution is not yet supported".into(),
        "conditioning ranges respected only inside the LH cube".into(),
    ]
}
