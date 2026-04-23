//! Parser for `bayronik-model/reports/validation_report.json`.
//!
//! Only the fields used by the registry builder and the regression tests are
//! pulled out — the rest is preserved as opaque JSON so we don't break when
//! the Python pipeline grows new metrics.

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationReport {
    pub weights_sha256: String,
    #[serde(default)]
    pub env: serde_json::Value,
    #[serde(default)]
    pub args: serde_json::Value,
    pub splits: Vec<SplitReport>,
    #[serde(default)]
    pub parameter_sensitivity: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplitReport {
    pub split: String,
    pub samples: u32,
    #[serde(default)]
    pub mean_infer_ms: f64,
    pub mean_log_mse: f64,
    pub mean_baseline_log_mse: f64,
    pub mse_improvement_x: f64,
    pub mean_log_mae: f64,
    pub mean_pk_log_mae: f64,
    #[serde(default)]
    pub mean_baseline_pk_log_mae: f64,
    pub pk_improvement_x: f64,
    pub mean_cross_corr: f64,
    #[serde(default)]
    pub p99_pdf_l1: f64,
    pub mean_suppression_err: f64,
}

impl ValidationReport {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        let parsed: ValidationReport = serde_json::from_str(&data)
            .with_context(|| format!("parse {}", path.display()))?;
        Ok(parsed)
    }

    pub fn split(&self, name: &str) -> Result<&SplitReport> {
        self.splits
            .iter()
            .find(|s| s.split.eq_ignore_ascii_case(name))
            .ok_or_else(|| anyhow!("validation report has no '{name}' split"))
    }
}
