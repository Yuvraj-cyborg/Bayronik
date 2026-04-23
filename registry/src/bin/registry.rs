//! `registry`: rebuild `weights/model_registry.json` from the
//! Python-generated `validation_report.json`. Pure Rust, no libtorch.

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use registry::report::{SplitReport, ValidationReport};
use registry::sha::sha256_file;
use registry::{
    FrozenMetrics, ModelRegistry, SplitMetrics, default_conditions, default_conditions_units,
    default_limitations,
};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Parser, Debug)]
#[command(
    name = "registry",
    about = "Build model_registry.json from validation_report.json"
)]
struct Cli {
    /// Validation report produced by `model/benchmarks/validation.py`.
    #[arg(long, default_value = "model/reports/validation_report.json")]
    report: PathBuf,

    /// Where to write `model_registry.json`.
    #[arg(long, default_value = "model/weights/model_registry.json")]
    out: PathBuf,

    /// Optional .pth weights file to checksum into the registry.
    #[arg(
        long,
        default_value = "model/weights/best_ufno_cond_LH_IllustrisTNG.pth"
    )]
    weights_pth: PathBuf,

    /// Optional TorchScript .pt file to checksum into the registry.
    #[arg(
        long,
        default_value = "model/weights/traced_ufno_cond_LH_IllustrisTNG.pt"
    )]
    weights_pt: PathBuf,

    /// Override the model_id (default: ufno-cond-illustristng).
    #[arg(long)]
    model_id: Option<String>,

    /// Override the version string (default: 1.0.0 or whatever the
    /// previous registry recorded).
    #[arg(long)]
    version: Option<String>,

    /// If true, ignore any existing registry at `--out` and rebuild from
    /// scratch (useful in CI / clean rebuilds).
    #[arg(long, default_value_t = false)]
    fresh: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let report = ValidationReport::load(&cli.report)
        .with_context(|| format!("loading validation report from {}", cli.report.display()))?;
    let lh = split_metrics(report.split("LH")?);
    let cv = split_metrics(report.split("CV")?);

    let pth_sha = optional_sha(&cli.weights_pth)?;
    let pt_sha = optional_sha(&cli.weights_pt)?;

    let prev = if cli.fresh {
        None
    } else {
        ModelRegistry::load(&cli.out).ok()
    };

    let git_rev = git_revision().or_else(|| prev.as_ref().and_then(|p| p.git_revision.clone()));
    let model_id = cli
        .model_id
        .or_else(|| prev.as_ref().map(|p| p.model_id.clone()))
        .unwrap_or_else(|| "ufno-cond-illustristng".to_string());
    let version = cli
        .version
        .or_else(|| prev.as_ref().map(|p| p.version.clone()))
        .unwrap_or_else(|| "1.0.0".to_string());

    let registry = ModelRegistry {
        model_id,
        version,
        architecture: prev
            .as_ref()
            .map(|p| p.architecture.clone())
            .unwrap_or_else(|| "UFNO2dConditional".into()),
        training_dataset: prev
            .as_ref()
            .map(|p| p.training_dataset.clone())
            .unwrap_or_else(|| "CAMELS-LH/IllustrisTNG (z=0)".into()),
        training_steps: prev.as_ref().and_then(|p| p.training_steps),
        training_epochs: prev.as_ref().and_then(|p| p.training_epochs),
        trained_on: prev.as_ref().map(|p| p.trained_on.clone()).unwrap_or_else(|| {
            report
                .env
                .get("device")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string()
        }),
        trained_at: prev
            .as_ref()
            .map(|p| p.trained_at.clone())
            .unwrap_or_else(|| chrono::Utc::now().to_rfc3339()),
        git_revision: git_rev,
        conditions: default_conditions(),
        conditions_units: Some(default_conditions_units()),
        input_shape: vec![1, 256, 256],
        output_shape: vec![1, 256, 256],
        weights_pth_sha256: pth_sha,
        weights_pt_sha256: pt_sha,
        paper_reference: prev.as_ref().and_then(|p| p.paper_reference.clone()),
        license: prev
            .as_ref()
            .map(|p| p.license.clone())
            .unwrap_or_else(|| "MIT".into()),
        frozen_metrics: FrozenMetrics {
            lh,
            cv,
            weights_sha256: report.weights_sha256.clone(),
            env: report.env.clone(),
            validation_args: report.args.clone(),
        },
        limitations: default_limitations(),
        citation: prev.as_ref().and_then(|p| p.citation.clone()),
    };

    registry.save(&cli.out)?;
    println!("wrote {}", cli.out.display());
    println!(
        "  LH log_mse={:.5}  r(k)={:.4}  Δχ²(MSE)={:.1}x",
        registry.frozen_metrics.lh.log_mse,
        registry.frozen_metrics.lh.cross_corr,
        registry.frozen_metrics.lh.mse_improvement_x
    );
    println!(
        "  CV log_mse={:.5}  r(k)={:.4}  Δχ²(MSE)={:.1}x",
        registry.frozen_metrics.cv.log_mse,
        registry.frozen_metrics.cv.cross_corr,
        registry.frozen_metrics.cv.mse_improvement_x
    );
    Ok(())
}

fn split_metrics(s: &SplitReport) -> SplitMetrics {
    SplitMetrics {
        n: s.samples,
        log_mse: s.mean_log_mse,
        log_mae: s.mean_log_mae,
        baseline_log_mse: s.mean_baseline_log_mse,
        mse_improvement_x: s.mse_improvement_x,
        pk_log_mae: s.mean_pk_log_mae,
        pk_improvement_x: s.pk_improvement_x,
        cross_corr: s.mean_cross_corr,
        suppression_err: s.mean_suppression_err,
    }
}

fn optional_sha(path: &Path) -> Result<Option<String>> {
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(sha256_file(path).map_err(|e| {
        anyhow!("hashing {}: {e}", path.display())
    })?))
}

fn git_revision() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}
