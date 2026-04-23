//! Regression suite. Gates `weights/model_registry.json` and the latest
//! `validation_report.json` on the frozen scientific bars from the release plan.

use registry::ModelRegistry;
use registry::report::ValidationReport;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace parent")
        .to_path_buf()
}

fn registry_path() -> PathBuf {
    workspace_root().join("model/weights/model_registry.json")
}

fn report_path() -> PathBuf {
    workspace_root().join("model/reports/validation_report.json")
}

fn must_load_registry() -> ModelRegistry {
    let path = registry_path();
    assert!(
        path.exists(),
        "model_registry.json missing at {} — run `make registry` first",
        path.display()
    );
    ModelRegistry::load(&path).expect("registry parses")
}

fn must_load_report() -> ValidationReport {
    let path = report_path();
    assert!(
        path.exists(),
        "validation_report.json missing at {} — run `make validation` first",
        path.display()
    );
    ValidationReport::load(&path).expect("report parses")
}

// Frozen scientific quality bars. These are intentionally tighter than the
// CAMELS LH→CV cross-validation noise floor so the suite catches regressions
// without drifting on rerun. Update only with reviewed evidence.
const LH_LOG_MSE_MAX: f64 = 0.010;
const CV_LOG_MSE_MAX: f64 = 0.010;
const LH_R_MIN: f64 = 0.99;
const CV_R_MIN: f64 = 0.99;
const MSE_IMPROVEMENT_MIN: f64 = 5.0;
const PK_IMPROVEMENT_MIN: f64 = 2.0;
const PDF_L1_P99_MAX: f64 = 0.10;

#[test]
fn lh_log_mse_below_threshold() {
    let r = must_load_registry();
    assert!(
        r.frozen_metrics.lh.log_mse <= LH_LOG_MSE_MAX,
        "LH log MSE = {} exceeds {}",
        r.frozen_metrics.lh.log_mse,
        LH_LOG_MSE_MAX
    );
}

#[test]
fn cv_log_mse_below_threshold() {
    let r = must_load_registry();
    assert!(
        r.frozen_metrics.cv.log_mse <= CV_LOG_MSE_MAX,
        "CV log MSE = {} exceeds {}",
        r.frozen_metrics.cv.log_mse,
        CV_LOG_MSE_MAX
    );
}

#[test]
fn cross_correlation_above_threshold() {
    let r = must_load_registry();
    assert!(
        r.frozen_metrics.lh.cross_corr >= LH_R_MIN,
        "LH r(k) = {} below {}",
        r.frozen_metrics.lh.cross_corr,
        LH_R_MIN
    );
    assert!(
        r.frozen_metrics.cv.cross_corr >= CV_R_MIN,
        "CV r(k) = {} below {}",
        r.frozen_metrics.cv.cross_corr,
        CV_R_MIN
    );
}

#[test]
fn improvement_over_baseline() {
    let r = must_load_registry();
    assert!(
        r.frozen_metrics.lh.mse_improvement_x >= MSE_IMPROVEMENT_MIN,
        "LH MSE improvement {}x below {}x",
        r.frozen_metrics.lh.mse_improvement_x,
        MSE_IMPROVEMENT_MIN
    );
    assert!(
        r.frozen_metrics.cv.mse_improvement_x >= MSE_IMPROVEMENT_MIN,
        "CV MSE improvement {}x below {}x",
        r.frozen_metrics.cv.mse_improvement_x,
        MSE_IMPROVEMENT_MIN
    );
    assert!(
        r.frozen_metrics.lh.pk_improvement_x >= PK_IMPROVEMENT_MIN,
        "LH P(k) improvement {}x below {}x",
        r.frozen_metrics.lh.pk_improvement_x,
        PK_IMPROVEMENT_MIN
    );
    assert!(
        r.frozen_metrics.cv.pk_improvement_x >= PK_IMPROVEMENT_MIN,
        "CV P(k) improvement {}x below {}x",
        r.frozen_metrics.cv.pk_improvement_x,
        PK_IMPROVEMENT_MIN
    );
}

#[test]
fn registry_matches_latest_report() {
    let r = must_load_registry();
    let report = must_load_report();
    assert_eq!(
        r.frozen_metrics.weights_sha256, report.weights_sha256,
        "registry weights_sha256 has drifted from validation_report.json — \
         rerun `make registry` after `make validation`"
    );
    let lh = report.split("LH").expect("LH split");
    let cv = report.split("CV").expect("CV split");
    assert!((r.frozen_metrics.lh.log_mse - lh.mean_log_mse).abs() < 1e-9);
    assert!((r.frozen_metrics.cv.log_mse - cv.mean_log_mse).abs() < 1e-9);
    assert!((r.frozen_metrics.lh.cross_corr - lh.mean_cross_corr).abs() < 1e-9);
    assert!((r.frozen_metrics.cv.cross_corr - cv.mean_cross_corr).abs() < 1e-9);
}

#[test]
fn registry_schema_complete() {
    let r = must_load_registry();
    assert!(!r.model_id.is_empty(), "model_id");
    assert!(!r.version.is_empty(), "version");
    assert_eq!(r.input_shape, vec![1, 256, 256]);
    assert_eq!(r.output_shape, vec![1, 256, 256]);
    assert_eq!(r.conditions.len(), 6);
    assert!(!r.frozen_metrics.weights_sha256.is_empty());
    assert!(
        r.weights_pth_sha256.is_some() || r.weights_pt_sha256.is_some(),
        "registry should record at least one weights checksum"
    );
    assert!(
        !r.limitations.is_empty(),
        "limitations must be enumerated for a science-facing model card"
    );
}

#[test]
fn pdf_distribution_close_enough() {
    let report = must_load_report();
    for split in &report.splits {
        assert!(
            split.p99_pdf_l1 <= PDF_L1_P99_MAX,
            "{}: 99th-percentile PDF L1 = {} exceeds {}",
            split.split,
            split.p99_pdf_l1,
            PDF_L1_P99_MAX
        );
    }
}

#[test]
fn weights_file_checksum_matches_registry() {
    let r = must_load_registry();
    let candidates = [
        "model/weights/best_ufno_cond_LH_IllustrisTNG.pth",
        "model/weights/model.pth",
    ];
    let pth = candidates
        .iter()
        .map(|p| workspace_root().join(p))
        .find(|p| p.exists());
    let Some(pth) = pth else {
        // No local weights checked into the workspace — fine for clean
        // checkouts. The training pipeline asserts this server-side.
        return;
    };
    let actual = registry::sha::sha256_file(&pth).expect("hash weights");
    assert_eq!(
        actual, r.frozen_metrics.weights_sha256,
        "weights file at {} no longer matches the SHA256 frozen in the registry",
        Path::display(&pth)
    );
}
