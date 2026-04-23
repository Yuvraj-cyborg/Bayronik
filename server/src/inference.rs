use anyhow::{Context, Result, anyhow};
use registry::sha::sha256_file;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tch::{CModule, Device, Kind, Tensor};
use tokio::sync::Mutex;

/// Wraps a TorchScript model loaded once and reused across requests.
///
/// `tch::CModule` is `Send + Sync`-incompatible due to the underlying torch
/// state. We protect it behind a `Mutex` and run inference inside
/// `spawn_blocking`, since libtorch will saturate threads anyway.
pub struct Model {
    inner: Mutex<CModule>,
    device: Device,
    weights_path: PathBuf,
    weights_sha: String,
}

impl Model {
    pub fn load(weights_path: impl AsRef<Path>) -> Result<Arc<Self>> {
        let path = weights_path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(anyhow!("model weights not found: {}", path.display()));
        }
        let device = Device::cuda_if_available();
        let mut module = CModule::load_on_device(&path, device)
            .with_context(|| format!("loading torchscript model: {}", path.display()))?;
        module.set_eval();

        let weights_sha = sha256_file(&path)?;
        tracing::info!(
            device = ?device,
            sha256 = %&weights_sha[..16],
            path = %path.display(),
            "loaded model"
        );
        Ok(Arc::new(Self {
            inner: Mutex::new(module),
            device,
            weights_path: path,
            weights_sha,
        }))
    }

    pub fn weights_sha(&self) -> &str {
        &self.weights_sha
    }

    pub fn weights_path(&self) -> &Path {
        &self.weights_path
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Run a single map (256×256, raw mass density) through the model.
    /// Applies `log1p` → forward → `expm1` so outputs are in raw mass
    /// units, matching the Python training pipeline.
    pub async fn infer_single(self: &Arc<Self>, input: Vec<f32>, conditions: [f32; 6]) -> Result<Vec<f32>> {
        let model = Arc::clone(self);
        tokio::task::spawn_blocking(move || model.infer_blocking(&input, conditions))
            .await
            .context("inference task panicked")?
    }

    fn infer_blocking(&self, input: &[f32], conditions: [f32; 6]) -> Result<Vec<f32>> {
        if input.len() != 256 * 256 {
            return Err(anyhow!(
                "expected 256x256 = 65536 floats, got {}",
                input.len()
            ));
        }

        let log_input: Vec<f32> = input.iter().map(|v| (v.max(0.0) + 1.0).ln()).collect();
        let x = Tensor::from_slice(&log_input)
            .reshape([1, 1, 256, 256])
            .to_kind(Kind::Float)
            .to_device(self.device);
        let cond = Tensor::from_slice(&conditions)
            .reshape([1, 6])
            .to_kind(Kind::Float)
            .to_device(self.device);

        let module = self.inner.blocking_lock();
        let out = module
            .forward_ts(&[x, cond])
            .context("model forward failed")?;
        drop(module);

        let mut buf: Vec<f32> = vec![0.0; 256 * 256];
        let len = buf.len();
        out.to_kind(Kind::Float)
            .to_device(Device::Cpu)
            .copy_data(&mut buf, len);

        for v in buf.iter_mut() {
            *v = v.exp() - 1.0;
        }
        Ok(buf)
    }
}
