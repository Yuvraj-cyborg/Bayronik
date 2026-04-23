use egui::{TextureHandle, TextureOptions, Vec2};
use egui_plot::{Line, Plot, PlotPoints, HLine};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

fn line(name: &str, points: Vec<[f64; 2]>) -> Line<'static> {
    Line::new(name.to_string(), PlotPoints::new(points))
}

use crate::analysis;
use crate::visualization::{array_to_colorimage, array_to_colorimage_diverging, compute_statistics, Colormap};

const API_URL: &str = "http://localhost:8000";

// ---- API types ----

#[derive(Serialize)]
struct InferenceRequest {
    input_map: Vec<Vec<f32>>,
    omega_m: f32,
    sigma_8: f32,
    a_sn1: f32,
    a_agn1: f32,
    a_sn2: f32,
    a_agn2: f32,
}

#[derive(Deserialize)]
struct InferenceResponse {
    output_map: Vec<Vec<f32>>,
    #[allow(dead_code)]
    input_shape: Vec<usize>,
    #[allow(dead_code)]
    output_shape: Vec<usize>,
}

#[derive(Deserialize)]
struct HealthResponse {
    #[allow(dead_code)]
    status: String,
    model_loaded: bool,
}

#[derive(Deserialize)]
struct DatasetInfo {
    dataset_type: String,
    n_samples: usize,
    resolution: usize,
}

#[derive(Deserialize)]
struct SampleResponse {
    input_map: Vec<Vec<f32>>,
    ground_truth: Vec<Vec<f32>>,
    params: Vec<f32>,
}

// ---- App state ----

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Camels,
    NBody,
    Sweep,
    About,
}

#[derive(Clone)]
struct MapData {
    flat: Vec<f32>,
    min: f32,
    max: f32,
    mean: f32,
}

impl MapData {
    fn from_flat(data: Vec<f32>) -> Self {
        let (mean, _std, min, max) = compute_statistics(&data);
        Self { flat: data, min, max, mean }
    }
}

pub struct BayronikApp {
    active_tab: Tab,
    resolution: usize,
    colormap: Colormap,
    log_scale: bool,
    status: String,

    // Physics parameters
    omega_m: f32,
    sigma_8: f32,
    a_sn1: f32,
    a_agn1: f32,
    a_sn2: f32,
    a_agn2: f32,

    // Server
    server_connected: bool,
    server_model_loaded: bool,
    inference_pending: bool,
    pending_results: Arc<Mutex<Vec<Result<InferenceResult, String>>>>,

    // CAMELS tab
    camels_info: Option<DatasetInfo>,
    camels_sample_idx: usize,
    camels_input: Option<MapData>,
    camels_output: Option<MapData>,
    camels_gt: Option<MapData>,
    camels_diff: Option<MapData>,
    camels_params: Option<Vec<f32>>,
    use_sample_params: bool,
    pending_sample: Arc<Mutex<Option<Result<SampleResponse, String>>>>,
    sample_loading: bool,
    pending_info: Arc<Mutex<Option<Result<DatasetInfo, String>>>>,

    // N-Body tab
    nbody_grid_res: usize,
    nbody_box_size: f32,
    nbody_steps: usize,
    nbody_seed: u64,
    nbody_input: Option<MapData>,
    nbody_output: Option<MapData>,
    nbody_diff: Option<MapData>,
    nbody_running: bool,

    // Sweep tab
    sweep_param_name: String,
    sweep_steps: usize,
    sweep_results: Vec<(f32, MapData)>,
    sweep_input: Option<MapData>,
    sweep_running: bool,

    // Textures cache
    tex_cache: std::collections::HashMap<String, TextureHandle>,
}

struct InferenceResult {
    output: Vec<f32>,
    tag: String,
}

impl Default for BayronikApp {
    fn default() -> Self {
        Self {
            active_tab: Tab::NBody,
            resolution: 256,
            colormap: Colormap::Inferno,
            log_scale: true,
            status: "Checking server...".into(),

            omega_m: 0.3,
            sigma_8: 0.8,
            a_sn1: 1.0,
            a_agn1: 1.0,
            a_sn2: 1.0,
            a_agn2: 1.0,

            server_connected: false,
            server_model_loaded: false,
            inference_pending: false,
            pending_results: Arc::new(Mutex::new(Vec::new())),

            camels_info: None,
            camels_sample_idx: 0,
            camels_input: None,
            camels_output: None,
            camels_gt: None,
            camels_diff: None,
            camels_params: None,
            use_sample_params: true,
            pending_sample: Arc::new(Mutex::new(None)),
            sample_loading: false,
            pending_info: Arc::new(Mutex::new(None)),

            nbody_grid_res: 32,
            nbody_box_size: 100.0,
            nbody_steps: 10,
            nbody_seed: 42,
            nbody_input: None,
            nbody_output: None,
            nbody_diff: None,
            nbody_running: false,

            sweep_param_name: "a_sn1".into(),
            sweep_steps: 3,
            sweep_results: Vec::new(),
            sweep_input: None,
            sweep_running: false,

            tex_cache: std::collections::HashMap::new(),
        }
    }
}

impl BayronikApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self::default();
        app.check_server_health(cc.egui_ctx.clone());
        app.fetch_dataset_info(cc.egui_ctx.clone());
        app
    }

    fn params_dict(&self) -> [f32; 6] {
        [self.omega_m, self.sigma_8, self.a_sn1, self.a_agn1, self.a_sn2, self.a_agn2]
    }

    fn params_string(&self) -> String {
        format!(
            "Ωm={:.2} σ8={:.2} ASN1={:.2} AAGN1={:.2}",
            self.omega_m, self.sigma_8, self.a_sn1, self.a_agn1
        )
    }

    // ---- Network calls ----

    fn check_server_health(&self, ctx: egui::Context) {
        let url = format!("{}/health", API_URL);
        let pending = self.pending_info.clone();
        ehttp::fetch(ehttp::Request::get(&url), move |result| {
            match result {
                Ok(response) => {
                    if let Ok(h) = serde_json::from_slice::<HealthResponse>(&response.bytes) {
                        log::info!("Server OK, model_loaded={}", h.model_loaded);
                    }
                }
                Err(e) => log::warn!("Server check failed: {e}"),
            }
            let _ = pending; // keep ref alive
            ctx.request_repaint();
        });
    }

    fn fetch_dataset_info(&self, ctx: egui::Context) {
        let url = format!("{}/dataset/info", API_URL);
        let pending = self.pending_info.clone();
        ehttp::fetch(ehttp::Request::get(&url), move |result| {
            let info = match result {
                Ok(resp) if resp.status == 200 => {
                    serde_json::from_slice::<DatasetInfo>(&resp.bytes)
                        .map_err(|e| e.to_string())
                }
                Ok(resp) => Err(format!("HTTP {}", resp.status)),
                Err(e) => Err(e),
            };
            if let Ok(mut guard) = pending.lock() {
                *guard = Some(info);
            }
            ctx.request_repaint();
        });
    }

    fn fetch_sample(&mut self, idx: usize, ctx: egui::Context) {
        self.sample_loading = true;
        let url = format!("{}/sample/{}", API_URL, idx);
        let pending = self.pending_sample.clone();
        ehttp::fetch(ehttp::Request::get(&url), move |result| {
            let sample = match result {
                Ok(resp) if resp.status == 200 => {
                    serde_json::from_slice::<SampleResponse>(&resp.bytes)
                        .map_err(|e| e.to_string())
                }
                Ok(resp) => Err(format!("HTTP {}", resp.status)),
                Err(e) => Err(e),
            };
            if let Ok(mut guard) = pending.lock() {
                *guard = Some(sample);
            }
            ctx.request_repaint();
        });
    }

    fn send_inference(&mut self, input: &[f32], tag: &str, ctx: egui::Context) {
        self.inference_pending = true;
        let n = self.resolution;
        let input_2d: Vec<Vec<f32>> = input.chunks(n).map(|r| r.to_vec()).collect();

        let request = InferenceRequest {
            input_map: input_2d,
            omega_m: self.omega_m,
            sigma_8: self.sigma_8,
            a_sn1: self.a_sn1,
            a_agn1: self.a_agn1,
            a_sn2: self.a_sn2,
            a_agn2: self.a_agn2,
        };

        let url = format!("{}/infer", API_URL);
        let body = serde_json::to_vec(&request).unwrap();
        let pending = self.pending_results.clone();
        let tag = tag.to_string();

        let mut http_req = ehttp::Request::post(&url, body);
        http_req.headers.insert("Content-Type", "application/json");

        ehttp::fetch(http_req, move |result| {
            let output = match result {
                Ok(resp) if resp.status == 200 => {
                    serde_json::from_slice::<InferenceResponse>(&resp.bytes)
                        .map(|r| InferenceResult {
                            output: r.output_map.into_iter().flatten().collect(),
                            tag,
                        })
                        .map_err(|e| e.to_string())
                }
                Ok(resp) => Err(format!("Server error: {}", resp.status)),
                Err(e) => Err(e),
            };
            if let Ok(mut guard) = pending.lock() {
                guard.push(output);
            }
            ctx.request_repaint();
        });
    }

    // ---- Async result polling ----

    fn poll_pending(&mut self) {
        // Dataset info
        if let Ok(mut guard) = self.pending_info.lock() {
            if let Some(result) = guard.take() {
                match result {
                    Ok(info) => {
                        self.server_connected = true;
                        self.camels_info = Some(info);
                    }
                    Err(_) => {}
                }
            }
        }

        // Sample fetch
        if let Ok(mut guard) = self.pending_sample.lock() {
            if let Some(result) = guard.take() {
                self.sample_loading = false;
                match result {
                    Ok(sample) => {
                        let input: Vec<f32> = sample.input_map.into_iter().flatten().collect();
                        let gt: Vec<f32> = sample.ground_truth.into_iter().flatten().collect();
                        self.camels_input = Some(MapData::from_flat(input));
                        self.camels_gt = Some(MapData::from_flat(gt));
                        self.camels_params = Some(sample.params);
                        self.camels_output = None;
                        self.camels_diff = None;
                    }
                    Err(e) => self.status = format!("Sample fetch failed: {e}"),
                }
            }
        }

        // Inference results (drain all queued results)
        let results: Vec<Result<InferenceResult, String>> = {
            if let Ok(mut guard) = self.pending_results.lock() {
                guard.drain(..).collect()
            } else {
                Vec::new()
            }
        };
        for result in results {
            self.inference_pending = false;
            match result {
                Ok(res) => {
                    self.server_connected = true;
                    self.server_model_loaded = true;
                    let output = MapData::from_flat(res.output.clone());

                    match res.tag.as_str() {
                        "camels" => {
                            if let Some(inp) = &self.camels_input {
                                let diff: Vec<f32> = res.output.iter()
                                    .zip(inp.flat.iter())
                                    .map(|(o, i)| o - i)
                                    .collect();
                                self.camels_diff = Some(MapData::from_flat(diff));
                            }
                            self.camels_output = Some(output);
                            self.status = format!("CAMELS inference done [{}]", self.params_string());
                        }
                        "nbody" => {
                            if let Some(inp) = &self.nbody_input {
                                let diff: Vec<f32> = res.output.iter()
                                    .zip(inp.flat.iter())
                                    .map(|(o, i)| o - i)
                                    .collect();
                                self.nbody_diff = Some(MapData::from_flat(diff));
                            }
                            self.nbody_output = Some(output);
                            self.nbody_running = false;
                            self.status = "N-body inference done".into();
                        }
                        tag if tag.starts_with("sweep_") => {
                            self.sweep_results.push((
                                tag.strip_prefix("sweep_").unwrap().parse().unwrap_or(0.0),
                                output,
                            ));
                            if self.sweep_results.len() >= self.sweep_steps {
                                self.sweep_running = false;
                                self.status = "Sweep complete".into();
                            }
                        }
                        _ => {}
                    }
                }
                Err(e) => {
                    self.status = format!("Inference failed: {e}");
                    self.nbody_running = false;
                    self.sweep_running = false;
                }
            }
        }
    }

    // ---- Texture helpers ----

    fn get_or_create_texture(
        &mut self,
        ctx: &egui::Context,
        key: &str,
        data: &MapData,
        colormap: &Colormap,
        log_scale: bool,
        diverging: bool,
    ) -> TextureHandle {
        let cache_key = format!("{}_{}_{}_{:?}_{}_{}", key, data.mean.to_bits(), data.max.to_bits(), colormap, log_scale, diverging);
        if let Some(tex) = self.tex_cache.get(&cache_key) {
            return tex.clone();
        }

        let n = self.resolution;
        let img = if diverging {
            array_to_colorimage_diverging(&data.flat, n, n, colormap)
        } else {
            array_to_colorimage(&data.flat, n, n, colormap, log_scale)
        };
        let opts = TextureOptions {
            magnification: egui::TextureFilter::Linear,
            minification: egui::TextureFilter::Linear,
            ..Default::default()
        };
        let tex = ctx.load_texture(&cache_key, img, opts);
        self.tex_cache.insert(cache_key, tex.clone());
        tex
    }

    fn show_map(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        key: &str,
        title: &str,
        data: &MapData,
        cmap: &Colormap,
        log_scale: bool,
        diverging: bool,
        map_size: Vec2,
    ) {
        ui.vertical(|ui| {
            ui.strong(title);
            let tex = self.get_or_create_texture(ctx, key, data, cmap, log_scale, diverging);
            let sized = egui::load::SizedTexture::new(tex.id(), map_size);
            ui.image(sized);
            let label = if log_scale {
                format!("log₁₀: [{:.1}, {:.1}]",
                    (data.min.max(0.0) + 1.0).log10(),
                    (data.max.max(0.0) + 1.0).log10())
            } else if diverging {
                let abs_max = data.min.abs().max(data.max.abs());
                format!("[{:.2e}, {:.2e}]", -abs_max, abs_max)
            } else {
                format!("[{:.2e}, {:.2e}]", data.min, data.max)
            };
            ui.small(label);
        });
    }

    fn tab_camels(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if self.camels_info.is_none() {
            ui.label("Connecting to server for CAMELS data...");
            ui.label("Start the inference server first: make server");
            if ui.button("Retry").clicked() {
                self.fetch_dataset_info(ctx.clone());
            }
            return;
        }

        let n_samples = self.camels_info.as_ref().map(|i| i.n_samples).unwrap_or(0);

        ui.horizontal(|ui| {
            ui.label("Sample index:");
            ui.add(egui::DragValue::new(&mut self.camels_sample_idx).range(0..=n_samples.saturating_sub(1)));
            if ui.button("Load Sample").clicked() {
                self.fetch_sample(self.camels_sample_idx, ctx.clone());
            }
            if self.sample_loading {
                ui.spinner();
            }
        });

        if let Some(params) = &self.camels_params {
            ui.colored_label(
                egui::Color32::from_rgb(100, 180, 255),
                format!(
                    "Ωm={:.3}  σ8={:.3}  ASN1={:.2}  AAGN1={:.2}  ASN2={:.2}  AAGN2={:.2}",
                    params.first().unwrap_or(&0.0),
                    params.get(1).unwrap_or(&0.0),
                    params.get(2).unwrap_or(&0.0),
                    params.get(3).unwrap_or(&0.0),
                    params.get(4).unwrap_or(&0.0),
                    params.get(5).unwrap_or(&0.0),
                ),
            );

            if self.use_sample_params {
                self.omega_m = *params.first().unwrap_or(&0.3);
                self.sigma_8 = *params.get(1).unwrap_or(&0.8);
                self.a_sn1 = *params.get(2).unwrap_or(&1.0);
                self.a_agn1 = *params.get(3).unwrap_or(&1.0);
                self.a_sn2 = *params.get(4).unwrap_or(&1.0);
                self.a_agn2 = *params.get(5).unwrap_or(&1.0);
            }
        }

        let can_infer = self.camels_input.is_some() && !self.inference_pending;
        ui.horizontal(|ui| {
            ui.add_enabled_ui(can_infer, |ui| {
                if ui.button(egui::RichText::new("⚡ Run Inference").strong().size(16.0)).clicked() {
                    if let Some(inp) = &self.camels_input {
                        self.send_inference(&inp.flat.clone(), "camels", ctx.clone());
                    }
                }
            });
            if self.inference_pending {
                ui.spinner();
                ui.label("Running...");
            }
        });

        ui.separator();

        // Maps display
        let available = ui.available_size();
        let map_w = ((available.x - 80.0) / 3.0).min(350.0);
        let map_size = Vec2::splat(map_w);
        let cmap = self.colormap;
        let ls = self.log_scale;

        if let Some(inp) = self.camels_input.clone() {
            ui.horizontal(|ui| {
                self.show_map(ui, ctx, "c_inp", "Input: Mcdm", &inp, &cmap, ls, false, map_size);
                if let Some(out) = self.camels_output.clone() {
                    self.show_map(ui, ctx, "c_out", "Predicted: Mtot", &out, &cmap, ls, false, map_size);
                }
                if let Some(diff) = self.camels_diff.clone() {
                    self.show_map(ui, ctx, "c_diff", "Baryonic Effect", &diff, &Colormap::DarkDiverging, false, true, map_size);
                }
            });
        }

        // Ground truth comparison
        if let (Some(out), Some(gt)) = (self.camels_output.clone(), self.camels_gt.clone()) {
            ui.separator();
            ui.strong("Ground Truth Comparison");
            ui.horizontal(|ui| {
                self.show_map(ui, ctx, "c_gt", "Ground Truth: Mtot", &gt, &cmap, ls, false, map_size);
                self.show_map(ui, ctx, "c_out2", "Prediction", &out, &cmap, ls, false, map_size);
                let error: Vec<f32> = out.flat.iter().zip(gt.flat.iter()).map(|(o, g)| o - g).collect();
                let error_data = MapData::from_flat(error);
                self.show_map(ui, ctx, "c_err", "Error", &error_data, &Colormap::DarkDiverging, false, true, map_size);
            });

            // Metrics
            let log_err: Vec<f64> = out.flat.iter().zip(gt.flat.iter())
                .map(|(&o, &g)| (o.max(0.0) + 1.0).log10() as f64 - (g.max(0.0) + 1.0).log10() as f64)
                .collect();
            let mse = log_err.iter().map(|e| e * e).sum::<f64>() / log_err.len() as f64;
            let mae = log_err.iter().map(|e| e.abs()).sum::<f64>() / log_err.len() as f64;

            ui.horizontal(|ui| {
                ui.label(format!("Log MSE: {:.4}", mse));
                ui.separator();
                ui.label(format!("Log MAE: {:.4}", mae));
            });

            // Power spectrum + S(k)
            self.show_analysis_plots(ui, &out.flat, &gt.flat, Some(&self.camels_input.clone().unwrap().flat));
        }
    }

    fn tab_nbody(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label("Generate a dark matter density map using the Particle-Mesh N-body simulator, then run the emulator.");
        ui.small("The N-body simulation runs entirely in your browser via WebAssembly.");

        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Grid:");
            egui::ComboBox::from_id_salt("nbody_grid")
                .selected_text(format!("{}³", self.nbody_grid_res))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.nbody_grid_res, 16, "16³");
                    ui.selectable_value(&mut self.nbody_grid_res, 32, "32³ (default)");
                    ui.selectable_value(&mut self.nbody_grid_res, 48, "48³");
                    ui.selectable_value(&mut self.nbody_grid_res, 64, "64³ (slow)");
                });
            ui.label("Box:");
            ui.add(egui::DragValue::new(&mut self.nbody_box_size).range(50.0..=500.0).suffix(" Mpc/h"));
            ui.label("Steps:");
            ui.add(egui::DragValue::new(&mut self.nbody_steps).range(3..=50));
            ui.label("Seed:");
            ui.add(egui::DragValue::new(&mut self.nbody_seed).range(0..=99999));
        });

        ui.horizontal(|ui| {
            let running = self.nbody_running || self.inference_pending;
            ui.add_enabled_ui(!running, |ui| {
                if ui.button(egui::RichText::new("🚀 Run N-Body + Emulator").strong().size(16.0)).clicked() {
                    self.nbody_running = true;
                    self.status = "Running N-body simulation...".into();
                    // Run simulation synchronously (fast for 32³)
                    let map = engine::run_simulation(
                        self.nbody_seed,
                        self.nbody_grid_res,
                        self.nbody_box_size,
                        0.01,
                        self.nbody_steps,
                        self.resolution,
                    );
                    self.nbody_input = Some(MapData::from_flat(map.clone()));
                    self.status = "N-body done, running emulator...".into();
                    self.send_inference(&map, "nbody", ctx.clone());
                }
            });
            if running {
                ui.spinner();
                ui.label(&self.status);
            }
        });

        ui.separator();

        let available = ui.available_size();
        let map_w = ((available.x - 80.0) / 3.0).min(350.0);
        let map_size = Vec2::splat(map_w);
        let cmap = self.colormap;
        let ls = self.log_scale;

        if let Some(inp) = self.nbody_input.clone() {
            ui.horizontal(|ui| {
                self.show_map(ui, ctx, "nb_inp", "N-Body: DM Density", &inp, &cmap, ls, false, map_size);
                if let Some(out) = self.nbody_output.clone() {
                    self.show_map(ui, ctx, "nb_out", "Emulated: Total Matter", &out, &cmap, ls, false, map_size);
                }
                if let Some(diff) = self.nbody_diff.clone() {
                    self.show_map(ui, ctx, "nb_diff", "Baryonic Effect", &diff, &Colormap::DarkDiverging, false, true, map_size);
                }
            });

            if let Some(out) = &self.nbody_output {
                self.show_analysis_plots(ui, &out.flat, &inp.flat, None);
            }
        }
    }

    fn tab_sweep(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label("Vary one physics parameter and see how the baryonic effect changes.");

        ui.horizontal(|ui| {
            ui.label("Parameter:");
            egui::ComboBox::from_id_salt("sweep_param")
                .selected_text(&self.sweep_param_name)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.sweep_param_name, "a_sn1".into(), "A_SN1 (stellar feedback)");
                    ui.selectable_value(&mut self.sweep_param_name, "a_agn1".into(), "A_AGN1 (AGN feedback)");
                    ui.selectable_value(&mut self.sweep_param_name, "omega_m".into(), "Omega_m");
                    ui.selectable_value(&mut self.sweep_param_name, "sigma_8".into(), "sigma_8");
                });
            ui.label("Steps:");
            ui.add(egui::DragValue::new(&mut self.sweep_steps).range(2..=5));
        });

        let has_input = self.nbody_input.is_some() || self.camels_input.is_some();
        ui.add_enabled_ui(has_input && !self.sweep_running && !self.inference_pending, |ui| {
            if ui.button(egui::RichText::new("▶ Run Sweep").strong().size(16.0)).clicked() {
                let input_flat = self.nbody_input.as_ref()
                    .or(self.camels_input.as_ref())
                    .unwrap()
                    .flat.clone();
                let input_data = MapData::from_flat(input_flat.clone());
                self.sweep_input = Some(input_data);
                self.sweep_results.clear();
                self.sweep_running = true;

                let ranges: std::collections::HashMap<&str, (f32, f32)> = [
                    ("omega_m", (0.1, 0.5)),
                    ("sigma_8", (0.6, 1.0)),
                    ("a_sn1", (0.25, 4.0)),
                    ("a_agn1", (0.25, 4.0)),
                    ("a_sn2", (0.5, 2.0)),
                    ("a_agn2", (0.5, 2.0)),
                ].into_iter().collect();

                let param_name = self.sweep_param_name.clone();
                let (lo, hi) = ranges.get(param_name.as_str()).copied().unwrap_or((0.0, 1.0));
                let n_steps = self.sweep_steps;

                for i in 0..n_steps {
                    let val = lo + (hi - lo) * i as f32 / (n_steps - 1).max(1) as f32;
                    let saved = self.params_dict();
                    match param_name.as_str() {
                        "omega_m" => self.omega_m = val,
                        "sigma_8" => self.sigma_8 = val,
                        "a_sn1" => self.a_sn1 = val,
                        "a_agn1" => self.a_agn1 = val,
                        "a_sn2" => self.a_sn2 = val,
                        "a_agn2" => self.a_agn2 = val,
                        _ => {}
                    }
                    self.send_inference(&input_flat, &format!("sweep_{}", val), ctx.clone());
                    self.omega_m = saved[0];
                    self.sigma_8 = saved[1];
                    self.a_sn1 = saved[2];
                    self.a_agn1 = saved[3];
                    self.a_sn2 = saved[4];
                    self.a_agn2 = saved[5];
                }
            }
        });

        if !has_input {
            ui.small("Run N-body or load a CAMELS sample first to have an input map.");
        }

        if self.sweep_running {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(format!("{}/{} done", self.sweep_results.len(), self.sweep_steps));
            });
        }

        if !self.sweep_results.is_empty() {
            ui.separator();
            let available = ui.available_size();
            let map_w = ((available.x - 40.0) / self.sweep_results.len() as f32).min(250.0);
            let map_size = Vec2::splat(map_w);
            let cmap = self.colormap;
            let ls = self.log_scale;

            let mut sorted = self.sweep_results.clone();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            ui.horizontal_wrapped(|ui| {
                for (val, data) in &sorted {
                    let key = format!("sw_{:.2}", val);
                    let title = format!("{}={:.2}", self.sweep_param_name, val);
                    self.show_map(ui, ctx, &key, &title, data, &cmap, ls, false, map_size);
                }
            });

            // Power spectrum overlay
            if let Some(inp) = &self.sweep_input {
                let log_inp = analysis::safe_log1p_field(&inp.flat);
                let (k_inp, pk_inp) = analysis::power_spectrum(&log_inp, self.resolution);

                let mut lines: Vec<Line> = vec![
                    line("Input DM", k_inp.iter().zip(pk_inp.iter()).map(|(&k, &p)| [k.ln(), p.ln()]).collect()),
                ];

                for (val, data) in &sorted {
                    let log_out = analysis::safe_log1p_field(&data.flat);
                    let (k_o, pk_o) = analysis::power_spectrum(&log_out, self.resolution);
                    lines.push(
                        line(&format!("{}={:.2}", self.sweep_param_name, val),
                             k_o.iter().zip(pk_o.iter()).map(|(&k, &p)| [k.ln(), p.ln()]).collect()),
                    );
                }

                ui.separator();
                ui.strong("Power Spectrum Comparison (log-log)");
                Plot::new("sweep_ps")
                    .height(250.0)
                    .legend(egui_plot::Legend::default())
                    .show(ui, |plot_ui| {
                        for line in lines {
                            plot_ui.line(line);
                        }
                    });
            }
        }
    }

    fn tab_about(&self, ui: &mut egui::Ui) {
        ui.heading("Bayronik");
        ui.label("Field-level baryonic emulator: dark matter → total matter density.");
        ui.add_space(8.0);

        ui.label("Architecture: U-FNO (Fourier Neural Operator) with FiLM conditioning on 6 astrophysical parameters.");
        ui.label("Training data: CAMELS IllustrisTNG suite (1,000 simulations × 15 projections = 15,000 maps at 256×256).");
        ui.add_space(8.0);

        ui.strong("Components");
        egui::Grid::new("about_grid").striped(true).show(ui, |ui| {
            ui.label("engine");
            ui.label("Particle-mesh N-body simulator (runs in-browser via WASM)");
            ui.end_row();
            ui.label("model");
            ui.label("Python U-FNO training and scientific validation pipeline");
            ui.end_row();
            ui.label("server");
            ui.label("HTTP inference backend (axum + tch, loads TorchScript .pt)");
            ui.end_row();
            ui.label("registry");
            ui.label("Model registry + frozen-threshold regression tests");
            ui.end_row();
            ui.label("client");
            ui.label("This app, egui + WASM");
            ui.end_row();
            ui.label("infer");
            ui.label("Terminal inference UI (ratatui + libtorch)");
            ui.end_row();
        });

        ui.add_space(8.0);
        ui.strong("Physics Parameters");
        egui::Grid::new("params_info").striped(true).show(ui, |ui| {
            ui.label("Ωm"); ui.label("Total matter density parameter"); ui.end_row();
            ui.label("σ₈"); ui.label("Amplitude of matter fluctuations"); ui.end_row();
            ui.label("A_SN1"); ui.label("Supernova feedback strength"); ui.end_row();
            ui.label("A_AGN1"); ui.label("AGN feedback strength"); ui.end_row();
            ui.label("A_SN2"); ui.label("Supernova feedback wind speed"); ui.end_row();
            ui.label("A_AGN2"); ui.label("AGN feedback burstiness"); ui.end_row();
        });

        ui.add_space(12.0);
        ui.separator();
        ui.small("MIT License • github.com/Yuvraj-cyborg/Bayronik");
    }

    // ---- Shared analysis plots ----

    fn show_analysis_plots(&self, ui: &mut egui::Ui, output: &[f32], reference: &[f32], dm_input: Option<&[f32]>) {
        let n = self.resolution;
        let log_out = analysis::safe_log1p_field(output);
        let log_ref = analysis::safe_log1p_field(reference);
        let (k_out, pk_out) = analysis::power_spectrum(&log_out, n);
        let (k_ref, pk_ref) = analysis::power_spectrum(&log_ref, n);

        ui.separator();
        ui.columns(2, |cols| {
            // Power spectrum
            cols[0].strong("Power Spectrum P(k)  [log-log]");
            Plot::new("ps_plot")
                .height(220.0)
                .legend(egui_plot::Legend::default())
                .show(&mut cols[0], |plot_ui| {
                    if let Some(dm) = dm_input {
                        let log_dm = analysis::safe_log1p_field(dm);
                        let (k_dm, pk_dm) = analysis::power_spectrum(&log_dm, n);
                        plot_ui.line(
                            line("Input DM", k_dm.iter().zip(pk_dm.iter()).map(|(&k, &p)| [k.ln(), p.ln()]).collect()).width(1.5),
                        );
                    }
                    plot_ui.line(
                        line("Reference", k_ref.iter().zip(pk_ref.iter()).map(|(&k, &p)| [k.ln(), p.ln()]).collect()).width(1.5),
                    );
                    plot_ui.line(
                        line("Prediction", k_out.iter().zip(pk_out.iter()).map(|(&k, &p)| [k.ln(), p.ln()]).collect()).width(2.0),
                    );
                });

            // Baryon suppression
            cols[1].strong("Baryon Suppression S(k)");
            let (k_s, s_k) = analysis::baryon_suppression(&k_ref, &pk_ref, &k_out, &pk_out);
            Plot::new("sk_plot")
                .height(220.0)
                .legend(egui_plot::Legend::default())
                .show(&mut cols[1], |plot_ui| {
                    plot_ui.hline(HLine::new("S=1", 1.0));
                    plot_ui.line(
                        line("S(k) = P_out/P_ref", k_s.iter().zip(s_k.iter()).map(|(&k, &s)| [k, s]).collect()).width(2.0),
                    );
                });
        });

        // 1-point PDF
        let (x_out, y_out) = analysis::pixel_pdf(output, 60);
        let (x_ref, y_ref) = analysis::pixel_pdf(reference, 60);

        ui.strong("1-Point PDF");
        Plot::new("pdf_plot")
            .height(180.0)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    line("Reference", x_ref.iter().zip(y_ref.iter()).map(|(&x, &y)| [x, y]).collect()).width(1.5),
                );
                plot_ui.line(
                    line("Prediction", x_out.iter().zip(y_out.iter()).map(|(&x, &y)| [x, y]).collect()).width(2.0),
                );
            });
    }
}

// ---- Main egui App trait ----

impl eframe::App for BayronikApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_pending();

        // Top bar
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(egui::RichText::new("Bayronik").strong());
                ui.label("-");
                ui.label("Field-level baryonic emulator");

                let status_color = if self.server_connected && self.server_model_loaded {
                    egui::Color32::GREEN
                } else if self.server_connected {
                    egui::Color32::YELLOW
                } else {
                    egui::Color32::from_rgb(120, 120, 120)
                };

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.colored_label(status_color, &self.status);
                });
            });
        });

        // Left panel — controls
        egui::SidePanel::left("controls")
            .min_width(260.0)
            .show(ctx, |ui| {
                ui.heading("Display");
                ui.checkbox(&mut self.log_scale, "Log scale");
                egui::ComboBox::from_label("Colormap")
                    .selected_text(self.colormap.name())
                    .show_ui(ui, |ui| {
                        for cmap in Colormap::ALL {
                            let changed = ui.selectable_value(&mut self.colormap, cmap, cmap.name()).changed();
                            if changed {
                                self.tex_cache.clear();
                            }
                        }
                    });

                if ui.button("Clear texture cache").clicked() {
                    self.tex_cache.clear();
                }

                ui.separator();
                ui.heading("Physics Parameters");

                if self.camels_params.is_some() {
                    ui.checkbox(&mut self.use_sample_params, "Use sample's params");
                }

                egui::Grid::new("params_grid").show(ui, |ui| {
                    ui.label("Ωm:");
                    ui.add(egui::Slider::new(&mut self.omega_m, 0.1..=0.5).fixed_decimals(2));
                    ui.end_row();
                    ui.label("σ₈:");
                    ui.add(egui::Slider::new(&mut self.sigma_8, 0.6..=1.0).fixed_decimals(2));
                    ui.end_row();
                    ui.label("A_SN1:");
                    ui.add(egui::Slider::new(&mut self.a_sn1, 0.25..=4.0).fixed_decimals(2));
                    ui.end_row();
                    ui.label("A_AGN1:");
                    ui.add(egui::Slider::new(&mut self.a_agn1, 0.25..=4.0).fixed_decimals(2));
                    ui.end_row();
                    ui.label("A_SN2:");
                    ui.add(egui::Slider::new(&mut self.a_sn2, 0.5..=2.0).fixed_decimals(2));
                    ui.end_row();
                    ui.label("A_AGN2:");
                    ui.add(egui::Slider::new(&mut self.a_agn2, 0.5..=2.0).fixed_decimals(2));
                    ui.end_row();
                });

                ui.separator();
                ui.heading("Server");
                ui.horizontal(|ui| {
                    let dot = if self.server_connected { "🟢" } else { "🔴" };
                    ui.label(dot);
                    ui.code(API_URL);
                });
                if ui.button("Check Connection").clicked() {
                    self.check_server_health(ctx.clone());
                    self.fetch_dataset_info(ctx.clone());
                }
            });

        // Central panel — tab content
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.active_tab, Tab::NBody, egui::RichText::new("N-Body Simulator").strong());
                ui.selectable_value(&mut self.active_tab, Tab::Camels, egui::RichText::new("CAMELS Data").strong());
                ui.selectable_value(&mut self.active_tab, Tab::Sweep, egui::RichText::new("Parameter Sweep").strong());
                ui.selectable_value(&mut self.active_tab, Tab::About, egui::RichText::new("About").strong());
            });
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                match self.active_tab {
                    Tab::NBody => self.tab_nbody(ui, ctx),
                    Tab::Camels => self.tab_camels(ui, ctx),
                    Tab::Sweep => self.tab_sweep(ui, ctx),
                    Tab::About => self.tab_about(ui),
                }
            });
        });
    }
}
