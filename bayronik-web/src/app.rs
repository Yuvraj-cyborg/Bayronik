use egui::{TextureHandle, TextureOptions, Vec2};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use crate::visualization::{array_to_colorimage, compute_statistics, Colormap};

const API_URL: &str = "http://localhost:8000";

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

#[derive(Clone)]
struct MapStats {
    mean: f32,
    std: f32,
    min: f32,
    max: f32,
}

pub struct BayronikApp {
    input_map: Option<Vec<f32>>,
    output_map: Option<Vec<f32>>,
    input_texture: Option<TextureHandle>,
    output_texture: Option<TextureHandle>,
    diff_texture: Option<TextureHandle>,
    resolution: usize,
    status: String,
    colormap: Colormap,

    omega_m: f32,
    sigma_8: f32,
    a_sn1: f32,
    a_agn1: f32,
    a_sn2: f32,
    a_agn2: f32,

    server_connected: bool,
    inference_pending: bool,
    pending_output: Arc<Mutex<Option<Result<Vec<f32>, String>>>>,

    input_stats: Option<MapStats>,
    output_stats: Option<MapStats>,
    diff_stats: Option<MapStats>,
    last_params: Option<String>,
}

impl Default for BayronikApp {
    fn default() -> Self {
        Self {
            input_map: None,
            output_map: None,
            input_texture: None,
            output_texture: None,
            diff_texture: None,
            resolution: 256,
            status: "Checking server connection...".to_string(),
            colormap: Colormap::Viridis,
            omega_m: 0.3,
            sigma_8: 0.8,
            a_sn1: 1.0,
            a_agn1: 1.0,
            a_sn2: 1.0,
            a_agn2: 1.0,
            server_connected: false,
            inference_pending: false,
            pending_output: Arc::new(Mutex::new(None)),
            input_stats: None,
            output_stats: None,
            diff_stats: None,
            last_params: None,
        }
    }
}

impl BayronikApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let app = Self::default();
        app.check_server_health(cc.egui_ctx.clone());
        app
    }

    fn check_server_health(&self, ctx: egui::Context) {
        let url = format!("{}/health", API_URL);

        ehttp::fetch(ehttp::Request::get(&url), move |result| {
            ctx.request_repaint();
            match result {
                Ok(response) => {
                    if let Ok(health) = serde_json::from_slice::<HealthResponse>(&response.bytes) {
                        log::info!("Server connected, model_loaded: {}", health.model_loaded);
                    }
                }
                Err(e) => {
                    log::error!("Server connection failed: {}", e);
                }
            }
        });
    }

    fn generate_test_input(&mut self) {
        let n = self.resolution;
        let mut data = vec![0.0f32; n * n];

        for y in 0..n {
            for x in 0..n {
                let dx = (x as f32 - n as f32 / 2.0) / (n as f32 / 4.0);
                let dy = (y as f32 - n as f32 / 2.0) / (n as f32 / 4.0);
                let r = (dx * dx + dy * dy).sqrt();
                let halo1 = 10.0 * (-r * r * 2.0).exp();

                let dx2 = (x as f32 - n as f32 / 4.0) / (n as f32 / 8.0);
                let dy2 = (y as f32 - n as f32 / 4.0) / (n as f32 / 8.0);
                let r2 = (dx2 * dx2 + dy2 * dy2).sqrt();
                let halo2 = 5.0 * (-r2 * r2 * 2.0).exp();

                let noise = ((x as f32 * 0.1).sin() * (y as f32 * 0.1).cos()) * 0.5;

                data[y * n + x] = (halo1 + halo2 + noise + 1.0).max(0.1);
            }
        }

        self.input_map = Some(data);
        self.status = "Test input generated - click Run Inference".to_string();
    }

    fn params_string(&self) -> String {
        format!(
            "Om={:.2} s8={:.2} ASN1={:.2} AAGN1={:.2} ASN2={:.2} AAGN2={:.2}",
            self.omega_m, self.sigma_8, self.a_sn1, self.a_agn1, self.a_sn2, self.a_agn2
        )
    }

    fn run_inference(&mut self, ctx: egui::Context) {
        let Some(input) = &self.input_map else {
            self.status = "Generate input first".to_string();
            return;
        };

        self.inference_pending = true;
        let params = self.params_string();
        self.status = format!("Running inference... [{}]", params);

        let n = self.resolution;
        let input_2d: Vec<Vec<f32>> = input.chunks(n).map(|row| row.to_vec()).collect();

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

        let pending_output = self.pending_output.clone();

        let http_request = ehttp::Request {
            method: "POST".to_string(),
            url,
            body,
            headers: ehttp::Headers::new(&[("Content-Type", "application/json")]),
        };

        ehttp::fetch(http_request, move |result| {
            let output = match result {
                Ok(response) => {
                    if response.status == 200 {
                        match serde_json::from_slice::<InferenceResponse>(&response.bytes) {
                            Ok(resp) => {
                                let flat: Vec<f32> = resp.output_map.into_iter().flatten().collect();
                                Ok(flat)
                            }
                            Err(e) => Err(format!("JSON parse error: {}", e)),
                        }
                    } else {
                        Err(format!("Server error: {}", response.status))
                    }
                }
                Err(e) => Err(format!("Request failed: {}", e)),
            };

            if let Ok(mut guard) = pending_output.lock() {
                *guard = Some(output);
            }
            ctx.request_repaint();
        });
    }

    fn check_pending_inference(&mut self, ctx: &egui::Context) {
        let result = {
            if let Ok(mut guard) = self.pending_output.lock() {
                guard.take()
            } else {
                None
            }
        };

        if let Some(result) = result {
            self.inference_pending = false;
            match result {
                Ok(output) => {
                    self.output_map = Some(output);
                    self.last_params = Some(self.params_string());
                    self.update_textures(ctx);
                    self.status = format!("Inference complete [{}]", self.params_string());
                    self.server_connected = true;
                }
                Err(e) => {
                    self.status = format!("Error: {}", e);
                }
            }
        }
    }

    fn update_textures(&mut self, ctx: &egui::Context) {
        let n = self.resolution;

        let tex_options = TextureOptions {
            magnification: egui::TextureFilter::Linear,
            minification: egui::TextureFilter::Linear,
            ..Default::default()
        };

        if let Some(input) = &self.input_map {
            let (mean, std, min, max) = compute_statistics(input);
            self.input_stats = Some(MapStats { mean, std, min, max });

            let img = array_to_colorimage(input, n, n, &self.colormap);
            self.input_texture = Some(ctx.load_texture("input_map", img, tex_options));
        }

        if let Some(output) = &self.output_map {
            let (mean, std, min, max) = compute_statistics(output);
            self.output_stats = Some(MapStats { mean, std, min, max });

            let img = array_to_colorimage(output, n, n, &self.colormap);
            self.output_texture = Some(ctx.load_texture("output_map", img, tex_options));

            if let Some(input) = &self.input_map {
                let diff: Vec<f32> = output
                    .iter()
                    .zip(input.iter())
                    .map(|(o, i)| o - i)
                    .collect();

                let (mean, std, min, max) = compute_statistics(&diff);
                self.diff_stats = Some(MapStats { mean, std, min, max });

                let diff_img = array_to_colorimage(&diff, n, n, &Colormap::Coolwarm);
                self.diff_texture = Some(ctx.load_texture("diff_map", diff_img, tex_options));
            }
        }
    }

    fn show_stats(&self, ui: &mut egui::Ui, label: &str, stats: &Option<MapStats>) {
        if let Some(s) = stats {
            ui.label(format!("{}: [{:.2}, {:.2}]", label, s.min, s.max));
        }
    }
}

impl eframe::App for BayronikApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.check_pending_inference(ctx);

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Bayronik");
                ui.label("|");
                ui.label("Baryonic Field Emulator");

                let status_color = if self.server_connected {
                    egui::Color32::GREEN
                } else if self.inference_pending {
                    egui::Color32::YELLOW
                } else {
                    egui::Color32::LIGHT_GRAY
                };

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.colored_label(status_color, &self.status);
                });
            });
        });

        egui::SidePanel::left("controls")
            .min_width(280.0)
            .show(ctx, |ui| {
                ui.heading("Controls");
                ui.separator();

                ui.heading("Server");
                ui.horizontal(|ui| {
                    ui.label("API:");
                    ui.code(API_URL);
                });
                if ui.button("Check Connection").clicked() {
                    self.check_server_health(ctx.clone());
                }

                ui.separator();
                ui.heading("Input");

                if ui.button("Generate Test Input").clicked() {
                    self.generate_test_input();
                    self.update_textures(ctx);
                }

                ui.separator();
                ui.heading("Physics Parameters");
                ui.small("Adjust and click Run Inference to see changes");

                egui::Grid::new("params_grid").show(ui, |ui| {
                    ui.label("Omega_m:");
                    ui.add(egui::Slider::new(&mut self.omega_m, 0.1..=0.5).fixed_decimals(2));
                    ui.end_row();

                    ui.label("sigma_8:");
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

                let can_infer = self.input_map.is_some() && !self.inference_pending;
                ui.add_enabled_ui(can_infer, |ui| {
                    if ui
                        .button(egui::RichText::new("Run Inference").strong())
                        .clicked()
                    {
                        self.run_inference(ctx.clone());
                    }
                });

                if self.inference_pending {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Processing...");
                    });
                }

                ui.separator();
                ui.heading("Display");

                egui::ComboBox::from_label("Colormap")
                    .selected_text(format!("{:?}", self.colormap))
                    .show_ui(ui, |ui| {
                        let changed = ui
                            .selectable_value(&mut self.colormap, Colormap::Viridis, "Viridis")
                            .changed()
                            || ui
                                .selectable_value(&mut self.colormap, Colormap::Inferno, "Inferno")
                                .changed()
                            || ui
                                .selectable_value(&mut self.colormap, Colormap::Plasma, "Plasma")
                                .changed()
                            || ui
                                .selectable_value(&mut self.colormap, Colormap::Coolwarm, "Coolwarm")
                                .changed();

                        if changed {
                            self.update_textures(ctx);
                        }
                    });

                ui.separator();
                ui.heading("Statistics");
                self.show_stats(ui, "Input", &self.input_stats);
                self.show_stats(ui, "Output", &self.output_stats);
                self.show_stats(ui, "Diff", &self.diff_stats);

                if let Some(params) = &self.last_params {
                    ui.separator();
                    ui.small(format!("Last run: {}", params));
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            let map_size = Vec2::splat((available.x / 3.0 - 20.0).min(available.y - 80.0));

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.heading("Input: Dark Matter");
                    if let Some(tex) = &self.input_texture {
                        ui.image((tex.id(), map_size));
                    } else {
                        ui.allocate_space(map_size);
                        ui.label("No input");
                    }
                });

                ui.vertical(|ui| {
                    ui.heading("Output: Total Matter");
                    if let Some(tex) = &self.output_texture {
                        ui.image((tex.id(), map_size));
                    } else {
                        ui.allocate_space(map_size);
                        ui.label("No output");
                    }
                });

                ui.vertical(|ui| {
                    ui.heading("Baryonic Effect");
                    if let Some(tex) = &self.diff_texture {
                        ui.image((tex.id(), map_size));
                    } else {
                        ui.allocate_space(map_size);
                        ui.label("Run inference to see");
                    }
                });
            });
        });
    }
}
