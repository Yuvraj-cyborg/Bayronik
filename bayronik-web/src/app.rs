use egui::{TextureHandle, TextureOptions, Vec2};

use crate::inference::Emulator;
use crate::visualization::{array_to_colorimage, Colormap};

pub struct BayronikApp {
    emulator: Option<Emulator>,
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
    
    model_loaded: bool,
    loading_model: bool,
}

impl Default for BayronikApp {
    fn default() -> Self {
        Self {
            emulator: None,
            input_map: None,
            output_map: None,
            input_texture: None,
            output_texture: None,
            diff_texture: None,
            resolution: 256,
            status: "Click 'Load Model' to begin".to_string(),
            colormap: Colormap::Viridis,
            omega_m: 0.3,
            sigma_8: 0.8,
            a_sn1: 1.0,
            a_agn1: 1.0,
            a_sn2: 1.0,
            a_agn2: 1.0,
            model_loaded: false,
            loading_model: false,
        }
    }
}

impl BayronikApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }

    fn get_conditions(&self) -> [f32; 6] {
        [
            self.omega_m,
            self.sigma_8,
            self.a_sn1,
            self.a_agn1,
            self.a_sn2,
            self.a_agn2,
        ]
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
        self.status = "Test input generated".to_string();
    }

    fn run_inference(&mut self, ctx: &egui::Context) {
        if let (Some(emulator), Some(input)) = (&self.emulator, &self.input_map) {
            let conditions = self.get_conditions();
            
            match emulator.run(input, Some(&conditions)) {
                Ok(output) => {
                    self.output_map = Some(output);
                    self.update_textures(ctx);
                    self.status = "Inference complete".to_string();
                }
                Err(e) => {
                    self.status = format!("Inference error: {}", e);
                }
            }
        } else {
            self.status = "Load model and input first".to_string();
        }
    }

    fn update_textures(&mut self, ctx: &egui::Context) {
        let n = self.resolution;
        
        if let Some(input) = &self.input_map {
            let img = array_to_colorimage(input, n, n, &self.colormap);
            self.input_texture = Some(ctx.load_texture(
                "input_map",
                img,
                TextureOptions::default(),
            ));
        }
        
        if let Some(output) = &self.output_map {
            let img = array_to_colorimage(output, n, n, &self.colormap);
            self.output_texture = Some(ctx.load_texture(
                "output_map",
                img,
                TextureOptions::default(),
            ));
            
            if let Some(input) = &self.input_map {
                let diff: Vec<f32> = output.iter()
                    .zip(input.iter())
                    .map(|(o, i)| o - i)
                    .collect();
                let diff_img = array_to_colorimage(&diff, n, n, &Colormap::Coolwarm);
                self.diff_texture = Some(ctx.load_texture(
                    "diff_map",
                    diff_img,
                    TextureOptions::default(),
                ));
            }
        }
    }
}

impl eframe::App for BayronikApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Bayronik");
                ui.label("|");
                ui.label("Baryonic Field Emulator");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(&self.status);
                });
            });
        });

        egui::SidePanel::left("controls").min_width(250.0).show(ctx, |ui| {
            ui.heading("Controls");
            ui.separator();

            ui.heading("Model");
            if self.model_loaded {
                ui.label("Model: Loaded");
            } else if self.loading_model {
                ui.label("Loading model...");
                ui.spinner();
            } else {
                if ui.button("Load Model").clicked() {
                    self.status = "Model loading not yet implemented in WASM".to_string();
                }
            }
            
            ui.separator();
            ui.heading("Input");
            
            if ui.button("Generate Test Input").clicked() {
                self.generate_test_input();
                self.update_textures(ctx);
            }
            
            ui.separator();
            ui.heading("Physics Parameters");
            
            ui.horizontal(|ui| {
                ui.label("Omega_m:");
                ui.add(egui::Slider::new(&mut self.omega_m, 0.1..=0.5));
            });
            
            ui.horizontal(|ui| {
                ui.label("sigma_8:");
                ui.add(egui::Slider::new(&mut self.sigma_8, 0.6..=1.0));
            });
            
            ui.horizontal(|ui| {
                ui.label("A_SN1:");
                ui.add(egui::Slider::new(&mut self.a_sn1, 0.25..=4.0));
            });
            
            ui.horizontal(|ui| {
                ui.label("A_AGN1:");
                ui.add(egui::Slider::new(&mut self.a_agn1, 0.25..=4.0));
            });
            
            ui.horizontal(|ui| {
                ui.label("A_SN2:");
                ui.add(egui::Slider::new(&mut self.a_sn2, 0.5..=2.0));
            });
            
            ui.horizontal(|ui| {
                ui.label("A_AGN2:");
                ui.add(egui::Slider::new(&mut self.a_agn2, 0.5..=2.0));
            });
            
            ui.separator();
            
            if ui.button("Run Inference").clicked() {
                self.run_inference(ctx);
            }
            
            ui.separator();
            ui.heading("Colormap");
            egui::ComboBox::from_label("")
                .selected_text(format!("{:?}", self.colormap))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.colormap, Colormap::Viridis, "Viridis");
                    ui.selectable_value(&mut self.colormap, Colormap::Inferno, "Inferno");
                    ui.selectable_value(&mut self.colormap, Colormap::Plasma, "Plasma");
                    ui.selectable_value(&mut self.colormap, Colormap::Coolwarm, "Coolwarm");
                });
            
            if self.input_map.is_some() || self.output_map.is_some() {
                if ui.button("Refresh View").clicked() {
                    self.update_textures(ctx);
                }
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            let map_size = Vec2::splat((available.x / 3.0 - 20.0).min(available.y - 60.0));
            
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
                        ui.label("No diff");
                    }
                });
            });
        });
    }
}
