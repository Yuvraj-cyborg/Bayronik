mod heatmap;

use anyhow::{Context, Result};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use heatmap::HeatmapWidget;
use npyz::NpyFile;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph},
};
use std::{fs::File, io::{self, Stdout}, time::{Duration, Instant}};
use tch::{kind::Kind, Device, Tensor};

enum DataSource {
    CamelsCV,
    NBodyGenerated,
}

struct App {
    input_map: Tensor,
    output_map: Tensor,
    all_input_maps: Vec<f32>,
    current_sim_idx: usize,
    total_sims: usize,
    model: tch::CModule,
    device: Device,
    data_source: DataSource,
    status_message: String,
}

impl App {
    fn new() -> Result<Self> {
        println!("Loading dataset and TorchScript model...");
        
        let npy_path = "../bayronik-model/data/Maps_Mcdm_IllustrisTNG_CV_z=0.00.npy";
        println!("   Loading: {}", npy_path);
        
        let reader = File::open(npy_path)
            .with_context(|| format!("Failed to open NPY file at '{}'", npy_path))?;
        let npy_file = NpyFile::new(reader)
            .with_context(|| "Failed to parse NPY file structure.")?;
        
        let all_data: Vec<f32> = npy_file.into_vec()?;
        let total_sims = all_data.len() / (256 * 256);
        println!("   ✓ Loaded {} simulations", total_sims);
        
        let model_path = "../bayronik-model/weights/traced_unet_LH.pt";
        println!("   Loading model: {}", model_path);
        let device = Device::cuda_if_available();
        let model = tch::CModule::load_on_device(model_path, device)
            .with_context(|| format!("Failed to load TorchScript model from '{}'", model_path))?;
        println!("   ✓ Model loaded on {:?}", device);
        
        let first_sim_data = &all_data[..256 * 256];
        let input_map_raw = Tensor::from_slice(first_sim_data)
            .reshape(&[1, 1, 256, 256])
            .to_kind(Kind::Float);
        
        let input_map = input_map_raw.log1p();
        let output_map = model.forward_ts(&[input_map.to(device)])?.to(Device::Cpu);
        
        println!("✅ Ready! Use ← → arrows to navigate simulations");
        
        Ok(Self {
            input_map,
            output_map,
            all_input_maps: all_data,
            current_sim_idx: 0,
            total_sims,
            model,
            device,
            data_source: DataSource::CamelsCV,
            status_message: String::new(),
        })
    }
    
    fn generate_nbody_map(&self) -> Vec<f32> {
        bayronik_core::run_simulation(
            131_072,  // 2^17 particles - reduces shot noise
            64,       // 64^3 grid - better force resolution
            100.0,    // 100 Mpc box
            0.005,    // Smaller timestep for stability
            80,       // 2× more steps to break grid symmetry
            256,      // 256x256 output resolution
        )
    }
    
    fn switch_to_nbody(&mut self) -> Result<()> {
        self.status_message = "Generating N-body simulation...".to_string();
        
        let mut nbody_map = self.generate_nbody_map();
        
        // Match CAMELS statistics in log-space
        let nbody_mean: f32 = nbody_map.iter().sum::<f32>() / nbody_map.len() as f32;
        let nbody_std: f32 = (nbody_map.iter()
            .map(|&x| (x - nbody_mean).powi(2))
            .sum::<f32>() / nbody_map.len() as f32).sqrt();
        
        // Scale and shift to match CAMELS Mcdm log-space statistics
        // Target: mean ~10^10, but after log1p we want log-space mean ~22-23, std ~2.5
        let target_mean = 1e10;
        let target_std_ratio = 0.7;  // Reduce variance to match CAMELS better
        
        for val in &mut nbody_map {
            // Standardize then rescale variance
            *val = (*val - nbody_mean) / nbody_std * (nbody_mean * target_std_ratio) + nbody_mean;
            *val = (*val).max(0.0);  // Ensure positive
            *val *= target_mean / nbody_mean;
        }
        
        let input_map_raw = Tensor::from_slice(&nbody_map)
            .reshape(&[1, 1, 256, 256])
            .to_kind(Kind::Float);
        
        self.input_map = input_map_raw.log1p();
        self.output_map = self.model
            .forward_ts(&[self.input_map.to(self.device)])?
            .to(Device::Cpu);
        
        self.data_source = DataSource::NBodyGenerated;
        self.status_message = "N-body simulation complete".to_string();
        Ok(())
    }
    
    fn switch_to_camels(&mut self) -> Result<()> {
        self.status_message = "Switching to CAMELS data...".to_string();
        self.load_simulation(0)?;
        self.data_source = DataSource::CamelsCV;
        self.status_message = String::new();
        Ok(())
    }
    
    fn load_simulation(&mut self, idx: usize) -> Result<()> {
        let start = idx * 256 * 256;
        let end = start + 256 * 256;
        let sim_data = &self.all_input_maps[start..end];
        
        let input_map_raw = Tensor::from_slice(sim_data)
            .reshape(&[1, 1, 256, 256])
            .to_kind(Kind::Float);
        
        self.input_map = input_map_raw.log1p();
        self.output_map = self.model
            .forward_ts(&[self.input_map.to(self.device)])?
            .to(Device::Cpu);
        
        self.current_sim_idx = idx;
        Ok(())
    }
}

fn main() -> Result<()> {
    let app = App::new()?;
    let mut terminal = setup_terminal()?;
    let result = run_app_logic(&mut terminal, app);
    restore_terminal(&mut terminal)?;
    if let Err(err) = result {
        println!("Error: {:?}", err);
    }
    Ok(())
}

fn run_app_logic(terminal: &mut Terminal<CrosstermBackend<Stdout>>, mut app: App) -> Result<()> {
    let mut last_tick = Instant::now();
    let tick_rate = Duration::from_millis(250);

    loop {
        terminal.draw(|f| ui(f, &app))?;
        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Right | KeyCode::Char('n') => {
                        if matches!(app.data_source, DataSource::CamelsCV) {
                            let next_idx = (app.current_sim_idx + 1) % app.total_sims;
                            app.load_simulation(next_idx)?;
                        }
                    }
                    KeyCode::Left | KeyCode::Char('p') => {
                        if matches!(app.data_source, DataSource::CamelsCV) {
                            let prev_idx = if app.current_sim_idx == 0 {
                                app.total_sims - 1
                            } else {
                                app.current_sim_idx - 1
                            };
                            app.load_simulation(prev_idx)?;
                        }
                    }
                    KeyCode::Char('r') => {
                        if matches!(app.data_source, DataSource::CamelsCV) {
                            use rand::Rng;
                            let random_idx = rand::thread_rng().gen_range(0..app.total_sims);
                            app.load_simulation(random_idx)?;
                        }
                    }
                    KeyCode::Char('g') => {
                        terminal.draw(|f| {
                            app.status_message = "Generating N-body simulation...".to_string();
                            ui(f, &app);
                        })?;
                        app.switch_to_nbody()?;
                    }
                    KeyCode::Char('c') => {
                        app.switch_to_camels()?;
                    }
                    _ => {}
                }
            }
        }
        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }
}

fn ui(frame: &mut Frame, app: &App) {
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(4),
            Constraint::Min(0),
        ])
        .split(frame.area());
    
    let title_block = Block::default().borders(Borders::ALL)
        .title("Bayronik: Baryonic Field Emulator")
        .style(Style::default().fg(Color::Cyan).bold());
    let title_text = format!("Model: Trained on 1000 LH sims | Demo: {} CV sims | Values in log-space | Press 'q' to quit", app.total_sims);
    let title = Paragraph::new(title_text).block(title_block);
    frame.render_widget(title, main_layout[0]);
    
    let controls_block = Block::default().borders(Borders::ALL)
        .title("Controls")
        .style(Style::default().fg(Color::Yellow));
    
    let source_label = match app.data_source {
        DataSource::CamelsCV => format!("CAMELS CV {}/{}", app.current_sim_idx + 1, app.total_sims),
        DataSource::NBodyGenerated => "N-Body Generated (Press 'c' to return to CAMELS)".to_string(),
    };
    
    let controls_text = if app.status_message.is_empty() {
        format!(
            "{} | [←/→] Nav | [r] Rand | [g] Gen N-body | [c] CAMELS | [q] Quit",
            source_label
        )
    } else {
        format!("{} | ⚡ {}", source_label, app.status_message)
    };
    
    let controls = Paragraph::new(controls_text)
        .block(controls_block)
        .style(Style::default().fg(Color::Green));
    frame.render_widget(controls, main_layout[1]);
    
    let maps_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(34),
        ])
        .split(main_layout[2]);
    
    let input_widget = HeatmapWidget::new(&app.input_map, "Input: Dark Matter (Mcdm)", Color::Blue);
    frame.render_widget(input_widget, maps_layout[0]);
    
    let output_widget = HeatmapWidget::new(&app.output_map, "Output: Total Matter (Mtot)", Color::Green);
    frame.render_widget(output_widget, maps_layout[1]);
    
    let diff_map = &app.output_map - &app.input_map;
    let diff_widget = HeatmapWidget::new(&diff_map, "Baryonic Effect (Δ)", Color::Red);
    frame.render_widget(diff_widget, maps_layout[2]);
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>, io::Error> {
    let mut stdout = io::stdout();
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    Terminal::new(CrosstermBackend::new(stdout))
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<(), io::Error> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()
}

