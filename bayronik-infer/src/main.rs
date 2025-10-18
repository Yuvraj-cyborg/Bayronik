use anyhow::{Context, Result};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use npyz::NpyFile;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph, Widget},
};
use std::{fs::File, io::{self, Stdout}, time::{Duration, Instant}};
use tch::{kind::Kind, Device, Tensor};

struct App {
    input_map: Tensor,
    output_map: Tensor,
    all_input_maps: Vec<f32>,  // Store all maps from LH dataset
    current_sim_idx: usize,     // Which simulation we're viewing
    total_sims: usize,          // Total number of simulations
    model: tch::CModule,        // Keep model loaded for quick inference
    device: Device,
}

impl App {
    fn new() -> Result<Self> {
        println!("🔄 Loading dataset and TorchScript model...");
        
        let npy_path = "../bayronik-model/data/Maps_Mcdm_IllustrisTNG_CV_z=0.00.npy";
        println!("   Loading: {}", npy_path);
        
        let reader = File::open(npy_path)
            .with_context(|| format!("Failed to open NPY file at '{}'", npy_path))?;
        let npy_file = NpyFile::new(reader)
            .with_context(|| "Failed to parse NPY file structure.")?;
        
        let all_data: Vec<f32> = npy_file.into_vec()?;
        let total_sims = all_data.len() / (256 * 256);
        println!("   ✓ Loaded {} simulations", total_sims);
        
        // Load model
        let model_path = "../bayronik-model/weights/traced_unet_LH.pt";
        println!("   Loading model: {}", model_path);
        let device = Device::cuda_if_available();
        let model = tch::CModule::load_on_device(model_path, device)
            .with_context(|| format!("Failed to load TorchScript model from '{}'", model_path))?;
        println!("   ✓ Model loaded on {:?}", device);
        
        // Load first simulation
        let first_sim_data = &all_data[..256 * 256];
        let input_map_raw = Tensor::from_slice(first_sim_data)
            .reshape(&[1, 1, 256, 256])
            .to_kind(Kind::Float);
        
        // Apply log1p transform (model was trained on log-space data!)
        let input_map = input_map_raw.log1p();
        
        // Run inference (model outputs in log-space)
        let output_map = model.forward_ts(&[input_map.to(device)])?.to(Device::Cpu);
        
        println!(" Ready! Use ← → arrows to navigate simulations");
        
        Ok(Self {
            input_map,
            output_map,
            all_input_maps: all_data,
            current_sim_idx: 0,
            total_sims,
            model,
            device,
        })
    }
    
    fn load_simulation(&mut self, idx: usize) -> Result<()> {
        // Load specific simulation from dataset
        let start = idx * 256 * 256;
        let end = start + 256 * 256;
        let sim_data = &self.all_input_maps[start..end];
        
        let input_map_raw = Tensor::from_slice(sim_data)
            .reshape(&[1, 1, 256, 256])
            .to_kind(Kind::Float);
        
        // Apply log1p transform (model expects log-space)
        self.input_map = input_map_raw.log1p();
        
        // Run inference (outputs in log-space)
        self.output_map = self.model
            .forward_ts(&[self.input_map.to(self.device)])?
            .to(Device::Cpu);
        
        self.current_sim_idx = idx;
        Ok(())
    }
}

fn main() -> Result<()> {
    let app = App::new()?;
    println!(" Starting TUI... (Press 'q' to quit)");
    std::thread::sleep(std::time::Duration::from_millis(500));

    let mut terminal = setup_terminal()?;
    let app_result = run_app_logic(&mut terminal, app);
    restore_terminal(&mut terminal)?;

    if let Err(err) = app_result {
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
                        let next_idx = (app.current_sim_idx + 1) % app.total_sims;
                        app.load_simulation(next_idx)?;
                    }
                    KeyCode::Left | KeyCode::Char('p') => {
                        let prev_idx = if app.current_sim_idx == 0 {
                            app.total_sims - 1
                        } else {
                            app.current_sim_idx - 1
                        };
                        app.load_simulation(prev_idx)?;
                    }
                    KeyCode::Char('r') => {
                        use rand::Rng;
                        let random_idx = rand::thread_rng().gen_range(0..app.total_sims);
                        app.load_simulation(random_idx)?;
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
        .title(" Bayronik: Baryonic Field Emulator")
        .style(Style::default().fg(Color::Cyan).bold());
    let title_text = format!("Model: Trained on 1000 LH sims | Demo: {} CV sims | Values in log-space | Press 'q' to quit", app.total_sims);
    let title = Paragraph::new(title_text).block(title_block);
    frame.render_widget(title, main_layout[0]);
    
    let controls_block = Block::default().borders(Borders::ALL)
        .title("Controls")
        .style(Style::default().fg(Color::Yellow));
    let controls_text = format!(
        "Simulation {}/{} | [←/→] Navigate | [r] Random | [q] Quit",
        app.current_sim_idx + 1,
        app.total_sims
    );
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
    
    let input_widget = MapWidget::new(&app.input_map, "Input: Dark Matter (Mcdm)", Color::Blue);
    frame.render_widget(input_widget, maps_layout[0]);
    
    let output_widget = MapWidget::new(&app.output_map, "Output: Total Matter (Mtot)", Color::Green);
    frame.render_widget(output_widget, maps_layout[1]);
    
    let diff_map = &app.output_map - &app.input_map;
    let diff_widget = MapWidget::new(&diff_map, "Baryonic Effect (Δ)", Color::Red);
    frame.render_widget(diff_widget, maps_layout[2]);
}


const CHAR_RAMP: [char; 16] = [
    ' ', '░', '▒', '▓', '█', '▀', '▄', '▌', '▐', '■', '▪', '●', '◆', '◈', '★', '█'
];

struct MapWidget<'a> {
    tensor: &'a Tensor,
    title: &'a str,
    title_color: Color,
}

impl<'a> MapWidget<'a> {
    fn new(tensor: &'a Tensor, title: &'a str, title_color: Color) -> Self {
        Self { tensor, title, title_color }
    }

    fn downsample(&self, target_width: usize, target_height: usize) -> Option<Vec<f64>> {
        let original_dims = self.tensor.size();
        
        if original_dims.len() < 2 { 
            eprintln!("Error: Tensor has less than 2 dimensions: {:?}", original_dims);
            return None; 
        }
        
        // Handle both [B, C, H, W] and [H, W] formats
        let (original_height, original_width) = if original_dims.len() == 4 {
            (original_dims[2] as usize, original_dims[3] as usize)
        } else if original_dims.len() == 3 {
            (original_dims[1] as usize, original_dims[2] as usize)
        } else {
            (original_dims[0] as usize, original_dims[1] as usize)
        };

        let mut downsampled = vec![0.0; target_width * target_height];
        
        // Squeeze and convert to contiguous before extracting data
        let squeezed = self.tensor.squeeze().contiguous();
        let data_size = squeezed.numel();
        
        let mut data = vec![0.0f32; data_size as usize];
        squeezed.copy_data(&mut data, data_size as usize);
        
        // Sanity check
        if data.len() != original_height * original_width {
            eprintln!("Data size mismatch: expected {}, got {}", 
                     original_height * original_width, data.len());
            return None;
        }

        for y_out in 0..target_height {
            for x_out in 0..target_width {
                let x_in_start = (x_out * original_width) / target_width;
                let x_in_end = ((x_out + 1) * original_width) / target_width;
                let y_in_start = (y_out * original_height) / target_height;
                let y_in_end = ((y_out + 1) * original_height) / target_height;

                let mut sum = 0.0;
                let mut count = 0;
                for y in y_in_start..y_in_end {
                    for x in x_in_start..x_in_end {
                        sum += data[y * original_width + x] as f64;
                        count += 1;
                    }
                }
                if count > 0 {
                    downsampled[y_out * target_width + x_out] = sum / count as f64;
                }
            }
        }
        Some(downsampled)
    }
}

impl Widget for MapWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Compute statistics for display
        let mean = self.tensor.mean(Kind::Float).double_value(&[]);
        let std_val = self.tensor.std(true).double_value(&[]);
        let min_val = self.tensor.min().double_value(&[]);
        let max_val = self.tensor.max().double_value(&[]);
        
        // Create title with stats
        let title_text = format!(
            "{} │ μ={:.3} σ={:.3} ⇵[{:.3},{:.3}]",
            self.title, mean, std_val, min_val, max_val
        );
        
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title_text)
            .style(Style::default().fg(self.title_color).bold());
        
        let inner_area = block.inner(area);
        block.render(area, buf);

        if inner_area.width < 1 || inner_area.height < 1 { return; }

        let target_width = inner_area.width as usize;
        let target_height = inner_area.height as usize;
        
        let downsampled = match self.downsample(target_width, target_height) {
            Some(d) => d,
            None => { // If downsampling fails, render an error and stop.
                Paragraph::new("Error: Could not process tensor data.")
                    .render(inner_area, buf);
                return;
            }
        };

        let min_val = downsampled.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = downsampled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = if max_val > min_val { max_val - min_val } else { 1.0 };

        // Enhanced rendering with color
        for y in 0..target_height {
            for x in 0..target_width {
                let val = downsampled[y * target_width + x];
                let normalized = ((val - min_val) / range).clamp(0.0, 1.0);
                let ramp_index = (normalized * (CHAR_RAMP.len() - 1) as f64).round() as usize;
                let char_to_draw = CHAR_RAMP[ramp_index.min(CHAR_RAMP.len() - 1)];
                
                // Color gradient based on value
                let color = if normalized < 0.2 {
                    Color::DarkGray
                } else if normalized < 0.4 {
                    Color::Gray
                } else if normalized < 0.6 {
                    Color::White
                } else if normalized < 0.8 {
                    Color::Yellow
                } else {
                    Color::Red
                };
                
                let cell_x = inner_area.x + x as u16;
                let cell_y = inner_area.y + y as u16;
                
                if cell_x < area.right() && cell_y < area.bottom() {
                    buf[(cell_x, cell_y)]
                        .set_char(char_to_draw)
                        .set_fg(color);
                }
            }
        }
    }
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

