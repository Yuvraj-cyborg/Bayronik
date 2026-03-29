mod heatmap;

use anyhow::{Context, Result};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use heatmap::HeatmapWidget;
use npyz::NpyFile;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph, Tabs},
};
use std::{
    fs::File,
    io::{self, Stdout},
    time::{Duration, Instant},
};
use tch::{Device, Tensor, kind::Kind};

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Maps,
    NBody,
    Help,
}

impl Tab {
    const ALL: [Tab; 3] = [Tab::Maps, Tab::NBody, Tab::Help];

    fn title(self) -> &'static str {
        match self {
            Tab::Maps => "CAMELS Maps",
            Tab::NBody => "N-Body Sim",
            Tab::Help => "Help",
        }
    }

    fn index(self) -> usize {
        match self {
            Tab::Maps => 0,
            Tab::NBody => 1,
            Tab::Help => 2,
        }
    }
}

struct App {
    tab: Tab,
    input_map: Tensor,
    output_map: Tensor,
    diff_map: Tensor,
    all_input_maps: Vec<f32>,
    current_idx: usize,
    total_sims: usize,
    model: tch::CModule,
    device: Device,

    omega_m: f32,
    sigma_8: f32,
    a_sn1: f32,
    a_agn1: f32,
    a_sn2: f32,
    a_agn2: f32,

    nbody_grid: usize,
    nbody_box: f32,
    nbody_steps: usize,
    nbody_seed: u64,
    is_nbody: bool,
    status: String,
}

impl App {
    fn new() -> Result<Self> {
        println!("Loading dataset and TorchScript model...");

        let npy_path = "../bayronik-model/data/Maps_Mcdm_IllustrisTNG_CV_z=0.00.npy";
        println!("  Loading: {}", npy_path);

        let reader = File::open(npy_path)
            .with_context(|| format!("Failed to open NPY file at '{}'", npy_path))?;
        let npy_file = NpyFile::new(reader).context("Failed to parse NPY")?;
        let all_data: Vec<f32> = npy_file.into_vec()?;
        let total_sims = all_data.len() / (256 * 256);
        println!("  Loaded {} simulations", total_sims);

        let model_path = "../bayronik-model/weights/bayronik_ufno_cond.pt";
        println!("  Loading model: {}", model_path);
        let device = Device::cuda_if_available();
        let model = tch::CModule::load_on_device(model_path, device)
            .with_context(|| format!("Failed to load TorchScript from '{}'", model_path))?;
        println!("  Model loaded on {:?}", device);

        let mut app = Self {
            tab: Tab::Maps,
            input_map: Tensor::zeros(&[1, 1, 256, 256], (Kind::Float, Device::Cpu)),
            output_map: Tensor::zeros(&[1, 1, 256, 256], (Kind::Float, Device::Cpu)),
            diff_map: Tensor::zeros(&[1, 1, 256, 256], (Kind::Float, Device::Cpu)),
            all_input_maps: all_data,
            current_idx: 0,
            total_sims,
            model,
            device,
            omega_m: 0.3,
            sigma_8: 0.8,
            a_sn1: 1.0,
            a_agn1: 1.0,
            a_sn2: 1.0,
            a_agn2: 1.0,
            nbody_grid: 64,
            nbody_box: 100.0,
            nbody_steps: 80,
            nbody_seed: 42,
            is_nbody: false,
            status: String::new(),
        };
        app.load_sim(0)?;
        println!("Ready!");
        Ok(app)
    }

    fn conditions_tensor(&self) -> Tensor {
        Tensor::from_slice(&[
            self.omega_m, self.sigma_8,
            self.a_sn1, self.a_agn1,
            self.a_sn2, self.a_agn2,
        ])
        .reshape(&[1, 6])
        .to_kind(Kind::Float)
    }

    fn run_model(&mut self) -> Result<()> {
        let cond = self.conditions_tensor();
        self.output_map = self
            .model
            .forward_ts(&[self.input_map.to(self.device), cond.to(self.device)])?
            .to(Device::Cpu);
        self.diff_map = &self.output_map - &self.input_map;
        Ok(())
    }

    fn load_sim(&mut self, idx: usize) -> Result<()> {
        let start = idx * 256 * 256;
        let sim_data = &self.all_input_maps[start..start + 256 * 256];
        let raw = Tensor::from_slice(sim_data)
            .reshape(&[1, 1, 256, 256])
            .to_kind(Kind::Float);
        self.input_map = raw.log1p();
        self.current_idx = idx;
        self.is_nbody = false;
        self.run_model()
    }

    fn run_nbody(&mut self) -> Result<()> {
        self.status = "Running N-body...".into();
        let nbody_map = bayronik_core::run_simulation(
            self.nbody_seed,
            self.nbody_grid,
            self.nbody_box,
            0.005,
            self.nbody_steps,
            256,
        );

        let camels_ref = &self.all_input_maps[..256 * 256];
        let camels_log: Vec<f32> = camels_ref.iter().map(|&x| (x + 1.0).ln()).collect();
        let c_mean = camels_log.iter().sum::<f32>() / camels_log.len() as f32;
        let c_std = (camels_log.iter().map(|x| (x - c_mean).powi(2)).sum::<f32>()
            / camels_log.len() as f32)
            .sqrt();

        let mut nb_log: Vec<f32> = nbody_map.iter().map(|&x| (x.max(0.0) + 1.0).ln()).collect();
        let n_mean = nb_log.iter().sum::<f32>() / nb_log.len() as f32;
        let n_std = (nb_log.iter().map(|x| (x - n_mean).powi(2)).sum::<f32>()
            / nb_log.len() as f32)
            .sqrt();

        for v in &mut nb_log {
            *v = (*v - n_mean) / n_std * c_std + c_mean;
        }

        self.input_map = Tensor::from_slice(&nb_log)
            .reshape(&[1, 1, 256, 256])
            .to_kind(Kind::Float);
        self.is_nbody = true;
        self.run_model()?;
        self.status = "N-body done".into();
        Ok(())
    }
}

fn main() -> Result<()> {
    let app = App::new()?;
    let mut terminal = setup_terminal()?;
    let result = run_loop(&mut terminal, app);
    restore_terminal(&mut terminal)?;
    if let Err(e) = result {
        eprintln!("Error: {e:?}");
    }
    Ok(())
}

fn run_loop(terminal: &mut Terminal<CrosstermBackend<Stdout>>, mut app: App) -> Result<()> {
    let tick = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| draw(f, &app))?;
        let timeout = tick.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),

                    // Tab switching
                    KeyCode::Tab => {
                        app.tab = match app.tab {
                            Tab::Maps => Tab::NBody,
                            Tab::NBody => Tab::Help,
                            Tab::Help => Tab::Maps,
                        };
                    }
                    KeyCode::BackTab => {
                        app.tab = match app.tab {
                            Tab::Maps => Tab::Help,
                            Tab::NBody => Tab::Maps,
                            Tab::Help => Tab::NBody,
                        };
                    }
                    KeyCode::Char('1') => app.tab = Tab::Maps,
                    KeyCode::Char('2') => app.tab = Tab::NBody,
                    KeyCode::Char('3') => app.tab = Tab::Help,

                    // Maps tab controls
                    KeyCode::Right | KeyCode::Char('n') if app.tab == Tab::Maps => {
                        let next = (app.current_idx + 1) % app.total_sims;
                        let _ = app.load_sim(next);
                    }
                    KeyCode::Left | KeyCode::Char('p') if app.tab == Tab::Maps => {
                        let prev = if app.current_idx == 0 { app.total_sims - 1 } else { app.current_idx - 1 };
                        let _ = app.load_sim(prev);
                    }
                    KeyCode::Char('r') if app.tab == Tab::Maps => {
                        use rand::Rng;
                        let idx = rand::rng().random_range(0..app.total_sims);
                        let _ = app.load_sim(idx);
                    }

                    // N-Body tab controls
                    KeyCode::Char('g') | KeyCode::Enter if app.tab == Tab::NBody => {
                        app.status = "Generating N-body...".into();
                        terminal.draw(|f| draw(f, &app))?;
                        let _ = app.run_nbody();
                    }
                    KeyCode::Up if app.tab == Tab::NBody => {
                        app.nbody_grid = (app.nbody_grid * 2).min(128);
                    }
                    KeyCode::Down if app.tab == Tab::NBody => {
                        app.nbody_grid = (app.nbody_grid / 2).max(16);
                    }

                    _ => {}
                }
            }
        }
        if last_tick.elapsed() >= tick {
            last_tick = Instant::now();
        }
    }
}

fn draw(frame: &mut Frame, app: &App) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // tabs
            Constraint::Min(0),    // content
            Constraint::Length(1), // footer
        ])
        .split(frame.area());

    // Tab bar
    let tab_titles: Vec<Line> = Tab::ALL.iter().map(|t| Line::from(t.title())).collect();
    let tabs = Tabs::new(tab_titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Bayronik ")
                .title_style(Style::default().fg(Color::Cyan).bold()),
        )
        .highlight_style(Style::default().fg(Color::Yellow).bold())
        .select(app.tab.index());
    frame.render_widget(tabs, root[0]);

    // Content
    match app.tab {
        Tab::Maps => draw_maps_tab(frame, app, root[1]),
        Tab::NBody => draw_nbody_tab(frame, app, root[1]),
        Tab::Help => draw_help_tab(frame, root[1]),
    }

    // Footer
    let footer_text = match app.tab {
        Tab::Maps => "←/→ navigate  r random  1-3 tabs  q quit",
        Tab::NBody => "g/Enter generate  ↑/↓ grid size  1-3 tabs  q quit",
        Tab::Help => "Tab switch  1-3 tabs  q quit",
    };
    let footer = Paragraph::new(footer_text)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center);
    frame.render_widget(footer, root[2]);
}

fn draw_maps_tab(frame: &mut Frame, app: &App, area: Rect) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // params + sample info
            Constraint::Min(0),   // heatmaps
        ])
        .split(area);

    // Info bar
    let source = if app.is_nbody {
        "N-Body Generated".to_string()
    } else {
        format!("CAMELS CV {}/{}", app.current_idx + 1, app.total_sims)
    };
    let info = format!(
        " {} │ Ωm={:.2}  σ8={:.2}  ASN1={:.1}  AAGN1={:.1}  ASN2={:.1}  AAGN2={:.1}",
        source, app.omega_m, app.sigma_8, app.a_sn1, app.a_agn1, app.a_sn2, app.a_agn2,
    );
    let info_block = Block::default().borders(Borders::ALL).title(" Parameters ");
    let info_widget = Paragraph::new(info)
        .block(info_block)
        .style(Style::default().fg(Color::Green));
    frame.render_widget(info_widget, layout[0]);

    // Three heatmaps
    let maps = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(34),
            Constraint::Percentage(33),
        ])
        .split(layout[1]);

    let inp = HeatmapWidget::new(&app.input_map, "Input: Mcdm", Color::Blue);
    frame.render_widget(inp, maps[0]);

    let out = HeatmapWidget::new(&app.output_map, "Output: Mtot", Color::Green);
    frame.render_widget(out, maps[1]);

    let diff = HeatmapWidget::new(&app.diff_map, "Baryonic Effect", Color::Red).diverging();
    frame.render_widget(diff, maps[2]);
}

fn draw_nbody_tab(frame: &mut Frame, app: &App, area: Rect) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5), // controls
            Constraint::Min(0),   // heatmaps
        ])
        .split(area);

    // N-body controls
    let ctrl_text = vec![
        Line::from(vec![
            Span::styled(" Grid: ", Style::default().fg(Color::Yellow)),
            Span::raw(format!("{}³", app.nbody_grid)),
            Span::styled("  Box: ", Style::default().fg(Color::Yellow)),
            Span::raw(format!("{:.0} Mpc/h", app.nbody_box)),
            Span::styled("  Steps: ", Style::default().fg(Color::Yellow)),
            Span::raw(format!("{}", app.nbody_steps)),
            Span::styled("  Seed: ", Style::default().fg(Color::Yellow)),
            Span::raw(format!("{}", app.nbody_seed)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(" Press ", Style::default().fg(Color::DarkGray)),
            Span::styled("g", Style::default().fg(Color::Yellow).bold()),
            Span::styled(" or ", Style::default().fg(Color::DarkGray)),
            Span::styled("Enter", Style::default().fg(Color::Yellow).bold()),
            Span::styled(" to generate │ ", Style::default().fg(Color::DarkGray)),
            Span::styled("↑/↓", Style::default().fg(Color::Yellow).bold()),
            Span::styled(" grid resolution │ ", Style::default().fg(Color::DarkGray)),
            if !app.status.is_empty() {
                Span::styled(&app.status, Style::default().fg(Color::Cyan))
            } else {
                Span::raw("")
            },
        ]),
    ];
    let ctrl_block = Block::default()
        .borders(Borders::ALL)
        .title(" N-Body Simulator ")
        .title_style(Style::default().fg(Color::Cyan));
    let ctrl = Paragraph::new(ctrl_text).block(ctrl_block);
    frame.render_widget(ctrl, layout[0]);

    if app.is_nbody {
        let maps = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(33),
                Constraint::Percentage(34),
                Constraint::Percentage(33),
            ])
            .split(layout[1]);

        let inp = HeatmapWidget::new(&app.input_map, "N-Body DM", Color::Blue);
        frame.render_widget(inp, maps[0]);

        let out = HeatmapWidget::new(&app.output_map, "Emulated Mtot", Color::Green);
        frame.render_widget(out, maps[1]);

        let diff = HeatmapWidget::new(&app.diff_map, "Baryonic Effect", Color::Red).diverging();
        frame.render_widget(diff, maps[2]);
    } else {
        let msg = Paragraph::new("\n  No N-body result yet. Press 'g' or Enter to generate.")
            .style(Style::default().fg(Color::DarkGray))
            .block(Block::default().borders(Borders::ALL).title(" Output "));
        frame.render_widget(msg, layout[1]);
    }
}

fn draw_help_tab(frame: &mut Frame, area: Rect) {
    let help = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Bayronik", Style::default().fg(Color::Cyan).bold()),
            Span::raw(" — Baryonic Field Emulator TUI"),
        ]),
        Line::from(""),
        Line::from(Span::styled("  Navigation", Style::default().fg(Color::Yellow).bold())),
        Line::from("    Tab / Shift+Tab    Switch tabs"),
        Line::from("    1  2  3            Jump to tab"),
        Line::from("    q / Esc            Quit"),
        Line::from(""),
        Line::from(Span::styled("  CAMELS Maps Tab", Style::default().fg(Color::Yellow).bold())),
        Line::from("    ← → / p n          Previous / next simulation"),
        Line::from("    r                  Random simulation"),
        Line::from(""),
        Line::from(Span::styled("  N-Body Tab", Style::default().fg(Color::Yellow).bold())),
        Line::from("    g / Enter          Generate N-body map + run emulator"),
        Line::from("    ↑ / ↓              Increase / decrease grid resolution"),
        Line::from(""),
        Line::from(Span::styled("  Display", Style::default().fg(Color::Yellow).bold())),
        Line::from("    Maps show log-space density fields (log1p transform)"),
        Line::from("    Colormap: Inferno (density) │ Diverging (baryonic effect)"),
        Line::from("    Baryonic Effect = Output − Input (zero-centered)"),
        Line::from(""),
        Line::from(Span::styled("  Architecture", Style::default().fg(Color::Yellow).bold())),
        Line::from("    Model:  Conditional U-FNO with FiLM conditioning"),
        Line::from("    Data:   CAMELS IllustrisTNG LH (15k maps, 1000 sims)"),
        Line::from("    N-body: Rust PM code (Zel'dovich + KDK + CIC + FFT Poisson)"),
        Line::from(""),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Help ")
        .title_style(Style::default().fg(Color::Cyan));
    let widget = Paragraph::new(help).block(block);
    frame.render_widget(widget, area);
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
