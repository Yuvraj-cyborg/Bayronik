use egui::{Color32, ColorImage, Rect, Ui, Vec2};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Colormap {
    Viridis,
    Inferno,
    Plasma,
    Magma,
    Coolwarm,
    DarkDiverging,
}

impl Colormap {
    pub const ALL: [Colormap; 6] = [
        Colormap::Inferno,
        Colormap::Viridis,
        Colormap::Plasma,
        Colormap::Magma,
        Colormap::Coolwarm,
        Colormap::DarkDiverging,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Colormap::Viridis => "Viridis",
            Colormap::Inferno => "Inferno",
            Colormap::Plasma => "Plasma",
            Colormap::Magma => "Magma",
            Colormap::Coolwarm => "Coolwarm",
            Colormap::DarkDiverging => "Dark Diverging",
        }
    }

    pub fn sample(&self, t: f32) -> Color32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Colormap::Viridis => viridis(t),
            Colormap::Inferno => inferno(t),
            Colormap::Plasma => plasma(t),
            Colormap::Magma => magma(t),
            Colormap::Coolwarm => coolwarm(t),
            Colormap::DarkDiverging => dark_diverging(t),
        }
    }
}

// ---- Heatmap rendering ----

pub fn array_to_colorimage(
    data: &[f32],
    width: usize,
    height: usize,
    colormap: &Colormap,
    log_scale: bool,
) -> ColorImage {
    array_to_colorimage_ex(data, width, height, colormap, log_scale, false)
}

/// Diverging variant: centers the colormap at zero so negative=blue, zero=center, positive=red.
pub fn array_to_colorimage_diverging(
    data: &[f32],
    width: usize,
    height: usize,
    colormap: &Colormap,
) -> ColorImage {
    array_to_colorimage_ex(data, width, height, colormap, false, true)
}

fn array_to_colorimage_ex(
    data: &[f32],
    width: usize,
    height: usize,
    colormap: &Colormap,
    log_scale: bool,
    diverging: bool,
) -> ColorImage {
    let transformed: Vec<f32> = if log_scale {
        data.iter().map(|&v| (v.max(0.0) + 1.0).log10()).collect()
    } else {
        data.to_vec()
    };

    let vals: Vec<f32> = transformed.iter().filter(|v| v.is_finite()).cloned().collect();
    if vals.is_empty() {
        let pixels = vec![Color32::BLACK; width * height];
        return ColorImage {
            size: [width, height],
            source_size: Vec2::new(width as f32, height as f32),
            pixels,
        };
    }

    let min_val = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let pixels: Vec<Color32> = if diverging {
        // Center at zero: t=0.5 is zero, symmetric around it
        let abs_max = min_val.abs().max(max_val.abs()).max(1e-8);
        transformed
            .iter()
            .map(|&v| {
                if !v.is_finite() {
                    Color32::BLACK
                } else {
                    let t = (v / abs_max + 1.0) * 0.5; // maps [-abs_max, +abs_max] → [0, 1]
                    colormap.sample(t)
                }
            })
            .collect()
    } else {
        let range = (max_val - min_val).max(1e-8);
        transformed
            .iter()
            .map(|&v| {
                if !v.is_finite() {
                    Color32::BLACK
                } else {
                    let t = (v - min_val) / range;
                    colormap.sample(t)
                }
            })
            .collect()
    };

    ColorImage {
        size: [width, height],
        source_size: Vec2::new(width as f32, height as f32),
        pixels,
    }
}

/// Render a vertical color bar beside a map image.
pub fn draw_color_bar(
    ui: &mut Ui,
    colormap: &Colormap,
    min_val: f64,
    max_val: f64,
    log_scale: bool,
    height: f32,
) {
    let bar_width = 18.0;
    let n_steps = 128usize;

    let (response, painter) =
        ui.allocate_painter(Vec2::new(bar_width + 50.0, height), egui::Sense::hover());
    let rect = response.rect;
    let bar_rect = Rect::from_min_size(rect.min, Vec2::new(bar_width, height));

    for i in 0..n_steps {
        let t = 1.0 - i as f32 / n_steps as f32;
        let color = colormap.sample(t);
        let y0 = bar_rect.min.y + (i as f32 / n_steps as f32) * height;
        let y1 = bar_rect.min.y + ((i + 1) as f32 / n_steps as f32) * height;
        painter.rect_filled(
            Rect::from_min_max(
                egui::pos2(bar_rect.min.x, y0),
                egui::pos2(bar_rect.max.x, y1),
            ),
            0.0,
            color,
        );
    }

    painter.rect_stroke(
        bar_rect,
        0.0,
        egui::Stroke::new(1.0, Color32::GRAY),
        egui::StrokeKind::Outside,
    );

    let label_x = bar_rect.max.x + 4.0;
    let n_labels = 5;
    for i in 0..=n_labels {
        let frac = i as f64 / n_labels as f64;
        let val = min_val + (max_val - min_val) * frac;
        let label = if log_scale {
            format!("1e{:.1}", val)
        } else {
            format_value(val)
        };
        let y = bar_rect.max.y - frac as f32 * height;
        painter.text(
            egui::pos2(label_x, y),
            egui::Align2::LEFT_CENTER,
            label,
            egui::FontId::proportional(10.0),
            Color32::LIGHT_GRAY,
        );
    }
}

fn format_value(v: f64) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        "0".into()
    } else if abs >= 1e6 || abs < 0.01 {
        format!("{:.1e}", v)
    } else {
        format!("{:.2}", v)
    }
}

pub fn compute_statistics(data: &[f32]) -> (f32, f32, f32, f32) {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (mean, std, min, max)
}

// ---- Colormap LUT functions ----

fn interpolate_colors(colors: &[(f32, f32, f32)], t: f32) -> Color32 {
    let n = colors.len() - 1;
    let idx = (t * n as f32).floor() as usize;
    let idx = idx.min(n - 1);
    let frac = t * n as f32 - idx as f32;
    let (r1, g1, b1) = colors[idx];
    let (r2, g2, b2) = colors[idx + 1];
    Color32::from_rgb(
        ((r1 + (r2 - r1) * frac) * 255.0) as u8,
        ((g1 + (g2 - g1) * frac) * 255.0) as u8,
        ((b1 + (b2 - b1) * frac) * 255.0) as u8,
    )
}

#[allow(clippy::approx_constant)]
fn viridis(t: f32) -> Color32 {
    interpolate_colors(
        &[
            (0.267, 0.004, 0.329), (0.282, 0.140, 0.458), (0.254, 0.265, 0.530),
            (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
            (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
            (0.741, 0.873, 0.150), (0.993, 0.906, 0.144),
        ],
        t,
    )
}

fn inferno(t: f32) -> Color32 {
    interpolate_colors(
        &[
            (0.001, 0.000, 0.014), (0.122, 0.047, 0.283), (0.329, 0.059, 0.406),
            (0.531, 0.135, 0.376), (0.716, 0.215, 0.292), (0.863, 0.337, 0.178),
            (0.954, 0.506, 0.059), (0.988, 0.699, 0.106), (0.961, 0.893, 0.336),
            (0.988, 1.000, 0.644),
        ],
        t,
    )
}

fn plasma(t: f32) -> Color32 {
    interpolate_colors(
        &[
            (0.050, 0.030, 0.528), (0.294, 0.012, 0.615), (0.492, 0.012, 0.658),
            (0.658, 0.134, 0.588), (0.798, 0.280, 0.470), (0.899, 0.424, 0.362),
            (0.963, 0.569, 0.259), (0.988, 0.722, 0.145), (0.940, 0.975, 0.131),
        ],
        t,
    )
}

fn magma(t: f32) -> Color32 {
    interpolate_colors(
        &[
            (0.001, 0.000, 0.014), (0.099, 0.063, 0.254), (0.270, 0.063, 0.425),
            (0.451, 0.081, 0.440), (0.632, 0.150, 0.400), (0.810, 0.260, 0.340),
            (0.930, 0.420, 0.330), (0.980, 0.610, 0.440), (0.990, 0.810, 0.640),
            (0.987, 0.991, 0.750),
        ],
        t,
    )
}

fn coolwarm(t: f32) -> Color32 {
    interpolate_colors(
        &[
            (0.230, 0.299, 0.754), (0.552, 0.689, 0.998),
            (0.866, 0.866, 0.866),
            (0.958, 0.606, 0.476), (0.706, 0.016, 0.150),
        ],
        t,
    )
}

/// Matches Streamlit's DARK_DIVERGING: blue → dark → red with near-black center.
fn dark_diverging(t: f32) -> Color32 {
    // Ported from webapp.py DARK_DIVERGING:
    // [0.0, rgb(20,60,180)]  [0.3, rgb(40,40,100)]  [0.5, rgb(15,15,15)]
    // [0.7, rgb(100,35,35)]  [1.0, rgb(200,30,30)]
    interpolate_colors(
        &[
            (20.0 / 255.0, 60.0 / 255.0, 180.0 / 255.0),  // deep blue
            (40.0 / 255.0, 40.0 / 255.0, 100.0 / 255.0),   // dark blue
            (15.0 / 255.0, 15.0 / 255.0, 15.0 / 255.0),    // near-black center
            (100.0 / 255.0, 35.0 / 255.0, 35.0 / 255.0),   // dark red
            (200.0 / 255.0, 30.0 / 255.0, 30.0 / 255.0),   // bright red
        ],
        t,
    )
}
