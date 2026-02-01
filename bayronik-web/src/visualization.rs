use egui::{Color32, ColorImage, Vec2};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Colormap {
    Viridis,
    Inferno,
    Plasma,
    Coolwarm,
}

impl Colormap {
    pub fn sample(&self, t: f32) -> Color32 {
        let t = t.clamp(0.0, 1.0);

        match self {
            Colormap::Viridis => viridis(t),
            Colormap::Inferno => inferno(t),
            Colormap::Plasma => plasma(t),
            Colormap::Coolwarm => coolwarm(t),
        }
    }
}

#[allow(clippy::approx_constant)]
fn viridis(t: f32) -> Color32 {
    let colors = [
        (0.267, 0.004, 0.329),
        (0.282, 0.140, 0.458),
        (0.254, 0.265, 0.530),
        (0.207, 0.372, 0.553),
        (0.164, 0.471, 0.558),
        (0.128, 0.567, 0.551),
        (0.135, 0.659, 0.518),
        (0.267, 0.749, 0.441),
        (0.478, 0.821, 0.318),
        (0.741, 0.873, 0.150),
        (0.993, 0.906, 0.144),
    ];
    interpolate_colors(&colors, t)
}

fn inferno(t: f32) -> Color32 {
    let colors = [
        (0.001, 0.000, 0.014),
        (0.122, 0.047, 0.283),
        (0.329, 0.059, 0.406),
        (0.531, 0.135, 0.376),
        (0.716, 0.215, 0.292),
        (0.863, 0.337, 0.178),
        (0.954, 0.506, 0.059),
        (0.988, 0.699, 0.106),
        (0.961, 0.893, 0.336),
        (0.988, 1.000, 0.644),
    ];
    interpolate_colors(&colors, t)
}

fn plasma(t: f32) -> Color32 {
    let colors = [
        (0.050, 0.030, 0.528),
        (0.294, 0.012, 0.615),
        (0.492, 0.012, 0.658),
        (0.658, 0.134, 0.588),
        (0.798, 0.280, 0.470),
        (0.899, 0.424, 0.362),
        (0.963, 0.569, 0.259),
        (0.988, 0.722, 0.145),
        (0.940, 0.975, 0.131),
    ];
    interpolate_colors(&colors, t)
}

fn coolwarm(t: f32) -> Color32 {
    let colors = [
        (0.230, 0.299, 0.754),
        (0.552, 0.689, 0.998),
        (0.866, 0.866, 0.866),
        (0.958, 0.606, 0.476),
        (0.706, 0.016, 0.150),
    ];
    interpolate_colors(&colors, t)
}

fn interpolate_colors(colors: &[(f32, f32, f32)], t: f32) -> Color32 {
    let n = colors.len() - 1;
    let idx = (t * n as f32).floor() as usize;
    let idx = idx.min(n - 1);
    let frac = t * n as f32 - idx as f32;

    let (r1, g1, b1) = colors[idx];
    let (r2, g2, b2) = colors[idx + 1];

    let r = (r1 + (r2 - r1) * frac) * 255.0;
    let g = (g1 + (g2 - g1) * frac) * 255.0;
    let b = (b1 + (b2 - b1) * frac) * 255.0;

    Color32::from_rgb(r as u8, g as u8, b as u8)
}

pub fn array_to_colorimage(
    data: &[f32],
    width: usize,
    height: usize,
    colormap: &Colormap,
) -> ColorImage {
    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_val - min_val).max(1e-8);

    let pixels: Vec<Color32> = data
        .iter()
        .map(|&v| {
            let t = (v - min_val) / range;
            colormap.sample(t)
        })
        .collect();

    ColorImage {
        size: [width, height],
        source_size: Vec2::new(width as f32, height as f32),
        pixels,
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
