use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color as RC, Style};
use ratatui::widgets::{Block, Borders, Widget};
use tch::Tensor;

/// Half-block heatmap with true-color inferno colormap.
/// Uses ▀ (upper half block) to pack 2 vertical pixels per terminal cell,
/// giving 2x vertical resolution with independent fg (top) and bg (bottom) colors.
pub struct HeatmapWidget<'a> {
    tensor: &'a Tensor,
    title: &'a str,
    border_color: RC,
    diverging: bool,
}

impl<'a> HeatmapWidget<'a> {
    pub fn new(tensor: &'a Tensor, title: &'a str, border_color: RC) -> Self {
        Self { tensor, title, border_color, diverging: false }
    }

    pub fn diverging(mut self) -> Self {
        self.diverging = true;
        self
    }

    fn extract_flat(&self) -> Option<(Vec<f32>, usize, usize)> {
        let dims = self.tensor.size();
        let (h, w) = match dims.len() {
            4 => (dims[2] as usize, dims[3] as usize),
            3 => (dims[1] as usize, dims[2] as usize),
            2 => (dims[0] as usize, dims[1] as usize),
            _ => return None,
        };
        let sq = self.tensor.squeeze().contiguous();
        let n = sq.numel() as usize;
        let mut buf = vec![0f32; n];
        sq.copy_data(&mut buf, n);
        if buf.len() == h * w { Some((buf, w, h)) } else { None }
    }

    fn downsample(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; dw * dh];
        for dy in 0..dh {
            for dx in 0..dw {
                let x0 = (dx * sw) / dw;
                let x1 = ((dx + 1) * sw) / dw;
                let y0 = (dy * sh) / dh;
                let y1 = ((dy + 1) * sh) / dh;
                let mut sum = 0.0f64;
                let mut cnt = 0u32;
                for y in y0..y1 {
                    for x in x0..x1 {
                        sum += src[y * sw + x] as f64;
                        cnt += 1;
                    }
                }
                out[dy * dw + dx] = if cnt > 0 { (sum / cnt as f64) as f32 } else { 0.0 };
            }
        }
        out
    }
}

impl Widget for HeatmapWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let stats = tensor_stats(self.tensor);
        let title = format!(
            " {} │ [{:.2}, {:.2}] μ={:.2} ",
            self.title, stats.min, stats.max, stats.mean
        );

        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(Style::default().fg(self.border_color));

        let inner = block.inner(area);
        block.render(area, buf);

        if inner.width < 2 || inner.height < 2 {
            return;
        }

        let (data, sw, sh) = match self.extract_flat() {
            Some(d) => d,
            None => return,
        };

        let pw = inner.width as usize;
        let ph = inner.height as usize * 2; // 2 pixels per cell vertically
        let pixels = Self::downsample(&data, sw, sh, pw, ph);

        let (vmin, vmax) = if self.diverging {
            let abs_max = stats.min.abs().max(stats.max.abs()).max(1e-8);
            (-abs_max, abs_max)
        } else {
            (stats.min, stats.max)
        };
        let range = (vmax - vmin).max(1e-8);

        for cy in 0..inner.height as usize {
            for cx in 0..pw {
                let top_idx = (cy * 2) * pw + cx;
                let bot_idx = (cy * 2 + 1) * pw + cx;

                let t_top = ((pixels[top_idx] - vmin) / range).clamp(0.0, 1.0);
                let t_bot = if bot_idx < pixels.len() {
                    ((pixels[bot_idx] - vmin) / range).clamp(0.0, 1.0)
                } else {
                    t_top
                };

                let fg = if self.diverging { diverging_color(t_top) } else { inferno(t_top) };
                let bg = if self.diverging { diverging_color(t_bot) } else { inferno(t_bot) };

                let sx = inner.x + cx as u16;
                let sy = inner.y + cy as u16;
                if sx < area.right() && sy < area.bottom() {
                    buf[(sx, sy)]
                        .set_char('▀')
                        .set_fg(fg)
                        .set_bg(bg);
                }
            }
        }
    }
}

pub struct Stats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
}

pub fn tensor_stats(t: &Tensor) -> Stats {
    Stats {
        min: t.min().double_value(&[]) as f32,
        max: t.max().double_value(&[]) as f32,
        mean: t.mean(tch::Kind::Float).double_value(&[]) as f32,
        std: t.std(true).double_value(&[]) as f32,
    }
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round().clamp(0.0, 255.0) as u8
}

fn color_lerp(stops: &[(f32, u8, u8, u8)], t: f32) -> RC {
    let t = t.clamp(0.0, 1.0);
    if t <= stops[0].0 {
        return RC::Rgb(stops[0].1, stops[0].2, stops[0].3);
    }
    for i in 1..stops.len() {
        if t <= stops[i].0 {
            let s = (t - stops[i - 1].0) / (stops[i].0 - stops[i - 1].0);
            let (_, r0, g0, b0) = stops[i - 1];
            let (_, r1, g1, b1) = stops[i];
            return RC::Rgb(lerp_u8(r0, r1, s), lerp_u8(g0, g1, s), lerp_u8(b0, b1, s));
        }
    }
    let last = stops.last().unwrap();
    RC::Rgb(last.1, last.2, last.3)
}

fn inferno(t: f32) -> RC {
    color_lerp(
        &[
            (0.00, 0, 0, 4),
            (0.13, 31, 12, 72),
            (0.25, 85, 15, 109),
            (0.38, 136, 34, 106),
            (0.50, 186, 54, 85),
            (0.63, 227, 89, 51),
            (0.75, 249, 140, 10),
            (0.88, 249, 201, 50),
            (1.00, 252, 255, 164),
        ],
        t,
    )
}

fn diverging_color(t: f32) -> RC {
    color_lerp(
        &[
            (0.00, 20, 60, 180),
            (0.30, 40, 40, 100),
            (0.50, 15, 15, 15),
            (0.70, 100, 35, 35),
            (1.00, 200, 30, 30),
        ],
        t,
    )
}
