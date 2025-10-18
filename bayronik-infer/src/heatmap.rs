use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color as RatatuiColor;
use ratatui::widgets::{Block, Borders, Widget};
use tch::Tensor;

pub struct HeatmapWidget<'a> {
    tensor: &'a Tensor,
    title: &'a str,
    border_color: RatatuiColor,
}

impl<'a> HeatmapWidget<'a> {
    pub fn new(tensor: &'a Tensor, title: &'a str, border_color: RatatuiColor) -> Self {
        Self {
            tensor,
            title,
            border_color,
        }
    }

    fn get_data(&self, width: usize, height: usize) -> Option<Vec<f64>> {
        let original_dims = self.tensor.size();
        if original_dims.len() < 2 {
            return None;
        }

        let (orig_h, orig_w) = if original_dims.len() == 4 {
            (original_dims[2] as usize, original_dims[3] as usize)
        } else if original_dims.len() == 3 {
            (original_dims[1] as usize, original_dims[2] as usize)
        } else {
            (original_dims[0] as usize, original_dims[1] as usize)
        };

        let squeezed = self.tensor.squeeze().contiguous();
        let data_size = squeezed.numel();
        let mut data = vec![0.0f32; data_size as usize];
        squeezed.copy_data(&mut data, data_size as usize);

        if data.len() != orig_h * orig_w {
            return None;
        }

        let mut downsampled = vec![0.0; width * height];
        for y_out in 0..height {
            for x_out in 0..width {
                let x_in_start = (x_out * orig_w) / width;
                let x_in_end = ((x_out + 1) * orig_w) / width;
                let y_in_start = (y_out * orig_h) / height;
                let y_in_end = ((y_out + 1) * orig_h) / height;

                let mut sum = 0.0;
                let mut count = 0;
                for y in y_in_start..y_in_end {
                    for x in x_in_start..x_in_end {
                        sum += data[y * orig_w + x] as f64;
                        count += 1;
                    }
                }
                if count > 0 {
                    downsampled[y_out * width + x_out] = sum / count as f64;
                }
            }
        }
        Some(downsampled)
    }
}

impl Widget for HeatmapWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mean = self.tensor.mean(tch::Kind::Float).double_value(&[]);
        let std_val = self.tensor.std(true).double_value(&[]);
        let min_val = self.tensor.min().double_value(&[]);
        let max_val = self.tensor.max().double_value(&[]);

        let title_text = format!(
            "{} │ μ={:.2} σ={:.2} [{:.2},{:.2}]",
            self.title, mean, std_val, min_val, max_val
        );

        let block = Block::default()
            .borders(Borders::ALL)
            .title(title_text)
            .border_style(ratatui::style::Style::default().fg(self.border_color));

        let inner_area = block.inner(area);
        block.render(area, buf);

        if inner_area.width < 4 || inner_area.height < 4 {
            return;
        }

        let plot_width = inner_area.width as usize * 2;
        let plot_height = inner_area.height as usize * 4;

        let data = match self.get_data(plot_width, plot_height) {
            Some(d) => d,
            None => return,
        };

        let data_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let data_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = if data_max > data_min { data_max - data_min } else { 1.0 };

        const BRAILLE: [[u8; 2]; 4] = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]];

        for cell_y in 0..inner_area.height {
            for cell_x in 0..inner_area.width {
                let mut braille_char = 0x2800u32;

                for dot_y in 0..4 {
                    for dot_x in 0..2 {
                        let pixel_x = cell_x as usize * 2 + dot_x;
                        let pixel_y = cell_y as usize * 4 + dot_y;

                        if pixel_x < plot_width && pixel_y < plot_height {
                            let idx = pixel_y * plot_width + pixel_x;
                            let val = data[idx];
                            let normalized = ((val - data_min) / range).clamp(0.0, 1.0);

                            if normalized > 0.25 {
                                braille_char |= BRAILLE[dot_y][dot_x] as u32;
                            }
                        }
                    }
                }

                let ch = char::from_u32(braille_char).unwrap_or('?');
                    let color = {
                    let mut sum = 0.0;
                    let mut count = 0;
                    for dot_y in 0..4 {
                        for dot_x in 0..2 {
                            let pixel_x = cell_x as usize * 2 + dot_x;
                            let pixel_y = cell_y as usize * 4 + dot_y;
                            if pixel_x < plot_width && pixel_y < plot_height {
                                sum += data[pixel_y * plot_width + pixel_x];
                                count += 1;
                            }
                        }
                    }
                    let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                    let normalized = ((avg - data_min) / range).clamp(0.0, 1.0);

                    if normalized < 0.2 {
                        RatatuiColor::Rgb(50, 50, 80)
                    } else if normalized < 0.4 {
                        RatatuiColor::Rgb(80, 100, 180)
                    } else if normalized < 0.6 {
                        RatatuiColor::Rgb(150, 180, 220)
                    } else if normalized < 0.8 {
                        RatatuiColor::Rgb(255, 200, 100)
                    } else {
                        RatatuiColor::Rgb(255, 100, 100)
                    }
                };

                let screen_x = inner_area.x + cell_x;
                let screen_y = inner_area.y + cell_y;

                if screen_x < area.right() && screen_y < area.bottom() {
                    buf[(screen_x, screen_y)]
                        .set_char(ch)
                        .set_fg(color);
                }
            }
        }
    }
}

