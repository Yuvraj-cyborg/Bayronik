use super::grid::Grid;
use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// FFT-based Poisson solver for the gravitational potential.
///
/// Uses batched 1D FFTs along each axis to perform a proper 3D transform,
/// then applies the continuous Green's function phi_k = -delta_k / k^2.
pub struct FftSolver {
    resolution: usize,
    forward_plan: Arc<dyn Fft<f32>>,
    inverse_plan: Arc<dyn Fft<f32>>,
    buffer: Vec<Complex<f32>>,
}

impl FftSolver {
    pub fn new(resolution: usize) -> Self {
        let mut planner = FftPlanner::new();
        let total_cells = resolution * resolution * resolution;

        let forward_plan = planner.plan_fft_forward(resolution);
        let inverse_plan = planner.plan_fft_inverse(resolution);

        Self {
            resolution,
            forward_plan,
            inverse_plan,
            buffer: vec![Complex::new(0.0, 0.0); total_cells],
        }
    }

    /// Solve nabla^2 phi = delta on the grid.
    ///
    /// Transforms density_contrast to Fourier space, applies the Green's
    /// function phi_k = -delta_k / k^2, and transforms back. The result
    /// is written into grid.potential.
    pub fn solve_potential(&mut self, grid: &mut Grid) {
        let n = self.resolution;

        for (i, d) in grid.density_contrast.iter().enumerate() {
            self.buffer[i] = Complex::new(*d, 0.0);
        }

        self.fft3d_forward();

        let k_factor = 2.0 * std::f32::consts::PI / grid.box_size;

        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let kx = freq(ix, n) as f32 * k_factor;
                    let ky = freq(iy, n) as f32 * k_factor;
                    let kz = freq(iz, n) as f32 * k_factor;
                    let k_sq = kx * kx + ky * ky + kz * kz;

                    let idx = (ix * n + iy) * n + iz;
                    if k_sq > 1e-10 {
                        self.buffer[idx] /= -k_sq;
                    } else {
                        self.buffer[idx] = Complex::new(0.0, 0.0);
                    }
                }
            }
        }

        self.fft3d_inverse();

        let norm = 1.0 / (n * n * n) as f32;
        for (i, pot) in grid.potential.iter_mut().enumerate() {
            *pot = self.buffer[i].re * norm;
        }
    }

    /// Forward 3D FFT via batched 1D transforms along z, y, x.
    fn fft3d_forward(&mut self) {
        let n = self.resolution;

        for ix in 0..n {
            for iy in 0..n {
                let off = (ix * n + iy) * n;
                let mut row: Vec<Complex<f32>> = self.buffer[off..off + n].to_vec();
                self.forward_plan.process(&mut row);
                self.buffer[off..off + n].copy_from_slice(&row);
            }
        }

        for ix in 0..n {
            for iz in 0..n {
                let mut col = vec![Complex::new(0.0, 0.0); n];
                for iy in 0..n {
                    col[iy] = self.buffer[(ix * n + iy) * n + iz];
                }
                self.forward_plan.process(&mut col);
                for iy in 0..n {
                    self.buffer[(ix * n + iy) * n + iz] = col[iy];
                }
            }
        }

        for iy in 0..n {
            for iz in 0..n {
                let mut col = vec![Complex::new(0.0, 0.0); n];
                for ix in 0..n {
                    col[ix] = self.buffer[(ix * n + iy) * n + iz];
                }
                self.forward_plan.process(&mut col);
                for ix in 0..n {
                    self.buffer[(ix * n + iy) * n + iz] = col[ix];
                }
            }
        }
    }

    /// Inverse 3D FFT via batched 1D transforms along x, y, z.
    fn fft3d_inverse(&mut self) {
        let n = self.resolution;

        for iy in 0..n {
            for iz in 0..n {
                let mut col = vec![Complex::new(0.0, 0.0); n];
                for ix in 0..n {
                    col[ix] = self.buffer[(ix * n + iy) * n + iz];
                }
                self.inverse_plan.process(&mut col);
                for ix in 0..n {
                    self.buffer[(ix * n + iy) * n + iz] = col[ix];
                }
            }
        }

        for ix in 0..n {
            for iz in 0..n {
                let mut col = vec![Complex::new(0.0, 0.0); n];
                for iy in 0..n {
                    col[iy] = self.buffer[(ix * n + iy) * n + iz];
                }
                self.inverse_plan.process(&mut col);
                for iy in 0..n {
                    self.buffer[(ix * n + iy) * n + iz] = col[iy];
                }
            }
        }

        for ix in 0..n {
            for iy in 0..n {
                let off = (ix * n + iy) * n;
                let mut row: Vec<Complex<f32>> = self.buffer[off..off + n].to_vec();
                self.inverse_plan.process(&mut row);
                self.buffer[off..off + n].copy_from_slice(&row);
            }
        }
    }
}

fn freq(i: usize, n: usize) -> i32 {
    if i > n / 2 {
        i as i32 - n as i32
    } else {
        i as i32
    }
}
