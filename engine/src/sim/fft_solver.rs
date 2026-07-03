use super::fft3::{freq, Fft3};
use super::grid::Grid;
use num_complex::Complex;

/// FFT-based Poisson solver for the comoving gravitational potential.
///
/// Solves nabla^2 phi = prefactor * delta with the continuous Green's
/// function phi_k = -prefactor * delta_k / k^2. For cosmological runs the
/// prefactor is (3/2) Omega_m / a (H0 = 1 units); pass 1.0 to solve the
/// plain normalized Poisson equation.
pub struct FftSolver {
    resolution: usize,
    fft: Fft3,
    buffer: Vec<Complex<f32>>,
}

impl FftSolver {
    pub fn new(resolution: usize) -> Self {
        let total_cells = resolution * resolution * resolution;
        Self {
            resolution,
            fft: Fft3::new(resolution),
            buffer: vec![Complex::new(0.0, 0.0); total_cells],
        }
    }

    /// Solve nabla^2 phi = prefactor * delta on the grid; the result is
    /// written into grid.potential.
    pub fn solve_potential(&mut self, grid: &mut Grid, prefactor: f32) {
        let n = self.resolution;

        for (i, d) in grid.density_contrast.iter().enumerate() {
            self.buffer[i] = Complex::new(*d, 0.0);
        }

        self.fft.forward(&mut self.buffer);

        let k_factor = 2.0 * std::f32::consts::PI / grid.box_size;

        for ix in 0..n {
            let kx = freq(ix, n) as f32 * k_factor;
            for iy in 0..n {
                let ky = freq(iy, n) as f32 * k_factor;
                for iz in 0..n {
                    let kz = freq(iz, n) as f32 * k_factor;
                    let k_sq = kx * kx + ky * ky + kz * kz;

                    let idx = (ix * n + iy) * n + iz;
                    if k_sq > 1e-10 {
                        self.buffer[idx] *= -prefactor / k_sq;
                    } else {
                        self.buffer[idx] = Complex::new(0.0, 0.0);
                    }
                }
            }
        }

        self.fft.inverse(&mut self.buffer);

        let norm = 1.0 / (n * n * n) as f32;
        for (i, pot) in grid.potential.iter_mut().enumerate() {
            *pot = self.buffer[i].re * norm;
        }
    }
}
