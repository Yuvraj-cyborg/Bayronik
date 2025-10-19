//! This module will contain the logic for solving gravity on the grid using FFT.

use super::grid::Grid;
use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// A solver that computes the gravitational potential from a density field
/// using a Fast Fourier Transform (FFT).
pub struct FftSolver {
    resolution: usize,
    forward_plan: Arc<dyn Fft<f32>>,
    inverse_plan: Arc<dyn Fft<f32>>,
    // Buffer to hold the data in complex form for the FFT
    fft_buffer: Vec<Complex<f32>>,
}

impl FftSolver {
    /// Creates a new FFT solver for a given grid resolution.
    pub fn new(resolution: usize) -> Self {
        let mut planner = FftPlanner::new();
        let total_cells = resolution * resolution * resolution;

        // Create plans for forward and inverse FFTs.
        // These are pre-computed for efficiency.
        let forward_plan = planner.plan_fft_forward(total_cells);
        let inverse_plan = planner.plan_fft_inverse(total_cells);

        Self {
            resolution,
            forward_plan,
            inverse_plan,
            fft_buffer: vec![Complex::new(0.0, 0.0); total_cells],
        }
    }

    /// Solves for the gravitational potential given the density grid.
    ///
    /// This is the core of the Particle-Mesh method. The steps are:
    /// 1. Forward FFT of the density grid.
    /// 2. Apply the Green's function for gravity in Fourier space.
    ///    (This is just a multiplication).
    /// 3. Inverse FFT to get the potential back in real space.
    pub fn solve_potential(&mut self, grid: &mut Grid) {
        // Step 1: Copy density data to our complex buffer
        for (i, density_val) in grid.density_contrast.iter().enumerate() {
            self.fft_buffer[i] = Complex::new(*density_val, 0.0);
        }

        // Step 2: Perform the forward FFT (in-place)
        self.forward_plan.process(&mut self.fft_buffer);

        // Step 3: Apply the Green's function in Fourier space.
        // The potential_k = - density_k / (k^2)
        // We need to calculate the wave vector 'k' for each mode.
        let _n = self.resolution as f32;
        let k_factor = 2.0 * std::f32::consts::PI / grid.box_size;

        for i in 0..self.fft_buffer.len() {
            let (kx, ky, kz) = self.get_k_vector(i, k_factor);
            let k_squared = kx * kx + ky * ky + kz * kz;

            // Avoid division by zero for the k=0 mode (the DC component).
            // The mean density contrast is zero, so this mode should be zero anyway.
            if k_squared > 1e-6 {
                self.fft_buffer[i] /= -k_squared;
            } else {
                self.fft_buffer[i] = Complex::new(0.0, 0.0);
            }
        }
        
        // The Nyquist frequency requires special handling in some FFT schemes,
        // but for our purposes, this approximation is sufficient.

        // Step 4: Perform the inverse FFT to get the potential in real space
        self.inverse_plan.process(&mut self.fft_buffer);

        // Step 5: Copy the real part of the result back to the grid's potential field
        // and normalize it.
        let normalization = 1.0 / (self.resolution * self.resolution * self.resolution) as f32;
        for i in 0..grid.potential.len() {
            grid.potential[i] = self.fft_buffer[i].re * normalization;
        }
    }

    /// Helper function to get the (kx, ky, kz) wave vector for a given index
    /// in the 1D flattened FFT output array.
    fn get_k_vector(&self, index: usize, k_factor: f32) -> (f32, f32, f32) {
        let n = self.resolution;
        let iz = index % n;
        let iy = (index / n) % n;
        let ix = index / (n * n);

        // FFT frequencies need to be "shifted" to represent negative and positive frequencies.
        // For a dimension of size N, indices 0..N/2 correspond to positive frequencies,
        // and N/2..N-1 correspond to negative frequencies.
        let kx = if ix > n / 2 { ix as i32 - n as i32 } else { ix as i32 } as f32;
        let ky = if iy > n / 2 { iy as i32 - n as i32 } else { iy as i32 } as f32;
        let kz = if iz > n / 2 { iz as i32 - n as i32 } else { iz as i32 } as f32;

        (kx * k_factor, ky * k_factor, kz * k_factor)
    }
}

