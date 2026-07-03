//! Shared in-place 3D FFT built from batched 1D transforms.
//!
//! rustfft transforms are unnormalized in both directions; callers divide by
//! n^3 after an inverse transform to recover physical amplitudes.

use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub struct Fft3 {
    n: usize,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
}

impl Fft3 {
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::new();
        Self {
            n,
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        }
    }

    pub fn forward(&self, buf: &mut [Complex<f32>]) {
        self.transform(buf, true);
    }

    /// Unnormalized inverse; divide by n^3 to complete the round trip.
    pub fn inverse(&self, buf: &mut [Complex<f32>]) {
        self.transform(buf, false);
    }

    fn transform(&self, buf: &mut [Complex<f32>], forward: bool) {
        let n = self.n;
        debug_assert_eq!(buf.len(), n * n * n);
        let plan = if forward { &self.forward } else { &self.inverse };

        // Along z (contiguous rows).
        for ix in 0..n {
            for iy in 0..n {
                let off = (ix * n + iy) * n;
                plan.process(&mut buf[off..off + n]);
            }
        }

        // Along y.
        let mut col = vec![Complex::new(0.0, 0.0); n];
        for ix in 0..n {
            for iz in 0..n {
                for iy in 0..n {
                    col[iy] = buf[(ix * n + iy) * n + iz];
                }
                plan.process(&mut col);
                for iy in 0..n {
                    buf[(ix * n + iy) * n + iz] = col[iy];
                }
            }
        }

        // Along x.
        for iy in 0..n {
            for iz in 0..n {
                for ix in 0..n {
                    col[ix] = buf[(ix * n + iy) * n + iz];
                }
                plan.process(&mut col);
                for ix in 0..n {
                    buf[(ix * n + iy) * n + iz] = col[ix];
                }
            }
        }
    }
}

/// Signed integer frequency for FFT index i on an n-point grid.
pub fn freq(i: usize, n: usize) -> i32 {
    if i > n / 2 {
        i as i32 - n as i32
    } else {
        i as i32
    }
}
