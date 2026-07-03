//! Cosmological initial conditions: Gaussian random field with the linear
//! LCDM power spectrum, displaced with the Zel'dovich approximation.
//!
//! Recipe (standard first-order Lagrangian perturbation theory):
//! 1. Draw unit white noise per grid cell, FFT to get W(k) with <|W|^2> = n^3.
//! 2. Scale each mode by sqrt(P(k) n^3 / V) * D(a_init) so the inverse
//!    transform is a realization of the linear density contrast at a_init.
//! 3. Displacement field psi(k) = i k delta(k) / k^2 (so div psi = -delta).
//! 4. Particles start at grid points q, displaced to x = q + psi(q), with
//!    growing-mode momenta p = a^2 E(a) f(a) psi(q)   (p = a^2 dx/dt, H0 = 1).

use num_complex::Complex;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use crate::cosmology::{Cosmology, LinearPower};
use crate::sim::fft3::{freq, Fft3};
use crate::sim::particle::{Particle, ParticleSet};

impl ParticleSet {
    /// Initialize `grid_res^3` particles in a periodic box of `box_size`
    /// (Mpc/h) as a Zel'dovich realization of the linear power spectrum at
    /// scale factor `a_init`. Particle masses are set so the box mean equals
    /// the comoving matter density Omega_m * rho_crit.
    pub fn initialize_zeldovich(
        &mut self,
        grid_res: usize,
        box_size: f32,
        seed: u64,
        cosmo: &Cosmology,
        a_init: f64,
    ) {
        self.box_size = box_size;
        let n = grid_res;
        let n_total = n * n * n;
        let volume = (box_size as f64).powi(3);

        let power = LinearPower::new(cosmo);
        let d_init = cosmo.growth_d(a_init);

        // 1. White noise field.
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, 1.0).unwrap();
        let mut delta: Vec<Complex<f32>> = (0..n_total)
            .map(|_| Complex::new(normal.sample(&mut rng), 0.0))
            .collect();

        let fft = Fft3::new(n);
        fft.forward(&mut delta);

        // 2. Imprint the power spectrum, scaled to a_init.
        //    delta_k = W_k * sqrt(P(k) n^3 / V) * D(a_init)
        let k_fund = 2.0 * std::f64::consts::PI / box_size as f64;
        let mode_norm = (n_total as f64).sqrt() * d_init;

        // 3. Displacement fields psi_k = i k delta_k / k^2.
        let mut disp = [
            vec![Complex::new(0.0f32, 0.0); n_total],
            vec![Complex::new(0.0f32, 0.0); n_total],
            vec![Complex::new(0.0f32, 0.0); n_total],
        ];

        for ix in 0..n {
            let kx = freq(ix, n) as f64 * k_fund;
            for iy in 0..n {
                let ky = freq(iy, n) as f64 * k_fund;
                for iz in 0..n {
                    let kz = freq(iz, n) as f64 * k_fund;
                    let k2 = kx * kx + ky * ky + kz * kz;
                    let idx = (ix * n + iy) * n + iz;

                    if k2 < 1e-12 {
                        delta[idx] = Complex::new(0.0, 0.0);
                        continue;
                    }

                    let k_mag = k2.sqrt();
                    let amp = (power.p(k_mag) / volume).sqrt() * mode_norm;
                    let d_k = delta[idx] * amp as f32;

                    // psi_j = i k_j / k^2 * delta_k
                    let i_over_k2 = Complex::new(0.0f32, (1.0 / k2) as f32);
                    let base = d_k * i_over_k2;
                    disp[0][idx] = base * kx as f32;
                    disp[1][idx] = base * ky as f32;
                    disp[2][idx] = base * kz as f32;
                }
            }
        }

        // 4. Back to real space (divide by n^3 for the unnormalized inverse).
        let inv_norm = 1.0 / n_total as f32;
        for d in disp.iter_mut() {
            fft.inverse(d);
        }

        // Growing-mode momentum coefficient: p = a^2 E(a) f(a) psi.
        let vel_coef = (a_init * a_init * cosmo.e(a_init) * cosmo.growth_f(a_init)) as f32;

        // Particle mass: total matter in the box divided evenly.
        let particle_mass = (cosmo.mean_matter_density() * volume / n_total as f64) as f32;

        let cell = box_size / n as f32;
        self.particles = Vec::with_capacity(n_total);

        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let idx = (ix * n + iy) * n + iz;
                    let psi = [
                        disp[0][idx].re * inv_norm,
                        disp[1][idx].re * inv_norm,
                        disp[2][idx].re * inv_norm,
                    ];
                    let q = [
                        (ix as f32 + 0.5) * cell,
                        (iy as f32 + 0.5) * cell,
                        (iz as f32 + 0.5) * cell,
                    ];
                    self.particles.push(Particle {
                        position: [
                            (q[0] + psi[0]).rem_euclid(box_size),
                            (q[1] + psi[1]).rem_euclid(box_size),
                            (q[2] + psi[2]).rem_euclid(box_size),
                        ],
                        velocity: [
                            psi[0] * vel_coef,
                            psi[1] * vel_coef,
                            psi[2] * vel_coef,
                        ],
                        force: [0.0; 3],
                        mass: particle_mass,
                    });
                }
            }
        }
    }
}
