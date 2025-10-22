use num_complex::Complex;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rustfft::FftPlanner;

pub struct Particle {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub force: [f32; 3],
    pub mass: f32,
}

pub struct ParticleSet {
    pub particles: Vec<Particle>,
    pub box_size: f32,
}

impl ParticleSet {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            box_size: 0.0,
        }
    }

    /// Initializes particle positions and velocities using the Zel'dovich Approximation.
    /// This is a physically motivated method based on linear perturbation theory.
    /// Uses proper 3D FFT via batched 1D transforms along each axis.
    pub fn initialize_zeldovich(&mut self, grid_res: usize, box_size: f32) {
        self.box_size = box_size;
        let n = grid_res;
        let num_particles = n * n * n;
        self.particles = Vec::with_capacity(num_particles);
        let mut rng = rand::rng();

        // Set up 1D FFT plans for 3D transform (batched along each axis)
        let mut planner = FftPlanner::new();
        let fwd_plan = planner.plan_fft_forward(n);
        let inv_plan = planner.plan_fft_inverse(n);

        // Generate Gaussian random field in real space, then FFT to get density_k
        let mut density = vec![Complex::new(0.0, 0.0); num_particles];
        
        // Initialize with Gaussian random values
        let normal = Normal::new(0.0, 1.0).unwrap();
        for val in &mut density {
            *val = Complex::new(normal.sample(&mut rng) as f32, 0.0);
        }

        // Forward 3D FFT: apply 1D FFT along each dimension
        // Step 1: FFT along z-axis (innermost, contiguous)
        for ix in 0..n {
            for iy in 0..n {
                let offset = (ix * n + iy) * n;
                let mut slice: Vec<Complex<f32>> = density[offset..offset + n].to_vec();
                fwd_plan.process(&mut slice);
                density[offset..offset + n].copy_from_slice(&slice);
            }
        }

        // Step 2: FFT along y-axis
        for ix in 0..n {
            for iz in 0..n {
                let mut slice = vec![Complex::new(0.0, 0.0); n];
                for iy in 0..n {
                    slice[iy] = density[(ix * n + iy) * n + iz];
                }
                fwd_plan.process(&mut slice);
                for iy in 0..n {
                    density[(ix * n + iy) * n + iz] = slice[iy];
                }
            }
        }

        // Step 3: FFT along x-axis
        for iy in 0..n {
            for iz in 0..n {
                let mut slice = vec![Complex::new(0.0, 0.0); n];
                for ix in 0..n {
                    slice[ix] = density[(ix * n + iy) * n + iz];
                }
                fwd_plan.process(&mut slice);
                for ix in 0..n {
                    density[(ix * n + iy) * n + iz] = slice[ix];
                }
            }
        }

        // Now density contains density_k in Fourier space
        // Apply power spectrum weighting P(k) ~ k^(-1.5) for CDM-like clustering
        let k_fundamental = 2.0 * std::f32::consts::PI / box_size;
        
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let kx = if ix > n / 2 { ix as i32 - n as i32 } else { ix as i32 } as f32 * k_fundamental;
                    let ky = if iy > n / 2 { iy as i32 - n as i32 } else { iy as i32 } as f32 * k_fundamental;
                    let kz = if iz > n / 2 { iz as i32 - n as i32 } else { iz as i32 } as f32 * k_fundamental;
                    
                    let k_mag_sq = kx * kx + ky * ky + kz * kz;
                    let idx = (ix * n + iy) * n + iz;
                    
                    if k_mag_sq < 1e-6 {
                        // DC mode: set to zero (no mean overdensity)
                        density[idx] = Complex::new(0.0, 0.0);
                    } else {
                        // Apply P(k) ~ k^(-1.5) weighting for realistic structure
                        let p_k = k_mag_sq.powf(-0.75); // sqrt(P(k))
                        density[idx] *= p_k;
                    }
                }
            }
        }

        // Compute displacement field: Ψ_i(k) = i * k_i * δ(k) / k²
        let mut disp_x = vec![Complex::new(0.0, 0.0); num_particles];
        let mut disp_y = vec![Complex::new(0.0, 0.0); num_particles];
        let mut disp_z = vec![Complex::new(0.0, 0.0); num_particles];

        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let kx = if ix > n / 2 { ix as i32 - n as i32 } else { ix as i32 } as f32 * k_fundamental;
                    let ky = if iy > n / 2 { iy as i32 - n as i32 } else { iy as i32 } as f32 * k_fundamental;
                    let kz = if iz > n / 2 { iz as i32 - n as i32 } else { iz as i32 } as f32 * k_fundamental;
                    
                    let k_mag_sq = kx * kx + ky * ky + kz * kz;
                    let idx = (ix * n + iy) * n + iz;
                    
                    if k_mag_sq > 1e-6 {
                        let factor = Complex::new(0.0, -1.0) / k_mag_sq; // -i/k² (note sign for IFFT convention)
                        disp_x[idx] = density[idx] * kx * factor;
                        disp_y[idx] = density[idx] * ky * factor;
                        disp_z[idx] = density[idx] * kz * factor;
                    }
                }
            }
        }

        // Inverse 3D FFT for each displacement component
        for disp in [&mut disp_x, &mut disp_y, &mut disp_z] {
            // IFFT along x-axis
            for iy in 0..n {
                for iz in 0..n {
                    let mut slice = vec![Complex::new(0.0, 0.0); n];
                    for ix in 0..n {
                        slice[ix] = disp[(ix * n + iy) * n + iz];
                    }
                    inv_plan.process(&mut slice);
                    for ix in 0..n {
                        disp[(ix * n + iy) * n + iz] = slice[ix];
                    }
                }
            }

            // IFFT along y-axis
            for ix in 0..n {
                for iz in 0..n {
                    let mut slice = vec![Complex::new(0.0, 0.0); n];
                    for iy in 0..n {
                        slice[iy] = disp[(ix * n + iy) * n + iz];
                    }
                    inv_plan.process(&mut slice);
                    for iy in 0..n {
                        disp[(ix * n + iy) * n + iz] = slice[iy];
                    }
                }
            }

            // IFFT along z-axis
            for ix in 0..n {
                for iy in 0..n {
                    let offset = (ix * n + iy) * n;
                    let mut slice: Vec<Complex<f32>> = disp[offset..offset + n].to_vec();
                    inv_plan.process(&mut slice);
                    disp[offset..offset + n].copy_from_slice(&slice);
                }
            }
        }

        // Normalization and particle placement
        let normalization = 1.0 / (num_particles as f32);
        let cell_size = box_size / n as f32;
        let displacement_scale = box_size * 0.05; // Scale displacements to ~5% of box
        let velocity_scale = 0.02; // Small initial velocities

        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let idx = (ix * n + iy) * n + iz;
                    
                    // Lagrangian position (regular grid)
                    let q = [
                        (ix as f32 + 0.5) * cell_size,
                        (iy as f32 + 0.5) * cell_size,
                        (iz as f32 + 0.5) * cell_size,
                    ];

                    // Zel'dovich displacement
                    let psi = [
                        disp_x[idx].re * normalization * displacement_scale,
                        disp_y[idx].re * normalization * displacement_scale,
                        disp_z[idx].re * normalization * displacement_scale,
                    ];

                    // Eulerian position: x = q + Ψ(q)
                    let position = [
                        (q[0] + psi[0]).rem_euclid(box_size),
                        (q[1] + psi[1]).rem_euclid(box_size),
                        (q[2] + psi[2]).rem_euclid(box_size),
                    ];

                    // Peculiar velocity v = H*f*Ψ (approximated as v ∝ Ψ)
                    let velocity = [
                        psi[0] * velocity_scale,
                        psi[1] * velocity_scale,
                        psi[2] * velocity_scale,
                    ];

                    self.particles.push(Particle {
                        position,
                        velocity,
                        force: [0.0, 0.0, 0.0],
                        mass: 1.0,
                    });
                }
            }
        }
    }

    #[allow(dead_code)]
    /// Initialize particles from density field with cosmological-like clustering.
    /// Uses rejection sampling to create initial overdensities.
    pub fn initialize_grid_with_perturbations(&mut self, num_particles: usize, box_size: f32) {
        self.box_size = box_size;
        let mut rng = rand::rng();

        self.particles = Vec::new();

        // Generate density field using Fourier modes (simplified power spectrum)
        let n_modes = 8;
        let mut modes = Vec::new();

        for i in 1..=n_modes {
            let k = (i as f32) * 2.0 * 3.14159 / box_size;
            let amplitude = 1.0 / (i as f32).sqrt(); // P(k) ~ k^(-1/2) approximation

            for _ in 0..3 {
                let kx = k * (rng.random::<f32>() * 2.0 - 1.0);
                let ky = k * (rng.random::<f32>() * 2.0 - 1.0);
                let kz = k * (rng.random::<f32>() * 2.0 - 1.0);
                let phase = rng.random::<f32>() * 6.28318;
                modes.push((kx, ky, kz, phase, amplitude));
            }
        }

        // Helper function to evaluate density at position
        let density_at = |x: f32, y: f32, z: f32| -> f32 {
            let mut rho = 1.0;
            for (kx, ky, kz, phase, amp) in &modes {
                rho += amp * (kx * x + ky * y + kz * z + phase).sin();
            }
            rho.max(0.1) // Ensure positive
        };

        // Find max density for rejection sampling
        let mut max_rho: f32 = 0.0;
        for _ in 0..1000 {
            let x = rng.random::<f32>() * box_size;
            let y = rng.random::<f32>() * box_size;
            let z = rng.random::<f32>() * box_size;
            max_rho = max_rho.max(density_at(x, y, z));
        }
        max_rho *= 1.2; // Safety margin

        // Rejection sampling: place particles according to density field
        let mut attempts = 0;
        let max_attempts = num_particles * 100;

        while self.particles.len() < num_particles && attempts < max_attempts {
            let x = rng.random::<f32>() * box_size;
            let y = rng.random::<f32>() * box_size;
            let z = rng.random::<f32>() * box_size;

            let rho = density_at(x, y, z);
            let accept_prob = rho / max_rho;

            if rng.random::<f32>() < accept_prob {
                self.particles.push(Particle {
                    position: [x, y, z],
                    velocity: [
                        (rng.random::<f32>() - 0.5) * 0.01,
                        (rng.random::<f32>() - 0.5) * 0.01,
                        (rng.random::<f32>() - 0.5) * 0.01,
                    ],
                    force: [0.0, 0.0, 0.0],
                    mass: 1.0,
                });
            }
            attempts += 1;
        }

        // Fill remaining with uniform if rejection sampling didn't get enough
        while self.particles.len() < num_particles {
            self.particles.push(Particle {
                position: [
                    rng.random::<f32>() * box_size,
                    rng.random::<f32>() * box_size,
                    rng.random::<f32>() * box_size,
                ],
                velocity: [
                    (rng.random::<f32>() - 0.5) * 0.01,
                    (rng.random::<f32>() - 0.5) * 0.01,
                    (rng.random::<f32>() - 0.5) * 0.01,
                ],
                force: [0.0, 0.0, 0.0],
                mass: 1.0,
            });
        }
    }

    /// Kick-drift-kick leapfrog integration.
    pub fn integrate(&mut self, dt: f32) {
        let half_dt = 0.5 * dt;

        for p in &mut self.particles {
            p.velocity[0] += p.force[0] * half_dt;
            p.velocity[1] += p.force[1] * half_dt;
            p.velocity[2] += p.force[2] * half_dt;

            p.position[0] = (p.position[0] + p.velocity[0] * dt).rem_euclid(self.box_size);
            p.position[1] = (p.position[1] + p.velocity[1] * dt).rem_euclid(self.box_size);
            p.position[2] = (p.position[2] + p.velocity[2] * dt).rem_euclid(self.box_size);
        }
    }

    /// Complete kick after force recalculation.
    pub fn kick(&mut self, dt: f32) {
        let half_dt = 0.5 * dt;
        for p in &mut self.particles {
            p.velocity[0] += p.force[0] * half_dt;
            p.velocity[1] += p.force[1] * half_dt;
            p.velocity[2] += p.force[2] * half_dt;
        }
    }

    /// Project particles to 2D along z-axis using CIC.
    pub fn project_to_2d(&self, resolution: usize) -> Vec<f32> {
        let mut map = vec![0.0; resolution * resolution];
        let cell_size = self.box_size / resolution as f32;
        let inv_cell_size = 1.0 / cell_size;

        for p in &self.particles {
            let x_grid = p.position[0] * inv_cell_size;
            let y_grid = p.position[1] * inv_cell_size;

            let i = x_grid.floor() as isize;
            let j = y_grid.floor() as isize;

            let dx = x_grid - i as f32;
            let dy = y_grid - j as f32;

            let w = [
                (1.0 - dx) * (1.0 - dy),
                dx * (1.0 - dy),
                (1.0 - dx) * dy,
                dx * dy,
            ];

            let n = resolution as isize;
            let idx = |x: isize, y: isize| (((x % n + n) % n) * n + ((y % n + n) % n)) as usize;

            map[idx(i, j)] += p.mass * w[0];
            map[idx(i + 1, j)] += p.mass * w[1];
            map[idx(i, j + 1)] += p.mass * w[2];
            map[idx(i + 1, j + 1)] += p.mass * w[3];
        }

        map
    }
}
