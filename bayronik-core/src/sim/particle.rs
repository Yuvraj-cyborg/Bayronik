use rand::Rng;

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

    /// Initialize particles with random uniform positions.
    #[allow(dead_code)]
    pub fn initialize_randomly(&mut self, num_particles: usize, box_size: f32) {
        self.box_size = box_size;
        let mut rng = rand::rng();
        self.particles = (0..num_particles)
            .map(|_| Particle {
                position: [
                    rng.random::<f32>() * box_size,
                    rng.random::<f32>() * box_size,
                    rng.random::<f32>() * box_size,
                ],
                velocity: [0.0, 0.0, 0.0],
                force: [0.0, 0.0, 0.0],
                mass: 1.0,
            })
            .collect();
    }
    
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
            let idx = |x: isize, y: isize| {
                (((x % n + n) % n) * n + ((y % n + n) % n)) as usize
            };
            
            map[idx(i, j)] += p.mass * w[0];
            map[idx(i + 1, j)] += p.mass * w[1];
            map[idx(i, j + 1)] += p.mass * w[2];
            map[idx(i + 1, j + 1)] += p.mass * w[3];
        }
        
        map
    }
}

