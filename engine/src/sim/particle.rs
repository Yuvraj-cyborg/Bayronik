//! Particle storage, symplectic drift/kick operators, and 2D projection.
//!
//! Code units: comoving positions in Mpc/h, masses in Msun/h, H0 = 1.
//! `velocity` stores the canonical momentum per unit mass p = a^2 dx/dt,
//! which satisfies dp/dt = -grad phi (no Hubble drag term). The scale-factor
//! dependence lives entirely in the drift/kick coefficients supplied by
//! `Cosmology::drift_factor` / `Cosmology::kick_factor`.

pub struct Particle {
    /// Comoving position in Mpc/h, periodic in [0, box_size).
    pub position: [f32; 3],
    /// Canonical momentum per unit mass, p = a^2 dx/dt (code units).
    pub velocity: [f32; 3],
    /// Force -grad phi interpolated to the particle (code units).
    pub force: [f32; 3],
    /// Particle mass in Msun/h.
    pub mass: f32,
}

pub struct ParticleSet {
    pub particles: Vec<Particle>,
    pub box_size: f32,
}

impl Default for ParticleSet {
    fn default() -> Self {
        Self::new()
    }
}

impl ParticleSet {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            box_size: 0.0,
        }
    }

    /// Total mass of all particles in Msun/h.
    pub fn total_mass(&self) -> f64 {
        self.particles.iter().map(|p| p.mass as f64).sum()
    }

    /// Drift: x += p * coef, with periodic wrapping.
    /// `coef` is int da / (a^3 E(a)) over the step.
    pub fn drift(&mut self, coef: f32) {
        let box_size = self.box_size;
        for p in &mut self.particles {
            p.position[0] = (p.position[0] + p.velocity[0] * coef).rem_euclid(box_size);
            p.position[1] = (p.position[1] + p.velocity[1] * coef).rem_euclid(box_size);
            p.position[2] = (p.position[2] + p.velocity[2] * coef).rem_euclid(box_size);
        }
    }

    /// Kick: p += F * coef, using the currently stored forces.
    /// `coef` is int da / (a E(a)) over the (half-)step.
    pub fn kick(&mut self, coef: f32) {
        for p in &mut self.particles {
            p.velocity[0] += p.force[0] * coef;
            p.velocity[1] += p.force[1] * coef;
            p.velocity[2] += p.force[2] * coef;
        }
    }

    /// Project particles along the z-axis onto a 2D surface-density map
    /// using CIC deposition.
    ///
    /// `slab_fraction` in (0, 1] selects a z-slab of thickness
    /// `slab_fraction * box_size` centered at the box midplane; 1.0 projects
    /// the full box. Output units: (Msun/h) / (Mpc/h)^2, matching the CAMELS
    /// map convention (surface density integrated through the slab).
    pub fn project_to_2d(&self, resolution: usize, slab_fraction: f32) -> Vec<f32> {
        let mut map = vec![0.0f32; resolution * resolution];
        let cell_size = self.box_size / resolution as f32;
        let inv_cell_size = 1.0 / cell_size;

        let slab_fraction = slab_fraction.clamp(0.0, 1.0);
        let half_thickness = 0.5 * slab_fraction * self.box_size;
        let center = 0.5 * self.box_size;
        let z_min = center - half_thickness;
        let z_max = center + half_thickness;

        for p in &self.particles {
            if slab_fraction < 1.0 && (p.position[2] < z_min || p.position[2] >= z_max) {
                continue;
            }

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

        // Mass per pixel -> surface density.
        let pixel_area = cell_size * cell_size;
        let inv_area = 1.0 / pixel_area;
        for v in map.iter_mut() {
            *v *= inv_area;
        }

        map
    }
}
