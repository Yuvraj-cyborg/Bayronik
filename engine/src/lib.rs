//! # Bayronik Engine
//!
//! Cosmological Particle-Mesh N-body code for dark-matter structure
//! formation in a flat LCDM universe.
//!
//! Physics:
//! - Zel'dovich initial conditions from an Eisenstein-Hu linear power
//!   spectrum normalized to sigma8, at a configurable starting redshift.
//! - Comoving equations of motion with canonical momentum p = a^2 dx/dt,
//!   integrated with symplectic KDK leapfrog in scale factor
//!   (kick/drift factors are exact integrals of 1/(aE) and 1/(a^3 E)).
//! - CIC mass assignment, FFT Poisson solver with the physical prefactor
//!   (3/2) Omega_m / a, CIC force interpolation.
//!
//! Units: lengths in comoving Mpc/h, masses in Msun/h, H0 = 1.
//! The output map is a CDM surface density in (Msun/h)/(Mpc/h)^2, the same
//! convention as CAMELS 2D mass maps.

pub mod cosmology;
#[cfg(feature = "npy")]
pub mod output;
pub mod sim;

pub use cosmology::{Cosmology, LinearPower, RHO_CRIT};
pub use sim::fft_solver::FftSolver;
pub use sim::forces;
pub use sim::gravity;
pub use sim::grid::Grid;
pub use sim::particle::ParticleSet;

/// Full configuration for one simulation run.
#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    pub seed: u64,
    /// Particles per dimension (grid_res^3 total) and PM mesh resolution.
    pub grid_res: usize,
    /// Comoving box side in Mpc/h. CAMELS uses 25.
    pub box_size: f32,
    /// Starting redshift for the Zel'dovich ICs.
    pub z_init: f64,
    /// Number of leapfrog steps from a_init to a = 1 (log-spaced in a).
    pub n_steps: usize,
    /// Output map resolution (upsampled from the PM mesh if larger).
    pub projection_res: usize,
    /// Fraction of the box depth to project (1.0 = full box). CAMELS maps
    /// project 5 Mpc/h slabs of a 25 Mpc/h box, i.e. 0.2.
    pub slab_fraction: f32,
    pub cosmo: Cosmology,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            grid_res: 64,
            box_size: 25.0,
            z_init: 49.0,
            n_steps: 32,
            projection_res: 256,
            slab_fraction: 1.0,
            cosmo: Cosmology::default(),
        }
    }
}

/// Run a complete N-body simulation to a = 1 and return the projected CDM
/// surface density map, `projection_res^2` values in (Msun/h)/(Mpc/h)^2.
pub fn run_simulation(config: &SimConfig) -> Vec<f32> {
    let cosmo = &config.cosmo;
    let a_init = 1.0 / (1.0 + config.z_init);

    let mut particles = ParticleSet::new();
    particles.initialize_zeldovich(
        config.grid_res,
        config.box_size,
        config.seed,
        cosmo,
        a_init,
    );

    let mut grid = Grid::new(config.grid_res, config.box_size);
    let mut solver = FftSolver::new(config.grid_res);

    // Log-spaced scale factor steps from a_init to 1.
    let n_steps = config.n_steps.max(1);
    let ln_ratio = (1.0 / a_init).ln() / n_steps as f64;
    let a_at = |i: usize| a_init * (ln_ratio * i as f64).exp();

    // KDK leapfrog: one force evaluation per step (force at the end of a
    // step is reused as the force at the start of the next).
    compute_forces(&mut particles, &mut grid, &mut solver, cosmo, a_init);

    for i in 0..n_steps {
        let a0 = a_at(i);
        let a1 = a_at(i + 1);
        let a_half = (a0 * a1).sqrt();

        particles.kick(cosmo.kick_factor(a0, a_half) as f32);
        particles.drift(cosmo.drift_factor(a0, a1) as f32);
        compute_forces(&mut particles, &mut grid, &mut solver, cosmo, a1);
        particles.kick(cosmo.kick_factor(a_half, a1) as f32);
    }

    // Project total matter, then scale to the CDM component to match the
    // CAMELS Mcdm map convention.
    let internal_res = config.projection_res.min(2 * config.grid_res);
    let sigma_total = particles.project_to_2d(internal_res, config.slab_fraction);

    let cdm_fraction = cosmo.cdm_fraction() as f32;
    let mut map = if internal_res < config.projection_res {
        upsample_bilinear(&sigma_total, internal_res, config.projection_res)
    } else {
        sigma_total
    };
    for v in map.iter_mut() {
        *v *= cdm_fraction;
    }
    map
}

/// One PM force evaluation: CIC deposit, Poisson solve with the physical
/// prefactor (3/2) Omega_m / a, finite-difference gradient, CIC gather.
fn compute_forces(
    particles: &mut ParticleSet,
    grid: &mut Grid,
    solver: &mut FftSolver,
    cosmo: &Cosmology,
    a: f64,
) {
    grid.clear_density();
    gravity::assign_mass_cic(particles, grid);
    solver.solve_potential(grid, cosmo.poisson_prefactor(a) as f32);
    let (fx, fy, fz) = forces::calculate_forces_from_potential(grid);
    forces::interpolate_forces_to_particles(particles, grid, &fx, &fy, &fz);
}

/// Periodic bilinear upsampling of a square map from `from`^2 to `to`^2
/// pixels. Interpolates at pixel centers, preserving the map mean, so the
/// PM-limited projection can be served at the emulator's input resolution
/// without inventing small-scale power.
pub fn upsample_bilinear(map: &[f32], from: usize, to: usize) -> Vec<f32> {
    debug_assert_eq!(map.len(), from * from);
    if from == to {
        return map.to_vec();
    }
    let mut out = vec![0.0f32; to * to];
    let scale = from as f32 / to as f32;
    let n = from as isize;
    let wrap = |v: isize| v.rem_euclid(n) as usize;

    for i in 0..to {
        let x = (i as f32 + 0.5) * scale - 0.5;
        let x0 = x.floor();
        let dx = x - x0;
        let (xa, xb) = (wrap(x0 as isize), wrap(x0 as isize + 1));
        for j in 0..to {
            let y = (j as f32 + 0.5) * scale - 0.5;
            let y0 = y.floor();
            let dy = y - y0;
            let (ya, yb) = (wrap(y0 as isize), wrap(y0 as isize + 1));

            out[i * to + j] = map[xa * from + ya] * (1.0 - dx) * (1.0 - dy)
                + map[xb * from + ya] * dx * (1.0 - dy)
                + map[xa * from + yb] * (1.0 - dx) * dy
                + map[xb * from + yb] * dx * dy;
        }
    }
    out
}
