//! # Bayronik Core
//!
//! Particle-Mesh N-body simulation engine for cosmological structure formation.
//!
//! Implements a PM scheme with CIC mass assignment, FFT-based Poisson solver,
//! and KDK symplectic leapfrog integration. Zel'dovich approximation provides
//! cosmologically-motivated initial conditions with a CDM-like power spectrum.
//!
//! The output is a 2D surface density map suitable as input for the Bayronik
//! baryonic field emulator.

#[cfg(feature = "npy")]
pub mod output;
pub mod sim;

pub use sim::fft_solver::FftSolver;
pub use sim::forces;
pub use sim::gravity;
pub use sim::grid::Grid;
pub use sim::particle::ParticleSet;

/// Run a complete N-body simulation and return a 2D projected density map.
///
/// The simulation uses Zel'dovich ICs on a grid of `grid_resolution^3`
/// particles, evolved with KDK leapfrog for `num_steps` timesteps.
pub fn run_simulation(
    seed: u64,
    grid_resolution: usize,
    box_size: f32,
    time_step: f32,
    num_steps: usize,
    projection_res: usize,
) -> Vec<f32> {
    let mut particles = ParticleSet::new();
    particles.initialize_zeldovich(grid_resolution, box_size, seed);

    let mut grid = Grid::new(grid_resolution, box_size);
    let mut fft_solver = FftSolver::new(grid_resolution);

    let gravity_strength = 2.5;

    for _ in 0..num_steps {
        // Compute forces at current positions
        grid.clear_density();
        gravity::assign_mass_cic(&particles, &mut grid);
        fft_solver.solve_potential(&mut grid);

        let (mut fx, mut fy, mut fz) = forces::calculate_forces_from_potential(&grid);
        for f in &mut fx {
            *f *= gravity_strength;
        }
        for f in &mut fy {
            *f *= gravity_strength;
        }
        for f in &mut fz {
            *f *= gravity_strength;
        }
        forces::interpolate_forces_to_particles(&mut particles, &grid, &fx, &fy, &fz);

        // Half-kick + full drift
        particles.integrate(time_step);

        // Recompute forces at new positions (proper KDK)
        grid.clear_density();
        gravity::assign_mass_cic(&particles, &mut grid);
        fft_solver.solve_potential(&mut grid);

        let (mut fx, mut fy, mut fz) = forces::calculate_forces_from_potential(&grid);
        for f in &mut fx {
            *f *= gravity_strength;
        }
        for f in &mut fy {
            *f *= gravity_strength;
        }
        for f in &mut fz {
            *f *= gravity_strength;
        }
        forces::interpolate_forces_to_particles(&mut particles, &grid, &fx, &fy, &fz);

        // Second half-kick
        particles.kick(time_step);
    }

    particles.project_to_2d(projection_res)
}
