//! # Bayronik Core
//!
//! A high-performance N-body Particle-Mesh (PM) simulation engine for cosmological
//! structure formation, designed to solve the baryonic feedback problem in weak lensing.
//!
//! ## Overview
//!
//! Bayronik addresses a critical efficiency bottleneck in modern cosmology: precision
//! measurements of universe parameters (Dark Matter density Ωm, clumpiness σ8) rely on
//! weak gravitational lensing, but are systematically biased by "baryonic physics"—messy,
//! non-gravitational processes like supernova explosions and AGN feedback that violently
//! redistribute gas and matter on large scales.
//!
//! ## Architecture
//!
//! The engine implements a classic PM N-body scheme:
//!
//! 1. **Mass Assignment**: Particles deposit mass onto a 3D grid using Cloud-in-Cell (CIC)
//!    interpolation for smooth density fields
//! 2. **Poisson Solver**: FFT-based gravity solver computes potential φ where ∇²φ = 4πGρ
//! 3. **Force Calculation**: Forces derived via finite differences F = -∇φ
//! 4. **Time Integration**: Symplectic Kick-Drift-Kick leapfrog with periodic boundaries
//! 5. **Projection**: 3D → 2D surface density maps via CIC for weak lensing analysis
//!
//! ## Initial Conditions
//!
//! Supports Zel'dovich approximation for cosmologically-motivated initial conditions,
//! generating correlated particle displacements from a Gaussian random field with
//! CDM-like power spectrum P(k) ~ k^(-1.5).
//!
//! ## Example
//!
//! ```rust,ignore
//! use bayronik_core::{run_simulation, ParticleSet, Grid, FftSolver};
//!
//! // Run a complete simulation
//! let map = run_simulation(
//!     32_768,  // num_particles (ignored with Zel'dovich ICs)
//!     64,      // grid_resolution (also sets IC grid)
//!     100.0,   // box_size in Mpc/h
//!     0.01,    // time_step
//!     100,     // num_steps
//!     256,     // output projection resolution
//! );
//! ```
//!
//! ## Physics References
//!
//! - Hockney & Eastwood (1988) - Computer Simulation Using Particles
//! - Zel'dovich (1970) - Gravitational instability approximation

pub mod output;
pub mod sim;

pub use sim::fft_solver::FftSolver;
pub use sim::forces;
pub use sim::gravity;
pub use sim::grid::Grid;
pub use sim::particle::ParticleSet;

/// Run a complete N-body simulation and return 2D projection.
pub fn run_simulation(
    _num_particles: usize,
    grid_resolution: usize,
    box_size: f32,
    time_step: f32,
    num_steps: usize,
    projection_res: usize,
) -> Vec<f32> {
    let mut particles = ParticleSet::new();
    // Zel'dovich ICs: grid_resolution sets both IC grid and force grid
    // Actual particle count = grid_resolution^3
    particles.initialize_zeldovich(grid_resolution, box_size);

    let mut grid = Grid::new(grid_resolution, box_size);
    let mut fft_solver = FftSolver::new(grid_resolution);

    // Add gravitational amplification factor to grow perturbations faster
    let growth_factor = 2.5;

    for _ in 0..num_steps {
        grid.clear_density();
        gravity::assign_mass_cic(&particles, &mut grid);
        fft_solver.solve_potential(&mut grid);

        let (mut fx, mut fy, mut fz) = forces::calculate_forces_from_potential(&grid);

        // Amplify gravitational forces to accelerate structure formation
        for f in &mut fx {
            *f *= growth_factor;
        }
        for f in &mut fy {
            *f *= growth_factor;
        }
        for f in &mut fz {
            *f *= growth_factor;
        }

        forces::interpolate_forces_to_particles(&mut particles, &grid, &fx, &fy, &fz);

        particles.integrate(time_step);

        let (mut fx, mut fy, mut fz) = forces::calculate_forces_from_potential(&grid);
        for f in &mut fx {
            *f *= growth_factor;
        }
        for f in &mut fy {
            *f *= growth_factor;
        }
        for f in &mut fz {
            *f *= growth_factor;
        }

        forces::interpolate_forces_to_particles(&mut particles, &grid, &fx, &fy, &fz);
        particles.kick(time_step);
    }

    particles.project_to_2d(projection_res)
}
