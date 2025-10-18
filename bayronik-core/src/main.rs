//! The main entry point for the core N-body simulation crate.

mod sim;

use sim::fft_solver::FftSolver;
use sim::grid::Grid;
use sim::gravity; 
use sim::particle::ParticleSet;

fn main() {
    println!("Initializing Baryonic Field Emulator - Core Simulation");

    // Simulation Params
    const NUM_PARTICLES: usize = 10_000; // Start with a small number
    const GRID_RESOLUTION: usize = 32;   // A 32x32x32 grid
    const BOX_SIZE: f32 = 100.0;         // Simulation box size in Mpc/h

    let mut particles = ParticleSet::new();
    particles.initialize_randomly(NUM_PARTICLES, BOX_SIZE);
    println!("Initialized {} particles randomly.", NUM_PARTICLES);

    let mut grid = Grid::new(GRID_RESOLUTION, BOX_SIZE);
    println!("Initialized a {}^3 grid.", GRID_RESOLUTION);

    let mut fft_solver = FftSolver::new(GRID_RESOLUTION);

    println!("\n--- Performing one simulation step ---");

    // Clear the density grid from the previous step.
    grid.clear_density();
    println!("1. Cleared density grid.");

    // Assign particle masses to the grid (CIC assignment).
    println!("2. Assigning particle mass to grid...");
    gravity::assign_mass_cic(&particles, &mut grid); 

    // Solve for gravitational potential using FFT.
    println!("3. Solving for potential...");
    fft_solver.solve_potential(&mut grid);

    // // 4. Calculate forces on the grid from the potential.
    // //    (We will implement this function next).
    // println!("4. Calculating forces on grid (TODO)...");
    // // calculate_forces(&mut grid);

    // // 5. Interpolate forces from the grid back to the particles.
    // //    (We will implement this function next).
    // println!("5. Interpolating forces back to particles (TODO)...");
    // // interpolate_forces_to_particles(&grid, &mut particles);

    // // 6. Evolve particles forward in time (kick-drift-kick).
    // //    (We will implement this function next).
    // println!("6. Evolving particles (TODO)...");
    // // evolve_particles(&mut particles, time_step);

    println!("\nSimulation step conceptualized. Next, we'll calculate forces from potential!");
}

