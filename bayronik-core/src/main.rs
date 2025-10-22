mod sim;

use sim::fft_solver::FftSolver;
use sim::forces;
use sim::gravity;
use sim::grid::Grid;
use sim::particle::ParticleSet;

fn main() {
    println!("bayronik-core: N-body PM simulation");

    const NUM_PARTICLES: usize = 32_768;
    const GRID_RESOLUTION: usize = 64;
    const BOX_SIZE: f32 = 100.0;
    const TIME_STEP: f32 = 0.01;
    const NUM_STEPS: usize = 10;

    let mut particles = ParticleSet::new();
    particles.initialize_grid_with_perturbations(NUM_PARTICLES, BOX_SIZE);
    println!(
        "Initialized {} particles in {}^3 Mpc/h box",
        NUM_PARTICLES, BOX_SIZE
    );

    let mut grid = Grid::new(GRID_RESOLUTION, BOX_SIZE);
    let mut fft_solver = FftSolver::new(GRID_RESOLUTION);
    println!("Grid resolution: {}^3\n", GRID_RESOLUTION);

    for step in 0..NUM_STEPS {
        println!("Step {}/{}", step + 1, NUM_STEPS);

        grid.clear_density();
        gravity::assign_mass_cic(&particles, &mut grid);
        fft_solver.solve_potential(&mut grid);

        let (fx, fy, fz) = forces::calculate_forces_from_potential(&grid);
        forces::interpolate_forces_to_particles(&mut particles, &grid, &fx, &fy, &fz);

        particles.integrate(TIME_STEP);

        let (fx_new, fy_new, fz_new) = forces::calculate_forces_from_potential(&grid);
        forces::interpolate_forces_to_particles(&mut particles, &grid, &fx_new, &fy_new, &fz_new);
        particles.kick(TIME_STEP);
    }

    println!("\nProjecting to 2D map (256x256)...");
    let map_2d = particles.project_to_2d(256);
    println!(
        "Mean surface density: {:.3e}",
        map_2d.iter().sum::<f32>() / map_2d.len() as f32
    );

    println!("\nSimulation complete. Use particle.project_to_2d() for outputs.");
}
