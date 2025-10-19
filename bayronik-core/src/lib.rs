pub mod output;
pub mod sim;

pub use sim::fft_solver::FftSolver;
pub use sim::forces;
pub use sim::gravity;
pub use sim::grid::Grid;
pub use sim::particle::ParticleSet;

/// Run a complete N-body simulation and return 2D projection.
pub fn run_simulation(
    num_particles: usize,
    grid_resolution: usize,
    box_size: f32,
    time_step: f32,
    num_steps: usize,
    projection_res: usize,
) -> Vec<f32> {
    let mut particles = ParticleSet::new();
    particles.initialize_grid_with_perturbations(num_particles, box_size);
    
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
        for f in &mut fx { *f *= growth_factor; }
        for f in &mut fy { *f *= growth_factor; }
        for f in &mut fz { *f *= growth_factor; }
        
        forces::interpolate_forces_to_particles(&mut particles, &grid, &fx, &fy, &fz);
        
        particles.integrate(time_step);
        
        let (mut fx, mut fy, mut fz) = forces::calculate_forces_from_potential(&grid);
        for f in &mut fx { *f *= growth_factor; }
        for f in &mut fy { *f *= growth_factor; }
        for f in &mut fz { *f *= growth_factor; }
        
        forces::interpolate_forces_to_particles(&mut particles, &grid, &fx, &fy, &fz);
        particles.kick(time_step);
    }
    
    particles.project_to_2d(projection_res)
}
