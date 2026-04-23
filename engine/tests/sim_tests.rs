use engine::{FftSolver, Grid, ParticleSet, forces, gravity};

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

#[test]
fn grid_initializes_to_zero() {
    let g = Grid::new(8, 100.0);
    assert_eq!(g.density_contrast.len(), 8 * 8 * 8);
    assert!(g.density_contrast.iter().all(|&v| v == 0.0));
    assert!(g.potential.iter().all(|&v| v == 0.0));
}

#[test]
fn grid_clear_resets_density() {
    let mut g = Grid::new(4, 50.0);
    g.density_contrast[0] = 42.0;
    g.density_contrast[63] = -1.0;
    g.clear_density();
    assert!(g.density_contrast.iter().all(|&v| v == 0.0));
}

// ---------------------------------------------------------------------------
// CIC mass assignment
// ---------------------------------------------------------------------------

#[test]
fn cic_conserves_total_mass() {
    let mut ps = ParticleSet::new();
    ps.initialize_zeldovich(8, 100.0, 123);
    let n_particles = ps.particles.len();
    assert_eq!(n_particles, 8 * 8 * 8);

    let mut grid = Grid::new(8, 100.0);
    gravity::assign_mass_cic(&ps, &mut grid);

    // density_contrast = rho/rho_mean - 1, so sum should be ~0
    let sum: f32 = grid.density_contrast.iter().sum();
    assert!(
        sum.abs() < 0.01,
        "CIC density contrast should sum to ~0, got {sum}"
    );
}

#[test]
fn cic_single_particle_at_center() {
    let mut ps = ParticleSet::new();
    ps.box_size = 4.0;
    ps.particles.push(engine::sim::particle::Particle {
        position: [2.0, 2.0, 2.0],
        velocity: [0.0; 3],
        force: [0.0; 3],
        mass: 1.0,
    });

    let mut grid = Grid::new(4, 4.0);
    gravity::assign_mass_cic(&ps, &mut grid);

    // Single particle: mean_density = 1/64; particle deposits mass=1 into surrounding cells.
    // After density contrast conversion, total should sum to ~0.
    let sum: f32 = grid.density_contrast.iter().sum();
    assert!(
        sum.abs() < 0.1,
        "Single-particle CIC sum should be ~0, got {sum}"
    );

    // The cell containing the particle should have positive density contrast
    let idx = (2 * 4 + 2) * 4 + 2; // cell (2,2,2) in a 4^3 grid
    assert!(
        grid.density_contrast[idx] > 0.0,
        "Cell containing particle should have positive density"
    );
}

// ---------------------------------------------------------------------------
// FFT Poisson solver
// ---------------------------------------------------------------------------

#[test]
fn poisson_solver_zero_density_gives_zero_potential() {
    let mut grid = Grid::new(8, 100.0);
    let mut solver = FftSolver::new(8);
    solver.solve_potential(&mut grid);
    let max_pot: f32 = grid.potential.iter().map(|v| v.abs()).fold(0.0, f32::max);
    assert!(
        max_pot < 1e-6,
        "Zero density should give zero potential, max = {max_pot}"
    );
}

#[test]
fn poisson_solver_produces_smooth_potential() {
    let mut ps = ParticleSet::new();
    ps.initialize_zeldovich(8, 100.0, 42);

    let mut grid = Grid::new(8, 100.0);
    gravity::assign_mass_cic(&ps, &mut grid);

    let mut solver = FftSolver::new(8);
    solver.solve_potential(&mut grid);

    // Potential should be finite and not all zeros
    assert!(grid.potential.iter().all(|v| v.is_finite()));

    let max_pot: f32 = grid.potential.iter().map(|v| v.abs()).fold(0.0, f32::max);
    assert!(
        max_pot > 1e-10,
        "Non-trivial density should give non-zero potential"
    );
}

#[test]
fn fft_roundtrip_preserves_data() {
    // Verifies that FFT forward + Green's function + inverse produces a valid result
    // by checking that the potential has the same structure dimensionality as the density
    let n = 16;
    let mut grid = Grid::new(n, 100.0);

    // Place a point mass
    let center = (n / 2 * n + n / 2) * n + n / 2;
    grid.density_contrast[center] = 1.0;

    let mut solver = FftSolver::new(n);
    solver.solve_potential(&mut grid);

    // Potential at the center should be the deepest (most negative)
    let pot_center = grid.potential[center];
    let pot_corner = grid.potential[0];
    assert!(
        pot_center < pot_corner,
        "Potential at point mass ({pot_center}) should be deeper than corner ({pot_corner})"
    );
}

// ---------------------------------------------------------------------------
// Forces
// ---------------------------------------------------------------------------

#[test]
fn forces_from_uniform_potential_are_zero() {
    let mut grid = Grid::new(8, 100.0);
    // Set all potential to a constant
    grid.potential.iter_mut().for_each(|v| *v = 5.0);

    let (fx, fy, fz) = forces::calculate_forces_from_potential(&grid);

    let max_f: f32 = fx
        .iter()
        .chain(fy.iter())
        .chain(fz.iter())
        .map(|v| v.abs())
        .fold(0.0, f32::max);
    assert!(
        max_f < 1e-6,
        "Uniform potential should give zero forces, max = {max_f}"
    );
}

#[test]
fn forces_are_antisymmetric() {
    // Away from the periodic boundary, a linear potential gradient should give
    // constant force in the opposite direction.
    let n = 8;
    let mut grid = Grid::new(n, 100.0);
    let cell_size = 100.0 / n as f32;

    // phi(i,j,k) = i * cell_size (linear in x)
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                grid.potential[(i * n + j) * n + k] = i as f32 * cell_size;
            }
        }
    }

    let (fx, fy, fz) = forces::calculate_forces_from_potential(&grid);

    // fx should be approximately -1 in interior cells (force = -grad phi).
    // Boundary cells include the periodic discontinuity and are not checked here.
    for i in 1..n - 1 {
        for j in 0..n {
            for k in 0..n {
                let idx = (i * n + j) * n + k;
                assert!(
                    (fx[idx] + 1.0).abs() < 1e-6,
                    "Interior x-force should be -1, got {} at ({i},{j},{k})",
                    fx[idx]
                );
            }
        }
    }

    // fy, fz should be ~0
    let fy_max: f32 = fy.iter().map(|v| v.abs()).fold(0.0, f32::max);
    let fz_max: f32 = fz.iter().map(|v| v.abs()).fold(0.0, f32::max);
    assert!(fy_max < 1e-6, "fy should be zero for x-only gradient");
    assert!(fz_max < 1e-6, "fz should be zero for x-only gradient");
}

#[test]
fn force_interpolation_works() {
    let n = 8;
    let mut grid = Grid::new(n, 100.0);

    // Linear potential: phi = x
    let cell_size = 100.0 / n as f32;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                grid.potential[(i * n + j) * n + k] = i as f32 * cell_size;
            }
        }
    }

    let (fx, fy, fz) = forces::calculate_forces_from_potential(&grid);

    let mut ps = ParticleSet::new();
    ps.box_size = 100.0;
    ps.particles.push(engine::sim::particle::Particle {
        position: [50.0, 50.0, 50.0],
        velocity: [0.0; 3],
        force: [0.0; 3],
        mass: 1.0,
    });

    forces::interpolate_forces_to_particles(&mut ps, &grid, &fx, &fy, &fz);

    let p = &ps.particles[0];
    assert!(p.force[0].abs() > 0.0, "Interpolated fx should be non-zero");
    assert!(p.force[1].abs() < 1e-4, "Interpolated fy should be ~zero");
    assert!(p.force[2].abs() < 1e-4, "Interpolated fz should be ~zero");
}

// ---------------------------------------------------------------------------
// Particle integration
// ---------------------------------------------------------------------------

#[test]
fn particles_wrap_periodically() {
    let mut ps = ParticleSet::new();
    ps.box_size = 10.0;
    ps.particles.push(engine::sim::particle::Particle {
        position: [9.5, 0.5, 5.0],
        velocity: [2.0, -2.0, 0.0],
        force: [0.0; 3],
        mass: 1.0,
    });

    ps.integrate(1.0);

    let p = &ps.particles[0];
    // 9.5 + 2.0*1.0 = 11.5 => 1.5 (wrapped)
    assert!(
        (p.position[0] - 1.5).abs() < 0.01,
        "x should wrap: got {}",
        p.position[0]
    );
    // 0.5 + (-2.0)*1.0 = -1.5 => 8.5 (wrapped)
    assert!(
        (p.position[1] - 8.5).abs() < 0.01,
        "y should wrap: got {}",
        p.position[1]
    );
}

#[test]
fn kick_updates_velocity() {
    let mut ps = ParticleSet::new();
    ps.box_size = 10.0;
    ps.particles.push(engine::sim::particle::Particle {
        position: [5.0, 5.0, 5.0],
        velocity: [0.0, 0.0, 0.0],
        force: [10.0, -5.0, 2.0],
        mass: 1.0,
    });

    ps.kick(0.1); // half_dt = 0.05
    let p = &ps.particles[0];
    assert!((p.velocity[0] - 0.5).abs() < 1e-5);
    assert!((p.velocity[1] - (-0.25)).abs() < 1e-5);
    assert!((p.velocity[2] - 0.1).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// Zel'dovich ICs
// ---------------------------------------------------------------------------

#[test]
fn zeldovich_produces_correct_particle_count() {
    let mut ps = ParticleSet::new();
    ps.initialize_zeldovich(16, 100.0, 7);
    assert_eq!(ps.particles.len(), 16 * 16 * 16);
    assert_eq!(ps.box_size, 100.0);
}

#[test]
fn zeldovich_particles_inside_box() {
    let mut ps = ParticleSet::new();
    ps.initialize_zeldovich(16, 50.0, 99);

    for (i, p) in ps.particles.iter().enumerate() {
        for dim in 0..3 {
            assert!(
                p.position[dim] >= 0.0 && p.position[dim] < 50.0,
                "Particle {i} dim {dim}: position {} out of box [0, 50)",
                p.position[dim]
            );
        }
    }
}

#[test]
fn zeldovich_is_deterministic() {
    let mut ps1 = ParticleSet::new();
    ps1.initialize_zeldovich(8, 100.0, 42);

    let mut ps2 = ParticleSet::new();
    ps2.initialize_zeldovich(8, 100.0, 42);

    for (a, b) in ps1.particles.iter().zip(ps2.particles.iter()) {
        assert_eq!(a.position, b.position);
        assert_eq!(a.velocity, b.velocity);
    }
}

#[test]
fn zeldovich_different_seeds_differ() {
    let mut ps1 = ParticleSet::new();
    ps1.initialize_zeldovich(8, 100.0, 1);

    let mut ps2 = ParticleSet::new();
    ps2.initialize_zeldovich(8, 100.0, 2);

    let differ = ps1
        .particles
        .iter()
        .zip(ps2.particles.iter())
        .any(|(a, b)| a.position != b.position);
    assert!(differ, "Different seeds should produce different ICs");
}

// ---------------------------------------------------------------------------
// 2D projection
// ---------------------------------------------------------------------------

#[test]
fn projection_conserves_mass() {
    let mut ps = ParticleSet::new();
    ps.initialize_zeldovich(8, 100.0, 55);

    let map = ps.project_to_2d(64);
    assert_eq!(map.len(), 64 * 64);

    let total: f32 = map.iter().sum();
    let expected = ps.particles.len() as f32; // each particle has mass 1
    let rel_err = (total - expected).abs() / expected;
    assert!(
        rel_err < 0.01,
        "Projection should conserve mass: got {total}, expected {expected} (err {rel_err:.4})"
    );
}

#[test]
fn projection_all_positive() {
    let mut ps = ParticleSet::new();
    ps.initialize_zeldovich(8, 100.0, 77);
    let map = ps.project_to_2d(32);
    assert!(
        map.iter().all(|&v| v >= 0.0),
        "Projected density should be non-negative"
    );
}

// ---------------------------------------------------------------------------
// Full simulation integration test
// ---------------------------------------------------------------------------

#[test]
fn run_simulation_produces_valid_output() {
    let map = engine::run_simulation(42, 8, 100.0, 0.01, 5, 32);
    assert_eq!(map.len(), 32 * 32);

    // All values should be finite and non-negative
    assert!(
        map.iter().all(|v| v.is_finite()),
        "All pixels should be finite"
    );
    assert!(
        map.iter().all(|&v| v >= 0.0),
        "All pixels should be non-negative"
    );

    // Should have some structure (not all identical)
    let min = map.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = map.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(max > min, "Output should have variation, not uniform");
}

#[test]
fn run_simulation_deterministic() {
    let m1 = engine::run_simulation(42, 8, 100.0, 0.01, 3, 16);
    let m2 = engine::run_simulation(42, 8, 100.0, 0.01, 3, 16);
    assert_eq!(m1, m2, "Same seed should produce identical output");
}

#[test]
fn run_simulation_different_seeds_differ() {
    let m1 = engine::run_simulation(1, 8, 100.0, 0.01, 3, 16);
    let m2 = engine::run_simulation(2, 8, 100.0, 0.01, 3, 16);
    assert_ne!(m1, m2, "Different seeds should produce different output");
}
