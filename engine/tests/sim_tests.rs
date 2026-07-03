use engine::{forces, gravity, Cosmology, FftSolver, Grid, ParticleSet, SimConfig};

const CAMELS_BOX: f32 = 25.0;
const Z_INIT: f64 = 49.0;
const A_INIT: f64 = 1.0 / (1.0 + Z_INIT);

fn ics(grid_res: usize, box_size: f32, seed: u64) -> ParticleSet {
    let mut ps = ParticleSet::new();
    ps.initialize_zeldovich(grid_res, box_size, seed, &Cosmology::default(), A_INIT);
    ps
}

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

#[test]
fn grid_initializes_to_zero() {
    let g = Grid::new(8, CAMELS_BOX);
    assert_eq!(g.density_contrast.len(), 8 * 8 * 8);
    assert!(g.density_contrast.iter().all(|&v| v == 0.0));
    assert!(g.potential.iter().all(|&v| v == 0.0));
}

#[test]
fn grid_clear_resets_density() {
    let mut g = Grid::new(4, CAMELS_BOX);
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
    let ps = ics(8, CAMELS_BOX, 123);
    assert_eq!(ps.particles.len(), 8 * 8 * 8);

    let mut grid = Grid::new(8, CAMELS_BOX);
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

    let sum: f32 = grid.density_contrast.iter().sum();
    assert!(
        sum.abs() < 0.1,
        "Single-particle CIC sum should be ~0, got {sum}"
    );

    let idx = (2 * 4 + 2) * 4 + 2;
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
    let mut grid = Grid::new(8, CAMELS_BOX);
    let mut solver = FftSolver::new(8);
    solver.solve_potential(&mut grid, 1.0);
    let max_pot: f32 = grid.potential.iter().map(|v| v.abs()).fold(0.0, f32::max);
    assert!(
        max_pot < 1e-6,
        "Zero density should give zero potential, max = {max_pot}"
    );
}

#[test]
fn poisson_point_mass_potential_well() {
    let n = 16;
    let mut grid = Grid::new(n, CAMELS_BOX);

    let center = (n / 2 * n + n / 2) * n + n / 2;
    grid.density_contrast[center] = 1.0;

    let mut solver = FftSolver::new(n);
    solver.solve_potential(&mut grid, 1.0);

    let pot_center = grid.potential[center];
    let pot_corner = grid.potential[0];
    assert!(
        pot_center < pot_corner,
        "Potential at point mass ({pot_center}) should be deeper than corner ({pot_corner})"
    );
}

#[test]
fn poisson_prefactor_scales_potential_linearly() {
    let n = 8;
    let mut solver = FftSolver::new(n);

    let mut grid1 = Grid::new(n, CAMELS_BOX);
    grid1.density_contrast[100] = 1.0;
    solver.solve_potential(&mut grid1, 1.0);

    let mut grid2 = Grid::new(n, CAMELS_BOX);
    grid2.density_contrast[100] = 1.0;
    solver.solve_potential(&mut grid2, 3.0);

    for (a, b) in grid1.potential.iter().zip(grid2.potential.iter()) {
        assert!(
            (3.0 * a - b).abs() < 1e-4 * b.abs().max(1e-6),
            "Potential must scale linearly with prefactor: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Forces
// ---------------------------------------------------------------------------

#[test]
fn forces_from_uniform_potential_are_zero() {
    let mut grid = Grid::new(8, CAMELS_BOX);
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
fn forces_oppose_potential_gradient() {
    let n = 8;
    let box_size = CAMELS_BOX;
    let mut grid = Grid::new(n, box_size);
    let cell_size = box_size / n as f32;

    // phi linear in x: interior force must be exactly -1.
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                grid.potential[(i * n + j) * n + k] = i as f32 * cell_size;
            }
        }
    }

    let (fx, fy, fz) = forces::calculate_forces_from_potential(&grid);

    for i in 1..n - 1 {
        for j in 0..n {
            for k in 0..n {
                let idx = (i * n + j) * n + k;
                assert!(
                    (fx[idx] + 1.0).abs() < 1e-5,
                    "Interior x-force should be -1, got {} at ({i},{j},{k})",
                    fx[idx]
                );
            }
        }
    }

    let fy_max: f32 = fy.iter().map(|v| v.abs()).fold(0.0, f32::max);
    let fz_max: f32 = fz.iter().map(|v| v.abs()).fold(0.0, f32::max);
    assert!(fy_max < 1e-5, "fy should be zero for x-only gradient");
    assert!(fz_max < 1e-5, "fz should be zero for x-only gradient");
}

#[test]
fn force_interpolation_works() {
    let n = 8;
    let box_size = CAMELS_BOX;
    let mut grid = Grid::new(n, box_size);

    let cell_size = box_size / n as f32;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                grid.potential[(i * n + j) * n + k] = i as f32 * cell_size;
            }
        }
    }

    let (fx, fy, fz) = forces::calculate_forces_from_potential(&grid);

    let mut ps = ParticleSet::new();
    ps.box_size = box_size;
    ps.particles.push(engine::sim::particle::Particle {
        position: [box_size / 2.0, box_size / 2.0, box_size / 2.0],
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
// Drift / kick operators
// ---------------------------------------------------------------------------

#[test]
fn drift_moves_and_wraps_periodically() {
    let mut ps = ParticleSet::new();
    ps.box_size = 10.0;
    ps.particles.push(engine::sim::particle::Particle {
        position: [9.5, 0.5, 5.0],
        velocity: [2.0, -2.0, 0.0],
        force: [0.0; 3],
        mass: 1.0,
    });

    ps.drift(1.0);

    let p = &ps.particles[0];
    assert!(
        (p.position[0] - 1.5).abs() < 1e-5,
        "x should wrap: got {}",
        p.position[0]
    );
    assert!(
        (p.position[1] - 8.5).abs() < 1e-5,
        "y should wrap: got {}",
        p.position[1]
    );
}

#[test]
fn kick_updates_momentum_by_force_times_coefficient() {
    let mut ps = ParticleSet::new();
    ps.box_size = 10.0;
    ps.particles.push(engine::sim::particle::Particle {
        position: [5.0, 5.0, 5.0],
        velocity: [0.0, 0.0, 0.0],
        force: [10.0, -5.0, 2.0],
        mass: 1.0,
    });

    ps.kick(0.1);
    let p = &ps.particles[0];
    assert!((p.velocity[0] - 1.0).abs() < 1e-5);
    assert!((p.velocity[1] - (-0.5)).abs() < 1e-5);
    assert!((p.velocity[2] - 0.2).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// Zel'dovich ICs
// ---------------------------------------------------------------------------

#[test]
fn zeldovich_produces_correct_particle_count() {
    let ps = ics(16, CAMELS_BOX, 7);
    assert_eq!(ps.particles.len(), 16 * 16 * 16);
    assert_eq!(ps.box_size, CAMELS_BOX);
}

#[test]
fn zeldovich_particles_inside_box() {
    let ps = ics(16, CAMELS_BOX, 99);
    for (i, p) in ps.particles.iter().enumerate() {
        for dim in 0..3 {
            assert!(
                p.position[dim] >= 0.0 && p.position[dim] < CAMELS_BOX,
                "Particle {i} dim {dim}: position {} out of box",
                p.position[dim]
            );
        }
    }
}

#[test]
fn zeldovich_total_mass_matches_mean_density() {
    let cosmo = Cosmology::default();
    let ps = ics(16, CAMELS_BOX, 3);

    let expected = cosmo.mean_matter_density() * (CAMELS_BOX as f64).powi(3);
    let total = ps.total_mass();
    let rel = (total - expected).abs() / expected;
    assert!(
        rel < 1e-4,
        "Total mass {total:.4e} should equal Omega_m rho_crit V = {expected:.4e}"
    );
}

#[test]
fn zeldovich_displacements_small_at_high_z() {
    // At z = 49 the rms displacement must be well below a grid cell,
    // otherwise the ICs are in the shell-crossing regime and invalid.
    let n = 16;
    let ps = ics(n, CAMELS_BOX, 11);
    let cell = CAMELS_BOX / n as f32;

    let mut sum_sq = 0.0f64;
    for (i, p) in ps.particles.iter().enumerate() {
        let iz = i % n;
        let iy = (i / n) % n;
        let ix = i / (n * n);
        let q = [
            (ix as f32 + 0.5) * cell,
            (iy as f32 + 0.5) * cell,
            (iz as f32 + 0.5) * cell,
        ];
        for d in 0..3 {
            let mut dx = p.position[d] - q[d];
            // minimal periodic distance
            if dx > CAMELS_BOX / 2.0 {
                dx -= CAMELS_BOX;
            }
            if dx < -CAMELS_BOX / 2.0 {
                dx += CAMELS_BOX;
            }
            sum_sq += (dx as f64) * (dx as f64);
        }
    }
    let rms = (sum_sq / ps.particles.len() as f64).sqrt();
    assert!(
        rms > 0.0 && rms < cell as f64 * 0.5,
        "rms displacement {rms:.4} Mpc/h should be < half a cell ({:.4})",
        cell * 0.5
    );
}

#[test]
fn zeldovich_is_deterministic() {
    let ps1 = ics(8, CAMELS_BOX, 42);
    let ps2 = ics(8, CAMELS_BOX, 42);
    for (a, b) in ps1.particles.iter().zip(ps2.particles.iter()) {
        assert_eq!(a.position, b.position);
        assert_eq!(a.velocity, b.velocity);
    }
}

#[test]
fn zeldovich_different_seeds_differ() {
    let ps1 = ics(8, CAMELS_BOX, 1);
    let ps2 = ics(8, CAMELS_BOX, 2);
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
    let ps = ics(8, CAMELS_BOX, 55);
    let res = 64;
    let map = ps.project_to_2d(res, 1.0);
    assert_eq!(map.len(), res * res);

    // Surface density * pixel area summed over the map = total mass.
    let pixel_area = (CAMELS_BOX / res as f32) as f64 * (CAMELS_BOX / res as f32) as f64;
    let total: f64 = map.iter().map(|&v| v as f64).sum::<f64>() * pixel_area;
    let expected = ps.total_mass();
    let rel_err = (total - expected).abs() / expected;
    assert!(
        rel_err < 0.01,
        "Projection should conserve mass: got {total:.4e}, expected {expected:.4e}"
    );
}

#[test]
fn projection_slab_selects_subvolume() {
    let ps = ics(16, CAMELS_BOX, 21);
    let res = 32;
    let pixel_area = (CAMELS_BOX / res as f32) as f64 * (CAMELS_BOX / res as f32) as f64;

    let full: f64 = ps
        .project_to_2d(res, 1.0)
        .iter()
        .map(|&v| v as f64)
        .sum::<f64>()
        * pixel_area;
    let slab: f64 = ps
        .project_to_2d(res, 0.2)
        .iter()
        .map(|&v| v as f64)
        .sum::<f64>()
        * pixel_area;

    let frac = slab / full;
    assert!(
        frac > 0.1 && frac < 0.35,
        "A 20% slab should hold roughly 20% of the mass, got {frac:.3}"
    );
}

#[test]
fn projection_all_non_negative() {
    let ps = ics(8, CAMELS_BOX, 77);
    let map = ps.project_to_2d(32, 1.0);
    assert!(
        map.iter().all(|&v| v >= 0.0),
        "Projected density should be non-negative"
    );
}

// ---------------------------------------------------------------------------
// Bilinear upsampling
// ---------------------------------------------------------------------------

#[test]
fn upsample_preserves_mean() {
    let map = vec![1.0f32, 2.0, 3.0, 4.0];
    let up = engine::upsample_bilinear(&map, 2, 8);
    assert_eq!(up.len(), 64);

    let mean_in: f32 = map.iter().sum::<f32>() / 4.0;
    let mean_out: f32 = up.iter().sum::<f32>() / 64.0;
    assert!(
        (mean_in - mean_out).abs() < 1e-5,
        "Upsampling must preserve the mean: {mean_in} vs {mean_out}"
    );
}

#[test]
fn upsample_constant_stays_constant() {
    let map = vec![7.5f32; 16];
    let up = engine::upsample_bilinear(&map, 4, 16);
    assert!(up.iter().all(|&v| (v - 7.5).abs() < 1e-5));
}

// ---------------------------------------------------------------------------
// Full simulation
// ---------------------------------------------------------------------------

fn quick_config(seed: u64) -> SimConfig {
    SimConfig {
        seed,
        grid_res: 16,
        box_size: CAMELS_BOX,
        z_init: Z_INIT,
        n_steps: 8,
        projection_res: 32,
        slab_fraction: 1.0,
        cosmo: Cosmology::default(),
    }
}

#[test]
fn run_simulation_produces_valid_output() {
    let map = engine::run_simulation(&quick_config(42));
    assert_eq!(map.len(), 32 * 32);
    assert!(
        map.iter().all(|v| v.is_finite()),
        "All pixels should be finite"
    );
    assert!(
        map.iter().all(|&v| v >= 0.0),
        "All pixels should be non-negative"
    );

    let min = map.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = map.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(max > min, "Output should have variation, not uniform");
}

#[test]
fn run_simulation_mean_surface_density_is_physical() {
    // Full-box projection: mean Sigma_cdm = (Omega_m - Omega_b) rho_crit L.
    let cfg = quick_config(7);
    let map = engine::run_simulation(&cfg);

    let cosmo = cfg.cosmo;
    let expected =
        (cosmo.omega_m - cosmo.omega_b) * engine::RHO_CRIT * cfg.box_size as f64;
    let mean: f64 = map.iter().map(|&v| v as f64).sum::<f64>() / map.len() as f64;
    let rel = (mean - expected).abs() / expected;
    assert!(
        rel < 0.02,
        "Mean surface density {mean:.4e} should match {expected:.4e} (rel err {rel:.4})"
    );
}

#[test]
fn run_simulation_grows_structure() {
    // Gravitational collapse must increase the clumping factor <S^2>/<S>^2
    // well beyond its near-unity initial value.
    let cfg = SimConfig {
        grid_res: 32,
        n_steps: 16,
        projection_res: 64,
        ..quick_config(42)
    };

    // Project ICs at the particle-lattice resolution: finer grids alias the
    // regular lattice into spurious clumping.
    let ps = ics(cfg.grid_res, cfg.box_size, cfg.seed);
    let ic_map = ps.project_to_2d(cfg.grid_res, 1.0);
    let clump = |m: &[f32]| {
        let mean: f64 = m.iter().map(|&v| v as f64).sum::<f64>() / m.len() as f64;
        let mean_sq: f64 = m.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / m.len() as f64;
        mean_sq / (mean * mean)
    };

    let final_map = engine::run_simulation(&cfg);
    let c_ic = clump(&ic_map);
    let c_final = clump(&final_map);

    assert!(
        c_ic < 1.05,
        "ICs at z=49 should be nearly uniform, clumping = {c_ic:.4}"
    );
    assert!(
        c_final > 1.2,
        "Evolved field should be clumpy: clumping = {c_final:.4} (IC {c_ic:.4})"
    );
}

#[test]
fn run_simulation_deterministic() {
    let m1 = engine::run_simulation(&quick_config(42));
    let m2 = engine::run_simulation(&quick_config(42));
    assert_eq!(m1, m2, "Same seed should produce identical output");
}

#[test]
fn incremental_stepping_matches_run_simulation() {
    // The frame-by-frame Simulation API must produce bit-identical output
    // to the batch entry point.
    let cfg = quick_config(42);
    let batch = engine::run_simulation(&cfg);

    let mut sim = engine::Simulation::new(cfg);
    assert_eq!(sim.n_steps(), cfg.n_steps);
    assert!(!sim.is_done());

    let mut steps_taken = 0;
    loop {
        let more = sim.step();
        steps_taken += 1;
        if !more {
            break;
        }
    }
    assert_eq!(steps_taken, cfg.n_steps);
    assert!(sim.is_done());
    assert!(!sim.step(), "step() after completion must be a no-op");

    assert_eq!(sim.projected_map(), batch);
}

#[test]
fn run_simulation_different_seeds_differ() {
    let m1 = engine::run_simulation(&quick_config(1));
    let m2 = engine::run_simulation(&quick_config(2));
    assert_ne!(m1, m2, "Different seeds should produce different output");
}

#[test]
fn run_simulation_sigma8_controls_clustering() {
    // Higher sigma8 -> more clustering in the final map.
    let mut lo = quick_config(42);
    lo.grid_res = 32;
    lo.n_steps = 12;
    lo.projection_res = 64;
    lo.cosmo.sigma8 = 0.6;

    let mut hi = lo;
    hi.cosmo.sigma8 = 1.0;

    let clump = |m: &[f32]| {
        let mean: f64 = m.iter().map(|&v| v as f64).sum::<f64>() / m.len() as f64;
        let mean_sq: f64 = m.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / m.len() as f64;
        mean_sq / (mean * mean)
    };

    let c_lo = clump(&engine::run_simulation(&lo));
    let c_hi = clump(&engine::run_simulation(&hi));
    assert!(
        c_hi > c_lo,
        "sigma8=1.0 should cluster more than 0.6: {c_hi:.4} vs {c_lo:.4}"
    );
}
