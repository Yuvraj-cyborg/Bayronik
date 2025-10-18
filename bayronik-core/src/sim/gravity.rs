//! This module handles gravity-related calculations like mass assignment
//! and force computation.

use super::{grid::Grid, particle::ParticleSet};

/// Assigns particle mass to the grid using the Cloud-in-Cell (CIC) scheme.
/// This method provides a smoother density field than Nearest Grid Point.
pub fn assign_mass_cic(particles: &ParticleSet, grid: &mut Grid) {
    let n = grid.resolution;
    let n_f32 = n as f32;
    let cell_size = grid.box_size / n_f32;
    let inv_cell_size = 1.0 / cell_size;

    // First, calculate the total mass and average density.
    let total_mass: f32 = particles.particles.iter().map(|p| p.mass).sum();
    let total_cells = (n * n * n) as f32;
    let mean_density = total_mass / total_cells;

    // Assign mass from each particle to the 8 surrounding grid cells.
    for p in &particles.particles {
        // Find the particle's position in grid coordinates.
        let pos_grid = [
            p.position[0] * inv_cell_size,
            p.position[1] * inv_cell_size,
            p.position[2] * inv_cell_size,
        ];

        // Find the integer index of the "base" grid cell (the bottom-left-front corner).
        let i = pos_grid[0].floor() as isize;
        let j = pos_grid[1].floor() as isize;
        let k = pos_grid[2].floor() as isize;

        // Find the particle's fractional distance within that cell (dx, dy, dz).
        let dx = pos_grid[0] - i as f32;
        let dy = pos_grid[1] - j as f32;
        let dz = pos_grid[2] - k as f32;

        // Calculate the weights for the 8 corners of the cube.
        let w_000 = (1.0 - dx) * (1.0 - dy) * (1.0 - dz);
        let w_100 = dx * (1.0 - dy) * (1.0 - dz);
        let w_010 = (1.0 - dx) * dy * (1.0 - dz);
        let w_001 = (1.0 - dx) * (1.0 - dy) * dz;
        let w_110 = dx * dy * (1.0 - dz);
        let w_101 = dx * (1.0 - dy) * dz;
        let w_011 = (1.0 - dx) * dy * dz;
        let w_111 = dx * dy * dz;

        // Distribute the particle's mass to the 8 cells, handling periodic boundaries.
        // We use isize and the modulo operator `%` to wrap around the grid edges.
        let n_isize = n as isize;
        let idx = |x: isize, y: isize, z: isize| {
            (((x.rem_euclid(n_isize)) * n_isize + y.rem_euclid(n_isize)) * n_isize + z.rem_euclid(n_isize))
                as usize
        };

        grid.density_contrast[idx(i, j, k)] += p.mass * w_000;
        grid.density_contrast[idx(i + 1, j, k)] += p.mass * w_100;
        grid.density_contrast[idx(i, j + 1, k)] += p.mass * w_010;
        grid.density_contrast[idx(i, j, k + 1)] += p.mass * w_001;
        grid.density_contrast[idx(i + 1, j + 1, k)] += p.mass * w_110;
        grid.density_contrast[idx(i + 1, j, k + 1)] += p.mass * w_101;
        grid.density_contrast[idx(i, j + 1, k + 1)] += p.mass * w_011;
        grid.density_contrast[idx(i + 1, j + 1, k + 1)] += p.mass * w_111;
    }

    // Convert the raw density to density contrast: (rho - rho_mean) / rho_mean
    if mean_density > 1e-6 {
        for val in &mut grid.density_contrast {
            *val = (*val / mean_density) - 1.0;
        }
    }
}
