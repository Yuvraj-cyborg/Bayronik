use super::{grid::Grid, particle::ParticleSet};

/// Compute gravitational forces on grid from potential using finite differences.
pub fn calculate_forces_from_potential(grid: &Grid) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = grid.resolution;
    let cell_size = grid.box_size / n as f32;
    let inv_2h = 0.5 / cell_size;
    
    let mut fx = vec![0.0; n * n * n];
    let mut fy = vec![0.0; n * n * n];
    let mut fz = vec![0.0; n * n * n];
    
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let idx = (i * n + j) * n + k;
                
                let ip = ((i + 1) % n * n + j) * n + k;
                let im = ((i + n - 1) % n * n + j) * n + k;
                let jp = (i * n + (j + 1) % n) * n + k;
                let jm = (i * n + (j + n - 1) % n) * n + k;
                let kp = (i * n + j) * n + (k + 1) % n;
                let km = (i * n + j) * n + (k + n - 1) % n;
                
                fx[idx] = -(grid.potential[ip] - grid.potential[im]) * inv_2h;
                fy[idx] = -(grid.potential[jp] - grid.potential[jm]) * inv_2h;
                fz[idx] = -(grid.potential[kp] - grid.potential[km]) * inv_2h;
            }
        }
    }
    
    (fx, fy, fz)
}

/// Interpolate forces from grid to particles using CIC.
pub fn interpolate_forces_to_particles(
    particles: &mut ParticleSet,
    grid: &Grid,
    fx: &[f32],
    fy: &[f32],
    fz: &[f32],
) {
    let n = grid.resolution;
    let n_f32 = n as f32;
    let cell_size = grid.box_size / n_f32;
    let inv_cell_size = 1.0 / cell_size;
    
    for p in &mut particles.particles {
        let pos_grid = [
            p.position[0] * inv_cell_size,
            p.position[1] * inv_cell_size,
            p.position[2] * inv_cell_size,
        ];
        
        let i = pos_grid[0].floor() as isize;
        let j = pos_grid[1].floor() as isize;
        let k = pos_grid[2].floor() as isize;
        
        let dx = pos_grid[0] - i as f32;
        let dy = pos_grid[1] - j as f32;
        let dz = pos_grid[2] - k as f32;
        
        let w = [
            (1.0 - dx) * (1.0 - dy) * (1.0 - dz),
            dx * (1.0 - dy) * (1.0 - dz),
            (1.0 - dx) * dy * (1.0 - dz),
            (1.0 - dx) * (1.0 - dy) * dz,
            dx * dy * (1.0 - dz),
            dx * (1.0 - dy) * dz,
            (1.0 - dx) * dy * dz,
            dx * dy * dz,
        ];
        
        let n_isize = n as isize;
        let indices = [
            (((i % n_isize + n_isize) % n_isize * n_isize + (j % n_isize + n_isize) % n_isize) * n_isize + (k % n_isize + n_isize) % n_isize) as usize,
            ((((i + 1) % n_isize + n_isize) % n_isize * n_isize + (j % n_isize + n_isize) % n_isize) * n_isize + (k % n_isize + n_isize) % n_isize) as usize,
            (((i % n_isize + n_isize) % n_isize * n_isize + ((j + 1) % n_isize + n_isize) % n_isize) * n_isize + (k % n_isize + n_isize) % n_isize) as usize,
            (((i % n_isize + n_isize) % n_isize * n_isize + (j % n_isize + n_isize) % n_isize) * n_isize + ((k + 1) % n_isize + n_isize) % n_isize) as usize,
            ((((i + 1) % n_isize + n_isize) % n_isize * n_isize + ((j + 1) % n_isize + n_isize) % n_isize) * n_isize + (k % n_isize + n_isize) % n_isize) as usize,
            ((((i + 1) % n_isize + n_isize) % n_isize * n_isize + (j % n_isize + n_isize) % n_isize) * n_isize + ((k + 1) % n_isize + n_isize) % n_isize) as usize,
            (((i % n_isize + n_isize) % n_isize * n_isize + ((j + 1) % n_isize + n_isize) % n_isize) * n_isize + ((k + 1) % n_isize + n_isize) % n_isize) as usize,
            ((((i + 1) % n_isize + n_isize) % n_isize * n_isize + ((j + 1) % n_isize + n_isize) % n_isize) * n_isize + ((k + 1) % n_isize + n_isize) % n_isize) as usize,
        ];
        
        p.force[0] = (0..8).map(|idx| w[idx] * fx[indices[idx]]).sum();
        p.force[1] = (0..8).map(|idx| w[idx] * fy[indices[idx]]).sum();
        p.force[2] = (0..8).map(|idx| w[idx] * fz[indices[idx]]).sum();
    }
}

