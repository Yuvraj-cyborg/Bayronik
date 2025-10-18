//! Defines the mesh for the Particle-Mesh gravity calculation.

/// Represents the 3D grid (mesh) used to compute gravitational forces.
pub struct Grid {
    /// Number of grid cells along one dimension.
    pub resolution: usize,
    /// The size of the simulation box.
    pub box_size: f32,
    /// A flattened 3D array storing the density at each grid point.
    /// The value is (density - average_density).
    pub density_contrast: Vec<f32>,
    /// A flattened 3D array storing the gravitational potential.
    pub potential: Vec<f32>,
    /// A flattened 3D array storing the force vector [fx, fy, fz] at each grid point.
    pub force: Vec<[f32; 3]>,
}

impl Grid {
    /// Creates a new grid for the simulation.
    pub fn new(resolution: usize, box_size: f32) -> Self {
        let num_cells = resolution * resolution * resolution;
        Self {
            resolution,
            box_size,
            density_contrast: vec![0.0; num_cells],
            potential: vec![0.0; num_cells],
            force: vec![[0.0, 0.0, 0.0]; num_cells],
        }
    }

    /// Clears the density field, resetting all values to zero.
    /// This is done at the start of each simulation step.
    pub fn clear_density(&mut self) {
        self.density_contrast.iter_mut().for_each(|d| *d = 0.0);
    }
}

