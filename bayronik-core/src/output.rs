use ndarray::Array2;
use std::fs::File;
use crate::sim::particle::ParticleSet;
use std::io::{self, Write};
use std::path::Path;

/// Save 2D map to NPY format compatible with Python/NumPy.
pub fn save_map_npy(map: &[f32], resolution: usize, path: &str) -> anyhow::Result<()> {
    let array = Array2::from_shape_vec((resolution, resolution), map.to_vec())?;
    ndarray_npy::write_npy(path, &array)?;
    Ok(())
}

/// Save 2D map as plain text for debugging.
pub fn save_map_txt(map: &[f32], resolution: usize, path: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    for row in map.chunks(resolution) {
        let line = row.iter()
            .map(|v| format!("{:.6e}", v))
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(file, "{}", line)?;
    }
    Ok(())
}

/// Saves the positions of all particles to a simple CSV file.
/// Each line will contain the x, y, and z coordinates of a particle.
pub fn save_particle_positions(particles: &ParticleSet, filepath: &str) -> io::Result<()> {
    let path = Path::new(filepath);
    let mut file = File::create(&path)?;

    // Write a header (optional, but good practice)
    writeln!(file, "x,y,z")?;

    // Write each particle's position
    for p in &particles.particles {
        writeln!(file, "{},{},{}", p.position[0], p.position[1], p.position[2])?;
    }

    Ok(())
}
