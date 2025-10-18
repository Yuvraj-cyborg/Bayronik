//! This module handles input/output for the simulation, like saving particle data.

use crate::sim::particle::ParticleSet;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

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
