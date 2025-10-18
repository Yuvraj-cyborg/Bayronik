//! Defines the basic particle structure for the N-body simulation.

use rand::Rng;

/// Represents a single particle in the simulation.
pub struct Particle {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub force: [f32; 3], // The force acting on the particle
    pub mass: f32,
}

/// A collection of all particles in the simulation.
pub struct ParticleSet {
    pub particles: Vec<Particle>,
    pub box_size: f32,
}

impl ParticleSet {
    /// Creates a new, empty particle set.
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            box_size: 0.0,
        }
    }

    /// Initializes a set of particles with random positions.
    pub fn initialize_randomly(&mut self, num_particles: usize, box_size: f32) {
        self.box_size = box_size;
        let mut rng = rand::rng();
        self.particles = (0..num_particles)
            .map(|_| Particle {
                position: [
                    rng.random::<f32>() * box_size,
                    rng.random::<f32>() * box_size,
                    rng.random::<f32>() * box_size,
                ],
                velocity: [0.0, 0.0, 0.0],
                force: [0.0, 0.0, 0.0],
                mass: 1.0, // Assuming all particles have the same mass for now
            })
            .collect();
    }
}

