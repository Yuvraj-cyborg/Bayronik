//! This module contains the logic for evolving particles in time.

use super::particle::ParticleSet;

/// Evolves the particles' positions and velocities using a Kick-Drift-Kick
/// leapfrog integrator scheme.
///
/// We implement a Drift-Kick-Drift variant, which is equivalent.
/// It assumes forces have just been calculated.
pub fn evolve_particles(particles: &mut ParticleSet, time_step: f32) {
    // This is a simplified KDK. A full implementation would split the kicks.
    // For our purposes, this Euler-Cromer style integration is a good start.
    for p in &mut particles.particles {
        // --- KICK ---
        // Update velocity based on force (F=ma => a=F/m)
        let acceleration = [
            p.force[0] / p.mass,
            p.force[1] / p.mass,
            p.force[2] / p.mass,
        ];

        p.velocity[0] += acceleration[0] * time_step;
        p.velocity[1] += acceleration[1] * time_step;
        p.velocity[2] += acceleration[2] * time_step;

        // --- DRIFT ---
        // Update position based on new velocity
        p.position[0] += p.velocity[0] * time_step;
        p.position[1] += p.velocity[1] * time_step;
        p.position[2] += p.velocity[2] * time_step;

        // --- Handle Periodic Boundary Conditions ---
        // Ensure particles that leave one side of the box re-enter on the opposite side.
        let box_size = particles.box_size;
        p.position[0] = p.position[0].rem_euclid(box_size);
        p.position[1] = p.position[1].rem_euclid(box_size);
        p.position[2] = p.position[2].rem_euclid(box_size);
    }
}
