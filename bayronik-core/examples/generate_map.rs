use bayronik_core::{run_simulation, output};

fn main() -> anyhow::Result<()> {
    println!("Generating N-body simulation map...");
    
    let map = run_simulation(
        32_768,  // particles
        64,      // grid resolution
        100.0,   // box size (Mpc/h)
        0.01,    // time step
        10,      // steps
        256,     // output resolution
    );
    
    let output_path = "nbody_map_256.npy";
    output::save_map_npy(&map, 256, output_path)?;
    println!("Saved to {}", output_path);
    
    let mean = map.iter().sum::<f32>() / map.len() as f32;
    let std = (map.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / map.len() as f32).sqrt();
    println!("Stats: mean={:.3e}, std={:.3e}", mean, std);
    
    Ok(())
}

