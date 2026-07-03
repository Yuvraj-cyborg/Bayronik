use engine::{output, run_simulation, SimConfig};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let grid_res = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let box_size = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(25.0);
    let steps = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
    let proj_res = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(256);
    let output_path = args
        .get(5)
        .cloned()
        .unwrap_or_else(|| "nbody_map_256.npy".to_string());
    let seed: u64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(42);
    let slab_fraction: f32 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(1.0);

    let config = SimConfig {
        seed,
        grid_res,
        box_size,
        n_steps: steps,
        projection_res: proj_res,
        slab_fraction,
        ..Default::default()
    };

    println!(
        "grid_res={grid_res}, box_size={box_size} Mpc/h, steps={steps}, proj_res={proj_res}, seed={seed}, slab={slab_fraction}"
    );

    let map = run_simulation(&config);

    output::save_map_npy(&map, proj_res, &output_path)?;
    println!("Saved to {}", output_path);

    let mean = map.iter().sum::<f32>() / map.len() as f32;
    let std = (map.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / map.len() as f32).sqrt();
    let max = map.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = map.iter().cloned().fold(f32::INFINITY, f32::min);
    println!(
        "Stats [Msun/h/(Mpc/h)^2]: mean={:.3e}, std={:.3e}, min={:.3e}, max={:.3e}",
        mean, std, min, max
    );

    Ok(())
}
