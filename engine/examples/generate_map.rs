use engine::{output, run_simulation};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let grid_res = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let box_size = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100.0);
    let steps = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let proj_res = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(256);
    let output_path = args
        .get(5)
        .cloned()
        .unwrap_or_else(|| "nbody_map_256.npy".to_string());
    let seed: u64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(42);

    println!(
        "grid_res={grid_res}, box_size={box_size}, steps={steps}, proj_res={proj_res}, seed={seed}"
    );

    let map = run_simulation(seed, grid_res, box_size, 0.01, steps, proj_res);

    output::save_map_npy(&map, proj_res, &output_path)?;
    println!("Saved to {}", output_path);

    let mean = map.iter().sum::<f32>() / map.len() as f32;
    let std = (map.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / map.len() as f32).sqrt();
    let max = map.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = map.iter().cloned().fold(f32::INFINITY, f32::min);
    println!(
        "Stats: mean={:.3e}, std={:.3e}, min={:.3e}, max={:.3e}",
        mean, std, min, max
    );

    Ok(())
}
