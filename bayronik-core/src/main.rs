use bayronik_core::run_simulation;

fn main() {
    println!("bayronik-core: N-body PM simulation");

    let map = run_simulation(42, 64, 100.0, 0.01, 10, 256);

    let mean = map.iter().sum::<f32>() / map.len() as f32;
    let max = map.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = map.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("Map: 256x256, mean={mean:.3e}, min={min:.3e}, max={max:.3e}");
}
