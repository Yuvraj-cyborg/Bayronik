use engine::{run_simulation, SimConfig};

fn main() {
    println!("engine: cosmological PM N-body simulation");

    let config = SimConfig::default();
    println!(
        "box={} Mpc/h, grid={}^3, z_init={}, steps={}, Om={}, s8={}",
        config.box_size,
        config.grid_res,
        config.z_init,
        config.n_steps,
        config.cosmo.omega_m,
        config.cosmo.sigma8
    );

    let map = run_simulation(&config);

    let mean = map.iter().sum::<f32>() / map.len() as f32;
    let max = map.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = map.iter().cloned().fold(f32::INFINITY, f32::min);
    println!(
        "Map: {res}x{res} Msun/h/(Mpc/h)^2, mean={mean:.3e}, min={min:.3e}, max={max:.3e}",
        res = config.projection_res
    );
}
