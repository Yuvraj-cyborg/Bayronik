use ndarray::Array2;
use std::fs::File;
use std::io::Write;

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

