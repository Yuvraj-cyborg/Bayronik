/// Cosmological analysis functions (pure Rust, WASM-compatible).
///
/// Mirrors the Python analysis in webapp.py but runs entirely client-side.

/// Isotropic 2D power spectrum P(k) of a square field.
///
/// Uses a simple DFT via the Cooley-Tukey-style approach with f64 arithmetic.
/// Returns (k_centers, pk) arrays with only bins where pk > 0.
pub fn power_spectrum(field: &[f32], n: usize) -> (Vec<f64>, Vec<f64>) {
    let nf = n as f64;

    // Compute 2D DFT magnitudes squared (real-valued DFT via direct sum for
    // modest resolutions; for 256x256 this is ~65k complex multiplies per
    // frequency -- we instead use a row-column 1D FFT approach).
    //
    // For WASM we use a simple radix-2 FFT when n is a power of 2, otherwise
    // fall back to DFT. 256 is a power of 2 so FFT path is taken.
    let pk2d = fft2d_power(field, n);

    let k_max = n / 2;
    let mut pk_sum = vec![0.0f64; k_max];
    let mut pk_cnt = vec![0usize; k_max];
    let mut k_sum = vec![0.0f64; k_max];

    for iy in 0..n {
        let ky = if iy <= n / 2 { iy as f64 } else { iy as f64 - nf };
        for ix in 0..n {
            let kx = if ix <= n / 2 { ix as f64 } else { ix as f64 - nf };
            let k_mag = (kx * kx + ky * ky).sqrt();
            let bin = k_mag.floor() as usize;
            if bin >= 1 && bin < k_max {
                pk_sum[bin] += pk2d[iy * n + ix];
                k_sum[bin] += k_mag;
                pk_cnt[bin] += 1;
            }
        }
    }

    let mut k_out = Vec::with_capacity(k_max);
    let mut pk_out = Vec::with_capacity(k_max);

    for i in 1..k_max {
        if pk_cnt[i] > 0 {
            let avg = pk_sum[i] / pk_cnt[i] as f64;
            if avg > 0.0 {
                k_out.push(k_sum[i] / pk_cnt[i] as f64);
                pk_out.push(avg);
            }
        }
    }

    (k_out, pk_out)
}

/// Baryon suppression ratio S(k) = P_tot(k) / P_dm(k).
pub fn baryon_suppression(
    k_dm: &[f64],
    pk_dm: &[f64],
    k_tot: &[f64],
    pk_tot: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    if k_dm.is_empty() || k_tot.is_empty() {
        return (vec![1.0], vec![1.0]);
    }

    let k_min = k_dm[0].max(k_tot[0]);
    let k_max = k_dm[k_dm.len() - 1].min(k_tot[k_tot.len() - 1]);
    if k_max <= k_min {
        return (vec![1.0], vec![1.0]);
    }

    let n = k_dm.len().min(k_tot.len()).max(2);
    let mut k_common = Vec::with_capacity(n);
    let mut suppression = Vec::with_capacity(n);

    for i in 0..n {
        let k = k_min + (k_max - k_min) * i as f64 / (n - 1) as f64;
        let dm = interp(k, k_dm, pk_dm);
        let tot = interp(k, k_tot, pk_tot);
        k_common.push(k);
        suppression.push(if dm > 0.0 { tot / dm } else { 1.0 });
    }

    (k_common, suppression)
}

/// 1-point PDF of log(1+rho).
pub fn pixel_pdf(field: &[f32], n_bins: usize) -> (Vec<f64>, Vec<f64>) {
    let log_vals: Vec<f64> = field
        .iter()
        .map(|&v| (v.max(0.0) as f64 + 1.0).ln())
        .filter(|v| v.is_finite())
        .collect();

    if log_vals.is_empty() {
        return (vec![0.0], vec![0.0]);
    }

    let min_v = log_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_v = log_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_v - min_v;
    if range < 1e-12 {
        return (vec![min_v], vec![1.0]);
    }

    let mut counts = vec![0usize; n_bins];
    for &v in &log_vals {
        let bin = ((v - min_v) / range * n_bins as f64).floor() as usize;
        let bin = bin.min(n_bins - 1);
        counts[bin] += 1;
    }

    let total = log_vals.len() as f64;
    let bin_width = range / n_bins as f64;

    let centers: Vec<f64> = (0..n_bins)
        .map(|i| min_v + (i as f64 + 0.5) * bin_width)
        .collect();
    let density: Vec<f64> = counts.iter().map(|&c| c as f64 / (total * bin_width)).collect();

    (centers, density)
}

/// Safe log1p that clamps negatives.
pub fn safe_log1p_field(field: &[f32]) -> Vec<f32> {
    field.iter().map(|&v| (v.max(0.0) + 1.0).ln()).collect()
}

/// Mean of log1p(Mcdm) over CAMELS IllustrisTNG LH z=0 column-density maps.
/// Measured from `model/data/Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy` (10 random
/// samples). Inputs to the U-FNO emulator are expected to live in this
/// distribution because the training pipeline applies `log1p` before the
/// forward pass.
pub const CAMELS_LOG1P_MEAN: f32 = 25.5;
pub const CAMELS_LOG1P_STD: f32 = 1.04;

/// Rescale a field so its `log1p` distribution matches the CAMELS Mcdm
/// training set. This makes out-of-distribution inputs (e.g., a particle-mesh
/// projection in raw particle counts) consumable by a model trained on
/// `log1p(column_density)` in `Msun/h/Mpc^2`.
///
/// The transform is affine in log space:
///     log1p(out) = (log1p(field) - μ_field) / σ_field * σ_camels + μ_camels
/// so spatial structure (clustering, Fourier modes) is preserved exactly;
/// only the absolute scale and contrast are normalized. Fields already in the
/// CAMELS distribution are left effectively unchanged.
pub fn calibrate_to_camels(field: &[f32]) -> Vec<f32> {
    let n = field.len();
    if n == 0 {
        return Vec::new();
    }

    let log_field: Vec<f32> = field.iter().map(|&v| (v.max(0.0) + 1.0).ln()).collect();

    let inv_n = 1.0 / n as f32;
    let mean: f32 = log_field.iter().sum::<f32>() * inv_n;
    let var: f32 = log_field.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() * inv_n;
    let std = var.sqrt().max(1e-6);

    log_field
        .iter()
        .map(|&v| {
            let z = (v - mean) / std;
            let rescaled = z * CAMELS_LOG1P_STD + CAMELS_LOG1P_MEAN;
            (rescaled.exp() - 1.0).max(0.0)
        })
        .collect()
}

/// Pixel-wise log-space difference: `log1p(out) - log1p(inp)`.
///
/// In CAMELS-scale units this is approximately `ln(M_tot / M_dm)`, the
/// standard way to visualize the baryonic effect. Positive values mark
/// pixels where baryons accumulated; negative values mark pixels where
/// feedback evacuated mass.
pub fn log_diff(out: &[f32], inp: &[f32]) -> Vec<f32> {
    out.iter()
        .zip(inp.iter())
        .map(|(&o, &i)| (o.max(0.0) + 1.0).ln() - (i.max(0.0) + 1.0).ln())
        .collect()
}

// ---- Internal helpers ----

fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let pos = xs.partition_point(|&v| v < x);
    let i = if pos == 0 { 0 } else { pos - 1 };
    let j = (i + 1).min(xs.len() - 1);
    if (xs[j] - xs[i]).abs() < 1e-15 {
        return ys[i];
    }
    let t = (x - xs[i]) / (xs[j] - xs[i]);
    ys[i] + t * (ys[j] - ys[i])
}

/// Compute |FFT2D(field)|^2 / N^4 using row-column 1D FFT.
fn fft2d_power(field: &[f32], n: usize) -> Vec<f64> {
    let n2 = n * n;
    let nf4 = (n as f64).powi(4);

    // Working buffers: real and imaginary parts
    let mut re = vec![0.0f64; n2];
    let mut im = vec![0.0f64; n2];

    for i in 0..n2 {
        re[i] = field[i] as f64;
    }

    // FFT along rows
    let mut row_re = vec![0.0f64; n];
    let mut row_im = vec![0.0f64; n];
    for y in 0..n {
        let off = y * n;
        row_re.copy_from_slice(&re[off..off + n]);
        row_im.copy_from_slice(&im[off..off + n]);
        fft1d(&mut row_re, &mut row_im, false);
        re[off..off + n].copy_from_slice(&row_re);
        im[off..off + n].copy_from_slice(&row_im);
    }

    // FFT along columns
    let mut col_re = vec![0.0f64; n];
    let mut col_im = vec![0.0f64; n];
    for x in 0..n {
        for y in 0..n {
            col_re[y] = re[y * n + x];
            col_im[y] = im[y * n + x];
        }
        fft1d(&mut col_re, &mut col_im, false);
        for y in 0..n {
            re[y * n + x] = col_re[y];
            im[y * n + x] = col_im[y];
        }
    }

    // |F|^2 / N^4
    let mut power = vec![0.0f64; n2];
    for i in 0..n2 {
        power[i] = (re[i] * re[i] + im[i] * im[i]) / nf4;
    }
    power
}

/// In-place radix-2 Cooley-Tukey FFT (or DFT fallback).
fn fft1d(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n <= 1 {
        return;
    }

    if n.is_power_of_two() {
        fft_radix2(re, im, inverse);
    } else {
        dft_naive(re, im, inverse);
    }
}

fn fft_radix2(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = sign * 2.0 * std::f64::consts::PI / len as f64;
        let wn_re = angle.cos();
        let wn_im = angle.sin();

        let mut i = 0;
        while i < n {
            let mut w_re = 1.0;
            let mut w_im = 0.0;
            for k in 0..half {
                let u_re = re[i + k];
                let u_im = im[i + k];
                let v_re = re[i + k + half] * w_re - im[i + k + half] * w_im;
                let v_im = re[i + k + half] * w_im + im[i + k + half] * w_re;
                re[i + k] = u_re + v_re;
                im[i + k] = u_im + v_im;
                re[i + k + half] = u_re - v_re;
                im[i + k + half] = u_im - v_im;
                let new_w_re = w_re * wn_re - w_im * wn_im;
                let new_w_im = w_re * wn_im + w_im * wn_re;
                w_re = new_w_re;
                w_im = new_w_im;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let nf = n as f64;
        for i in 0..n {
            re[i] /= nf;
            im[i] /= nf;
        }
    }
}

fn dft_naive(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut out_re = vec![0.0f64; n];
    let mut out_im = vec![0.0f64; n];

    for k in 0..n {
        for j in 0..n {
            let angle = sign * 2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
            out_re[k] += re[j] * angle.cos() - im[j] * angle.sin();
            out_im[k] += re[j] * angle.sin() + im[j] * angle.cos();
        }
    }

    if inverse {
        let nf = n as f64;
        for i in 0..n {
            re[i] = out_re[i] / nf;
            im[i] = out_im[i] / nf;
        }
    } else {
        re.copy_from_slice(&out_re);
        im.copy_from_slice(&out_im);
    }
}
