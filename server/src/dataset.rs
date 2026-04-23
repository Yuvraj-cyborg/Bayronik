use anyhow::{Context, Result, anyhow};
use memmap2::Mmap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Memory-mapped CAMELS map dataset (numpy .npy file with shape (N, 256, 256)).
pub struct MapsDataset {
    #[allow(dead_code)]
    pub path: PathBuf,
    pub n_samples: usize,
    pub resolution: usize,
    data_offset: usize,
    mmap: Mmap,
}

impl MapsDataset {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap {}", path.display()))?;

        let (header, offset) = parse_npy_header(&mmap)
            .with_context(|| format!("parsing npy header for {}", path.display()))?;

        if !header.fortran_order
            && header.dtype == NpyDtype::F32
            && header.shape.len() == 3
            && header.shape[1] == 256
            && header.shape[2] == 256
        {
            Ok(Self {
                path,
                n_samples: header.shape[0],
                resolution: header.shape[1],
                data_offset: offset,
                mmap,
            })
        } else {
            Err(anyhow!(
                "unsupported npy layout: dtype={:?} shape={:?} fortran={}",
                header.dtype,
                header.shape,
                header.fortran_order
            ))
        }
    }

    pub fn get(&self, idx: usize) -> Result<Vec<f32>> {
        if idx >= self.n_samples {
            return Err(anyhow!("index {idx} out of range [0,{})", self.n_samples));
        }
        let bytes_per_map = self.resolution * self.resolution * 4;
        let start = self.data_offset + idx * bytes_per_map;
        let end = start + bytes_per_map;
        if end > self.mmap.len() {
            return Err(anyhow!("data offset out of bounds at idx {idx}"));
        }
        let slice = &self.mmap[start..end];
        let mut out = vec![0.0f32; self.resolution * self.resolution];
        for (i, chunk) in slice.chunks_exact(4).enumerate() {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(out)
    }
}

#[derive(Debug, PartialEq)]
enum NpyDtype {
    F32,
    F64,
    Other(String),
}

#[derive(Debug)]
struct NpyHeader {
    dtype: NpyDtype,
    fortran_order: bool,
    shape: Vec<usize>,
}

fn parse_npy_header(buf: &[u8]) -> Result<(NpyHeader, usize)> {
    if buf.len() < 10 || &buf[..6] != b"\x93NUMPY" {
        return Err(anyhow!("not a numpy .npy file"));
    }
    let major = buf[6];
    let minor = buf[7];
    let header_len_bytes = if major == 1 {
        u16::from_le_bytes([buf[8], buf[9]]) as usize
    } else if major >= 2 {
        u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize
    } else {
        return Err(anyhow!("unsupported npy version {major}.{minor}"));
    };
    let header_start = if major == 1 { 10 } else { 12 };
    let header_end = header_start + header_len_bytes;
    if header_end > buf.len() {
        return Err(anyhow!("npy header truncated"));
    }
    let header = std::str::from_utf8(&buf[header_start..header_end])
        .context("header not utf-8")?
        .trim_end()
        .trim_end_matches('\n');

    let dtype = parse_dict_str(header, "descr")?;
    let fortran_order = parse_dict_bool(header, "fortran_order")?;
    let shape = parse_dict_shape(header, "shape")?;

    let dtype = match dtype.as_str() {
        "<f4" | "|f4" => NpyDtype::F32,
        "<f8" | "|f8" => NpyDtype::F64,
        other => NpyDtype::Other(other.to_string()),
    };
    Ok((
        NpyHeader {
            dtype,
            fortran_order,
            shape,
        },
        header_end,
    ))
}

fn parse_dict_str(header: &str, key: &str) -> Result<String> {
    let needle = format!("'{key}':");
    let i = header
        .find(&needle)
        .ok_or_else(|| anyhow!("missing key {key}"))?;
    let rest = &header[i + needle.len()..];
    let i = rest.find('\'').ok_or_else(|| anyhow!("malformed {key}"))?;
    let rest = &rest[i + 1..];
    let end = rest.find('\'').ok_or_else(|| anyhow!("malformed {key}"))?;
    Ok(rest[..end].to_string())
}

fn parse_dict_bool(header: &str, key: &str) -> Result<bool> {
    let needle = format!("'{key}':");
    let i = header
        .find(&needle)
        .ok_or_else(|| anyhow!("missing key {key}"))?;
    let rest = header[i + needle.len()..].trim_start();
    if rest.starts_with("True") {
        Ok(true)
    } else if rest.starts_with("False") {
        Ok(false)
    } else {
        Err(anyhow!("bad bool for {key}"))
    }
}

fn parse_dict_shape(header: &str, key: &str) -> Result<Vec<usize>> {
    let needle = format!("'{key}':");
    let i = header
        .find(&needle)
        .ok_or_else(|| anyhow!("missing key {key}"))?;
    let rest = &header[i + needle.len()..];
    let lo = rest.find('(').ok_or_else(|| anyhow!("missing ("))?;
    let hi = rest.find(')').ok_or_else(|| anyhow!("missing )"))?;
    let inside = &rest[lo + 1..hi];
    inside
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| anyhow!("bad dim {s}: {e}")))
        .collect()
}
