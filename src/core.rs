use ndarray::Array2;
use rayon::prelude::*;
use simsimd::SpatialSimilarity; // gives f32::euclidean, etc.
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::seq::index::sample;
use anyhow::{Result, Context};


pub fn similarityrs(x: &Array2<f32>, bandwidth: f64, nsample: usize, seed: u64) -> Result<Vec<f64>> {
    let (rows, cols) = x.dim();
    anyhow::ensure!(cols >= 2, "Need at least 2 columns");

    // One-time: ensure contiguous standard layout (row-major).
    let x = x.as_standard_layout().to_owned();
    let data = x.as_slice().context("Array not contiguous")?;

    // Pre-calc nan cells to skip processing
    let valid: Vec<bool> = (0..rows)
        .map(|i| data[i * cols..(i + 1) * cols].iter().all(|x| x.is_finite()))
        .collect();

    let valid_rows: Vec<usize> = valid.iter().enumerate()
        .filter_map(|(i, &is_valid)| is_valid.then_some(i))
        .collect();

    // Select random samples
    let sampled_rows: Vec<usize> = if nsample > 0 {
        let k = nsample.min(valid_rows.len());
        let mut rng = StdRng::seed_from_u64(seed);
        sample(&mut rng, valid_rows.len(), k)
            .iter()
            .map(|p| valid_rows[p])
            .collect()
    } else {
        valid_rows.clone()
    };

    // Compute only for sampled i; everything else stays NaN
    let mut out = vec![f64::NAN; rows];

    let results: Vec<(usize, f64)> = valid_rows
        .par_iter()
        .map(|&i| {
            let a = &data[i * cols..(i + 1) * cols];
            let mut acc = 0.0f64;

            for &j in &sampled_rows {
                if j == i { continue; }
                let b = &data[j * cols..(j + 1) * cols];
                let d = f32::euclidean(a, b).expect("Unequal length") as f64;
                acc += (-d / bandwidth).exp();
            }
            (i, acc)
        })
        .collect();

    for (i, v) in results {
        out[i] = v;
    }

    Ok(out)
}

