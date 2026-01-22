use ndarray::Array2;
use rayon::prelude::*;
use simsimd::SpatialSimilarity; // gives f32::euclidean, etc.

pub fn similarityrs(x: &Array2<f32>) -> Result<Vec<f64>, String> {
    let (rows, cols) = x.dim();
    if cols < 2 {
        return Err("Need at least 2 columns".into());
    }

    // One-time: ensure contiguous standard layout (row-major).
    let x = x.as_standard_layout().to_owned();
    let data = x.as_slice().ok_or("Array not contiguous")?;

    // Pre-calc nan cells to skip processing
    let invalid: Vec<bool> = (0..rows)
        .map(|i| {
            data[i * cols..(i + 1) * cols]
                .iter()
                .any(|x| x.is_nan())
        })
        .collect();

    let out: Vec<f64> = (0..rows)
        .into_par_iter()
        .map(|i| {
            let a = &data[i * cols..(i + 1) * cols];

            // If a contains NaN → whole result is NaN
            if invalid[i] {
                return f64::NAN;
            }

            let mut acc = 0.0f64;
            for j in 0..rows {
                if j == i { continue; }
                let b = &data[j * cols..(j + 1) * cols];

                // Skip b if it contains NaN
                if invalid[j] {
                    continue;
                }

                // simsimd pattern (Result<f32, _>)
                let d = f32::euclidean(a, b).expect("Unequal length") as f64;

                // Ecological distance
                acc += (-d).exp();
            }
            acc
        })
        .collect();

    Ok(out)
}

