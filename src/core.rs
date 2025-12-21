use ndarray::{Array3, Array2, s};
use rayon::prelude::*;
use std::ops::Neg;
use simsimd::SpatialSimilarity;


/// Core fn for calculating similarity
pub fn similarity(x: Array3<f32>) -> Array2<f64> {
    let (rows, cols, _bands) = x.dim();
    assert!(cols >= 2, "Need at least 2 columns");

    let out_len = rows * (cols - 1);
    let mut out = vec![0.0f64; out_len];

    out.par_iter_mut()
        .enumerate()
        .for_each(|(k, out_cell)| {
            let i = k / (cols - 1);
            let j = k % (cols - 1);

            let va = x.slice(s![i, j, ..]).as_slice().unwrap();
            let vb = x.slice(s![i, j + 1, ..]).as_slice().unwrap();

            *out_cell = f32::euclidean(va, vb).expect("Unequal length").neg().exp();
        });

    Array2::from_shape_vec((rows, cols - 1), out).unwrap()
}

