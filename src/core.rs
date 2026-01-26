use ndarray::Array2;
use rayon::prelude::*;
use simsimd::SpatialSimilarity; // gives f32::euclidean, etc.
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::seq::index::sample;
use anyhow::{Result, Context};
use crate::affines::Affine;
use crate::utils;
use crate::distance;


pub fn dissimrs(
    x: &Array2<f32>, 
    trans: &Affine,
    is_geo: bool, 
    bandwidth: f64,
    radius: Option<f32>,
    nsample: usize, 
    ncols: usize,
    seed: u64
) -> Result<Vec<f64>> {
    // Flatten array rows: number of cells; cols: number of raster bands
    let (rows, cols) = x.dim();
    anyhow::ensure!(cols >= 2, "Need at least 2 columns");
    anyhow::ensure!(bandwidth.is_finite() && bandwidth > 0.0, "bandwidth must be > 0 and finite");

    // Ensure contiguous standard layout (row-major).
    let x = x.as_standard_layout().to_owned();
    let data = x.as_slice().context("Array not contiguous")?;

    // Get the XY for distance calc
    let xy: Vec<(f64, f64)> = utils::get_xy(rows, ncols, trans)?;

    // Pre-calc valid cells to skip processing nan cells
    let valid_rows: Vec<usize> = (0..rows)
        .filter(|&i| data[i * cols..(i + 1) * cols].iter().all(|v| v.is_finite()))
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
        .into_par_iter()
        .map(|i| {
            let a = &data[i * cols..(i + 1) * cols];
            let mut acc = 0.0f64;

            let (x1, y1) = xy[i];

            for &j in &sampled_rows {
                if j == i { continue; }
                
                // Filter cells within the raduis
                if let Some(r) = radius {
                    let (x2, y2) = xy[j];
                    let dist = distance::distance_km(x1, y1, x2, y2, is_geo);
                    if dist > r { continue; }
                }

                let b = &data[j * cols..(j + 1) * cols];
                let d = f32::euclidean(a, b).expect("Unequal length") as f64;
                acc += 1.0 - (-d / bandwidth).exp();
            }
            (i, acc)
        })
        .collect();

    for (i, v) in results {
        out[i] = v;
    }

    Ok(out)
}



#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use approx::assert_relative_eq;

    fn create_test_affine() -> Affine {
        let aff = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        aff.into()
    }

    #[test]
    fn test_dissimrs_basic() {
        let x = Array2::from_shape_vec((3, 2), vec![
            1.0, 2.0,
            1.5, 2.5,
            10.0, 20.0
        ]).unwrap();

        let trans = create_test_affine();
        let result = dissimrs(&x, &trans, false, 1.0, None, 0, 2, 42).unwrap();

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|&v| v.is_finite()));
        assert!(result[2] > result[0]);
        assert!(result[2] > result[1]);
    }

    #[test]
    fn test_dissimrs_with_nan() {
        let x = Array2::from_shape_vec((4, 2), vec![
            1.0, 2.0,
            f32::NAN, 2.0,
            3.0, 4.0,
            5.0, f32::NAN,
        ]).unwrap();

        let trans = create_test_affine();
        let result = dissimrs(&x, &trans, false, 1.0, None, 0, 2, 42).unwrap();

        assert!(result[0].is_finite());
        assert!(result[1].is_nan());
        assert!(result[2].is_finite());
        assert!(result[3].is_nan());
    }

    #[test]
    fn test_dissimrs_identical_cells() {
        let x = Array2::from_shape_vec((3, 2), vec![
            1.0, 2.0,
            1.0, 2.0,
            1.0, 2.0,
        ]).unwrap();

        let trans = create_test_affine();
        let result = dissimrs(&x, &trans, false, 1.0, None, 0, 2, 42).unwrap();

        for &val in &result {
            assert_relative_eq!(val, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dissimrs_invalid_bandwidth() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let trans = create_test_affine();

        assert!(dissimrs(&x, &trans, false, 0.0, None, 0, 2, 42).is_err());
        assert!(dissimrs(&x, &trans, false, -1.0, None, 0, 2, 42).is_err());
    }
}
