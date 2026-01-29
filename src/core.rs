use ndarray::Array2;
use rayon::prelude::*;
use simsimd::SpatialSimilarity; // gives f32::euclidean, etc.
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::seq::index::sample;
use anyhow::Result;
use crate::affines::Affine;
use crate::utils::Matrix;
use crate::utils;
use crate::distance;


/// Calculate the min climatic/environmental distance to sample point
pub fn climdistrs(
    x: &Array2<f32>, 
    trans: &Affine,     
    samples_x: &Vec<f64>,
    samples_y: &Vec<f64>,
    nrows: usize,
    ncols: usize,
) -> Result<Vec<f64>> {
    // Ensure contiguous standard layout (row-major) for faster mem access.
    let data = Matrix::from_array(x)?;

    // Flatten array rows: number of cells; cols: number of raster bands
    let (total_cells, bands) = data.dim();
    anyhow::ensure!(bands >= 2, "Need at least 2 columns");
    
    // Pre-calc valid cells to skip processing nan cells
    let valid_rows: Vec<usize> = (0..total_cells)
        .filter(|&i| data.row(i).iter().all(|v| v.is_finite()))
        .collect();

    // Get flatten array index from xy
    let mut samples: Vec<usize> = samples_x.iter()
        .zip(samples_y.iter())
        .filter_map(|(x, y)| {
            let (row, col) = trans.ij(*x, *y, nrows, ncols)?;
            Some(row * ncols + col)
        })
        .collect();

    anyhow::ensure!(!samples.is_empty(), "No valid sample points found");
    // Sort the samples to lower cache misses
    if !samples.is_empty() {
        samples.sort_unstable();
    }
    
    // Compute only for sampled i; everything else stays NaN
    let mut out = vec![f64::NAN; total_cells];

    let results: Vec<(usize, f64)> = valid_rows
        .into_par_iter()
        .map(|i| {
            let a = data.row(i);

            let mut mn = f64::INFINITY;

            for &j in &samples {              
                let b = data.row(j);
                // NOTE: make this an Option<f64> to ignore sample points with NAN values  
                let d = f32::euclidean(a, b).expect("Unequal length") as f64;
                mn = mn.min(d)
            }
            (i, mn)
        })
        .collect();

    for (i, v) in results {
        out[i] = v;
    }

    Ok(out)
}


// Climate dissimilarity index; both local and gloabl measures
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
    // Ensure contiguous standard layout (row-major) for faster mem access.
    let data = Matrix::from_array(x)?;

    // Flatten array rows: number of cells; cols: number of raster bands
    let (total_cells, bands) = data.dim();
    anyhow::ensure!(bands >= 2, "Need at least 2 columns");
    anyhow::ensure!(bandwidth.is_finite() && bandwidth > 0.0, "bandwidth must be > 0 and finite");

    // Get the XY for distance calc
    let xy: Vec<(f64, f64)> = utils::get_xy(total_cells, ncols, trans)?;

    // Pre-calc valid cells to skip processing nan cells
    let valid_rows: Vec<usize> = (0..total_cells)
        .filter(|&i| data.row(i).iter().all(|v| v.is_finite()))
        .collect();

    // Select random samples
    let mut sampled_rows: Vec<usize> = if nsample > 0 {
        let k = nsample.min(valid_rows.len());
        let mut rng = StdRng::seed_from_u64(seed);
        sample(&mut rng, valid_rows.len(), k)
            .iter()
            .map(|p| valid_rows[p])
            .collect()
    } else {
        valid_rows.clone()
    };
    // Sort the samples to lower cache misses
    if nsample > 0 {
        sampled_rows.sort_unstable();
    }

    // Compute only for sampled i; everything else stays NaN
    let mut out = vec![f64::NAN; total_cells];

    let results: Vec<(usize, f64)> = valid_rows
        .into_par_iter()
        .map(|i| {
            let a = data.row(i);
            let (x1, y1) = xy[i];
            
            let mut acc = 0.0f64;

            for &j in &sampled_rows {
                if j == i { continue; }
                
                // Filter cells within the raduis
                if let Some(r) = radius {
                    let (x2, y2) = xy[j];
                    let dist = distance::distance_km(x1, y1, x2, y2, is_geo);
                    if dist > r { continue; }
                }

                let b = data.row(j);
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
