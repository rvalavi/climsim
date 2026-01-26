use ndarray::Array2;
use numpy::PyArray2;
use pyo3::{Bound, PyAny, PyResult};
use pyo3::types::PyAnyMethods;
use crate::affines::Affine;

/// Simple way to convert a Python 2D numpy array into a Rust `Array2<f32>`.
pub fn to_array<'py>(x: &Bound<'py, PyAny>) -> PyResult<Array2<f32>> {
    let py_array: &PyArray2<f32> = x.extract()?;
    let arr = unsafe { py_array.as_array().to_owned() };
    Ok(arr)
}

/// Get xy from rows of a flatten array
/// n: number of cells in the 2D array/raster
/// ncols: number of columns in the array
pub fn get_xy(n: usize, ncols: usize, transform: &Affine) -> anyhow::Result<Vec<(f64, f64)>> {
    anyhow::ensure!(ncols > 0, "ncols must be > 0");

    let out: Vec<(f64, f64)> = (0..n).into_iter()
        .map(|i| {
            let row = i / ncols;
            let col = i % ncols;
            transform.xy(row, col)
        }).collect();

    Ok(out)
}

