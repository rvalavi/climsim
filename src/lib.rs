use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use numpy::{PyArray2, ToPyArray};
use ndarray::Array2;
// local mods
mod core;
mod utils;


/// Python interface for Rust function
#[pyfunction]
fn similaritypy(
    arr: &Bound<PyAny>,
    nrows: usize,
    ncols: usize,
    // radius_km: f32,
    // is_geo: bool,
    bandwidth: f64,
    nsample: usize, 
    seed: u64,
    n_cores: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let array: Array2<f32> = utils::to_array(arr)?;

    // local pool is safest in PyO3 contexts
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_cores.max(1))
        .build()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let out: Vec<f64> = pool
        .install(|| core::similarityrs(&array, bandwidth, nsample, seed))
        .map_err(|er| PyRuntimeError::new_err(format!("Climsim failed: {er}")))?;

    let outarray: Array2<f64> = Array2::from_shape_vec((nrows, ncols), out)
        .map_err(|e| PyValueError::new_err(format!("Output shape mismatch: {e}")))?;

    // Python::with_gil(|py| Ok(outarray.to_pyarray(py).to_owned()))
    // Convert back to Python array with gil
    Python::with_gil(|py| {
        let pyarray = outarray.to_pyarray_bound(py);
        Ok(pyarray.unbind())
    })
}


/// A Python module implemented in Rust.
#[pymodule]
fn climsim_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(similaritypy, m)?)?;
    Ok(())
}
