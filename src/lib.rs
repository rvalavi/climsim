use pyo3::prelude::*;
use numpy::{PyArray2, ToPyArray};
use ndarray::{Array3, Array2};
// local mods
mod core;
mod utils;


/// Python interface for Rust function
#[pyfunction]
fn similaritypy(
    arr: &Bound<PyAny>,
    radius_km: f32,
    is_geo: bool,
    n_cores: i32,
) -> PyResult<Py<PyArray2<f64>>> {

    let array: Array3<f32> = utils::to_array3(arr).unwrap();

    // Process the similarities in parallel
    let outarray: Array2<f64> = core::similarityrs(
        array,
        is_geo,
    );

    // Convert back to Python array with gil
    Python::with_gil(|py| {
        let pyarray = outarray.to_pyarray_bound(py);
        Ok(pyarray.unbind())
    })
}


/// A Python module implemented in Rust.
#[pymodule]
fn climsim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(similaritypy, m)?)?;
    Ok(())
}
