use pyo3::prelude::*;
use numpy::{PyArray2, ToPyArray};
use ndarray::{Array3, Array1};
// local mods
mod core;
mod utils;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sumilarityrs(
    arr: &Bound<PyAny>, 
    is_geo: bool,
) -> PyResult<Py<PyArray2<f64>>> {

    let array: Array3<f32> = utils::to_array3(arr).unwrap();

    // Process the similarities
    let outarray = core::similarity(
        array,
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
    m.add_function(wrap_pyfunction!(sumilarityrs, m)?)?;
    Ok(())
}
