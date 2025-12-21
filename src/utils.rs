use ndarray::Array3;
use numpy::PyArray3;
use pyo3::{Bound, PyAny, PyResult};
use pyo3::types::PyAnyMethods;
            
/// Convert a Python 3D numpy array into a Rust `Array3<f32>`.
pub fn to_array3<'py>(x: &Bound<'py, PyAny>) -> PyResult<Array3<f32>> {
    let py_array: &PyArray3<f32> = x.extract()?;
    let arr = unsafe { py_array.as_array().to_owned() };
    Ok(arr)
}

