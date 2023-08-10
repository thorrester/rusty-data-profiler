use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// add(a, b, /)
/// --
///
/// This function adds two 64-bit floats.
/// Args:
///     a:
///     The first number to add.
///    b:
///     The second number to add.
/// Returns:
///     float: The sum of ``a`` and ``b``.
#[pyfunction]
fn add(a: f64, b: f64) -> PyResult<f64> {
    Ok(a + b)
}

#[pyfunction]
fn compute_mean(values: Vec<f64>) -> PyResult<f64> {
    let sum: f64 = values.iter().sum();
    let count = values.len() as f64;
    Ok(sum / count)
}

/// A Python module implemented in Rust.
#[pymodule]
fn rusty_data_profiler(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(compute_mean, m)?)?;
    Ok(())
}
