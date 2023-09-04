mod math;
use crate::math::types::{Bin, Distinct, FeatureStat, Infinity, Stats};
use math::stats::compute_2d_array_stats;
use numpy::PyReadonlyArray2;
use pyo3::panic::PanicException;
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
pub fn parse_array(
    feature_names: Vec<String>,
    array: PyReadonlyArray2<f64>,
    bins: Option<Vec<Vec<f64>>>,
    num_bins: Option<u32>,
) -> Result<Vec<FeatureStat>, &PanicException> {
    let feature_bins = compute_2d_array_stats(&feature_names, &array.as_array(), &bins, &num_bins);

    let features = match feature_bins {
        Ok(features) => features,
        Err(error) => panic!("Error while parsing array: {:?}", error),
    };

    Ok(features)
}

/// add(a, b, /
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
    //let mean = values.as_array().mean().unwrap();
    //Ok(mean)
}

/// A Python module implemented in Rust.
/// Name must match cargo lib name
#[pymodule]
fn rusty_data_profiler(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_array, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(compute_mean, m)?)?;
    m.add_class::<FeatureStat>()?;
    m.add_class::<Bin>()?;
    m.add_class::<Distinct>()?;
    m.add_class::<Infinity>()?;
    m.add_class::<Stats>()?;
    Ok(())
}
