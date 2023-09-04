use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FeatureBin {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub bins: Vec<f64>,
    #[pyo3(get, set)]
    pub bin_counts: Vec<i32>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FeatureBins {
    pub feature: HashMap<String, FeatureBins>,
}

#[pymethods]
impl FeatureBin {
    #[new]
    fn new(name: String, bins: Vec<f64>, bin_counts: Vec<i32>) -> Self {
        Self {
            name: name,
            bins: bins,
            bin_counts: bin_counts,
        }
    }
}
