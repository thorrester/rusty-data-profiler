use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FeatureBin {
    pub name: String,
    pub bins: Vec<f64>,
    pub bin_counts: Vec<i32>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FeatureBins {
    pub feature: HashMap<String, FeatureBins>,
}
