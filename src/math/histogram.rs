use numpy::ndarray::Array;
use numpy::ndarray::{Array1, ArrayD, ArrayViewD, ArrayViewMutD};
use std::iter::FromIterator;

// create a function in rust that computes the bins of a given vector
pub fn compute_bins(data: &Array1<f64>, num_bins: usize) -> Vec<f64> {
    // find the min and max of the data

    let min = data.fold(f64::INFINITY, |a, &b| a.min(b));
    let max = data.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("min: {}", min);

    // create a vector of bins
    let mut bins = Vec::with_capacity(num_bins);

    // compute the bin width
    let bin_width = (max - min) / num_bins as f64;

    // create the bins
    for i in 0..num_bins {
        bins.push(min + bin_width * i as f64);
    }

    // return the bins
    bins
}

// use the following function to compute the bin counts
pub fn compute_bin_counts(data: &Vec<f64>, bins: &Vec<f64>) -> Vec<usize> {
    // create a vector to hold the bin counts
    let mut bin_counts = vec![0; bins.len()];

    // iterate over the data
    for &datum in data.iter() {
        // iterate over the bins
        for (i, bin) in bins.iter().enumerate() {
            if datum < *bin {
                bin_counts[i] += 1;
                break;
            }
        }
    }

    // return the bin counts
    bin_counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_compute_bins() {
        let test_array = Array1::random(10000, Uniform::new(0., 10.));

        compute_bins(&test_array, 10);
    }
}
