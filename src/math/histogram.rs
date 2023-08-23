use crate::math::types::FeatureBin;
use numpy::ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// Compute the bins for a 1d array of data
///
/// # Arguments
///
/// * `data` - A 1d array of data
/// * `num_bins` - The number of bins to use
///
/// # Returns
/// * `Vec<f64>` - A vector of bins
pub fn compute_bins(data: &Array1<f64>, num_bins: i32) -> Vec<f64> {
    // find the min and max of the data

    let min = data.fold(f64::INFINITY, |a, &b| a.min(b));
    let max = data.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // create a vector of bins
    let mut bins = Vec::<f64>::with_capacity(num_bins as usize);

    // compute the bin width
    let bin_width = (max - min) / num_bins as f64;

    // create the bins
    for i in 0..num_bins {
        bins.push(min + bin_width * i as f64);
    }

    // return the bins
    bins
}

/// Compute the bin counts for a 1d array of data
///
/// # Arguments
///
/// * `data` - A 1d array of data
/// * `bins` - A vector of bins
///
/// # Returns
/// * `Vec<i32>` - A vector of bin counts
pub fn compute_bin_counts(data: &Array1<f64>, bins: &Vec<f64>) -> Vec<i32> {
    // create a vector to hold the bin counts
    let bin_counts = Arc::new(Mutex::new(vec![0; bins.len()]));
    let max_bin = bins.last().unwrap();
    let data = data.to_vec();

    data.par_iter().for_each(|datum| {
        // iterate over the bins
        for (i, bin) in bins.iter().enumerate() {
            if bin != max_bin {
                let bin_range = bin..&bins[i + 1];

                if bin_range.contains(&datum) {
                    bin_counts.lock().unwrap()[i] += 1;
                    break;
                }
                continue;
            } else if bin == max_bin {
                if datum > bin {
                    bin_counts.lock().unwrap()[i] += 1;
                    break;
                }
                continue;
            } else {
                continue;
            }
        }
    });

    return bin_counts.lock().unwrap().to_vec();
}

/// Compute the bin counts for a 2d array of data
///
/// # Arguments
///
/// * `feature_names` - A vector of feature names
/// * `array_data` - A 2d array of data
/// * `bins` - An optional vector of bins
/// * `num_bins` - The number of bins to use
///
/// # Returns
/// * `Vec<FeatureBin>` - A vector of feature bins
pub fn compute_bin_counts_from_2d_array(
    feature_names: &Vec<String>,
    array_data: &Array2<f64>,
    bins: Option<&Vec<f64>>,
    num_bins: Option<i32>,
) -> Vec<FeatureBin> {
    let bin_vec = Arc::new(Mutex::new(Vec::new()));
    let columns = array_data.columns().into_iter().collect::<Vec<_>>();
    columns.par_iter().enumerate().for_each(|(index, column)| {
        let feature_name = feature_names[index].clone();

        let bins = match bins {
            Some(bins) => bins.to_vec(),
            None => compute_bins(&column.into_owned(), num_bins.unwrap_or(10)),
        };

        let bin_counts = compute_bin_counts(&column.into_owned(), &bins);

        let bins = FeatureBin {
            name: feature_name,
            bins: bins,
            bin_counts: bin_counts,
        };

        bin_vec.lock().unwrap().push(bins);
    });

    return bin_vec.lock().unwrap().to_vec();
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::{Alphanumeric, Normal};
    use ndarray_rand::RandomExt;
    use numpy::ndarray::{Array1, Array2};

    #[test]
    fn test_compute_bins() {
        let test_array = Array1::random(10_000, Normal::new(0.0, 1.0).unwrap());

        let bins = compute_bins(&test_array, 10);

        let counts = compute_bin_counts(&test_array, &bins);
        assert_eq!(10000, counts.iter().sum::<i32>());
    }

    #[test]
    fn test_compute_bin_counts_from_2d_array() {
        let num_features = 30;
        let num_records = 100_000;
        let num_bins = 10;
        let test_array =
            Array2::random((num_records, num_features), Normal::new(0.0, 1.0).unwrap());
        let feature_names = Array1::random(num_features, Alphanumeric)
            .to_vec()
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>();

        // test no feature bins specified
        let feature_bins =
            compute_bin_counts_from_2d_array(&feature_names, &test_array, None, Some(num_bins));
        assert_eq!(feature_bins.len(), num_features);

        // test with feature bins
        let feature_bins_with_bins = compute_bin_counts_from_2d_array(
            &feature_names,
            &test_array,
            Some(&feature_bins[0].bins.to_vec()),
            None,
        );
        assert_eq!(feature_bins_with_bins.len(), num_features);
    }
}
