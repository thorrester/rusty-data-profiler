use crate::math::histogram::{compute_bin_counts, compute_bins};
use crate::math::logger::Logger;
use crate::math::types::{Bin, Distinct, FeatureStat, Infinity, Stats};
use medians::{MStats, Median, Medianf64};
use ndarray::prelude::*;
use ndarray_stats::{interpolate::Nearest, QuantileExt};
use noisy_float::prelude::*;
use noisy_float::types::n64;
use num_traits::Float;
use numpy::ndarray::{aview1, ArrayView1, ArrayView2};
use rayon::prelude::*;
use std::collections::HashSet;

pub fn compute_quantiles(array: &ArrayView2<f64>) -> Array2<N64> {
    let axis = Axis(0);
    let qs = &[n64(0.25), n64(0.5), n64(0.75), n64(0.99)];
    array
        .map(|x| n64(*x))
        .quantiles_axis_mut(axis, &aview1(qs), &Nearest)
        .expect("failed to compute quantiles")
}

pub fn compute_mean(array: &ArrayView2<f64>) -> Array1<f64> {
    array.mean_axis(Axis(0)).expect("Failed to calculate mean")
}

pub fn compute_stddev(array: &ArrayView2<f64>) -> Array1<f64> {
    array.std_axis(Axis(0), 1.0)
}

pub fn compute_min(array: &ArrayView2<f64>) -> Array1<f64> {
    array.map_axis(Axis(0), |view| *view.min().unwrap())
}

pub fn compute_max(array: &ArrayView2<f64>) -> Array1<f64> {
    array.map_axis(Axis(0), |view| *view.max().unwrap())
}

struct DataProfiler<'a> {
    array_data: &'a ArrayView2<'a, f64>,
    feature_names: &'a [String],
}

impl<'a> DataProfiler<'a> {
    pub fn new(
        array_data: &'a ArrayView2<'a, f64>,
        feature_names: &'a [String],
    ) -> DataProfiler<'a> {
        DataProfiler {
            array_data: array_data,
            feature_names: feature_names,
        }
    }

    pub fn compute_profile(&self) {
        let funcs = [compute_max, compute_min, compute_mean, compute_stddev];
        let base_stats = funcs
            .par_iter()
            .map(|func| func(&self.array_data))
            .collect::<Vec<_>>();
    }

    pub fn compute_2d_array_stats(
        &self,
        feature_names: &[String],
    ) -> Result<Vec<FeatureStat>, String> {
        let columns = self.array_data.columns().into_iter().collect::<Vec<_>>();

        let feature_vec = columns
            .par_iter()
            .enumerate()
            .map(|(index, x)| {
                let feature_name = &feature_names[index];
                let data_bins = compute_bins(x, 10);

                let data_bin_counts = compute_bin_counts(x, &data_bins);
                let base_stats = compute_base_stats(x).unwrap();

                let bins = FeatureStat {
                    name: feature_name.clone(),
                    bins: Bin {
                        bins: data_bins.to_vec(),
                        bin_counts: data_bin_counts,
                    },
                    stats: base_stats,
                };

                return bins;
            })
            .collect::<Vec<FeatureStat>>();
        Ok(feature_vec)
    }
}

/// Compute the number of distinct values in a 1d array of data
///
/// # Arguments
///
/// * `feature_array` - A 1d array of data
///
/// # Returns
/// * `Result<Distinct, String>` - A tuple containing the number of distinct values and the percentage of distinct values
pub fn count_distinct<T: Send + Sync + std::fmt::Display>(
    feature_array: &ArrayView1<T>,
) -> Result<Distinct, String> {
    let unique: HashSet<String> = feature_array
        .into_par_iter()
        .map(|x| x.to_string())
        .collect();
    let count = unique.len();
    let count_perc = count as f64 / feature_array.len() as f64;

    Ok(Distinct {
        count,
        percent: count_perc,
    })
}

/// Compute the number of missing values in a 1d array of data
///
/// # Arguments
///
/// * `feature_array` - A 1d array of data
///
/// # Returns
/// * `Result<(f64, f64), String>` - A tuple containing the number of missing values and the percentage of missing values
pub fn count_missing<T>(feature_array: &[Option<T>]) -> Result<(usize, f64), String>
where
    T: PartialEq + Clone + Send + Sync + Float + From<i32> + From<f64> + 'static,
{
    let count = feature_array.iter().filter(|x| x.is_none()).count();
    let count_perc = count as f64 / feature_array.len() as f64;

    Ok((count, count_perc))
}

/// Compute the number of infinite values in a 1d array of data
///
/// # Arguments
///
/// * `feature_array` - A 1d array of data
///
/// # Returns
/// * `Result<(f64, f64), String>` - A tuple containing the number of infinite values and the percentage of infinite values
pub fn count_infinity(feature_array: &ArrayView1<f64>) -> Result<Infinity, String> {
    let count = feature_array
        .into_par_iter()
        .filter(|x| x.is_infinite())
        .count();

    let count_perc = count as f64 / feature_array.len() as f64;

    Ok(Infinity {
        count: count,
        percent: count_perc,
    })
}

/// Compute the base stats for a 1d array of data
///
/// # Arguments
///
/// * `feature_array` - A 1d array of data
pub fn compute_base_stats(feature_array: &ArrayView1<f64>) -> Result<Stats, String> {
    let computed_mean = feature_array.mean().unwrap();
    let computed_stddev = feature_array.std(1.0);
    let computed_median = feature_array.to_vec().as_slice().medstats().unwrap();

    let inf_meta = count_infinity(feature_array).unwrap();
    let distinct_meta = count_distinct(feature_array).unwrap();

    feature_array
        .to_vec()
        .sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = feature_array[0];
    let max = feature_array[feature_array.len() - 1];

    // calculate infinity

    Ok(Stats {
        median: computed_median.centre,
        mean: computed_mean,
        standard_dev: computed_stddev,
        min,
        max,
        distinct: distinct_meta,
        infinity: inf_meta,
        missing: None,
    })
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
pub fn compute_2d_array_stats(
    feature_names: &[String],
    array_data: &ArrayView2<f64>,
    bins: &Option<Vec<Vec<f64>>>,
    num_bins: &Option<u32>,
) -> Result<Vec<FeatureStat>, String> {
    let columns = array_data.columns().into_iter().collect::<Vec<_>>();

    let feature_vec = columns
        .par_iter()
        .enumerate()
        .map(|(index, x)| {
            let feature_name = &feature_names[index];

            let data_bins = match bins {
                Some(bins) => bins[index].to_owned(),
                None => compute_bins(x, num_bins.unwrap_or(10)),
            };

            let data_bin_counts = compute_bin_counts(x, &data_bins);
            let base_stats = compute_base_stats(x).unwrap();

            let bins = FeatureStat {
                name: feature_name.clone(),
                bins: Bin {
                    bins: data_bins.to_vec(),
                    bin_counts: data_bin_counts,
                },
                stats: base_stats,
            };

            return bins;
        })
        .collect::<Vec<FeatureStat>>();
    Ok(feature_vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use numpy::ndarray::{arr1, Array2};
    use rstats::Median;

    #[test]
    fn test_count_distinct() {
        // Test floats

        //test ints
        let array = arr1(&[1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1]);
        let distinct = count_distinct(&array.view()).unwrap();

        assert_eq!(distinct.count, 5);
        assert_eq!(distinct.percent, 5.0 / 11.0);

        // test string
        let array = arr1(&["a", "b", "c", "d", "e", "a", "a", "a", "a", "a", "a"]);
        let distinct = count_distinct(&array.view()).unwrap();

        assert_eq!(distinct.count, 5);
        assert_eq!(distinct.percent, 5.0 / 11.0);

        // test float
        let array = arr1(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.2, 2.3, 3.4, 8.9]);
        let distinct = count_distinct(&array.view()).unwrap();

        assert_eq!(distinct.count, 10);
        assert_eq!(distinct.percent, 10.0 / 10.0);
    }

    #[test]
    fn test_count_missing() {
        let test_array = [
            Some(1.0),
            Some(2.0),
            Some(3.0),
            None,
            Some(5.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.2),
            Some(-1.5),
            Some(100.0),
        ];

        let (count, count_perc) = count_missing(&test_array).unwrap();

        assert_eq!(count, 1);
        assert_eq!(count_perc, 1.0 / 11.0);
    }

    #[test]
    fn test_count_infinity() {
        let test_array = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 1.0, f64::INFINITY, f64::INFINITY]);

        let infinity = count_infinity(&test_array.view()).unwrap();

        assert_eq!(infinity.count, 2);
        assert_eq!(infinity.percent, 2.0 / 8.0);
    }

    #[test]
    fn test_median() {
        let v1 = vec![
            1_u8, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
        ];

        let median = v1.as_slice().medstats(|&x| x.into()).expect("median");

        println!("median: {}", median);
    }

    #[test]
    fn test_compute_2d_array() {
        let array = ndarray::arr2(&[
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, f64::INFINITY, 3.0, 4.0, 5.0],
        ]);
        let feature_names = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];

        let features = compute_2d_array_stats(&feature_names, &array.view(), &None, &None).unwrap();
        assert_eq!(features.len(), 5);

        let array = Array2::random((1_000, 10), Normal::new(0.0, 1.0).unwrap());
        let feature_names = (0..10).map(|x| x.to_string()).collect::<Vec<_>>();
        let features = compute_2d_array_stats(&feature_names, &array.view(), &None, &None).unwrap();
        assert_eq!(features.len(), 10);
    }
}
