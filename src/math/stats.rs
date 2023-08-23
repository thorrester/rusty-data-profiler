use itertools::Itertools;
use num_traits::Float;
use numpy::ndarray::Array1;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use std::cmp::Eq;
use std::cmp::Ord;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{AddAssign, MulAssign};
use std::sync::{Arc, Mutex};

/// Compute the number of distinct values in a 1d array of data
///
/// # Arguments
///
/// * `feature_array` - A 1d array of data
///
/// # Returns
/// * `Result<(f64, f64), String>` - A tuple containing the number of distinct values and the percentage of distinct values
pub fn count_distinct<T>(feature_array: &Vec<T>) -> Result<(usize, f64, Vec<T>), String>
where
    T: PartialEq + Clone + Debug + PartialOrd,
{
    let mut unique = feature_array.clone();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique.dedup();

    let count = unique.len();
    let count_perc = count as f64 / feature_array.len() as f64;

    Ok((count, count_perc, unique))
}

// Create a rust function that takes a vector of strings, integers, or floats and checks their types

// Need to figure out how to do this generically with above function
pub fn count_distinct_str(
    feature_array: &[Option<&str>],
) -> Result<(usize, f64, Vec<String>), String> {
    let distinct = Arc::new(Mutex::new(Vec::new()));

    feature_array.into_par_iter().for_each(|feature| {
        if feature.is_some() {
            let feat = feature.unwrap().to_string();
            if !distinct.lock().unwrap().contains(&feat) {
                distinct.lock().unwrap().push(feat);
            }
        }
    });

    let unique = distinct.lock().unwrap().to_vec();
    let count = unique.len();
    let count_perc = count as f64 / feature_array.len() as f64;

    Ok((count, count_perc, unique))
}

/// Compute the number of missing values in a 1d array of data
///
/// # Arguments
///
/// * `feature_array` - A 1d array of data
///
/// # Returns
/// * `Result<(f64, f64), String>` - A tuple containing the number of missing values and the percentage of missing values
pub fn count_missing<T>(feature_array: &[Option<T>]) -> Result<(f64, f64), String>
where
    T: PartialEq + Clone + Send + Sync + Float + From<i32> + From<f64> + 'static,
{
    let missing = Arc::new(Mutex::new(Vec::<i32>::new()));

    feature_array.par_iter().for_each(|feature| {
        if feature.is_none() {
            missing.lock().unwrap().push(1);
        }
    });

    let count = missing.lock().unwrap().to_vec().len() as f64;
    let count_perc = count / feature_array.len() as f64;

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
pub fn count_infinity(feature_array: &Array1<f64>) -> Result<(i32, f32), String> {
    let infinity = Arc::new(Mutex::new(Vec::<i32>::new()));

    feature_array.par_iter().for_each(|feature| {
        if feature.is_infinite() {
            infinity.lock().unwrap().push(1);
        }
    });

    let count = infinity.lock().unwrap().to_vec().len() as i32;
    let count_perc = count as f32 / feature_array.len() as f32;

    Ok((count, count_perc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::{Alphanumeric, Normal};
    use ndarray_rand::RandomExt;
    use numpy::ndarray::arr1;

    #[test]
    fn test_count_distinct() {
        // Test floats
        let test_array = Array1::random(1_000_000, Normal::new(0.0, 1.0).unwrap());
        let (count, count_perc, _unique) = count_distinct(&test_array.to_vec()).unwrap();

        assert_eq!(count, 8);
        assert_eq!(count_perc, 8.0 / 11.0);

        // Test strings
        //let test_array = arr1(&[
        //    None,
        //    Some("a"),
        //    Some("b"),
        //    Some("c"),
        //    Some("d"),
        //    Some("e"),
        //    Some("a"),
        //    Some("a"),
        //    Some("a"),
        //    Some("a"),
        //    Some("a"),
        //    Some("a"),
        //]);
        //let (count, count_perc, _unique) = count_distinct_str(&test_array.to_vec()).unwrap();
        //
        //assert_eq!(count, 5);
        //assert_eq!(count_perc, 5.0 / 12.0);
    }

    #[test]
    fn test_count_missing() {
        let test_array: Array1<Option<f64>> = arr1(&[
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
        ]);

        let (count, count_perc) = count_missing(&test_array.to_vec()).unwrap();

        assert!(count == 1.0);
        assert_eq!(count_perc, 1.0 / 11.0);
    }

    #[test]
    fn test_count_infinity() {
        let test_array = arr1(&[
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            1.0,
            1.0,
            1.0,
            1.2,
            -1.5,
            f64::INFINITY,
            f64::INFINITY,
        ]);

        let (count, count_perc) = count_infinity(&test_array).unwrap();

        assert_eq!(count, 2);
        assert_eq!(count_perc, 8.0 / 11.0)
    }
}
