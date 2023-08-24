use num_traits::Float;
use std::collections::HashSet;

/// Compute the number of distinct values in a 1d array of data
///
/// # Arguments
///
/// * `feature_array` - A 1d array of data
///
/// # Returns
/// * `Result<(f64, f64), String>` - A tuple containing the number of distinct values and the percentage of distinct values
pub fn count_distinct<T>(feature_array: &[T]) -> Result<(usize, f64), String>
where
    T: PartialEq + PartialOrd + std::fmt::Display,
{
    let unique: HashSet<String> = feature_array.into_iter().map(|x| x.to_string()).collect();
    let count = unique.len();
    let count_perc = count as f64 / feature_array.len() as f64;
    Ok((count, count_perc))
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
    let count = feature_array.into_iter().filter(|x| x.is_none()).count();
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
pub fn count_infinity(feature_array: &[f64]) -> Result<(usize, f32), String> {
    let count = feature_array
        .into_iter()
        .filter(|x| x.is_infinite())
        .count();

    let count_perc = count as f32 / feature_array.len() as f32;

    Ok((count, count_perc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_distinct() {
        // Test floats

        //test ints
        let array = [0, 1, 2, 3, 4, 5, 1, 2, 3, 8];
        let (count, count_perc) = count_distinct(&array).unwrap();

        assert_eq!(count, 7);
        assert_eq!(count_perc, 7.0 / 10.0);

        // test string
        let array = ["a", "b", "c", "d", "e", "a", "a", "a", "a", "a", "a"];
        let (count, count_perc) = count_distinct(&array).unwrap();

        assert_eq!(count, 5);
        assert_eq!(count_perc, 5.0 / 11.0);

        // test float
        let array = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.2, 2.3, 3.4, 8.9];
        let (count, count_perc) = count_distinct(&array).unwrap();

        assert_eq!(count, 10);
        assert_eq!(count_perc, 10.0 / 10.0);
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
        let test_array = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, f64::INFINITY, f64::INFINITY];

        let (count, count_perc) = count_infinity(&test_array).unwrap();

        assert_eq!(count, 2);
        assert_eq!(count_perc, 2.0 / 8.0);
    }
}
