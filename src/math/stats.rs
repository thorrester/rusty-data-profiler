use numpy::ndarray::Array1;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

pub fn count_distinct(feature_array: &Array1<f64>) -> Result<(f64, f64), String> {
    let distinct = Arc::new(Mutex::new(Vec::<f64>::new()));

    feature_array.par_iter().for_each(|feature| {
        if !distinct.lock().unwrap().contains(feature) {
            distinct.lock().unwrap().push(*feature);
        }
    });

    let count = distinct.lock().unwrap().to_vec().len() as f64;
    let count_perc = count / feature_array.len() as f64;

    Ok((count, count_perc))
}

pub fn count_missing(feature_array: &Array1<Option<f64>>) -> Result<(f64, f64), String> {
    let missing = Arc::new(Mutex::new(Vec::<f64>::new()));

    feature_array.par_iter().for_each(|feature| {
        if feature.is_none() {
            missing.lock().unwrap().push(1.0);
        }
    });

    let count = missing.lock().unwrap().to_vec().len() as f64;
    let count_perc = count / feature_array.len() as f64;

    Ok((count, count_perc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::arr1;

    #[test]
    fn test_count_distinct() {
        let test_array = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 1.0, 1.2, -1.5, 100.0]);

        let (count, count_perc) = count_distinct(&test_array).unwrap();

        assert_eq!(count, 8.0);
        assert_eq!(count_perc, 8.0 / 11.0)
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

        let (count, count_perc) = count_missing(&test_array).unwrap();

        assert!(count == 1.0);
        assert_eq!(count_perc, 1.0 / 11.0);
    }
}
