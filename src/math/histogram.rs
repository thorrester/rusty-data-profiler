use numpy::ndarray::Zip;
use numpy::ndarray::{arr2, Array1, Array2};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

// create a function in rust that computes the bins of a given vector
pub fn compute_bins(data: &Array1<f64>, num_bins: i32) -> Vec<f64> {
    // find the min and max of the data

    let min = data.fold(f64::INFINITY, |a, &b| a.min(b));
    let max = data.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // create a vector of bins
    let mut bins = Vec::with_capacity(num_bins as usize);

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

pub async fn compute_bin_counts_from_2d_array(array_data: &Array2<f64>, num_bins: i32) {
    // need features as an arg
    // build hashmap of features, bins and bin counts

    let bin_vec = Arc::new(Mutex::new(Vec::new()));

    //let bin_vec = Arc::new(Mutex::new(vec![
    //    vec![0; num_bins as usize];
    //    array_data.ncols() as f64 as usize
    //]));

    let columns = array_data.columns().into_iter().collect::<Vec<_>>();

    columns.par_iter().for_each(|column| {
        let bins = compute_bins(&column.into_owned(), num_bins);
        bin_vec.lock().unwrap().push(bins);
    });

    println!("{:?}", bin_vec.lock().unwrap().len());
    println!("{:?}", bin_vec.lock().unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use tokio;

    #[tokio::test]
    async fn test_compute_bins() {
        let test_array = Array1::random(10000, Uniform::new(0., 10.));

        let bins = compute_bins(&test_array, 10);

        let counts = compute_bin_counts(&test_array, &bins);
        assert_eq!(10000, counts.iter().sum::<i32>());
    }

    #[tokio::test]
    async fn test_compute_bin_counts_from_2d_array() {
        let test_array = Array2::random((1000, 10), Uniform::new(0., 10.));
        let bins = 10;

        compute_bin_counts_from_2d_array(&test_array, bins).await;
        assert_eq!(10, 11)
    }
}
