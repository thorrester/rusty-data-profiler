use ndarray::{ArrayBase, ViewRepr};
use numpy::ndarray::{Array1, Array2};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::collections::HashMap;
use std::io;

#[derive(Debug)]
enum GradType {
    Array2(Array2<f64>),
    F64(f64),
}

fn sigmoid(a: &f64) -> f64 {
    1.0 / (1.0 + (-a).exp())
}

/// Initialize weights for logistic regression
fn initialize_weights(dim: usize) -> Array2<f64> {
    let mut theta = Array2::<f64>::zeros((dim, 1));
    theta
}

pub fn accuracy_score(preds: &Vec<f64>, targets: &Vec<f64>) -> f64 {
    let errors = preds
        .par_iter()
        .zip(targets.par_iter())
        .map(|(pred, target)| (pred - target).abs())
        .collect::<Vec<f64>>();

    let total_error: f64 = errors.par_iter().sum();
    let accuracy = 1.0 - total_error / preds.len() as f64;
    accuracy
}

struct LogisticRegression {
    max_iterations: i32,
    learning_rate: f64,
    pub weights: Array2<f64>,
}

impl LogisticRegression {
    pub fn new() -> LogisticRegression {
        // create empty weights
        let weights = Array2::<f64>::zeros((1, 1));

        return LogisticRegression {
            max_iterations: 2,
            learning_rate: 1.0,
            weights: weights,
        };
    }

    /// Get prediction hypothesis
    ///
    /// # Arguments
    ///
    /// * `weight` - A 2d array of weights
    /// * `bias` - float64
    /// * `feature_array` - A 2d array of features
    ///
    /// # Returns
    /// * `Array2<f64>` - A 2d array of scores
    fn hypothesis(&mut self, feature_array: &Array2<f64>, weights: &Array2<f64>) -> Vec<f64> {
        println!("{:?}", feature_array.shape());
        println!("{:?}", weights.shape());
        println!("{:?}", feature_array.dot(weights).shape());
        let scores = feature_array
            .dot(weights)
            .into_raw_vec()
            .par_iter()
            .map(|x| sigmoid(x))
            .collect::<Vec<f64>>();
        // println!("{:?}", scores);
        scores
    }

    pub fn fit(&mut self, feature_array: &Array2<f64>, target: &Vec<f64>) {
        let mut weights = initialize_weights(feature_array.ncols());
        let dim = feature_array.nrows() as f64;
        // predict on feature array and update weights
        for _ in 0..self.max_iterations {
            let model_error = target
                .par_iter()
                .zip(self.hypothesis(feature_array, &weights).par_iter())
                .map(|(target, yhat)| target - yhat)
                .collect::<Vec<f64>>();

            println!("{:?}", model_error);
            let gradients = feature_array.t().dot(&Array1::from_vec(model_error)) / dim;

            weights = weights - (self.learning_rate * gradients);
        }
        self.weights = weights;
    }

    //pub fn predict(&mut self, feature_array: &Array2<f64>) -> Result<Vec<f64>, io::Error> {
    //    let scores = self.hypothesis(feature_array, &self.weights);
    //    Ok(scores)
    //}
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::{Normal, Uniform};
    use ndarray_rand::RandomExt;
    use numpy::ndarray::{Array1, Array2};
    use tokio;

    #[tokio::test]
    async fn test_log_reg() {
        let mut test_array = Array2::random((20, 1), Normal::new(0.0, 1.0).unwrap());
        let intercept = Array1::ones(20);
        test_array.push_column(intercept.view()).unwrap();

        let target = Array1::random(20, Uniform::new(0, 1))
            .to_vec()
            .par_iter()
            .zip(test_array.clone().into_raw_vec().par_iter())
            .map(|(_x, y)| if *y as f64 >= 1.0 { 1.0 } else { 0.0 })
            .collect::<Vec<f64>>();

        let mut log_reg = LogisticRegression::new();
        log_reg.fit(&test_array, &target.to_vec());
        //let predictions = log_reg.predict(&mut test_array).unwrap();

        //let acc = accuracy_score(&predictions, &target.to_vec());

        assert_eq!(10, 11);
    }
}
