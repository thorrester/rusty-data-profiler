use ndarray::{ArrayBase, ViewRepr};
use numpy::ndarray::{Array1, Array2};
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
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
fn initialize_theta(dim: usize) -> Array2<f64> {
    let mut theta = Array2::<f64>::zeros((dim, 1));
    theta
}

pub fn accuracy_score(preds: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    let mut errors = preds - targets;
    errors.par_map_inplace(|x| *x = x.abs());

    let total_error = errors.sum() as f64;
    let accuracy = 1.0 - total_error / preds.len() as f64;
    accuracy
}

struct LogisticRegression {
    max_iterations: i32,
    alpha: f64,
    pub theta: Array2<f64>,
    pub cost: Vec<f64>,
}

impl LogisticRegression {
    pub fn new() -> LogisticRegression {
        // create empty weights
        let theta = Array2::<f64>::zeros((1, 1));

        return LogisticRegression {
            max_iterations: 2,
            alpha: 1.0,
            theta: theta,
            cost: Vec::new(),
        };
    }

    /// Calculate the gradient for a set of predictions and targets
    ///
    /// # Arguments
    ///
    /// * `hypothesis` - A 2d array of predictions
    /// * `feature_array` - A 2d array of features
    /// * `targets` - A 1d array of targets
    ///
    /// # Returns
    /// * `HashMap<String, GradType>` - A hashmap of gradients
    fn calculate_gradient(
        &mut self,
        hypothesis: &Array2<f64>,
        feature_array: &Array2<f64>,
        target: &Array1<f64>,
        dim1: f64,
    ) -> Result<HashMap<String, GradType>, io::Error> {
        let error = (hypothesis - target);
        let dw = feature_array.t().dot(&error.t()) / dim1;
        let db = (hypothesis - target).sum() / dim1;

        let grads = HashMap::from([
            ("dw".to_string(), GradType::Array2(dw)),
            ("db".to_string(), GradType::F64(db)),
        ]);

        Ok(grads)
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
    fn hypothesis(&mut self, feature_array: &Array2<f64>) -> Array2<f64> {
        let mut scores = feature_array.dot(&self.theta);
        scores.par_map_inplace(|x| *x = sigmoid(x));
        println!("scores: {:?}", scores);
        scores
    }

    /// Compute the cost for a set of predictions and targets
    ///
    /// # Arguments
    ///
    /// * `hypothesis` - A 2d array of predictions
    /// * `targets` - A 1d array of targets
    ///
    /// # Returns
    /// * `f64` - A float64 of the cost
    fn compute_cost(
        &mut self,
        hypothesis: Array2<f64>,
        targets: &Array1<f64>,
        dim0: f64,
    ) -> Result<(f64, Vec<f64>), io::Error> {
        let ln_pred: Vec<f64> = hypothesis
            .into_raw_vec()
            .par_iter()
            .map(|x| x.ln_1p())
            .collect();
        //let left_side = targets * ln_pred;
        //let right_side = (1.0 - targets) * (1.0 - hypothesis).mapv(|x| x.ln_1p());
        //let cost = (-1.0 / dim0 as f64) * (left_side + right_side).sum();
        //let gradient = (1.0 / dim0) * (hypothesis - targets).t().dot(&(hypothesis - targets));

        Ok((1.0, ln_pred))
    }

    pub fn fit(&mut self, feature_array: &Array2<f64>, target: &Array1<f64>) {
        self.theta = initialize_theta(feature_array.ncols());
        let dim0 = feature_array.nrows() as f64;

        // predict on feature array and update weights
        for _ in 0..self.max_iterations {
            let hypoth = self.hypothesis(feature_array);
            let (cost, gradients) = self
                .compute_cost(hypoth, target, dim0)
                .expect("Failed to compute cost");
            //self.theta = &self.theta - self.alpha * gradients;
            //self.cost.push(cost);
            //println!("theta: {:?}", self.theta);
        }

        println!("cost: {:?}", self.cost);
    }
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
        let mut test_array = Array2::random((1_000_000, 1), Normal::new(0.0, 1.0).unwrap());
        let intercept = Array1::ones(1_000_000);
        test_array.push_column(intercept.view()).unwrap();

        let target =
            Array1::random(10_000, Uniform::new(0, 1))
                .map(|x| if *x as i32 >= 1 { 1.0 } else { 0.0 });

        let mut log_reg = LogisticRegression::new();
        log_reg.fit(&test_array, &target);
        //let predictions = log_reg.predict(&mut test_array).unwrap();

        //let acc = accuracy_score(&predictions, &target);

        //println!("{:?}", acc);

        assert_eq!(10, 11);
    }
}
