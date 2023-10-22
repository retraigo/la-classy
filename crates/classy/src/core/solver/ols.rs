/**
 * Ordinary Least Squares
 */

use ndarray::{Array2, Array1};
use ndarray_linalg::Inverse;

pub struct OrdinaryLeastSquares;

impl OrdinaryLeastSquares {
    pub fn train(&self, data: &Array2<f64>, targets: &Array1<f64>, silent: bool) -> Array1<f64> {
        let n = targets.len();

        let x_transpose = data.t();

        let xtx = &x_transpose * data;
        let xty = &x_transpose.dot(targets);

        let xtx_inverse= xtx.clone().inv().unwrap();
        let y_mean = targets.mean().unwrap();

        // OLS Weights = (XtX)' * XtY
        let weights = (xtx_inverse.dot(xty)).t().to_owned();

        // Calculate R2 only if `silent` is set to false
        if !silent {
            let mut sse = 0f64;
            let mut sst = 0f64;

            for i in 0..n {
                let _sse = targets[i] - weights.dot(&data.row(i));
                let _sst = targets[i] - y_mean;
                sse += _sse * _sse;
                sst += _sst * _sst;
            }
            let r2 = 1f64 - (sse / sst);
            println!("R2 Score: {}", r2);
        }
        weights
    }
}
