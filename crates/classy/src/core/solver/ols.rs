/**
 * Ordinary Least Squares
 */
extern crate nalgebra as na;
use ndarray::{Array2, Axis};

pub struct OrdinaryLeastSquares;

impl OrdinaryLeastSquares {
    pub fn train(&self, data: &Array2<f64>, targets: &Array2<f64>, silent: bool) -> Array2<f64> {
        let y_mean = targets.mean_axis(Axis(0));
        let mut sst = 0f64;
        for row in targets.axis_iter(Axis(0)) {
            sst += (row - &y_mean).map(|x| x.powi(2)).sum();
        }

        let x_transpose = data.t();

        let xtx = &x_transpose.dot(data);
        let xty = &x_transpose.dot(targets);

        let xtx_inverse = match xtx.try_inverse() {
            Some(inv) => inv,
            None => panic!("Unable to invert XtX"),
        };
        // OLS Weights = (XtX)' * XtY
        let weights: Array2<f64> = xtx_inverse * xty;

        // Calculate R2 only if `silent` is set to false
        if !silent {
            let sse = (targets - (data * &weights))
                .map(|x| x.powi(2))
                .sum();
            let r2 = 1f64 - (sse / sst);
            println!("R2 Score: {}", r2);
        }

        weights
    }
    pub fn predict(&self, data: &Array2<f64>, weights: &Array2<f64>) -> Array2<f64> {
        data * weights
    }
}
