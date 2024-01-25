/**
 * Ordinary Least Squares
 */
extern crate nalgebra as na;
use na::DMatrix;

pub struct OrdinaryLeastSquares;

impl OrdinaryLeastSquares {
    pub fn train(&self, data: &DMatrix<f64>, targets: &DMatrix<f64>, silent: bool) -> DMatrix<f64> {
        let y_mean = targets.row_mean();
        let mut sst = 0f64;
        for row in targets.row_iter() {
            sst += (row - &y_mean).map(|x| x.powi(2)).sum();
        }

        let x_transpose = data.transpose();

        let xtx = &x_transpose * data;
        let xty = &x_transpose * targets;

        let xtx_inverse = match xtx.try_inverse() {
            Some(inv) => inv,
            None => panic!("Unable to invert XtX"),
        };
        // OLS Weights = (XtX)' * XtY
        let weights: DMatrix<f64> = xtx_inverse * xty;

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
    pub fn predict(&self, data: &DMatrix<f64>, weights: &DMatrix<f64>) -> DMatrix<f64> {
        data * weights
    }
}
