/**
 * Ordinary Least Squares
 */
extern crate nalgebra as na;
use na::{DMatrix, DVector};

pub struct OrdinaryLeastSquares;

impl OrdinaryLeastSquares {
    pub fn train(&self, data: &DMatrix<f64>, targets: &DVector<f64>, silent: bool) -> DVector<f64> {
        let n = targets.len();

        let y: DMatrix<f64> = DMatrix::from_row_slice(targets.len(), 1, targets.data.as_slice());
        let x_transpose = data.transpose();

        let xtx = &x_transpose * data;
        let xty = &x_transpose * y;

        let xtx_inverse = match xtx.try_inverse() {
            Some(inv) => inv,
            None => panic!("Unable to invert XtX"),
        };
        let y_mean = targets.mean();

        // OLS Weights = (XtX)' * XtY
        let weights: DMatrix<f64> = (xtx_inverse * xty).transpose();

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
        DVector::from_row_slice(weights.as_slice())
    }
}
