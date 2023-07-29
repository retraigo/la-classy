/**
 * Ordinary Least Squares
 */
extern crate nalgebra as na;
use na::{DMatrix, DVector};

#[no_mangle]
pub unsafe extern "C" fn ordinary_least_squares(
    res_ptr: *mut f64,
    x_ptr: *const f64,
    y_ptr: *const f64,
    x_len: usize,
    y_len: usize,
    n_features: usize,
    silent: bool,
) -> f64 {
    // Get input values
    let x = std::slice::from_raw_parts(x_ptr, x_len * n_features);
    let y = std::slice::from_raw_parts(y_ptr, y_len);
    let data = DMatrix::from_row_slice(x_len, n_features, x);
    let target = DVector::from_column_slice(y);

    let n = x_len; // The TS part should take care of making sure lengths are equal

    let x_transpose = data.transpose();

    let xtx = &x_transpose * &data;
    let xty = &x_transpose * &target;

    let xtx_inverse = match xtx.try_inverse() {
        Some(inv) => inv,
        None => panic!("Unable to invert XtX"),
    };
    let y_mean = target.mean();

    // OLS Weights = (XtX)' * XtY
    let weights = (xtx_inverse * xty).transpose();
    // Intercept = y_mean - y_predicted_from_mean
    let intercept = y_mean - weights.dot(&data.row_mean());

    // Calculate R2 only if `silent` is set to false
    if !silent {
        let mut sse = 0f64;
        let mut sst = 0f64;

        for i in 0..n {
            let _sse = y[i] - weights.dot(&data.row(i)) + intercept;
            let _sst = y[i] - y_mean;
            sse += _sse * _sse;
            sst += _sst * _sst;
        }
        let r2 = 1f64 - (sse / sst);
        println!("R2 Score: {}", r2);
    }
    let out = std::slice::from_raw_parts_mut(res_ptr, n_features);
    for i in 0..weights.ncols() {
        out[i] = *weights.get_unchecked(i);
    }
    intercept
}
