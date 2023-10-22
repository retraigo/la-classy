use ndarray::Array1;

// Mean Absolute Error
pub fn mae(y: &Array1<f64>, y1: &Array1<f64>) -> Array1<f64> {
    (y1 - y).map(|x| x.abs())
}

// First Derivate of MAE
pub fn mae_d(y: &Array1<f64>, y1: &Array1<f64>) -> Array1<f64> {
    y1 - y
}

// Mean Squared Error
pub fn mse(y: &Array1<f64>, y1: &Array1<f64>) -> Array1<f64> {
    let diff = y1 - y;
    diff.clone() * &diff
}

// First Derivate of MSE
pub fn mse_d(y: &Array1<f64>, y1: &Array1<f64>) -> Array1<f64> {
    y1 - y
}