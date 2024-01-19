use nalgebra::DMatrix;

// Mean Absolute Error
pub fn mae(y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
    (y1 - y).map(|x| x.abs())
}

// First Derivate of MAE
pub fn mae_d(y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
    y1 - y
}

// Mean Squared Error
pub fn mse(y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
    let diff = y1 - y;
    diff.component_mul(&diff)
}

// First Derivate of MSE
pub fn mse_d(y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
    y1 - y
}