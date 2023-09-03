use nalgebra::DVector;

// Mean Absolute Error
pub fn mae(y: &DVector<f64>, y1: &DVector<f64>) -> DVector<f64> {
    (y1 - y).map(|x| x.abs())
}

// First Derivate of MAE
pub fn mae_d(y: &DVector<f64>, y1: &DVector<f64>) -> DVector<f64> {
    y1 - y
}

// Mean Squared Error
pub fn mse(y: &DVector<f64>, y1: &DVector<f64>) -> DVector<f64> {
    let diff = y1 - y;
    diff.component_mul(&diff)
}

// First Derivate of MSE
pub fn mse_d(y: &DVector<f64>, y1: &DVector<f64>) -> DVector<f64> {
    y1 - y
}