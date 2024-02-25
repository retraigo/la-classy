use nalgebra::DMatrix;

// Mean Absolute Error
pub fn mae(y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
    (y1 - y).map(|x| x.abs())
}

// First Derivate of MAE
pub fn mae_d(y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
    (y1 - y).map(|x| x.signum())
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

pub fn huber(y: &DMatrix<f64>, y1: &DMatrix<f64>, delta: f64) -> DMatrix<f64> {
    let mut loss = DMatrix::zeros(y.nrows(), y.ncols());
    for i in 0..y.len() {
        let residual = y[i] - y1[i];
        if residual.abs() <= delta {
            loss[i] = 0.5 * residual.powi(2);
        } else {
            loss[i] = delta * (residual.abs() - 0.5 * delta);
        }
    }
    loss
}

pub fn huber_d(y: &DMatrix<f64>, y1: &DMatrix<f64>, delta: f64) -> DMatrix<f64> {
    let mut gradient = DMatrix::zeros(y.nrows(), y.ncols());
    for i in 0..y.len() {
        let residual = y[i] - y1[i];
        if residual.abs() <= delta {
            gradient[i] = -residual;
        } else {
            gradient[i] = -delta * residual.signum();
        }
    }
    gradient
}
