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

pub fn tukey(y: &DMatrix<f64>, y1: &DMatrix<f64>, c: f64) -> DMatrix<f64> {
    let c_squared = c * c / 6.0;
    (y1 - y).map(|el| {
        let r = el.abs();
        if r <= c {
            c_squared * (1.0 - (1.0 - (r / c).powi(2)).powi(3))
        } else {
            c_squared
        }
    })
}

pub fn tukey_d(y: &DMatrix<f64>, y1: &DMatrix<f64>, c: f64) -> DMatrix<f64> {
    (y1 - y).map(|el| {
        let r = el.abs();
        if r <= c {
            let t = r / c;
            let t_squared_complement = 1.0 - (t.powi(2));
            t_squared_complement.powi(2) * r
        } else {
            0f64
        }
    })
}
