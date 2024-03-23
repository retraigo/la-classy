use ndarray::{Array1, Array2};

// Mean Absolute Error
pub fn mae(y: &Array2<f64>, y1: &Array2<f64>) -> Array2<f64> {
    (y1 - y).map(|x| x.abs())
}

// First Derivate of MAE
pub fn mae_d(y: &Array2<f64>, y1: &Array2<f64>) -> Array2<f64> {
    (y1 - y).map(|x| x.signum())
}

// Mean Squared Error
pub fn mse(y: &Array2<f64>, y1: &Array2<f64>) -> Array2<f64> {
    let diff = y1 - y;
    &diff * &diff
}

// First Derivate of MSE
pub fn mse_d(y: &Array2<f64>, y1: &Array2<f64>) -> Array2<f64> {
    y1 - y
}

pub fn huber(y: &Array2<f64>, y1: &Array2<f64>, delta: f64) -> Array2<f64> {
    let loss: Array1<f64> = y1
        .iter()
        .zip(y.iter())
        .map(|(y1_i, y_i)| {
            let residual = y_i - y1_i;
            if residual.abs() <= delta {
                0.5 * residual.powi(2)
            } else {
                delta * (residual.abs() - 0.5 * delta)
            }
        })
        .collect();
    loss.to_shape((y.nrows(), y.ncols())).unwrap().to_owned()
}

pub fn huber_d(y: &Array2<f64>, y1: &Array2<f64>, delta: f64) -> Array2<f64> {
    let gradient: Array1<f64> = y1
        .iter()
        .zip(y.iter())
        .map(|(y1_i, y_i)| {
            let residual = y_i - y1_i;
            if residual.abs() <= delta {
                -residual
            } else {
                -delta * residual.signum()
            }
        })
        .collect();
    gradient.to_shape((y.nrows(), y.ncols())).unwrap().to_owned()
}

pub fn tukey(y: &Array2<f64>, y1: &Array2<f64>, c: f64) -> Array2<f64> {
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

pub fn tukey_d(y: &Array2<f64>, y1: &Array2<f64>, c: f64) -> Array2<f64> {
    (y1 - y).map(|el| {
        let r = el.abs();
        if r <= c {
            r * (1.0 - ((r / c).powi(2))).powi(2)
        } else {
            0f64
        }
    })
}
