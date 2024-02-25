use nalgebra::DMatrix;

pub fn hinge(y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
    y.zip_map(y1, |y_i, y1_i| {
        let margin = 1.0 - y_i * y1_i;
        margin.max(0.0)
    })
}

pub fn hinge_d(y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
    y.zip_map(y1, |y_i, y1_i| {
        let margin = 1.0 - y_i * y1_i;
        if margin > 0.0 {
            -y_i
        } else {
            0.0
        }
    })
}

pub fn smooth_hinge(y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
    y.zip_map(y1, |y_i, y1_i| {
        let margin = y_i * y1_i;
        if margin > -1f64 {
            (1.0 - margin).max(0.0)
        } else {
            -4f64 * margin
        }
    })
}

pub fn smooth_hinge_d(y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
    y.zip_map(y1, |y_i, y1_i| {
        let margin = y_i * y1_i;
        if margin > -1f64 {
            -y_i
        } else {
            -4f64 * y_i
        }
    })
}
