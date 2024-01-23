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
