use nalgebra::DVector;

pub fn mean_absolute_error(y: &DVector<f64>, y1: &DVector<f64>) -> f64 {
    let diff = (y - y1).map(|x| x.abs());
    diff.sum() / y.len() as f64
}

pub fn mean_squared_error(y: &DVector<f64>, y1: &DVector<f64>) -> f64 {
    let diff = y - y1;
    diff.dot(&diff) / y.len() as f64
}

pub fn root_mean_squared_error(y: &DVector<f64>, y1: &DVector<f64>) -> f64 {
    mean_squared_error(y, y1).sqrt()
}

pub fn binary_cross_entropy(y: &DVector<f64>, y1: &DVector<f64>) -> f64 {
    let epsilon = 1e-15;
    y.zip_map(y1, |y_i, y1_i| {
        // Clipping to avoid log(0)
        let clipped_y1_i = y1_i.max(epsilon).min(1.0 - epsilon);
        -(y_i * clipped_y1_i.ln() + (1.0 - y_i) * (1.0 - clipped_y1_i).ln())
    })
    .sum()
        / y.len() as f64
}
pub fn cross_entropy(y: &DVector<f64>, y1: &DVector<f64>) -> f64 {
    let epsilon = 1e-15;
    let clipped_y1 = y1.map(|x| x.max(epsilon).min(1.0 - epsilon));
    -y.component_mul(&clipped_y1.map(|x| x.ln())).sum()
}
