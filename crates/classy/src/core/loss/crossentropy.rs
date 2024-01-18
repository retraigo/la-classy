use nalgebra::DVector;

const EPSILON: f64 = 1e-15;

pub fn bin_cross_entropy(y: &DVector<f64>, y1: &DVector<f64>) -> DVector<f64> {
    y.zip_map(y1, |y_i, y1_i| {
        // Clipping to avoid log(0)
        let clipped_y1_i = y1_i.max(EPSILON).min(1.0 - EPSILON);
        let clipped_y_i = y_i.max(EPSILON).min(1.0 - EPSILON);
        -(clipped_y_i * clipped_y1_i.ln() + (1.0 - clipped_y_i) * (1.0 - clipped_y1_i).ln())
    })
}

pub fn bin_cross_entropy_d(y: &DVector<f64>, y1: &DVector<f64>) -> DVector<f64> {
    y1 - y
}
