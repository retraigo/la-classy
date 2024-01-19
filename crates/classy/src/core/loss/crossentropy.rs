use nalgebra::DMatrix;

const EPSILON: f64 = 1e-15;

pub fn bin_cross_entropy(y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
    y.zip_map(y1, |y_i, y1_i| {
        // Clipping to avoid log(0)
        let clipped_y1_i = y1_i.max(EPSILON).min(1.0 - EPSILON);
        let clipped_y_i = y_i.max(EPSILON).min(1.0 - EPSILON);
        -(clipped_y_i * clipped_y1_i.ln() + (1.0 - clipped_y_i) * (1.0 - clipped_y1_i).ln())
    })
}

pub fn bin_cross_entropy_d(y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
    y1 - y
}

pub fn cross_entropy(y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
    DMatrix::from_row_slice(
        y.nrows(),
        1,
        (-y.component_mul(&y1.map(|x| x.max(EPSILON).min(1f64 - EPSILON).ln())))
            .column_sum()
            .as_slice(),
    )
}

pub fn cross_entropy_d(y: &DMatrix<f64>, y1: &DMatrix<f64>) -> DMatrix<f64> {
    y1 - y
}
