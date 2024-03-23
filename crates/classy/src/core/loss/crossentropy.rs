use ndarray::{Array1, Array2, Axis};

const EPSILON: f64 = 1e-15;

pub fn bin_cross_entropy(y: &Array2<f64>, y1: &Array2<f64>) -> Array2<f64> {
    let res: Array1<f64> = y1
        .iter()
        .zip(y.iter())
        .map(|(y1_i, y_i)| {
            // Clipping to avoid log(0)
            let clipped_y1_i = y1_i.max(EPSILON).min(1.0 - EPSILON);
            let clipped_y_i = y_i.max(EPSILON).min(1.0 - EPSILON);
            -(clipped_y_i * clipped_y1_i.ln() + (1.0 - clipped_y_i) * (1.0 - clipped_y1_i).ln())
        })
        .collect();
    res.to_shape((y.nrows(), y.ncols()))
        .unwrap()
        .to_owned()
}

pub fn bin_cross_entropy_d(y: &Array2<f64>, y1: &Array2<f64>) -> Array2<f64> {
    y1 - y
}

pub fn cross_entropy(y: &Array2<f64>, y1: &Array2<f64>) -> Array2<f64> {
    (-y * (&y1.map(|x| x.max(EPSILON).min(1f64 - EPSILON).ln())))
        .sum_axis(Axis(1))
        .to_shape((y.nrows(), 1))
        .unwrap()
        .to_owned()
}

pub fn cross_entropy_d(y: &Array2<f64>, y1: &Array2<f64>) -> Array2<f64> {
    y1 - y
}
