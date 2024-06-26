use ndarray::{Array1, Array2};

pub fn hinge(y: &Array2<f32>, y1: &Array2<f32>) -> Array2<f32> {
    y1.iter()
        .zip(y.iter())
        .map(|(y1_i, y_i)| {
            let margin = 1.0 - y_i * y1_i;
            margin.max(0.0)
        })
        .collect::<Array1<f32>>()
        .to_shape((y.nrows(), y.ncols()))
        .unwrap()
        .to_owned()
}

pub fn hinge_d(y: &Array2<f32>, y1: &Array2<f32>) -> Array2<f32> {
    y1.iter()
        .zip(y.iter())
        .map(|(y1_i, y_i)| {
            let margin = 1.0 - y_i * y1_i;
            if margin > 0.0 {
                -y_i
            } else {
                0.0
            }
        })
        .collect::<Array1<f32>>()
        .to_shape((y.nrows(), y.ncols()))
        .unwrap()
        .to_owned()
}

pub fn smooth_hinge(y: &Array2<f32>, y1: &Array2<f32>) -> Array2<f32> {
    y1.iter()
        .zip(y.iter())
        .map(|(y1_i, y_i)| {
            let margin = y_i * y1_i;
            if margin > -1f32 {
                (1.0 - margin).max(0.0)
            } else {
                -4f32 * margin
            }
        })
        .collect::<Array1<f32>>()
        .to_shape((y.nrows(), y.ncols()))
        .unwrap()
        .to_owned()
}

pub fn smooth_hinge_d(y: &Array2<f32>, y1: &Array2<f32>) -> Array2<f32> {
    y1.iter()
        .zip(y.iter())
        .map(|(y1_i, y_i)| {
            let margin = y_i * y1_i;
            if margin > -1f32 {
                -y_i
            } else {
                -4f32 * y_i
            }
        })
        .collect::<Array1<f32>>()
        .to_shape((y.nrows(), y.ncols()))
        .unwrap()
        .to_owned()
}
