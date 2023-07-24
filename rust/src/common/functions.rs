pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
pub fn mean_squared_error(y: &[f64], y1: &[f64]) -> f64 {
    (0..y.len()).map(|i| (y[i] - y1[i]).powi(2)).sum::<f64>() / y.len() as f64
}
