pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}