use ndarray::Array1;

pub fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    // Calculate the exponential of each element in the vector
    let logit_exps: Array1<f32> = logits.map(|val| val.exp());

    // Calculate the sum of exponential values
    let sum_exp: f32 = logit_exps.sum();

    // Normalize by dividing each element by the sum
    logit_exps / sum_exp
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}