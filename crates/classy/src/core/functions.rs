use nalgebra::DVector;

pub fn softmax(logits: &DVector<f64>) -> DVector<f64> {
    // Calculate the exponential of each element in the vector
    let logit_exps: DVector<f64> = logits.map(|val| val.exp());

    // Calculate the sum of exponential values
    let sum_exp: f64 = logit_exps.sum();

    // Normalize by dividing each element by the sum
    logit_exps / sum_exp
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}