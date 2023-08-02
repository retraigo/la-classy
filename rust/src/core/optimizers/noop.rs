use nalgebra::DVector;

#[derive(Debug)]
pub struct NoOptimizer;

impl NoOptimizer {
    pub fn new() -> NoOptimizer {
        NoOptimizer
    }
    pub fn optimize(
        &mut self,
        weights: &mut DVector<f64>,
        gradient: DVector<f64>,
        learning_rate: f64,
        l1: DVector<f64>,
    ) {
        *weights -= (gradient + l1) * learning_rate;
    }
}
