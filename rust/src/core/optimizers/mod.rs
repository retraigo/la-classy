use nalgebra::DVector;

pub mod adam;
pub mod noop;

pub enum Optimizer {
    Adam(adam::AdamOptimizer),
    NoOptimizer(noop::NoOptimizer),
}

impl Optimizer {
    pub fn optimize(
        &mut self,
        weights: &mut DVector<f64>,
        gradient: DVector<f64>,
        learning_rate: f64,
        l1: DVector<f64>,
    ) {
        match self {
            Self::Adam(adam) => adam.optimize(weights, gradient, learning_rate, l1),
            Self::NoOptimizer(noop) => noop.optimize(weights, gradient, learning_rate, l1)
        }
    }
}
